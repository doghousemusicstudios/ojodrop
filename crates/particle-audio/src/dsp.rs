//! DSP worker (spec §1/§5). Runs on a dedicated background thread, drains the
//! capture ring into a sliding analysis window, and per-hop computes the full
//! [`Features`] set, then publishes a snapshot through the lock-free triple buffer.
//!
//! Cached once, never re-allocated per hop (the classic perf pitfall the spec
//! warns about): the FFT plan, its scratch + spectrum buffers, the Hann window,
//! and both filterbanks.

#[cfg(feature = "capture")]
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
#[cfg(feature = "capture")]
use std::time::Duration;

use realfft::num_complex::Complex;
use realfft::{RealFftPlanner, RealToComplex};
#[cfg(feature = "capture")]
use rtrb::Consumer;

use crate::analysis::{
    self, FilterBank, MACRO_BANDS_HZ, SPECTRUM_BANDS, SPECTRUM_HI_HZ, SPECTRUM_LO_HZ,
};
use crate::butterchurn::{self, ButterchurnLevels};
use crate::complex_onset::ComplexOnsetDetector;
use crate::hpss::{HpssHistory, HpssSeparator};
use crate::hpss_bus::HpssBus;
use crate::linkwitz_riley::LinkwitzRileyBank;
use crate::onset::OnsetDetector;
use crate::predictive_drop::DropPredictor;
use crate::smoothing::{flush_denormal, Agc, AsymEnv, OnePole, ReactiveLevel, SilenceGate};
#[cfg(feature = "capture")]
use crate::spectrogram::SpectrogramPublisher;
use crate::spectrogram::SpectrogramTrail;
use crate::structure::StructureTracker;
use crate::tempo::TempoTracker;
use crate::tonal;
#[cfg(feature = "capture")]
use crate::triple_buffer::Writer;
use crate::{
    CaptureFrame, Features, CHROMA_BINS, FREQ_SPECTRUM_BINS, WAVEFORM_SAMPLES,
    WAVEFORM_SAMPLES_FULL,
};

/// FFT window size and hop. 2048 @ ~48k ≈ 43 ms window; hop 512 → 75% overlap,
/// ~10.7 ms between frames (well under the <50 ms glass-to-light target).
pub const FFT_LEN: usize = 2048;
pub const HOP: usize = 512;

/// Butterchurn band reactivity edges in Hz (audioLevels.js): bass 20-320,
/// mid 320-2800, treb 2800-11025. Deliberately wider than the 6 macro bands.
const REACT_BAND_EDGES_HZ: [(f32, f32); 3] = [(20.0, 320.0), (320.0, 2800.0), (2800.0, 11025.0)];

/// The three FFT macro bands actually consumed downstream: `sub_bass` (index 0),
/// `low_mid` (index 2), and `presence` (index 4) from [`MACRO_BANDS_HZ`]. The
/// `bass`/`mid`/`air` lanes (1/3/5) were computed and normalized every hop but
/// their outputs were discarded — those `Features` fields carry the
/// Butterchurn-normalized rails instead — so we build and normalize only these
/// three (P2-AUD-025).
const USED_MACRO_BANDS_HZ: [(f32, f32); 3] =
    [MACRO_BANDS_HZ[0], MACRO_BANDS_HZ[2], MACRO_BANDS_HZ[4]];

#[inline]
fn finite_or_zero(value: f32) -> f32 {
    if value.is_finite() {
        value
    } else {
        0.0
    }
}

#[inline]
fn sanitize_pcm_sample(value: f32) -> f32 {
    finite_or_zero(value).clamp(-1.0, 1.0)
}

/// Parameters the worker needs that come from the live capture device.
#[cfg(feature = "capture")]
pub struct DspParams {
    pub sample_rate: u32,
    /// User detection sensitivity (~0.1..3). 1.0 = neutral.
    pub sensitivity: f32,
}

/// Run the DSP loop until `running` is cleared (or the producer is abandoned —
/// i.e. the capture stream/engine was dropped).
#[cfg(feature = "capture")]
pub fn run(
    mut consumer: Consumer<CaptureFrame>,
    mut writer: Writer<Features>,
    mut spectrogram: SpectrogramPublisher,
    params: DspParams,
    running: Arc<AtomicBool>,
    overruns: Arc<AtomicU64>,
) {
    // Anti-alias + decimate the native stream down to a canonical analysis rate so
    // the fixed FFT/hop geometry and every seconds-derived history stay rate-stable
    // (see `crate::resample`). At 44.1/48 kHz the factor is 1 (passthrough).
    let native_rate = params.sample_rate as f32;
    let analysis_rate = crate::resample::effective_rate(native_rate);
    let hop_dt = HOP as f32 / analysis_rate;
    let mut decimator = crate::resample::FrameDecimator::new(native_rate);

    let mut state = DspState::new(analysis_rate, hop_dt, params.sensitivity);

    // Sliding windows: mono feeds the existing FFT/onsets; left/right are kept for
    // sampled PCM scope/audio-texture consumers.
    let mut window = vec![0.0f32; FFT_LEN];
    let mut window_left = vec![0.0f32; FFT_LEN];
    let mut window_right = vec![0.0f32; FFT_LEN];
    // Staging area for newly drained samples before they enter the window.
    let mut pending: Vec<CaptureFrame> = Vec::with_capacity(HOP * 4);
    // Scratch for the frames drained this pass, held separately from `pending` so
    // `drain_pass` can fold in a capture discontinuity BEFORE they enter the window
    // (P2-AUD-008). Reused every pass; `clear` keeps the allocation.
    let mut incoming: Vec<CaptureFrame> = Vec::with_capacity(HOP * 4);

    // ~hop cadence; if the ring under-runs we simply wait and retry.
    let poll = Duration::from_micros((hop_dt * 1e6 * 0.5) as u64).max(Duration::from_micros(500));

    // Capture-ring overrun watermark. When the realtime callback drops frames on a
    // full ring it bumps this counter; an increase means the next samples to arrive
    // are non-adjacent to what we already drained (P2-AUD-008).
    let mut last_overruns = overruns.load(Ordering::Relaxed);

    while running.load(Ordering::Relaxed) {
        if consumer.is_abandoned() && consumer.slots() == 0 {
            // Capture gone and nothing left to process.
            break;
        }

        // Drain everything currently available (cheap; pop is lock-free), feeding
        // each frame through the anti-alias decimator on the way in. Staged into
        // `incoming` (not straight into `pending`) so `drain_pass` can honor a
        // capture discontinuity BEFORE these frames enter the sliding window.
        incoming.clear();
        while let Ok(s) = consumer.pop() {
            if let Some(frame) = decimator.push(s) {
                incoming.push(frame);
            }
        }

        // Fold this pass into the window one whole hop at a time. `drain_pass`
        // first honors any capture-ring discontinuity (P2-AUD-008): if the realtime
        // callback dropped frames since the last pass it clears the stale pre-gap
        // leftover BEFORE `incoming` is staged, so no hop can straddle the gap. It
        // then emits every whole hop through the closure below.
        last_overruns = drain_pass(
            &mut pending,
            &incoming,
            last_overruns,
            overruns.load(Ordering::Relaxed),
            |hop| {
                // Shift window left by HOP, append the next HOP samples.
                window.copy_within(HOP.., 0);
                window_left.copy_within(HOP.., 0);
                window_right.copy_within(HOP.., 0);
                let tail = FFT_LEN - HOP;
                let mut hop_mono = [0.0f32; HOP];
                for (i, frame) in hop.iter().enumerate() {
                    let mono = sanitize_pcm_sample(frame.mono);
                    window[tail + i] = mono;
                    window_left[tail + i] = sanitize_pcm_sample(frame.left);
                    window_right[tail + i] = sanitize_pcm_sample(frame.right);
                    hop_mono[i] = mono;
                }

                let features = state.analyze_hop(&window, &hop_mono, &window_left, &window_right);
                writer.write(features);
                // Publish the scrolling-spectrogram trail for the render thread. The
                // trail was advanced inside `analyze_hop`; the publisher fills a
                // recycled immutable page off-lock and swaps it in with a single Arc
                // store — no full-ring copy under the lock, no per-hop allocation.
                spectrogram.publish(&state.spectrogram);
            },
        );

        if pending.len() < HOP {
            std::thread::sleep(poll);
        }
    }

    log::info!("particle-audio: DSP worker exiting");
}

/// Fold one worker pass of freshly drained `incoming` frames into `pending` and
/// emit every whole hop through `on_hop`, honoring capture-ring discontinuities
/// (P2-AUD-008).
///
/// The discontinuity check runs FIRST, before `incoming` is staged: if the
/// realtime capture callback dropped frames since the previous pass
/// (`now_overruns` advanced past `last_overruns`), the sub-hop leftover already in
/// `pending` is a stale PRE-gap tail — non-adjacent to the post-gap frames about
/// to arrive — so it is dropped here. Clearing it before the append (rather than
/// after the hop split, as the old code did) is what guarantees a pre-gap tail and
/// a post-gap head can never be spliced into one false-continuous hop: with the
/// old post-drain ordering, a post-gap head pushed during the same drain that
/// consumed the pre-gap tail would already have been analyzed as one straddling
/// hop before the flush ran. Only the sub-hop `pending` leftover is cleared; the
/// caller's sliding FFT window ages the pre-gap samples out on its own, so it is
/// left untouched (hard-zeroing it would inject a click).
///
/// With no discontinuity (`now_overruns == last_overruns`) the leftover is
/// retained and combined with `incoming` exactly as before — the steady-state path
/// is byte-identical. Returns the watermark to carry into the next pass.
#[cfg(any(feature = "capture", test))]
fn drain_pass(
    pending: &mut Vec<CaptureFrame>,
    incoming: &[CaptureFrame],
    last_overruns: u64,
    now_overruns: u64,
    mut on_hop: impl FnMut(&[CaptureFrame]),
) -> u64 {
    if now_overruns != last_overruns {
        // Capture discontinuity: drop the stale pre-gap leftover before the
        // non-adjacent post-gap frames are staged behind it.
        pending.clear();
    }
    pending.extend_from_slice(incoming);

    // Guard against unbounded growth / stale backlog if the worker fell behind:
    // shed the oldest staged frames so we analyze recent audio. Never fires on the
    // steady-state path (pending stays a few hops deep).
    if pending.len() > FFT_LEN * 8 {
        let drop_to = pending.len() - FFT_LEN * 4;
        pending.drain(0..drop_to);
    }

    while pending.len() >= HOP {
        on_hop(&pending[..HOP]);
        pending.drain(0..HOP);
    }
    now_overruns
}

/// Synchronous, thread-free, cpal-free analyzer: push raw stereo PCM per frame
/// and read the latest [`Features`]. Mirrors [`run`]'s windowing (2048-sample
/// window, 512-sample hop, 75% overlap) but consumes host-supplied samples
/// instead of a capture ring — for hosts that already capture audio and only
/// want the analysis (e.g. an FFI seam feeding raw stereo across each frame).
///
/// STATEFUL: hold ONE instance for the whole session and feed it continuously.
/// The tempo/onset/AGC/HPSS/structure trackers accumulate history across hops;
/// recreating it per frame throws away all warm-up and beat tracking.
pub struct Analyzer {
    state: DspState,
    /// Anti-aliasing decimator bringing high native rates down to a canonical
    /// analysis rate before windowing (see [`crate::resample`]).
    decimator: crate::resample::FrameDecimator,
    /// Effective analysis rate after decimation (Hz).
    analysis_rate: f32,
    // Sliding windows: `window` (mono) feeds the FFT/onset chain; left/right are
    // kept for the sampled PCM scope / audio-texture fields.
    window: Vec<f32>,
    window_left: Vec<f32>,
    window_right: Vec<f32>,
    // Staging for pushed samples awaiting a whole hop.
    pending: Vec<CaptureFrame>,
    latest: Features,
    /// Total whole hops analyzed over this analyzer's lifetime. Monotonic; used to
    /// verify large blocks are fully drained rather than trimmed (P2-AUD-004).
    hops_processed: u64,
}

impl Analyzer {
    /// Create an analyzer for a fixed PCM `sample_rate` (Hz). `sensitivity`
    /// matches [`DspParams::sensitivity`] (~0.1..3, 1.0 = neutral).
    ///
    /// Infallible: a pathological rate is clamped into the supported band so
    /// construction never allocates unbounded history buffers. Prefer
    /// [`Analyzer::try_new`] to reject a bad rate explicitly.
    pub fn new(sample_rate: u32, sensitivity: f32) -> Self {
        Self::build(crate::resample::clamp_native_rate(sample_rate), sensitivity)
    }

    /// Fallibly create an analyzer, rejecting `0` / out-of-range sample rates
    /// rather than building histories for a pathological rate.
    pub fn try_new(sample_rate: u32, sensitivity: f32) -> Result<Self, crate::AudioError> {
        crate::resample::validate_native_rate(sample_rate)?;
        Ok(Self::build(sample_rate as f32, sensitivity))
    }

    /// Build an analyzer for an already-validated native `sample_rate` (Hz). The
    /// DSP runs at the decimated analysis rate so the FFT geometry / hop cadence
    /// stay rate-stable.
    fn build(sample_rate: f32, sensitivity: f32) -> Self {
        let analysis_rate = crate::resample::effective_rate(sample_rate);
        let hop_dt = HOP as f32 / analysis_rate;
        Self {
            state: DspState::new(analysis_rate, hop_dt, sensitivity),
            decimator: crate::resample::FrameDecimator::new(sample_rate),
            analysis_rate,
            window: vec![0.0f32; FFT_LEN],
            window_left: vec![0.0f32; FFT_LEN],
            window_right: vec![0.0f32; FFT_LEN],
            pending: Vec::with_capacity(HOP * 4),
            latest: Features::default(),
            hops_processed: 0,
        }
    }

    /// Effective analysis sample rate in Hz. Equals the native rate at 44.1/48 kHz
    /// and a decimated ~44.1/48 kHz for higher native rates.
    pub fn analysis_sample_rate(&self) -> f32 {
        self.analysis_rate
    }

    /// Push a block of PLANAR stereo samples (equal-length `left`/`right`). Runs
    /// analysis for every whole hop the block completes and returns the most
    /// recent [`Features`]; a block shorter than a hop is buffered and the
    /// previous features are returned unchanged until a hop completes.
    pub fn push_planar(&mut self, left: &[f32], right: &[f32]) -> &Features {
        let n = left.len().min(right.len());
        for i in 0..n {
            let l = sanitize_pcm_sample(left[i]);
            let r = sanitize_pcm_sample(right[i]);
            // Same downmix as the cpal callback (channel mean; see `capture.rs`).
            let mono = 0.5 * (l + r);
            // Anti-alias + decimate to the canonical analysis rate before staging.
            if let Some(frame) = self.decimator.push(CaptureFrame {
                mono,
                left: l,
                right: r,
            }) {
                self.pending.push(frame);
                // Bound the staging buffer on very large blocks by draining the
                // whole hops accumulated so far — never discard valid samples
                // ahead of the hop loop (P2-AUD-004). `drain_hops` leaves < HOP
                // frames pending, so this caps memory without dropping audio.
                if self.pending.len() >= FFT_LEN * 4 {
                    self.drain_hops();
                }
            }
        }
        self.drain_hops();
        &self.latest
    }

    /// Push INTERLEAVED PCM with `channels` channels. The first two channels are
    /// preserved for stereo scope fields, matching media PCM analysis; mono
    /// sources duplicate left to right. Samples are repaired to finite clipped
    /// PCM before entering the analyzer.
    pub fn push_interleaved(&mut self, samples: &[f32], channels: usize) -> &Features {
        if channels == 0 {
            return &self.latest;
        }
        for frame in samples.chunks_exact(channels) {
            let l = sanitize_pcm_sample(frame[0]);
            let r = if channels > 1 {
                sanitize_pcm_sample(frame[1])
            } else {
                l
            };
            let mono = 0.5 * (l + r);
            // Anti-alias + decimate to the canonical analysis rate before staging.
            if let Some(frame) = self.decimator.push(CaptureFrame {
                mono,
                left: l,
                right: r,
            }) {
                self.pending.push(frame);
                // Bound the staging buffer on very large blocks by draining the
                // whole hops accumulated so far — never discard valid samples
                // ahead of the hop loop (P2-AUD-004). `drain_hops` leaves < HOP
                // frames pending, so this caps memory without dropping audio.
                if self.pending.len() >= FFT_LEN * 4 {
                    self.drain_hops();
                }
            }
        }
        self.drain_hops();
        &self.latest
    }

    /// Process every buffered whole hop, advancing the sliding window and the
    /// analyzer state — identical to the inner loop of [`run`].
    fn drain_hops(&mut self) {
        while self.pending.len() >= HOP {
            self.window.copy_within(HOP.., 0);
            self.window_left.copy_within(HOP.., 0);
            self.window_right.copy_within(HOP.., 0);
            let tail = FFT_LEN - HOP;
            let mut hop_mono = [0.0f32; HOP];
            for (i, frame) in self.pending.iter().take(HOP).enumerate() {
                self.window[tail + i] = frame.mono;
                self.window_left[tail + i] = frame.left;
                self.window_right[tail + i] = frame.right;
                hop_mono[i] = frame.mono;
            }
            self.latest = self.state.analyze_hop(
                &self.window,
                &hop_mono,
                &self.window_left,
                &self.window_right,
            );
            self.hops_processed += 1;
            self.pending.drain(0..HOP);
        }
    }

    /// The most recent computed features (unchanged between completed hops).
    pub fn latest(&self) -> &Features {
        &self.latest
    }
}

/// All cached buffers + per-feature smoothing state. One per worker.
struct DspState {
    sample_rate: f32,
    sensitivity: f32,

    // FFT (cached plan + scratch — never re-planned).
    fft: Arc<dyn RealToComplex<f32>>,
    fft_input: Vec<f32>,
    fft_scratch: Vec<Complex<f32>>,
    spectrum: Vec<Complex<f32>>,
    mag: Vec<f32>,
    prev_mag: Vec<f32>,

    hann: Vec<f32>,
    bin_hz: f32,

    macro_bank: FilterBank,
    spectrum_bank: FilterBank,
    /// Shared rolling magnitude history + time-axis (harmonic) median. Advanced
    /// once per hop and consumed by both `hpss` and `hpss_bus`, so the median-
    /// filtered history is stored/sorted a single time (P2-AUD-003).
    hpss_history: HpssHistory,
    hpss: HpssSeparator,
    /// Public median-filtering harmonic/percussive dual-bus separator (scalar +
    /// ratio rails). Reuses the worker's STFT magnitude frame — no second FFT.
    hpss_bus: HpssBus,
    /// Rolling scrolling-spectrogram trail (log-spaced, log-magnitude). Advanced
    /// once per hop from the same `self.mag` frame; published to the render thread.
    spectrogram: SpectrogramTrail,
    /// Rolling self-similarity / Foote novelty structure detector.
    structure: StructureTracker,

    // Butterchurn-parity bass/mid/treb + `_att` follower. Computed alongside the
    // 6 macro bands and routed into the `bass`/`mid`/`air` Features fields so the
    // MilkDrop drop path sees Butterchurn-normalized rails.
    butterchurn: ButterchurnLevels,
    /// Frame counter for the Butterchurn long-average warmup (frame < 50 → 0.9).
    bc_frame: u64,
    /// Effective frame rate of the hop cadence, fed to the Butterchurn follower.
    bc_fps: f32,

    // Smoothing / normalization for the three consumed macro bands
    // (sub_bass / low_mid / presence — see `USED_MACRO_BANDS_HZ`).
    band_lp: [OnePole; 3],
    band_agc: [Agc; 3],
    spectrum_rails: SpectrumRailBank,
    chroma_lp: [OnePole; CHROMA_BINS],
    key_smoother: tonal::KeySmoother,
    /// FFT-autocorrelation monophonic pitch detector (bounded per-hop work,
    /// P2-AUD-002). Owns its FFT plan + scratch; reused every hop.
    pitch: tonal::NsdfPitchDetector,
    rms_agc: Agc,
    rms_lp: OnePole,
    brightness_lp: OnePole,
    pitch_hz_lp: OnePole,
    pitch_norm_lp: OnePole,
    pitch_conf_lp: OnePole,
    rolloff_lp: OnePole,
    flatness_lp: OnePole,
    spread_lp: OnePole,
    contrast_lp: OnePole,
    superflux_lp: OnePole,
    k_weighting: KWeightingFilter,
    loudness: LoudnessTracker,
    gate: SilenceGate,

    // Onsets.
    kick: OnsetDetector,
    snare: OnsetDetector,
    hat: OnsetDetector,
    superflux_onset: OnsetDetector,
    // Precomputed band bin ranges for onset flux.
    kick_bins: (usize, usize),
    snare_bins: (usize, usize),
    hat_bins: (usize, usize),

    tempo: TempoTracker,

    // Complex-domain (phase+energy) onset detector — sharper transients, also
    // fires on soft/tonal phase-only onsets.
    complex_onset: ComplexOnsetDetector,
    // Steep 4th-order Linkwitz-Riley crossover filterbank (clean per-band energy).
    lr_bank: LinkwitzRileyBank,
    // Build-up / drop anticipation predictor over a short history window.
    drop_predictor: DropPredictor,
    // --- Butterchurn-faithful reactivity ---
    /// Per-band (bass/mid/treb) adaptive imm/longAvg normalizers.
    react: [ReactiveLevel; 3],
    /// Precomputed [start, stop) bin ranges over the 512-bin freq_spectrum for the
    /// three Butterchurn bands, using bucketHz = sample_rate / 1024.
    react_bins: [(usize, usize); 3],
    /// Seconds per hop, needed by the reactivity FPS adjustment.
    hop_dt: f32,
}

impl DspState {
    fn new(sample_rate: f32, hop_dt: f32, sensitivity: f32) -> Self {
        let mut planner = RealFftPlanner::<f32>::new();
        let fft = planner.plan_fft_forward(FFT_LEN);
        let fft_input = fft.make_input_vec();
        let fft_scratch = fft.make_scratch_vec();
        let spectrum = fft.make_output_vec();
        let n_bins = spectrum.len();

        let bin_hz = sample_rate / FFT_LEN as f32;

        // Only the three consumed macro bands are built (P2-AUD-025).
        let macro_bank = FilterBank::from_edges(&USED_MACRO_BANDS_HZ, FFT_LEN, sample_rate);
        let spectrum_bank = FilterBank::log_spaced(
            SPECTRUM_BANDS,
            SPECTRUM_LO_HZ,
            SPECTRUM_HI_HZ,
            FFT_LEN,
            sample_rate,
        );

        // Onset band bin ranges (spec §4): kick 20-150, snare 150-2k, hat >5k.
        let bin = |hz: f32| analysis::hz_to_bin(hz, FFT_LEN, sample_rate).min(n_bins - 1);
        let kick_bins = (bin(20.0), bin(150.0).max(bin(20.0) + 1));
        let snare_bins = (bin(150.0), bin(2000.0).max(bin(150.0) + 1));
        let hat_bins = (bin(5000.0), n_bins);

        // Onset detector tunings: short median window (~0.2 s), fast attack, decays
        // per spec (kick 250 ms, snare 200 ms, hat 120 ms), refractory ~60-70 ms.
        let med = ((0.2 / hop_dt) as usize).max(8);
        let kick = OnsetDetector::new(med, 1.6, 1e-4, 70.0, 8.0, 250.0, hop_dt);
        let snare = OnsetDetector::new(med, 1.7, 1e-4, 60.0, 6.0, 200.0, hop_dt);
        let hat = OnsetDetector::new(med, 1.8, 1e-4, 50.0, 4.0, 120.0, hop_dt);

        // Per-band smoothing for the three consumed macro bands: light LP + AGC
        // with slow decay & a small floor (identical per-lane construction, so the
        // used-lane outputs are unchanged by dropping the discarded lanes).
        let band_lp = std::array::from_fn(|_| OnePole::new(0.4));
        let band_agc = std::array::from_fn(|_| Agc::new(0.9995, 1e-3));
        let spectrum_rails = SpectrumRailBank::new(SPECTRUM_BANDS, hop_dt);
        let chroma_lp = std::array::from_fn(|_| OnePole::new(0.72));

        // --- Butterchurn reactivity band ranges over the 512-bin freq_spectrum ---
        // Butterchurn's freqArray has FREQ_SPECTRUM_BINS bins at bucketHz =
        // sample_rate / fftSize = sample_rate / 1024 (since fftSize = 2*numSamps).
        // edge bin = clamp(round(hz / bucketHz) - 1, 0, 511).
        let bucket_hz = sample_rate / (FREQ_SPECTRUM_BINS as f32 * 2.0);
        let react_bin = |hz: f32| -> usize {
            (((hz / bucket_hz).round() as i64 - 1).clamp(0, FREQ_SPECTRUM_BINS as i64 - 1)) as usize
        };
        let react_bins: [(usize, usize); 3] = std::array::from_fn(|i| {
            let (lo, hi) = REACT_BAND_EDGES_HZ[i];
            (react_bin(lo), react_bin(hi))
        });

        Self {
            sample_rate,
            sensitivity,
            fft,
            fft_input,
            fft_scratch,
            spectrum,
            mag: vec![0.0; n_bins],
            prev_mag: vec![0.0; n_bins],
            hann: analysis::hann_window(FFT_LEN),
            bin_hz,
            // Butterchurn band bins depend on the live device sample rate; its
            // `_att` follower is FPS-aware, so feed it the hop cadence.
            butterchurn: ButterchurnLevels::new(sample_rate),
            bc_frame: 0,
            bc_fps: 1.0 / hop_dt,
            macro_bank,
            spectrum_bank,
            hpss_history: HpssHistory::new(n_bins, hop_dt),
            hpss: HpssSeparator::new(n_bins, sample_rate, bin_hz, hop_dt),
            // Sized to the worker's STFT: `n_bins = FFT_LEN/2 + 1`. Both reuse
            // `self.mag` and the shared `hpss_history` each hop — no extra FFT and
            // no duplicate time-median sort.
            hpss_bus: HpssBus::new(n_bins),
            spectrogram: SpectrogramTrail::new(FFT_LEN, sample_rate),
            structure: StructureTracker::new(),
            band_lp,
            band_agc,
            spectrum_rails,
            chroma_lp,
            key_smoother: tonal::KeySmoother::new(),
            pitch: tonal::NsdfPitchDetector::new(FFT_LEN),
            rms_agc: Agc::new(0.9998, 1e-3),
            rms_lp: OnePole::new(0.5),
            brightness_lp: OnePole::new(0.5),
            pitch_hz_lp: OnePole::new(0.35),
            pitch_norm_lp: OnePole::new(0.35),
            pitch_conf_lp: OnePole::new(0.45),
            rolloff_lp: OnePole::new(0.5),
            flatness_lp: OnePole::new(0.55),
            spread_lp: OnePole::new(0.55),
            contrast_lp: OnePole::new(0.55),
            superflux_lp: OnePole::new(0.35),
            k_weighting: KWeightingFilter::new(sample_rate),
            loudness: LoudnessTracker::new(hop_dt),
            // RMS gate: enter-silence below 0.0008 linear, exit above 0.003 (hysteresis).
            gate: SilenceGate::new(0.0008, 0.003),
            kick,
            snare,
            hat,
            superflux_onset: OnsetDetector::new(med, 1.45, 1e-4, 55.0, 7.0, 180.0, hop_dt),
            kick_bins,
            snare_bins,
            hat_bins,
            tempo: TempoTracker::new(hop_dt),
            // Complex-domain onset operates on the full complex FFT output.
            complex_onset: ComplexOnsetDetector::new(n_bins, hop_dt),
            lr_bank: LinkwitzRileyBank::new(sample_rate),
            drop_predictor: DropPredictor::new(hop_dt),
            react: std::array::from_fn(|_| ReactiveLevel::new()),
            react_bins,
            hop_dt,
        }
    }

    /// Analyze one full window and return the smoothed, normalized features.
    fn analyze_hop(
        &mut self,
        window: &[f32],
        hop_samples: &[f32],
        left: &[f32],
        right: &[f32],
    ) -> Features {
        // --- time-domain RMS (pre-window, true signal level) ---
        let rms = {
            let sum_sq: f32 = window.iter().map(|s| s * s).sum();
            (sum_sq / window.len() as f32).sqrt()
        };
        let is_silent = self.gate.update(rms);

        // --- windowed FFT (cached plan + scratch) ---
        for (dst, (&s, &w)) in self
            .fft_input
            .iter_mut()
            .zip(window.iter().zip(self.hann.iter()))
        {
            *dst = s * w;
        }
        // process_with_scratch consumes the input as scratch; that's fine, we
        // refill fft_input every hop above.
        if self
            .fft
            .process_with_scratch(
                &mut self.fft_input,
                &mut self.spectrum,
                &mut self.fft_scratch,
            )
            .is_ok()
        {
            for (m, c) in self.mag.iter_mut().zip(self.spectrum.iter()) {
                *m = c.norm();
            }
        }

        // --- Butterchurn-parity bass/mid/treb + `_att`, freq_spectrum, reactivity ---
        // Feed the most-recent FFT_SIZE mono samples (scaled to Butterchurn's signed
        // -128..127 domain) into the faithful Butterchurn levels follower. Its FFT is
        // FFT_SIZE-point, so feeding only NUM_SAMPS would zero the second transform
        // half — feed the full window. `bc` hovers around ~1.0 because each band is
        // divided by its long-term running average, matching the reference renderer.
        let mut bc_time = [0.0f32; butterchurn::FFT_SIZE];
        let tail = window.len().saturating_sub(butterchurn::FFT_SIZE);
        for (dst, &s) in bc_time.iter_mut().zip(window[tail..].iter()) {
            *dst = (s * 128.0).clamp(-128.0, 127.0);
        }
        // Silence forces the MilkDrop convention val=att=1.0 (longAvg floor); the
        // gate flag is published separately so consumers can still react.
        let bc = self
            .butterchurn
            .update_signed(&bc_time, self.bc_fps, self.bc_frame);
        self.bc_frame = self.bc_frame.saturating_add(1);

        // Publish the reference-compatible Butterchurn freqArray to `bSpectrum`
        // custom waves verbatim — the exact array Butterchurn produces (no window,
        // signed -128..127 scaling, `equalize` curve), not a realfft approximation.
        let mut freq_spectrum = [0.0f32; FREQ_SPECTRUM_BINS];
        freq_spectrum.copy_from_slice(self.butterchurn.freq_array());
        // imm[i] = raw SUM of equalized bins over the band (not a mean). On silence
        // feed imm=0 so the ratios drift toward 1.0 (matches Butterchurn idle).
        let mut react_val = [1.0f32; 3];
        let mut react_att = [1.0f32; 3];
        for i in 0..3 {
            let (start, stop) = self.react_bins[i];
            let imm: f32 = if is_silent {
                0.0
            } else {
                freq_spectrum[start..stop.min(FREQ_SPECTRUM_BINS)]
                    .iter()
                    .sum()
            };
            let (val, att) = self.react[i].process(imm, self.hop_dt);
            react_val[i] = val;
            react_att[i] = att;
        }
        let vol_react = (react_val[0] + react_val[1] + react_val[2]) / 3.0;
        let vol_react_att = (react_att[0] + react_att[1] + react_att[2]) / 3.0;

        // Advance the shared magnitude history once, then route both HPSS
        // consumers through its per-bin harmonic reference so the time-axis median
        // is stored and sorted a single time (P2-AUD-003).
        self.hpss_history.advance(&self.mag);
        let hpss = self.hpss.analyze(
            &self.mag,
            self.hpss_history.harm_ref(),
            is_silent,
            self.sensitivity,
        );

        // Public median-filtering HPSS dual-bus rails + scrolling spectrogram.
        // Both consume the same STFT magnitude frame computed above (no second
        // FFT) and own their smoothing/normalization internally.
        let hpss_bus = self
            .hpss_bus
            .process(&self.mag, self.hpss_history.harm_ref(), is_silent);
        self.spectrogram.push(&self.mag, is_silent);

        // --- macro bands → dB → AGC → LP (only the 3 consumed lanes: sub_bass,
        //     low_mid, presence; P2-AUD-025) ---
        let mut raw_bands = [0.0f32; 3];
        self.macro_bank.apply(&self.mag, &mut raw_bands);
        let mut bands = [0.0f32; 3];
        for i in 0..3 {
            let db = lin_to_db_norm(raw_bands[i]);
            let agc = self.band_agc[i].process(db);
            bands[i] = self.band_lp[i].process(agc);
        }
        // Named views onto the three consumed macro-band rails.
        let (macro_sub_bass, macro_low_mid, macro_presence) = (bands[0], bands[1], bands[2]);

        // --- 32-band coarse spectrum ---
        let mut spectrum = [0.0f32; 32];
        let mut raw_spec = [0.0f32; SPECTRUM_BANDS];
        self.spectrum_bank.apply(&self.mag, &mut raw_spec);
        self.spectrum_rails
            .process(&raw_spec, is_silent, lin_to_db_norm, &mut spectrum);

        // --- tonal slice (FFT chroma + Krumhansl/key + triad chord) ---
        let raw_chroma = if is_silent {
            [0.0f32; CHROMA_BINS]
        } else {
            tonal::chroma_from_spectrum(self.hpss.harmonic_mag(), self.bin_hz)
        };
        let mut chroma = [0.0f32; CHROMA_BINS];
        for i in 0..CHROMA_BINS {
            chroma[i] = self.chroma_lp[i].process(raw_chroma[i]);
        }
        let key = if is_silent {
            self.key_smoother.reset();
            tonal::TonalEstimate::default()
        } else {
            self.key_smoother.update(&chroma)
        };
        let chord = if is_silent {
            tonal::TonalEstimate::default()
        } else {
            tonal::estimate_chord(&chroma)
        };
        let (harmony_hue, harmony_mood) = tonal::palette_from_key(key);
        let raw_pitch = if is_silent {
            tonal::PitchEstimate::default()
        } else {
            self.pitch.estimate(window, self.sample_rate, 45.0, 1600.0)
        };
        let pitch_gate = if is_silent {
            0.0
        } else {
            (0.35 + 0.65 * hpss_bus.harmonic_ratio.clamp(0.0, 1.0))
                * (hpss_bus.harmonic_level * 1.5).clamp(0.0, 1.0)
        };
        let pitch_confidence = self
            .pitch_conf_lp
            .process((raw_pitch.confidence * pitch_gate).clamp(0.0, 1.0));
        let pitch_norm_target = if pitch_confidence > 0.05 {
            raw_pitch.normalized
        } else {
            0.0
        };
        let pitch_hz_target = if pitch_confidence > 0.05 {
            raw_pitch.hz
        } else {
            0.0
        };
        let pitch_norm = self.pitch_norm_lp.process(pitch_norm_target);
        let pitch_hz = self.pitch_hz_lp.process(pitch_hz_target);

        // --- dynamics ---
        let brightness = self.brightness_lp.process(analysis::spectral_centroid(
            &self.mag,
            self.bin_hz,
            self.sample_rate,
        ));
        let flux_raw = analysis::spectral_flux(&self.prev_mag, &self.mag);
        // Normalize flux roughly by spectrum size for a 0..1-ish display value.
        let flux = (flux_raw / (self.mag.len() as f32 * 0.05)).clamp(0.0, 1.0);
        let rolloff = self
            .rolloff_lp
            .process(analysis::spectral_rolloff(&self.mag, 0.85));
        let spectral_flatness = self
            .flatness_lp
            .process(analysis::spectral_flatness(&self.mag));
        let spectral_spread = self.spread_lp.process(analysis::spectral_spread(&self.mag));
        let spectral_contrast = self
            .contrast_lp
            .process(analysis::spectral_contrast(&self.mag, self.bin_hz));
        let superflux_raw = if is_silent {
            0.0
        } else {
            analysis::superflux(&self.prev_mag, &self.mag, 3)
        };
        let superflux = self
            .superflux_lp
            .process((superflux_raw / (self.mag.len() as f32 * 0.012)).clamp(0.0, 1.0));

        // --- onsets (band-limited flux, gated by silence) ---
        let perc_mag = self.hpss.percussive_mag();
        let prev_perc_mag = self.hpss.prev_percussive_mag();
        let (kf, sf, hf) = if is_silent {
            (0.0, 0.0, 0.0)
        } else {
            (
                analysis::band_superflux(
                    prev_perc_mag,
                    perc_mag,
                    self.kick_bins.0,
                    self.kick_bins.1,
                    2,
                ),
                analysis::band_superflux(
                    prev_perc_mag,
                    perc_mag,
                    self.snare_bins.0,
                    self.snare_bins.1,
                    2,
                ),
                analysis::band_superflux(
                    prev_perc_mag,
                    perc_mag,
                    self.hat_bins.0,
                    self.hat_bins.1,
                    2,
                ),
            )
        };
        let (kick_onset, _) = self.kick.process(kf, self.sensitivity);
        let (snare_onset, _) = self.snare.process(sf, self.sensitivity);
        let (hat_onset, _) = self.hat.process(hf, self.sensitivity);
        let (superflux_onset, _) = self
            .superflux_onset
            .process(superflux_raw, self.sensitivity);

        // --- tempo / beat (broadband onset strength = vibrato-suppressed SuperFlux) ---
        let onset_strength = if is_silent { 0.0 } else { superflux_raw };
        let tempo = self.tempo.process(onset_strength);

        // --- rms level normalized ---
        let rms_norm = self
            .rms_lp
            .process(self.rms_agc.process(lin_to_db_norm(rms)));
        let k_weighted_energy = self.k_weighting.process_energy(hop_samples);
        let (lufs_momentary, lufs_short, loudness_build, lufs_range) =
            self.loudness.process(k_weighted_energy, is_silent);

        // --- complex-domain onset (phase + energy), Bello et al. 2004 ---
        // Operates on the full complex FFT output; keeps its own phase/mag
        // history so it's independent of the magnitude-only flux path above.
        let complex_onset = self
            .complex_onset
            .process(&self.spectrum, is_silent, self.sensitivity);

        // --- Linkwitz-Riley 4th-order multiband (clean per-band energy) ---
        // Fed the raw mono hop samples (pre-window) so the IIR crossover sees a
        // continuous time-domain stream.
        let lr = self.lr_bank.process(hop_samples, is_silent);

        // --- predictive drop / build-up anticipation ---
        // Energy must PRESERVE loudness dynamics: `rms_norm` is peak-AGC'd and sits
        // near 1.0 during any sustained section, so its window slope collapses and
        // the predictor sees no build-up. Feed the absolute dB-normalized loudness
        // instead (non-saturating over the -80 dB floor), so a quiet→loud build
        // actually ramps the prediction (P2-AUD-019). centroid = brightness;
        // high = presence band; sub = sub-bass band; flux drives the activity term.
        let drop_energy = lin_to_db_norm(rms);
        let drop_anticipation = self.drop_predictor.process(
            drop_energy,
            flux,
            brightness,
            macro_presence,
            macro_sub_bass,
            is_silent,
        );
        let structure = self.structure.process(
            &spectrum,
            &chroma,
            rms_norm,
            brightness,
            flux,
            hpss_bus.harmonic_ratio,
            is_silent,
        );

        // Save magnitude history for next hop's flux.
        self.hpss.finish_frame();
        self.prev_mag.copy_from_slice(&self.mag);

        Features {
            sub_bass: macro_sub_bass,
            // `bass`/`mid`/`air` carry the Butterchurn-normalized rails (~1.0
            // baseline) that the MilkDrop drop path reads as `bass`/`mid`/`treb`.
            // `sub_bass`/`low_mid`/`presence` keep the 0..1 AGC macro bands.
            bass: bc.bass,
            low_mid: macro_low_mid,
            mid: bc.mid,
            presence: macro_presence,
            air: bc.treb,
            // Real Butterchurn attenuated (slow-follower) envelopes carried
            // straight through to the MilkDrop `*_att` rails (`air` maps treb).
            bass_att: bc.bass_att,
            mid_att: bc.mid_att,
            treb_att: bc.treb_att,
            vol_att: bc.vol_att,
            rms_level: rms_norm,
            brightness,
            flux,
            rolloff,
            lufs_momentary,
            lufs_short,
            loudness_build,
            lufs_range,
            spectral_flatness,
            spectral_spread,
            spectral_contrast,
            superflux,
            superflux_onset,
            kick_onset,
            snare_onset,
            hat_onset,
            beat_confidence: tempo.confidence,
            beat_phase: tempo.beat_phase,
            bar_phase: tempo.bar_phase,
            bpm: tempo.bpm,
            beat_impulse: tempo.beat_impulse,
            is_silent: if is_silent { 1.0 } else { 0.0 },
            spectrum,
            chroma,
            key_root: key.root as f32 / CHROMA_BINS as f32,
            key_is_minor: if key.is_minor { 1.0 } else { 0.0 },
            key_confidence: key.confidence,
            chord_root: chord.root as f32 / CHROMA_BINS as f32,
            chord_is_minor: if chord.is_minor { 1.0 } else { 0.0 },
            chord_confidence: chord.confidence,
            harmony_hue,
            harmony_mood,
            waveform_left: downsample_waveform(left),
            waveform_right: downsample_waveform(right),
            perc_rms: hpss.perc_rms,
            perc_flux: hpss.perc_flux,
            perc_onset: hpss.perc_onset,
            perc_ratio: hpss.perc_ratio,
            harm_rms: hpss.harm_rms,
            harm_flux: hpss.harm_flux,
            harm_brightness: hpss.harm_brightness,
            harm_ratio: hpss.harm_ratio,
            complex_onset,
            lr_sub: lr.sub,
            lr_low: lr.low,
            lr_mid: lr.mid,
            lr_high: lr.high,
            lr_air: lr.air,
            drop_anticipation,
            harmonic_level: hpss_bus.harmonic_level,
            percussive_level: hpss_bus.percussive_level,
            harmonic_ratio: hpss_bus.harmonic_ratio,
            percussive_ratio: hpss_bus.percussive_ratio,
            pitch_hz,
            pitch_norm,
            pitch_confidence,
            structure_novelty: structure.novelty,
            structure_change: structure.change,
            structure_confidence: structure.confidence,
            // Butterchurn-faithful reactivity (the signal the MilkDrop engine wants).
            bass_react: react_val[0],
            mid_react: react_val[1],
            treb_react: react_val[2],
            bass_react_att: react_att[0],
            mid_react_att: react_att[1],
            treb_react_att: react_att[2],
            vol_react,
            vol_react_att,
            freq_spectrum,
            // The most recent WAVEFORM_SAMPLES_FULL raw samples from the sliding
            // window — the verbatim recent waveform the field documents, at
            // Butterchurn's native resolution (not a peak-decimated summary of the
            // whole window; P2-AUD-007). The window is longer than the field, so
            // this is the newest slice with no truncation.
            waveform_left_full: recent_waveform_full(left),
            waveform_right_full: recent_waveform_full(right),
        }
    }
}

/// Copy the most recent [`WAVEFORM_SAMPLES_FULL`] raw PCM samples from the
/// analysis window, in stream order (oldest → newest). This is the recent raw
/// waveform the `waveform_*_full` fields promise — unlike the coarse 32-sample
/// scope fields it is *not* peak-decimated. Inputs shorter than the field are
/// right-aligned (newest at the end) and zero-padded at the front.
fn recent_waveform_full(samples: &[f32]) -> [f32; WAVEFORM_SAMPLES_FULL] {
    let mut out = [0.0f32; WAVEFORM_SAMPLES_FULL];
    let n = samples.len().min(WAVEFORM_SAMPLES_FULL);
    out[WAVEFORM_SAMPLES_FULL - n..].copy_from_slice(&samples[samples.len() - n..]);
    out
}

fn downsample_waveform(samples: &[f32]) -> [f32; WAVEFORM_SAMPLES] {
    let mut out = [0.0f32; WAVEFORM_SAMPLES];
    if samples.is_empty() {
        return out;
    }
    for (i, dst) in out.iter_mut().enumerate() {
        let start = i * samples.len() / WAVEFORM_SAMPLES;
        let end = ((i + 1) * samples.len() / WAVEFORM_SAMPLES)
            .max(start + 1)
            .min(samples.len());
        let mut peak = 0.0f32;
        for &sample in &samples[start..end] {
            if sample.abs() > peak.abs() {
                peak = sample;
            }
        }
        *dst = peak.clamp(-1.0, 1.0);
    }
    out
}

/// Map a linear magnitude to a normalized `0..1` loudness via dBFS over a fixed
/// floor. `-80 dB → 0`, `0 dB → 1`. Cheap and monotonic; AGC handles per-source
/// scaling afterward.
#[inline]
pub(crate) fn lin_to_db_norm(lin: f32) -> f32 {
    const FLOOR_DB: f32 = -80.0;
    if lin <= 1e-9 {
        return 0.0;
    }
    let db = 20.0 * lin.log10();
    ((db - FLOOR_DB) / -FLOOR_DB).clamp(0.0, 1.0)
}

/// BS.1770-style K-weighting filter chain used before the EBU-R128 loudness
/// windows. Capture is already downmixed to mono, so the channel weighting term
/// is 1.0 and the loudness energy is just mean square of this filtered signal.
struct KWeightingFilter {
    shelf: Biquad,
    high_pass: Biquad,
}

impl KWeightingFilter {
    fn new(sample_rate: f32) -> Self {
        Self {
            // ITU-R BS.1770 K-weighting shape: +4 dB high shelf followed by
            // the RLB high-pass. Coefficients are generated for the live device
            // rate instead of assuming 48 kHz.
            shelf: Biquad::high_shelf(sample_rate, 1681.9745, 4.0),
            high_pass: Biquad::high_pass(sample_rate, 38.13547, 0.500327),
        }
    }

    fn process_energy(&mut self, samples: &[f32]) -> f32 {
        if samples.is_empty() {
            return 0.0;
        }
        let mut sum = 0.0f32;
        for &x in samples {
            let y = self.high_pass.process(self.shelf.process(x));
            sum += y * y;
        }
        sum / samples.len() as f32
    }
}

#[derive(Clone, Copy, Debug)]
struct Biquad {
    b0: f32,
    b1: f32,
    b2: f32,
    a1: f32,
    a2: f32,
    z1: f32,
    z2: f32,
}

impl Biquad {
    fn high_pass(sample_rate: f32, f0: f32, q: f32) -> Self {
        let (sin_w0, cos_w0) = biquad_sin_cos(sample_rate, f0);
        let alpha = sin_w0 / (2.0 * q.max(1e-3));
        let b0 = (1.0 + cos_w0) * 0.5;
        let b1 = -(1.0 + cos_w0);
        let b2 = (1.0 + cos_w0) * 0.5;
        let a0 = 1.0 + alpha;
        let a1 = -2.0 * cos_w0;
        let a2 = 1.0 - alpha;
        Self::from_coeffs(b0, b1, b2, a0, a1, a2)
    }

    fn high_shelf(sample_rate: f32, f0: f32, gain_db: f32) -> Self {
        let (sin_w0, cos_w0) = biquad_sin_cos(sample_rate, f0);
        let a = 10.0f32.powf(gain_db / 40.0);
        let sqrt_a = a.sqrt();
        // RBJ high-shelf with slope S=1.0. This tracks the BS.1770 pre-filter
        // closely enough for a realtime visual loudness rail while remaining
        // sample-rate independent.
        let alpha = sin_w0 * 0.5 * 2.0f32.sqrt();
        let b0 = a * ((a + 1.0) + (a - 1.0) * cos_w0 + 2.0 * sqrt_a * alpha);
        let b1 = -2.0 * a * ((a - 1.0) + (a + 1.0) * cos_w0);
        let b2 = a * ((a + 1.0) + (a - 1.0) * cos_w0 - 2.0 * sqrt_a * alpha);
        let a0 = (a + 1.0) - (a - 1.0) * cos_w0 + 2.0 * sqrt_a * alpha;
        let a1 = 2.0 * ((a - 1.0) - (a + 1.0) * cos_w0);
        let a2 = (a + 1.0) - (a - 1.0) * cos_w0 - 2.0 * sqrt_a * alpha;
        Self::from_coeffs(b0, b1, b2, a0, a1, a2)
    }

    fn from_coeffs(b0: f32, b1: f32, b2: f32, a0: f32, a1: f32, a2: f32) -> Self {
        let inv_a0 = if a0.abs() > 1e-12 { 1.0 / a0 } else { 1.0 };
        Self {
            b0: b0 * inv_a0,
            b1: b1 * inv_a0,
            b2: b2 * inv_a0,
            a1: a1 * inv_a0,
            a2: a2 * inv_a0,
            z1: 0.0,
            z2: 0.0,
        }
    }

    #[inline]
    fn process(&mut self, x: f32) -> f32 {
        let x = finite_or_zero(x);
        if !(self.z1.is_finite() && self.z2.is_finite()) {
            self.z1 = 0.0;
            self.z2 = 0.0;
        }
        let y = self.b0 * x + self.z1;
        // Flush the recursive state into a hard zero once it decays into the
        // denormal range so the filter never pays the subnormal CPU penalty
        // (P2-AUD-024). NaN/Inf still fall through to the finite guard below.
        self.z1 = flush_denormal(self.b1 * x - self.a1 * y + self.z2);
        self.z2 = flush_denormal(self.b2 * x - self.a2 * y);
        if y.is_finite() && self.z1.is_finite() && self.z2.is_finite() {
            y
        } else {
            self.z1 = 0.0;
            self.z2 = 0.0;
            0.0
        }
    }
}

#[inline]
fn biquad_sin_cos(sample_rate: f32, f0: f32) -> (f32, f32) {
    let nyquist_safe = (sample_rate * 0.45).max(10.0);
    let f0 = f0.clamp(1.0, nyquist_safe);
    let w0 = std::f32::consts::TAU * f0 / sample_rate.max(1.0);
    w0.sin_cos()
}

const ABS_GATE_LUFS: f32 = -70.0;
const REL_GATE_LU: f32 = 10.0;
const LRA_REL_GATE_LU: f32 = 20.0;

/// EBU-R128-shaped loudness tracker: K-weighted momentary (~400 ms),
/// relative-gated short-term (~3 s), build above the short-term floor, and a
/// rolling loudness range rail derived from short-term LUFS percentiles.
struct LoudnessTracker {
    energy: Vec<f32>,
    write: usize,
    filled: usize,
    momentary_len: usize,
    range_lufs: Vec<f32>,
    range_write: usize,
    range_filled: usize,
    /// Preallocated scratch reused by [`LoudnessTracker::loudness_range_lu`] so the
    /// per-hop loudness-range estimate never allocates or fully sorts (P2-AUD-012).
    range_scratch: Vec<f32>,
    build_env: AsymEnv,
}

impl LoudnessTracker {
    fn new(hop_dt: f32) -> Self {
        let short_len = ((3.0 / hop_dt).round() as usize).max(1);
        let momentary_len = ((0.4 / hop_dt).round() as usize).max(1).min(short_len);
        let range_len = ((20.0 / hop_dt).round() as usize).max(short_len);
        Self {
            energy: vec![0.0; short_len],
            write: 0,
            filled: 0,
            momentary_len,
            range_lufs: vec![-100.0; range_len],
            range_write: 0,
            range_filled: 0,
            range_scratch: Vec::with_capacity(range_len),
            build_env: AsymEnv::new(20.0, 650.0, hop_dt),
        }
    }

    fn process(&mut self, k_weighted_energy: f32, is_silent: bool) -> (f32, f32, f32, f32) {
        let e = if is_silent {
            0.0
        } else {
            k_weighted_energy.max(0.0)
        };
        self.energy[self.write] = e;
        self.write = (self.write + 1) % self.energy.len();
        self.filled = (self.filled + 1).min(self.energy.len());

        let momentary = lufs_energy_to_norm(self.window_average(self.momentary_len));
        let short_energy = self.gated_window_average(self.energy.len(), REL_GATE_LU);
        let short_lufs = energy_to_lufs(short_energy);
        let short = lufs_to_norm(short_lufs);

        self.range_lufs[self.range_write] = short_lufs;
        self.range_write = (self.range_write + 1) % self.range_lufs.len();
        self.range_filled = (self.range_filled + 1).min(self.range_lufs.len());
        let lufs_range = (self.loudness_range_lu() / 20.0).clamp(0.0, 1.0);

        let build = self
            .build_env
            .process((momentary - short).max(0.0).clamp(0.0, 1.0));
        (momentary, short, build, lufs_range)
    }

    fn window_average(&self, len: usize) -> f32 {
        let count = len.min(self.filled);
        if count == 0 {
            return 0.0;
        }
        let mut sum = 0.0f32;
        for offset in 0..count {
            let idx = (self.write + self.energy.len() - 1 - offset) % self.energy.len();
            sum += self.energy[idx];
        }
        sum / count as f32
    }

    fn gated_window_average(&self, len: usize, relative_gate_lu: f32) -> f32 {
        let count = len.min(self.filled);
        if count == 0 {
            return 0.0;
        }

        let abs_gate = lufs_to_energy(ABS_GATE_LUFS);
        let mut prelim_sum = 0.0f32;
        let mut prelim_count = 0usize;
        for offset in 0..count {
            let idx = (self.write + self.energy.len() - 1 - offset) % self.energy.len();
            let e = self.energy[idx];
            if e >= abs_gate {
                prelim_sum += e;
                prelim_count += 1;
            }
        }
        if prelim_count == 0 {
            return 0.0;
        }

        let prelim = prelim_sum / prelim_count as f32;
        let relative_gate = lufs_to_energy(energy_to_lufs(prelim) - relative_gate_lu);
        let gate = abs_gate.max(relative_gate);
        let mut sum = 0.0f32;
        let mut gated_count = 0usize;
        for offset in 0..count {
            let idx = (self.write + self.energy.len() - 1 - offset) % self.energy.len();
            let e = self.energy[idx];
            if e >= gate {
                sum += e;
                gated_count += 1;
            }
        }
        if gated_count == 0 {
            0.0
        } else {
            sum / gated_count as f32
        }
    }

    fn loudness_range_lu(&mut self) -> f32 {
        let count = self.range_filled;
        if count < 2 {
            return 0.0;
        }

        // Gather gated short-term LUFS into the reused scratch (no per-hop
        // allocation): `clear` keeps the capacity, so this never reallocates.
        self.range_scratch.clear();
        let mut energy_sum = 0.0f32;
        let ring = self.range_lufs.len();
        for offset in 0..count {
            let idx = (self.range_write + ring - 1 - offset) % ring;
            let lufs = self.range_lufs[idx];
            if lufs > ABS_GATE_LUFS {
                energy_sum += lufs_to_energy(lufs);
                self.range_scratch.push(lufs);
            }
        }
        if self.range_scratch.len() < 2 {
            return 0.0;
        }

        let preliminary_lufs = energy_to_lufs(energy_sum / self.range_scratch.len() as f32);
        let threshold = (preliminary_lufs - LRA_REL_GATE_LU).max(ABS_GATE_LUFS);
        self.range_scratch.retain(|&v| v >= threshold);
        if self.range_scratch.len() < 2 {
            return 0.0;
        }

        // Bounded order statistics via partial selection instead of a full sort.
        // Both percentiles use the same linear interpolation as the old sorted
        // path, so the result is identical to within float rounding.
        let p10 = percentile_select(&mut self.range_scratch, 0.10);
        let p95 = percentile_select(&mut self.range_scratch, 0.95);
        (p95 - p10).max(0.0)
    }
}

#[inline]
fn energy_to_lufs(energy: f32) -> f32 {
    if energy <= 1e-12 {
        return -100.0;
    }
    -0.691 + 10.0 * energy.log10()
}

#[inline]
fn lufs_to_energy(lufs: f32) -> f32 {
    10.0f32.powf((lufs + 0.691) / 10.0)
}

#[inline]
fn lufs_to_norm(lufs: f32) -> f32 {
    ((lufs - ABS_GATE_LUFS) / -ABS_GATE_LUFS).clamp(0.0, 1.0)
}

#[inline]
fn lufs_energy_to_norm(energy: f32) -> f32 {
    lufs_to_norm(energy_to_lufs(energy))
}

#[cfg(test)]
fn percentile_sorted(values: &[f32], p: f32) -> f32 {
    debug_assert!(!values.is_empty());
    if values.len() == 1 {
        return values[0];
    }
    let pos = p.clamp(0.0, 1.0) * (values.len() - 1) as f32;
    let lo = pos.floor() as usize;
    let hi = pos.ceil() as usize;
    if lo == hi {
        values[lo]
    } else {
        let frac = pos - lo as f32;
        values[lo] * (1.0 - frac) + values[hi] * frac
    }
}

/// Linear-interpolation percentile over an *unsorted* slice, using partial
/// selection instead of a full sort. Reorders `values` in place but never
/// allocates. Returns exactly the same value as [`percentile_sorted`] applied to
/// the sorted slice (same bracketing order statistics, same interpolation).
fn percentile_select(values: &mut [f32], p: f32) -> f32 {
    debug_assert!(!values.is_empty());
    let n = values.len();
    if n == 1 {
        return values[0];
    }
    let pos = p.clamp(0.0, 1.0) * (n - 1) as f32;
    let lo = pos.floor() as usize;
    let hi = pos.ceil() as usize;
    // `select_nth_unstable_by` places the lo-th smallest element at index `lo`
    // and returns the strictly-greater partition (buf[lo+1..]).
    let (_, lo_elem, greater) = values.select_nth_unstable_by(lo, |a, b| a.total_cmp(b));
    let lo_val = *lo_elem;
    if hi == lo {
        return lo_val;
    }
    // ceil - floor is at most 1, so the hi-th order statistic is the minimum of
    // the greater partition.
    let hi_val = greater.iter().copied().fold(f32::INFINITY, f32::min);
    let frac = pos - lo as f32;
    lo_val * (1.0 - frac) + hi_val * frac
}

/// Stateful post-processing for the 32 log spectrum bands.
///
/// Each rail is normalized against its own running peak, compared to a slow
/// running mean, and then fed through a fast-attack / slow-decay envelope. The
/// small absolute component keeps a sustained EQ band visible; the mean contrast
/// makes new band energy pop as emitter-bank impulses.
struct SpectrumRailBank {
    peak: Vec<Agc>,
    mean: Vec<OnePole>,
    env: Vec<AsymEnv>,
}

impl SpectrumRailBank {
    fn new(count: usize, hop_dt: f32) -> Self {
        Self {
            peak: (0..count).map(|_| Agc::new(0.9995, 1e-3)).collect(),
            // Slow enough to learn the track's long-term EQ profile without
            // swallowing musical attacks.
            mean: (0..count).map(|_| OnePole::new(0.995)).collect(),
            env: (0..count)
                .map(|_| AsymEnv::new(12.0, 240.0, hop_dt))
                .collect(),
        }
    }

    fn process<F>(&mut self, raw: &[f32], is_silent: bool, to_level: F, out: &mut [f32])
    where
        F: Fn(f32) -> f32,
    {
        debug_assert_eq!(raw.len(), self.peak.len());
        debug_assert_eq!(out.len(), self.peak.len());

        for i in 0..self.peak.len() {
            let level = if is_silent { 0.0 } else { to_level(raw[i]) };
            let peak_norm = self.peak[i].process(level);
            let mean = self.mean[i].process(peak_norm);
            let contrast = (peak_norm - mean).max(0.0);
            let emitter = (contrast * 0.85 + peak_norm * 0.15).clamp(0.0, 1.0);
            out[i] = self.env[i].process(emitter);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn spectrum_rails_attack_then_decay() {
        let hop_dt = HOP as f32 / 48_000.0;
        let mut rails = SpectrumRailBank::new(1, hop_dt);
        let mut out = [0.0f32; 1];

        rails.process(&[1.0], false, |x| x, &mut out);
        let attack = out[0];
        assert!(attack > 0.45, "rail should attack quickly, got {attack}");

        rails.process(&[0.0], false, |x| x, &mut out);
        let after_one_decay = out[0];
        assert!(
            after_one_decay > attack * 0.85,
            "rail should release slowly: attack={attack}, next={after_one_decay}"
        );
    }

    #[test]
    fn spectrum_rails_learn_steady_mean_but_keep_eq_floor() {
        let hop_dt = HOP as f32 / 48_000.0;
        let mut rails = SpectrumRailBank::new(1, hop_dt);
        let mut out = [0.0f32; 1];

        rails.process(&[1.0], false, |x| x, &mut out);
        let first = out[0];
        for _ in 0..900 {
            rails.process(&[1.0], false, |x| x, &mut out);
        }
        let steady = out[0];

        assert!(first > 0.45);
        assert!(
            steady < first * 0.5,
            "steady bands should lose transient contrast: first={first}, steady={steady}"
        );
        assert!(
            steady > 0.12,
            "steady bands should retain a visible EQ-bank floor, got {steady}"
        );
    }

    #[test]
    fn spectrum_rails_silence_drains_input_without_resetting_state() {
        let hop_dt = HOP as f32 / 48_000.0;
        let mut rails = SpectrumRailBank::new(1, hop_dt);
        let mut out = [0.0f32; 1];

        rails.process(&[1.0], false, |x| x, &mut out);
        let audible = out[0];
        for _ in 0..120 {
            rails.process(&[1.0], true, |x| x, &mut out);
        }

        assert!(
            out[0] < audible * 0.2,
            "silence should drain the rail envelope: audible={audible}, silent={}",
            out[0]
        );
    }

    #[test]
    fn analyzer_push_interleaved_sanitizes_and_clips_samples() {
        let mut analyzer = Analyzer::new(48_000, 1.0);

        analyzer.push_interleaved(&[2.0, f32::NAN, -3.0, 0.25], 2);

        assert_eq!(analyzer.pending.len(), 2);
        assert_eq!(analyzer.pending[0].left, 1.0);
        assert_eq!(analyzer.pending[0].right, 0.0);
        assert_eq!(analyzer.pending[0].mono, 0.5);
        assert_eq!(analyzer.pending[1].left, -1.0);
        assert_eq!(analyzer.pending[1].right, 0.25);
        assert_eq!(analyzer.pending[1].mono, -0.375);
    }

    /// A `CaptureFrame` tagged uniformly so a hop's provenance (pre-gap vs
    /// post-gap) is visible from any single sample.
    fn tagged_frame(tag: f32) -> CaptureFrame {
        CaptureFrame {
            mono: tag,
            left: tag,
            right: tag,
        }
    }

    #[test]
    fn drain_pass_no_overrun_keeps_leftover_and_stays_continuous() {
        // A sub-hop leftover (tag -1.0) carried from the previous pass is genuinely
        // adjacent to this pass's frames (tag +1.0): with no discontinuity it must
        // be retained and spliced into the next hop — the unchanged steady-state
        // path. Also proves the flush is gated strictly on the overrun watermark
        // rather than fired unconditionally.
        let leftover = HOP - 64;
        let mut pending: Vec<CaptureFrame> = (0..leftover).map(|_| tagged_frame(-1.0)).collect();
        let incoming: Vec<CaptureFrame> = (0..HOP * 2).map(|_| tagged_frame(1.0)).collect();
        let watermark = 9u64;

        let mut hops = 0usize;
        let mut first_hop_carries_leftover = false;
        let now = drain_pass(&mut pending, &incoming, watermark, watermark, |hop| {
            if hops == 0 {
                first_hop_carries_leftover = hop.iter().any(|f| f.mono == -1.0);
            }
            hops += 1;
        });

        assert_eq!(now, watermark, "no overrun leaves the watermark unchanged");
        assert!(
            first_hop_carries_leftover,
            "no-overrun path must carry the sub-hop leftover into the next hop"
        );
        // (HOP-64) + 2*HOP → 2 whole hops, HOP-64 remainder still staged.
        assert_eq!(hops, 2);
        assert_eq!(pending.len(), leftover);
    }

    #[test]
    fn drain_pass_discontinuity_flushes_leftover_without_splicing() {
        // Reproduce one worker pass across a capture-ring gap. The pre-gap leftover
        // (tag -1.0) is shorter than a hop, so on the continuous path it would
        // splice with the post-gap head (tag +1.0) into a single straddling hop.
        let leftover = HOP - 64;
        let mut pending: Vec<CaptureFrame> = (0..leftover).map(|_| tagged_frame(-1.0)).collect();
        let incoming: Vec<CaptureFrame> = (0..HOP * 3).map(|_| tagged_frame(1.0)).collect();
        let last = 3u64;

        let mut straddled = false;
        let mut all_post_gap = true;
        let mut hops = 0usize;
        // Watermark advanced by one → the callback dropped frames since last pass.
        let now = drain_pass(&mut pending, &incoming, last, last + 1, |hop| {
            let first = hop[0].mono;
            if hop.iter().any(|f| f.mono != first) {
                straddled = true;
            }
            if first != 1.0 {
                all_post_gap = false;
            }
            hops += 1;
        });

        assert_eq!(now, last + 1, "watermark carried forward");
        assert!(
            !straddled,
            "no hop may splice pre-gap and post-gap samples across a capture gap"
        );
        assert!(
            all_post_gap,
            "after the discontinuity flush only post-gap audio is analyzed"
        );
        // Pre-gap leftover flushed → exactly the 3 whole post-gap hops remain.
        assert_eq!(hops, 3);
        assert!(pending.is_empty());

        // The flush is load-bearing: had the leftover NOT been dropped (the old
        // post-drain ordering), the first hop WOULD straddle the gap. Prove that
        // counterfactual so a regression that removes/reorders the flush is caught.
        let mut spliced: Vec<CaptureFrame> = (0..leftover).map(|_| tagged_frame(-1.0)).collect();
        spliced.extend((0..HOP * 3).map(|_| tagged_frame(1.0)));
        let first_hop = &spliced[..HOP];
        assert!(
            first_hop.iter().any(|f| f.mono == -1.0) && first_hop.iter().any(|f| f.mono == 1.0),
            "sanity: without the flush the first hop splices pre/post-gap audio"
        );
    }

    #[test]
    fn loudness_tracker_separates_momentary_short_and_build() {
        let hop_dt = HOP as f32 / 48_000.0;
        let mut loudness = LoudnessTracker::new(hop_dt);

        for _ in 0..300 {
            let (_, short, build, range) = loudness.process(lufs_to_energy(-30.0), false);
            assert!(build <= 1.0);
            assert!(short >= 0.0);
            assert!(range >= 0.0);
        }
        let (_, steady_short, _, _) = loudness.process(lufs_to_energy(-30.0), false);
        let (loud_momentary, loud_short, build, _) = loudness.process(lufs_to_energy(-12.0), false);

        assert!(loud_momentary > steady_short);
        assert!(loud_short >= steady_short);
        assert!(build > 0.0, "momentary jump should create a build rail");
    }

    #[test]
    fn k_weighting_emphasizes_presence_over_sub_bass() {
        fn weighted_sine_energy(freq: f32) -> f32 {
            let sample_rate = 48_000.0;
            let mut filter = KWeightingFilter::new(sample_rate);
            let mut sum = 0.0f32;
            let mut count = 0usize;
            let total = sample_rate as usize;
            let warmup = (sample_rate * 0.1) as usize;
            for n in 0..total {
                let t = n as f32 / sample_rate;
                let x = 0.2 * (std::f32::consts::TAU * freq * t).sin();
                let y = filter.high_pass.process(filter.shelf.process(x));
                if n >= warmup {
                    sum += y * y;
                    count += 1;
                }
            }
            sum / count as f32
        }

        let sub_bass = weighted_sine_energy(60.0);
        let presence = weighted_sine_energy(3_000.0);
        assert!(
            presence > sub_bass * 3.0,
            "K-weighting should favor presence: sub={sub_bass}, presence={presence}"
        );
    }

    #[test]
    fn loudness_tracker_gates_absolute_silence() {
        let hop_dt = HOP as f32 / 48_000.0;
        let mut loudness = LoudnessTracker::new(hop_dt);

        let mut last = (1.0, 1.0, 1.0, 1.0);
        for _ in 0..500 {
            last = loudness.process(lufs_to_energy(-85.0), false);
        }

        assert_eq!(
            last.0, 0.0,
            "momentary should clamp below the -70 LUFS floor"
        );
        assert_eq!(
            last.1, 0.0,
            "short-term should be removed by the absolute gate"
        );
        assert_eq!(last.3, 0.0, "range should stay empty below the gate");
    }

    /// Build a `DspState` at the canonical 48 kHz / hop cadence for analysis-path
    /// tests that drive the full per-hop pipeline (including the newly wired
    /// `HpssBus` + `SpectrogramTrail`).
    fn analysis_state() -> DspState {
        let sample_rate = 48_000.0;
        let hop_dt = HOP as f32 / sample_rate;
        DspState::new(sample_rate, hop_dt, 1.0)
    }

    /// Push a fresh hop of `gen(sample_index)` through the sliding window and run
    /// `analyze_hop`, returning the published features. Mirrors the real worker's
    /// window-shift logic so the FFT sees a continuous stream.
    fn drive_hop<F: Fn(usize) -> f32>(
        state: &mut DspState,
        window: &mut Vec<f32>,
        sample_clock: &mut usize,
        gen: F,
    ) -> Features {
        window.copy_within(HOP.., 0);
        let tail = FFT_LEN - HOP;
        let mut hop = [0.0f32; HOP];
        for i in 0..HOP {
            let s = gen(*sample_clock);
            window[tail + i] = s;
            hop[i] = s;
            *sample_clock += 1;
        }
        state.analyze_hop(window, &hop, window, window)
    }

    /// A sustained sinusoid drives the public HPSS harmonic bus above the
    /// percussive bus (sustained partials are smooth across time).
    #[test]
    fn analysis_path_sustained_tone_is_harmonic() {
        let mut state = analysis_state();
        let mut window = vec![0.0f32; FFT_LEN];
        let mut clock = 0usize;
        let sample_rate = 48_000.0f32;
        let freq = 440.0f32;
        let tone = |n: usize| 0.5 * (std::f32::consts::TAU * freq * n as f32 / sample_rate).sin();

        let mut out = Features::default();
        for _ in 0..200 {
            out = drive_hop(&mut state, &mut window, &mut clock, tone);
        }

        assert_eq!(out.is_silent, 0.0, "tone should not be gated as silent");
        assert!(
            out.harmonic_ratio > out.percussive_ratio,
            "sustained tone should favor the harmonic bus: {:?} vs {:?}",
            out.harmonic_ratio,
            out.percussive_ratio
        );
        assert!(
            out.harmonic_ratio > 0.6,
            "harmonic ratio should clearly dominate for a pure tone: {}",
            out.harmonic_ratio
        );
        assert!(
            out.harmonic_level > out.percussive_level,
            "harmonic level should dominate for a pure tone: {} vs {}",
            out.harmonic_level,
            out.percussive_level
        );
    }

    /// A broadband impulse train (sharp clicks against near-silence) drives the
    /// public HPSS percussive bus above the harmonic bus on average. Single-sample
    /// impulses are spectrally flat (broadband), which the frequency-axis median
    /// routes onto the percussive bus, while the long gaps keep the time-axis
    /// (harmonic) median low.
    #[test]
    fn analysis_path_impulse_train_is_percussive() {
        let mut state = analysis_state();
        let mut window = vec![0.0f32; FFT_LEN];
        let mut clock = 0usize;
        let sample_rate = 48_000.0f32;
        // One sharp broadband impulse every ~200 ms — well beyond the harmonic
        // time-median window (~150 ms) so each click reads as a transient.
        let click_period = (sample_rate * 0.20) as usize;
        let signal = move |n: usize| {
            if n % click_period == 0 {
                1.0
            } else {
                0.0
            }
        };

        let mut perc_sum = 0.0f32;
        let mut harm_sum = 0.0f32;
        for _ in 0..240 {
            let out = drive_hop(&mut state, &mut window, &mut clock, signal);
            perc_sum += out.percussive_ratio;
            harm_sum += out.harmonic_ratio;
        }

        assert!(
            perc_sum > harm_sum,
            "an impulse train should be percussive on average: perc={perc_sum}, harm={harm_sum}"
        );
    }

    /// The wired spectrogram trail advances as hops are processed and its snapshot
    /// reflects the most-recent column. (Exercises `SpectrogramTrail` filling the
    /// shared `SpectrogramSnapshot` path the engine publishes.)
    #[test]
    fn analysis_path_spectrogram_advances() {
        let mut state = analysis_state();
        let mut window = vec![0.0f32; FFT_LEN];
        let mut clock = 0usize;
        let sample_rate = 48_000.0f32;
        let tone =
            |n: usize| 0.5 * (std::f32::consts::TAU * 1_000.0 * n as f32 / sample_rate).sin();

        assert_eq!(state.spectrogram.filled(), 0, "trail starts empty");
        let mut snap = state.spectrogram.snapshot();

        for hop in 1..=8 {
            drive_hop(&mut state, &mut window, &mut clock, tone);
            assert_eq!(
                state.spectrogram.filled(),
                hop,
                "trail should advance one column per hop"
            );
        }

        // Snapshot must mirror the live trail and expose a non-empty newest column.
        state.spectrogram.fill_snapshot(&mut snap);
        assert_eq!(snap.filled(), 8);
        let newest = snap.column(0).expect("newest spectrogram column");
        assert!(
            newest.iter().any(|&v| v > 0.0),
            "a 1 kHz tone should light up the spectrogram column: {newest:?}"
        );
        let (ring, write) = snap.raw_ring();
        assert_eq!(
            ring.len(),
            snap.frames() * snap.bins(),
            "ring sized frames×bins"
        );
        assert!(write < snap.frames(), "write cursor within ring");
    }

    /// Sustained loud audio then a long silence drains the public HPSS absolute
    /// rails toward zero (the gate forwards `is_silent` into the bus).
    #[test]
    fn analysis_path_silence_drains_hpss_levels() {
        let mut state = analysis_state();
        let mut window = vec![0.0f32; FFT_LEN];
        let mut clock = 0usize;
        let sample_rate = 48_000.0f32;
        let tone = |n: usize| 0.5 * (std::f32::consts::TAU * 440.0 * n as f32 / sample_rate).sin();

        for _ in 0..120 {
            drive_hop(&mut state, &mut window, &mut clock, tone);
        }
        let mut out = Features::default();
        for _ in 0..120 {
            out = drive_hop(&mut state, &mut window, &mut clock, |_| 0.0);
        }

        assert_eq!(out.is_silent, 1.0, "pure silence should gate");
        assert!(
            out.harmonic_level < 0.05 && out.percussive_level < 0.05,
            "gated silence should drain both HPSS bus levels: {:?}",
            out
        );
    }

    /// P2-AUD-025: only the three consumed macro lanes (sub_bass / low_mid /
    /// presence) are built and normalized — the discarded bass/mid/air FFT lanes
    /// and their per-lane state are gone.
    #[test]
    fn macro_bank_builds_only_used_lanes() {
        let state = analysis_state();
        assert_eq!(
            state.band_agc.len(),
            3,
            "only the 3 consumed macro lanes should carry AGC state"
        );
        assert_eq!(
            state.band_lp.len(),
            3,
            "only the 3 consumed macro lanes should carry LP state"
        );
        assert_eq!(
            state.macro_bank.len(),
            3,
            "the macro filterbank should build only the 3 consumed bands"
        );
    }

    /// P2-AUD-025: dropping the unused bass/mid/air macro lanes leaves the three
    /// consumed lanes bit-for-bit unchanged. Golden values were captured from the
    /// pre-fix 6-lane path on this exact deterministic signal.
    #[test]
    fn used_macro_lanes_match_prefix_golden() {
        let mut state = analysis_state();
        let mut window = vec![0.0f32; FFT_LEN];
        let mut clock = 0usize;
        let sr = 48_000.0f32;
        let gen = |n: usize| {
            let t = n as f32 / sr;
            0.45 * (std::f32::consts::TAU * 40.0 * t).sin()
                + 0.30 * (std::f32::consts::TAU * 300.0 * t).sin()
                + 0.20 * (std::f32::consts::TAU * 3000.0 * t).sin()
        };
        // Asymmetric tail: keep 40 Hz loud, drop 300 Hz / 3 kHz so `presence` falls
        // off its AGC peak to a discriminating sub-1.0 value.
        let tail = |n: usize| {
            let t = n as f32 / sr;
            0.45 * (std::f32::consts::TAU * 40.0 * t).sin()
                + 0.03
                    * (0.30 * (std::f32::consts::TAU * 300.0 * t).sin()
                        + 0.20 * (std::f32::consts::TAU * 3000.0 * t).sin())
        };
        let mut out = Features::default();
        for _ in 0..260 {
            out = drive_hop(&mut state, &mut window, &mut clock, gen);
        }
        for _ in 0..40 {
            out = drive_hop(&mut state, &mut window, &mut clock, tail);
        }
        assert_eq!(out.sub_bass, 1.0, "sub_bass lane changed vs pre-fix golden");
        assert_eq!(out.low_mid, 1.0, "low_mid lane changed vs pre-fix golden");
        assert!(
            (out.presence - 0.651_565_85).abs() < 1e-5,
            "presence lane changed vs pre-fix golden 0.65156585, got {}",
            out.presence
        );
    }

    /// P2-AUD-019: the drop predictor must see real loudness dynamics. A
    /// quiet→loud build should drive `drop_anticipation` up meaningfully; pre-fix
    /// the energy fed in was the peak-AGC'd `rms_norm` (stuck near 1.0), so the
    /// build produced no rising slope and the rail stayed near zero.
    #[test]
    fn drop_anticipation_rises_on_energy_build() {
        let mut state = analysis_state();
        let mut window = vec![0.0f32; FFT_LEN];
        let mut clock = 0usize;
        let sr = 48_000.0f32;
        let osc = |n: usize, amp: f32| amp * (std::f32::consts::TAU * 200.0 * n as f32 / sr).sin();

        // Quiet steady section (audible, not gated) — establishes a low baseline.
        let mut quiet_max = 0.0f32;
        for _ in 0..70 {
            let out = drive_hop(&mut state, &mut window, &mut clock, |n| osc(n, 0.03));
            quiet_max = quiet_max.max(out.drop_anticipation);
        }
        // Sustained build: amplitude ramps up over ~1 s.
        let build_hops = 100usize;
        let mut build_peak = 0.0f32;
        for i in 0..build_hops {
            let amp = 0.03 + 0.75 * (i as f32 / build_hops as f32);
            let out = drive_hop(&mut state, &mut window, &mut clock, move |n| osc(n, amp));
            build_peak = build_peak.max(out.drop_anticipation);
        }

        assert!(
            quiet_max < 0.15,
            "a quiet steady section should not read as a build: {quiet_max}"
        );
        assert!(
            build_peak > 0.25,
            "drop_anticipation should rise meaningfully during an energy build \
             (pre-fix it stayed stuck near zero): build_peak={build_peak}, quiet={quiet_max}"
        );
        assert!(
            build_peak > quiet_max + 0.2,
            "the build must lift the rail well above the quiet baseline: \
             build={build_peak}, quiet={quiet_max}"
        );
    }

    #[test]
    fn loudness_range_tracks_contrasting_sections() {
        let hop_dt = HOP as f32 / 48_000.0;
        let mut loudness = LoudnessTracker::new(hop_dt);
        let section_hops = (3.0 / hop_dt).round() as usize;
        let mut max_range = 0.0f32;

        for _ in 0..3 {
            for _ in 0..section_hops {
                let (_, _, _, range) = loudness.process(lufs_to_energy(-36.0), false);
                max_range = max_range.max(range);
            }
            for _ in 0..section_hops {
                let (_, _, _, range) = loudness.process(lufs_to_energy(-16.0), false);
                max_range = max_range.max(range);
            }
        }

        assert!(
            max_range > 0.35,
            "contrasting sections should produce a visible LRA rail, got {max_range}"
        );
    }

    /// P2-AUD-012: the bounded online loudness-range estimator returns exactly the
    /// same value as the previous allocate-and-full-sort path across a long,
    /// varied loudness stream, and never reallocates its scratch (no per-hop
    /// allocation).
    #[test]
    fn loudness_range_matches_full_sort_without_reallocating() {
        // Reference: the pre-fix gather + full-sort + interpolated percentiles.
        fn reference_lra(t: &LoudnessTracker) -> f32 {
            let count = t.range_filled;
            if count < 2 {
                return 0.0;
            }
            let ring = t.range_lufs.len();
            let mut values = Vec::new();
            let mut energy_sum = 0.0f32;
            for offset in 0..count {
                let idx = (t.range_write + ring - 1 - offset) % ring;
                let lufs = t.range_lufs[idx];
                if lufs > ABS_GATE_LUFS {
                    energy_sum += lufs_to_energy(lufs);
                    values.push(lufs);
                }
            }
            if values.len() < 2 {
                return 0.0;
            }
            let preliminary = energy_to_lufs(energy_sum / values.len() as f32);
            let threshold = (preliminary - LRA_REL_GATE_LU).max(ABS_GATE_LUFS);
            values.retain(|&v| v >= threshold);
            if values.len() < 2 {
                return 0.0;
            }
            values.sort_by(|a, b| a.total_cmp(b));
            let p10 = percentile_sorted(&values, 0.10);
            let p95 = percentile_sorted(&values, 0.95);
            (p95 - p10).max(0.0)
        }

        let hop_dt = HOP as f32 / 48_000.0;
        let mut loudness = LoudnessTracker::new(hop_dt);

        // A long, non-trivial loudness stream so the gated percentile path (retain
        // + interpolation) is actually exercised, not just the < 2 early return.
        let lufs_at = |i: usize| -18.0 - 22.0 * ((i % 11) as f32 / 10.0);

        // Warm past the range ring so the scratch is at full working size, then
        // freeze the capacity we expect to hold.
        for i in 0..2500 {
            loudness.process(lufs_to_energy(lufs_at(i)), false);
        }
        let warm_capacity = loudness.range_scratch.capacity();

        let mut exercised_nonzero = false;
        for i in 2500..5000 {
            loudness.process(lufs_to_energy(lufs_at(i)), false);
            let got = loudness.loudness_range_lu();
            let want = reference_lra(&loudness);
            assert_eq!(
                got, want,
                "online LRA diverged from the full-sort reference at hop {i}: {got} vs {want}"
            );
            exercised_nonzero |= got > 0.0;
            assert_eq!(
                loudness.range_scratch.capacity(),
                warm_capacity,
                "range scratch reallocated at hop {i} — per-hop allocation regressed"
            );
        }
        assert!(
            exercised_nonzero,
            "test signal never produced a non-zero LRA; percentile path not exercised"
        );
    }

    /// P2-AUD-001: at a high native rate the stream is decimated to a canonical
    /// analysis rate, so the fixed FFT/hop geometry keeps enough low-frequency
    /// resolution to detect a bass-register pitch. Pre-fix, a 2048-sample window
    /// at 192 kHz spanned only ~10 ms — less than one period of 55 Hz — so the
    /// autocorrelation could not resolve the fundamental (pitch_hz stayed 0).
    #[test]
    fn high_native_rate_resolves_low_pitch() {
        let native = 192_000u32;
        let mut analyzer = Analyzer::new(native, 1.0);
        // Decimation brings the analysis rate into the canonical band.
        let eff = analyzer.analysis_sample_rate();
        assert!(
            (eff - 48_000.0).abs() < 1.0,
            "192 kHz should decimate to ~48 kHz, got {eff}"
        );

        // ~1.4 s of a strong 55 Hz sine at the native rate.
        let freq = 55.0f32;
        let n = (native as f32 * 1.4) as usize;
        let mut ch = vec![0.0f32; n];
        for (i, s) in ch.iter_mut().enumerate() {
            *s = 0.5 * (std::f32::consts::TAU * freq * i as f32 / native as f32).sin();
        }
        let out = *analyzer.push_planar(&ch, &ch);

        assert_eq!(out.is_silent, 0.0, "a loud tone must not gate as silent");
        assert!(
            (40.0..75.0).contains(&out.pitch_hz),
            "55 Hz fundamental should be detected after decimation, got {} Hz",
            out.pitch_hz
        );
    }

    /// P2-AUD-004: a large valid block must be fully drained into hops, not trimmed
    /// down to a handful of samples before the hop loop runs. A 65,536-frame block
    /// at a passthrough (48 kHz) rate is exactly 128 whole hops.
    #[test]
    fn large_block_drains_every_hop_without_trimming() {
        let mut analyzer = Analyzer::new(48_000u32, 1.0);
        let block = FFT_LEN * 32; // 65,536 frames
        let ch = vec![0.05f32; block]; // any finite audio; count is what matters
        analyzer.push_planar(&ch, &ch);

        let expected = (block / HOP) as u64; // 128
        assert_eq!(
            analyzer.hops_processed, expected,
            "a {block}-frame block should drain into {expected} hops, got {} (samples were trimmed)",
            analyzer.hops_processed
        );
        // Staging must be left below one hop — nothing buffered, nothing dropped.
        assert!(
            analyzer.pending.len() < HOP,
            "pending should hold < HOP frames after draining, got {}",
            analyzer.pending.len()
        );
    }

    /// P2-AUD-004 (companion): the eager mid-block drain is numerically transparent —
    /// pushing one giant block yields the same final features and hop count as
    /// pushing the identical signal in many small chunks.
    #[test]
    fn eager_drain_matches_chunked_push() {
        let gen = |i: usize| 0.6 * (std::f32::consts::TAU * 220.0 * i as f32 / 48_000.0).sin();
        let total = FFT_LEN * 20;
        let signal: Vec<f32> = (0..total).map(gen).collect();

        let mut whole = Analyzer::new(48_000u32, 1.0);
        let whole_out = *whole.push_planar(&signal, &signal);

        let mut chunked = Analyzer::new(48_000u32, 1.0);
        for chunk in signal.chunks(333) {
            chunked.push_planar(chunk, chunk);
        }
        let chunked_out = *chunked.latest();

        assert_eq!(
            whole.hops_processed, chunked.hops_processed,
            "hop counts must match regardless of push granularity"
        );
        assert_eq!(
            whole_out.pitch_hz, chunked_out.pitch_hz,
            "final features must be identical regardless of push granularity"
        );
        assert_eq!(whole_out.rms_level, chunked_out.rms_level);
        assert_eq!(whole_out.harmonic_ratio, chunked_out.harmonic_ratio);
    }

    /// P2-AUD-007: `waveform_*_full` must be the verbatim most-recent raw waveform
    /// (the newest WAVEFORM_SAMPLES_FULL samples), not signed peak buckets decimated
    /// over the whole analysis window.
    #[test]
    fn waveform_full_is_recent_raw_samples_not_peak_buckets() {
        let sample_rate = 48_000u32; // passthrough decimator — bit-identical staging
        let mut analyzer = Analyzer::new(sample_rate, 1.0);

        // A deterministic, non-monotonic signal so raw samples differ clearly from
        // any peak-bucket decimation of the surrounding window.
        let total = FFT_LEN * 2; // whole hops, window fully primed
        let gen = |i: usize| 0.9 * (std::f32::consts::TAU * 300.0 * i as f32 / 48_000.0).sin();
        let ch: Vec<f32> = (0..total).map(gen).collect();
        let out = *analyzer.push_planar(&ch, &ch);

        // The published full waveform is exactly the last WAVEFORM_SAMPLES_FULL
        // samples of the stream, in order.
        for (j, &got) in out.waveform_left_full.iter().enumerate() {
            let want = gen(total - WAVEFORM_SAMPLES_FULL + j);
            assert!(
                (got - want).abs() < 1e-6,
                "waveform_left_full[{j}] = {got}, expected recent raw sample {want}"
            );
        }
        assert_eq!(
            out.waveform_right_full, out.waveform_left_full,
            "mono-duplicated channels should match"
        );

        // Guard against a regression back to peak buckets: a 4:1 peak-decimation of
        // the 2048-sample window would not reproduce the raw tail sample-for-sample.
        let mut peak_buckets = [0.0f32; WAVEFORM_SAMPLES_FULL];
        let window: Vec<f32> = (total - FFT_LEN..total).map(gen).collect();
        for (i, dst) in peak_buckets.iter_mut().enumerate() {
            let start = i * FFT_LEN / WAVEFORM_SAMPLES_FULL;
            let end = ((i + 1) * FFT_LEN / WAVEFORM_SAMPLES_FULL).max(start + 1);
            let mut peak = 0.0f32;
            for &s in &window[start..end] {
                if s.abs() > peak.abs() {
                    peak = s;
                }
            }
            *dst = peak;
        }
        assert!(
            out.waveform_left_full != peak_buckets,
            "waveform_full must not equal the old peak-bucket decimation"
        );
    }

    /// P2-AUD-005: the spectrum published to `bSpectrum` custom waves must be the
    /// exact Butterchurn `freqArray` (no window, signed -128..127 scaling, equalize
    /// curve) — i.e. a verbatim copy of the faithful port's output, not a
    /// Hann-windowed realfft approximation on a [-1,1] scale.
    #[test]
    fn freq_spectrum_is_reference_butterchurn_freq_array() {
        let mut state = analysis_state();
        let mut window = vec![0.0f32; FFT_LEN];
        let mut clock = 0usize;
        let sr = 48_000.0f32;
        let tone = |n: usize| 0.4 * (std::f32::consts::TAU * 220.0 * n as f32 / sr).sin();

        let mut out = Features::default();
        for _ in 0..8 {
            out = drive_hop(&mut state, &mut window, &mut clock, tone);
        }

        let reference = state.butterchurn.freq_array();
        assert_eq!(
            out.freq_spectrum.len(),
            reference.len(),
            "freq_spectrum must be the full Butterchurn freqArray"
        );
        for (i, (&got, &want)) in out.freq_spectrum.iter().zip(reference.iter()).enumerate() {
            assert_eq!(got, want, "freq_spectrum[{i}] must equal the freqArray bin");
        }
        assert!(
            out.freq_spectrum.iter().any(|&v| v > 0.0),
            "a sustained tone should populate the published freqArray"
        );
    }

    /// P2-AUD-006: analyzer construction validates the sample rate and fails for
    /// pathological rates instead of allocating (unbounded) histories, while the
    /// infallible constructor clamps into the supported band.
    #[test]
    fn analyzer_construction_rejects_pathological_rates() {
        // Fallible path rejects zero and out-of-range rates.
        assert!(
            Analyzer::try_new(0, 1.0).is_err(),
            "zero rate must be rejected"
        );
        assert!(
            Analyzer::try_new(2_000, 1.0).is_err(),
            "sub-audio rate must be rejected"
        );
        assert!(
            Analyzer::try_new(2_000_000, 1.0).is_err(),
            "absurdly high rate must be rejected"
        );
        assert!(
            Analyzer::try_new(48_000, 1.0).is_ok(),
            "48 kHz must be accepted"
        );

        // Infallible path clamps a pathological rate into a bounded analysis rate.
        let clamped = Analyzer::new(0, 1.0).analysis_sample_rate();
        assert!(
            clamped.is_finite() && clamped >= 4_000.0,
            "new() must clamp a bad rate to a bounded analysis rate, got {clamped}"
        );
    }
}

//! DSP worker (spec §1/§5). Runs on a dedicated background thread, drains the
//! capture ring into a sliding analysis window, and per-hop computes the full
//! [`Features`] set, then publishes a snapshot through the lock-free triple buffer.
//!
//! Cached once, never re-allocated per hop (the classic perf pitfall the spec
//! warns about): the FFT plan, its scratch + spectrum buffers, the Hann window,
//! and both filterbanks.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use realfft::num_complex::Complex;
use realfft::{RealFftPlanner, RealToComplex};
use rtrb::Consumer;

use crate::analysis::{
    self, FilterBank, MACRO_BANDS_HZ, SPECTRUM_BANDS, SPECTRUM_HI_HZ, SPECTRUM_LO_HZ,
};
use crate::butterchurn::{self, ButterchurnLevels};
use crate::complex_onset::ComplexOnsetDetector;
use crate::hpss::HpssSeparator;
use crate::hpss_bus::HpssBus;
use crate::linkwitz_riley::LinkwitzRileyBank;
use crate::onset::OnsetDetector;
use crate::predictive_drop::DropPredictor;
use crate::smoothing::{Agc, AsymEnv, OnePole, ReactiveLevel, SilenceGate};
use crate::spectrogram::{SpectrogramSnapshot, SpectrogramTrail};
use crate::structure::StructureTracker;
use crate::tempo::TempoTracker;
use crate::tonal;
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

/// Parameters the worker needs that come from the live capture device.
pub struct DspParams {
    pub sample_rate: u32,
    /// User detection sensitivity (~0.1..3). 1.0 = neutral.
    pub sensitivity: f32,
}

/// Run the DSP loop until `running` is cleared (or the producer is abandoned —
/// i.e. the capture stream/engine was dropped).
pub fn run(
    mut consumer: Consumer<CaptureFrame>,
    mut writer: Writer<Features>,
    spectrogram: Arc<Mutex<SpectrogramSnapshot>>,
    params: DspParams,
    running: Arc<AtomicBool>,
) {
    let sample_rate = params.sample_rate as f32;
    let hop_dt = HOP as f32 / sample_rate;

    let mut state = DspState::new(sample_rate, hop_dt, params.sensitivity);

    // Sliding windows: mono feeds the existing FFT/onsets; left/right are kept for
    // sampled PCM scope/audio-texture consumers.
    let mut window = vec![0.0f32; FFT_LEN];
    let mut window_left = vec![0.0f32; FFT_LEN];
    let mut window_right = vec![0.0f32; FFT_LEN];
    // Staging area for newly drained samples before they enter the window.
    let mut pending: Vec<CaptureFrame> = Vec::with_capacity(HOP * 4);

    // ~hop cadence; if the ring under-runs we simply wait and retry.
    let poll = Duration::from_micros((hop_dt * 1e6 * 0.5) as u64).max(Duration::from_micros(500));

    while running.load(Ordering::Relaxed) {
        if consumer.is_abandoned() && consumer.slots() == 0 {
            // Capture gone and nothing left to process.
            break;
        }

        // Drain everything currently available (cheap; pop is lock-free).
        while let Ok(s) = consumer.pop() {
            pending.push(s);
            // Guard against unbounded growth if the worker ever falls behind.
            if pending.len() > FFT_LEN * 8 {
                let drop_to = pending.len() - FFT_LEN * 4;
                pending.drain(0..drop_to);
            }
        }

        // Process as many whole hops as we have buffered.
        while pending.len() >= HOP {
            // Shift window left by HOP, append the next HOP samples.
            window.copy_within(HOP.., 0);
            window_left.copy_within(HOP.., 0);
            window_right.copy_within(HOP.., 0);
            let tail = FFT_LEN - HOP;
            let mut hop_mono = [0.0f32; HOP];
            for (i, frame) in pending.iter().take(HOP).enumerate() {
                window[tail + i] = frame.mono;
                window_left[tail + i] = frame.left;
                window_right[tail + i] = frame.right;
                hop_mono[i] = frame.mono;
            }

            let features = state.analyze_hop(&window, &hop_mono, &window_left, &window_right);
            writer.write(features);
            // Publish the scrolling-spectrogram trail for the render thread. The
            // trail was advanced inside `analyze_hop`; copy its ring into the
            // shared snapshot under a brief lock (a single memcpy of preallocated
            // storage — no allocation on the audio thread).
            if let Ok(mut snap) = spectrogram.lock() {
                state.spectrogram.fill_snapshot(&mut snap);
            }
            pending.drain(0..HOP);
        }

        if pending.len() < HOP {
            std::thread::sleep(poll);
        }
    }

    log::info!("particle-audio: DSP worker exiting");
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

    // Smoothing / normalization.
    band_lp: [OnePole; 6],
    band_agc: [Agc; 6],
    spectrum_rails: SpectrumRailBank,
    chroma_lp: [OnePole; CHROMA_BINS],
    key_smoother: tonal::KeySmoother,
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
    /// Butterchurn equalize curve `-0.02*ln((512-i)/512)`, applied per freq bin.
    equalize: [f32; FREQ_SPECTRUM_BINS],
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

        let macro_bank = FilterBank::from_edges(&MACRO_BANDS_HZ, FFT_LEN, sample_rate);
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

        // Per-band smoothing: light LP + AGC with slow decay & a small floor.
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
        // Butterchurn equalize: eq[i] = -0.02 * ln((numSamps - i) / numSamps).
        let inv_n = 1.0 / FREQ_SPECTRUM_BINS as f32;
        let equalize: [f32; FREQ_SPECTRUM_BINS] =
            std::array::from_fn(|i| -0.02 * (((FREQ_SPECTRUM_BINS - i) as f32) * inv_n).ln());

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
            hpss: HpssSeparator::new(n_bins, sample_rate, bin_hz, hop_dt),
            // Sized to the worker's STFT: `n_bins = FFT_LEN/2 + 1`, hop period in
            // seconds. Both reuse `self.mag` each hop — no extra FFT.
            hpss_bus: HpssBus::new(n_bins, hop_dt),
            spectrogram: SpectrogramTrail::new(FFT_LEN, sample_rate),
            structure: StructureTracker::new(),
            band_lp,
            band_agc,
            spectrum_rails,
            chroma_lp,
            key_smoother: tonal::KeySmoother::new(),
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
            equalize,
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

        // --- Butterchurn freq_spectrum + volume-independent band reactivity ---
        // Derive Butterchurn's 512-bin freqArray from our 2048-pt FFT: its bin j at
        // bucketHz = sr/1024 = 2j*(sr/2048), i.e. our self.mag[2j]. Apply its
        // equalize curve so the highs read at the same relative level as Butterchurn.
        let mut freq_spectrum = [0.0f32; FREQ_SPECTRUM_BINS];
        for j in 0..FREQ_SPECTRUM_BINS {
            let m = self.mag.get(2 * j).copied().unwrap_or(0.0);
            freq_spectrum[j] = self.equalize[j] * m;
        }
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

        let hpss = self.hpss.analyze(&self.mag, is_silent, self.sensitivity);

        // Public median-filtering HPSS dual-bus rails + scrolling spectrogram.
        // Both consume the same STFT magnitude frame computed above (no second
        // FFT) and own their smoothing/normalization internally.
        let hpss_bus = self.hpss_bus.process(&self.mag, is_silent);
        self.spectrogram.push(&self.mag, is_silent);

        // --- macro bands → dB → AGC → LP ---
        let mut raw_bands = [0.0f32; 6];
        self.macro_bank.apply(&self.mag, &mut raw_bands);
        let mut bands = [0.0f32; 6];
        for i in 0..6 {
            let db = lin_to_db_norm(raw_bands[i]);
            let agc = self.band_agc[i].process(db);
            bands[i] = self.band_lp[i].process(agc);
        }

        // --- Butterchurn-parity bass/mid/treb + `_att` ---
        // Feed the most-recent FFT_SIZE mono samples (scaled to Butterchurn's
        // signed -128..127 domain) into the FPS-aware levels follower. Butterchurn's
        // FFT is FFT_SIZE-point, so feeding only NUM_SAMPS would zero the second
        // transform half — feed the full window. The result hovers around ~1.0
        // because each band is divided by its long-term running average, matching
        // the reference renderer's normalization.
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
            tonal::estimate_mono_pitch(window, self.sample_rate, 45.0, 1600.0)
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
        // Energy = RMS-level rail; centroid = brightness; high = presence band;
        // sub = sub-bass band; flux drives the activity term.
        let drop_anticipation = self
            .drop_predictor
            .process(rms_norm, flux, brightness, bands[4], bands[0], is_silent);
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
            sub_bass: bands[0],
            // `bass`/`mid`/`air` carry the Butterchurn-normalized rails (~1.0
            // baseline) that the MilkDrop drop path reads as `bass`/`mid`/`treb`.
            // `sub_bass`/`low_mid`/`presence` keep the 0..1 AGC macro bands.
            bass: bc.bass,
            low_mid: bands[2],
            mid: bc.mid,
            presence: bands[4],
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
            // 512-sample waveform from the 2048-sample sliding window (peak-decimated,
            // same scheme as the 32-sample fields). The window is long enough that
            // 512 samples is fully available — no truncation.
            waveform_left_full: downsample_waveform_full(left),
            waveform_right_full: downsample_waveform_full(right),
        }
    }
}

fn downsample_waveform_full(samples: &[f32]) -> [f32; WAVEFORM_SAMPLES_FULL] {
    let mut out = [0.0f32; WAVEFORM_SAMPLES_FULL];
    if samples.is_empty() {
        return out;
    }
    for (i, dst) in out.iter_mut().enumerate() {
        let start = i * samples.len() / WAVEFORM_SAMPLES_FULL;
        let end = ((i + 1) * samples.len() / WAVEFORM_SAMPLES_FULL)
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
        let y = self.b0 * x + self.z1;
        self.z1 = self.b1 * x - self.a1 * y + self.z2;
        self.z2 = self.b2 * x - self.a2 * y;
        y
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

    fn loudness_range_lu(&self) -> f32 {
        let count = self.range_filled;
        if count < 2 {
            return 0.0;
        }

        let mut values = Vec::with_capacity(count);
        let mut energy_sum = 0.0f32;
        for offset in 0..count {
            let idx =
                (self.range_write + self.range_lufs.len() - 1 - offset) % self.range_lufs.len();
            let lufs = self.range_lufs[idx];
            if lufs > ABS_GATE_LUFS {
                energy_sum += lufs_to_energy(lufs);
                values.push(lufs);
            }
        }
        if values.len() < 2 {
            return 0.0;
        }

        let preliminary_lufs = energy_to_lufs(energy_sum / values.len() as f32);
        let threshold = (preliminary_lufs - LRA_REL_GATE_LU).max(ABS_GATE_LUFS);
        values.retain(|&v| v >= threshold);
        if values.len() < 2 {
            return 0.0;
        }

        values.sort_by(|a, b| a.total_cmp(b));
        let p10 = percentile_sorted(&values, 0.10);
        let p95 = percentile_sorted(&values, 0.95);
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
}

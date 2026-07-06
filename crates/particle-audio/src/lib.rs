//! particle-audio — self-contained real-time audio capture + DSP.
//!
//! The engine captures audio from the default input device (mic / line-in) and
//! runs its own DSP — FFT, log/mel macro bands, spectral dynamics, per-band
//! onset detection, and a pure-Rust autocorrelation tempo + phase-locked beat
//! tracker — entirely on background threads. The render/UI thread calls
//! [`AudioEngine::latest`] for a lock-free snapshot of the most recent analysis;
//! it never blocks.
//!
//! Thread model (spec §1):
//! ```text
//!   cpal callback  ──push──▶  rtrb SPSC ring  ──drain──▶  DSP worker
//!   (no alloc/FFT/lock)                                  (FFT + features +
//!                                                         smoothing/AGC)
//!                                                              │ publish
//!                                                              ▼
//!                                                    lock-free triple buffer
//!                                                              │ read
//!                                                              ▼
//!                                                    AudioEngine::latest()
//! ```
//!
//! ## License posture
//! cpal (Apache-2.0), realfft/rustfft (MIT/Apache), rtrb (MIT/Apache). Onset,
//! tempo, and beat tracking are reimplemented in pure Rust — **no GPL aubio**.

mod analysis;
pub mod butterchurn;
#[cfg(feature = "capture")]
mod capture;
#[cfg(all(feature = "capture", target_os = "macos"))]
mod capture_macos;
mod complex_onset;
mod dsp;
mod hpss;
mod hpss_bus;
mod linkwitz_riley;
mod onset;
mod predictive_drop;
mod smoothing;
mod spectrogram;
mod structure;
mod tempo;
mod tonal;
mod triple_buffer;

pub use butterchurn::{
    AudioLevels as ButterchurnAudioLevels, ButterchurnLevels, DEFAULT_SAMPLE_RATE,
};
#[cfg(feature = "capture")]
pub use capture::{CaptureConfig, CaptureSource};
pub use dsp::Analyzer;
pub use hpss_bus::{HpssBus, HpssLevels};
pub use spectrogram::{
    SpectrogramSnapshot, SpectrogramTrail, DEFAULT_BINS as SPECTROGRAM_DEFAULT_BINS,
    DEFAULT_FRAMES as SPECTROGRAM_DEFAULT_FRAMES, SPECTROGRAM_HI_HZ, SPECTROGRAM_LO_HZ,
};

/// FFT window size used by the DSP worker — exposed so an integrator wiring
/// [`HpssBus`] / [`SpectrogramTrail`] alongside the worker can size them to the
/// same STFT (`n_bins = FFT_LEN / 2 + 1`) and hop period.
pub const FFT_LEN: usize = dsp::FFT_LEN;
/// STFT hop in samples (frames advance every `HOP / sample_rate` seconds).
pub const HOP: usize = dsp::HOP;

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;

/// PCM samples published with every analysis snapshot for GPU audio-texture consumers.
/// Small enough to keep the lock-free snapshot cheap, large enough for a scope trace.
pub const WAVEFORM_SAMPLES: usize = 32;
/// Butterchurn-native full-resolution waveform length (512), for MilkDrop waveform
/// modes that need more samples than the 32-sample scope fields.
pub const WAVEFORM_SAMPLES_FULL: usize = 512;
/// Butterchurn-shaped FFT magnitude array length (its `freqArray`), 512 bins.
pub const FREQ_SPECTRUM_BINS: usize = 512;
pub const CHROMA_BINS: usize = tonal::CHROMA_BINS;

/// One realtime capture frame. `mono` feeds the existing FFT/onset analysis; `left` and
/// `right` preserve the first two input channels before downmixing for scope-style looks.
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct CaptureFrame {
    pub mono: f32,
    pub left: f32,
    pub right: f32,
}

/// Per-frame analysis result. All band/level fields normalized 0..1 (AGC'd, smoothed).
///
/// Field names/types are a FROZEN contract: they mirror the host engine's
/// `AudioInput` so a consumer can map one to the other trivially. Do not rename.
#[derive(Clone, Copy, Debug)]
pub struct Features {
    // 6 macro bands (0..1)
    pub sub_bass: f32,
    pub bass: f32,
    pub low_mid: f32,
    pub mid: f32,
    pub presence: f32,
    pub air: f32,
    // dynamics
    pub rms_level: f32,
    pub brightness: f32, // spectral centroid 0..1
    pub flux: f32,
    pub rolloff: f32,
    pub lufs_momentary: f32,
    pub lufs_short: f32,
    pub loudness_build: f32,
    pub lufs_range: f32,
    pub spectral_flatness: f32,
    pub spectral_spread: f32,
    pub spectral_contrast: f32,
    pub superflux: f32,
    pub superflux_onset: f32,
    // transients (impulse 0..1 with attack/decay envelopes)
    pub kick_onset: f32,
    pub snare_onset: f32,
    pub hat_onset: f32,
    pub beat_confidence: f32,
    // tempo
    pub beat_phase: f32, // continuous 0..1 sawtooth between beats
    pub bar_phase: f32,  // 0..1 over 4 beats
    pub bpm: f32,
    pub beat_impulse: f32,
    // gate
    pub is_silent: f32, // 0.0 = audio present, 1.0 = silent
    /// Smoothed log-band emitter/EQ rails, 0..1 each (optional; zeroed if unused).
    pub spectrum: [f32; 32],
    /// FFT-derived 12-bin chromagram, C=0 through B=11, fed from the causal
    /// harmonic HPSS mask. This is still not the backlog's full CQT tier.
    pub chroma: [f32; CHROMA_BINS],
    /// Detected key root around the 12-tone wheel: C=0.0, C#=1/12, ..., B=11/12.
    pub key_root: f32,
    /// 0.0 = major, 1.0 = minor.
    pub key_is_minor: f32,
    pub key_confidence: f32,
    /// Detected triad/chord root around the 12-tone wheel.
    pub chord_root: f32,
    /// 0.0 = major, 1.0 = minor.
    pub chord_is_minor: f32,
    pub chord_confidence: f32,
    /// Palette-oriented key hue and major/minor mood rails.
    pub harmony_hue: f32,
    pub harmony_mood: f32,
    /// Downsampled raw PCM from the latest analysis window. Mono sources duplicate into
    /// both channels; stereo/loopback sources preserve L/R before the mono analysis fold.
    pub waveform_left: [f32; WAVEFORM_SAMPLES],
    pub waveform_right: [f32; WAVEFORM_SAMPLES],
    /// Causal HPSS approximation rails. `perc_*` comes from a frequency-median mask
    /// and is meant for drum/transient motion; `harm_*` comes from a short time-median
    /// mask and is meant for sustained tone/color motion.
    pub perc_rms: f32,
    pub perc_flux: f32,
    pub perc_onset: f32,
    pub perc_ratio: f32,
    pub harm_rms: f32,
    pub harm_flux: f32,
    pub harm_brightness: f32,
    pub harm_ratio: f32,
    /// Butterchurn-parity attenuated (`att = avg / longAvg`) slow-follower rails,
    /// mirroring the host engine's `AudioInput`. The MilkDrop drop path reads these
    /// as `bass_att`/`mid_att`/`treb_att`, with `vol_att` the band mean.
    pub bass_att: f32,
    pub mid_att: f32,
    pub treb_att: f32,
    pub vol_att: f32,
    /// Complex-domain onset rail (Bello et al. 2004): a sharper transient spike
    /// than spectral-flux onsets — it also fires on phase-only (soft/tonal)
    /// onsets, not just magnitude bursts. Normalized 0..1.
    pub complex_onset: f32,
    /// Linkwitz-Riley 4th-order crossover band energies (0..1 each). Steeper and
    /// cleaner than the FFT triangular macro bands: sub / low / mid / high / air.
    pub lr_sub: f32,
    pub lr_low: f32,
    pub lr_mid: f32,
    pub lr_high: f32,
    pub lr_air: f32,
    /// Build-up / drop anticipation rail (0..1): rises as a drop approaches
    /// (rising energy + sustained flux + filter-sweep tension) and collapses
    /// once the drop lands.
    pub drop_anticipation: f32,
    /// Median-filtering harmonic/percussive dual-bus separation rails (the public
    /// [`HpssBus`] block, distinct from the older internal `perc_*`/`harm_*` rails
    /// above). All normalized 0..1. `*_level` are AGC'd absolute bus energies;
    /// `*_ratio` are the loudness-independent balance between the two buses
    /// (`harm/(harm+perc)` and its complement) — ~1.0 for pure tone, ~0.0 for pure
    /// percussion.
    pub harmonic_level: f32,
    pub percussive_level: f32,
    pub harmonic_ratio: f32,
    pub percussive_ratio: f32,
    /// Monophonic f0 estimate from the current analysis window. `0.0` means
    /// unvoiced/unknown; `pitch_norm` is the log-frequency 40 Hz..2 kHz rail.
    pub pitch_hz: f32,
    pub pitch_norm: f32,
    pub pitch_confidence: f32,
    /// Live structure rails from a rolling self-similarity / Foote-novelty
    /// tracker. `structure_change` is a decaying section-change impulse,
    /// `structure_confidence` gates history/energy readiness.
    pub structure_novelty: f32,
    pub structure_change: f32,
    pub structure_confidence: f32,

    // --- Butterchurn-faithful volume-independent reactivity (the signal the native
    // MilkDrop/Butterchurn renderer reacts to). Each is a ratio of the band's
    // immediate energy to its long-term rolling average (~1.0 at the recent average,
    // 0 when quiet, 2-4 on a hit) — independent of absolute volume.
    /// `imm / longAvg` for the bass band (20-320 Hz) — MilkDrop `bass`.
    pub bass_react: f32,
    /// `imm / longAvg` for the mid band (320-2800 Hz) — MilkDrop `mid`.
    pub mid_react: f32,
    /// `imm / longAvg` for the treb band (2800-11025 Hz) — MilkDrop `treb`.
    pub treb_react: f32,
    /// `avg / longAvg` for the bass band — MilkDrop `bass_att`.
    pub bass_react_att: f32,
    /// `avg / longAvg` for the mid band — MilkDrop `mid_att`.
    pub mid_react_att: f32,
    /// `avg / longAvg` for the treb band — MilkDrop `treb_att`.
    pub treb_react_att: f32,
    /// Mean of the three react ratios — Butterchurn `vol`.
    pub vol_react: f32,
    /// Mean of the three react_att ratios — Butterchurn `vol_att`.
    pub vol_react_att: f32,
    /// Butterchurn-shaped FFT magnitude array (`freqArray`), 512 bins, for
    /// `bSpectrum` custom waveforms. Mono.
    pub freq_spectrum: [f32; FREQ_SPECTRUM_BINS],
    /// Full-resolution per-sample waveform (512 samples, ~[-1,1]) at Butterchurn's
    /// native resolution so MilkDrop waveform modes have enough samples. Mono
    /// sources duplicate L/R.
    pub waveform_left_full: [f32; WAVEFORM_SAMPLES_FULL],
    pub waveform_right_full: [f32; WAVEFORM_SAMPLES_FULL],
}

impl Default for Features {
    fn default() -> Self {
        Self {
            sub_bass: 0.0,
            bass: 0.0,
            low_mid: 0.0,
            mid: 0.0,
            presence: 0.0,
            air: 0.0,
            rms_level: 0.0,
            brightness: 0.0,
            flux: 0.0,
            rolloff: 0.0,
            lufs_momentary: 0.0,
            lufs_short: 0.0,
            loudness_build: 0.0,
            lufs_range: 0.0,
            spectral_flatness: 0.0,
            spectral_spread: 0.0,
            spectral_contrast: 0.0,
            superflux: 0.0,
            superflux_onset: 0.0,
            kick_onset: 0.0,
            snare_onset: 0.0,
            hat_onset: 0.0,
            beat_confidence: 0.0,
            beat_phase: 0.0,
            bar_phase: 0.0,
            bpm: 0.0,
            beat_impulse: 0.0,
            is_silent: 0.0,
            spectrum: [0.0; 32],
            chroma: [0.0; CHROMA_BINS],
            key_root: 0.0,
            key_is_minor: 0.0,
            key_confidence: 0.0,
            chord_root: 0.0,
            chord_is_minor: 0.0,
            chord_confidence: 0.0,
            harmony_hue: 0.0,
            harmony_mood: 0.0,
            waveform_left: [0.0; WAVEFORM_SAMPLES],
            waveform_right: [0.0; WAVEFORM_SAMPLES],
            perc_rms: 0.0,
            perc_flux: 0.0,
            perc_onset: 0.0,
            perc_ratio: 0.0,
            harm_rms: 0.0,
            harm_flux: 0.0,
            harm_brightness: 0.0,
            harm_ratio: 0.0,
            bass_att: 0.0,
            mid_att: 0.0,
            treb_att: 0.0,
            vol_att: 0.0,
            complex_onset: 0.0,
            lr_sub: 0.0,
            lr_low: 0.0,
            lr_mid: 0.0,
            lr_high: 0.0,
            lr_air: 0.0,
            drop_anticipation: 0.0,
            harmonic_level: 0.0,
            percussive_level: 0.0,
            harmonic_ratio: 0.0,
            percussive_ratio: 0.0,
            pitch_hz: 0.0,
            pitch_norm: 0.0,
            pitch_confidence: 0.0,
            structure_novelty: 0.0,
            structure_change: 0.0,
            structure_confidence: 0.0,
            // Reactivity ratios default to 1.0 (band at its average) — never 0,
            // which would read as silence on the first frame.
            bass_react: 1.0,
            mid_react: 1.0,
            treb_react: 1.0,
            bass_react_att: 1.0,
            mid_react_att: 1.0,
            treb_react_att: 1.0,
            vol_react: 1.0,
            vol_react_att: 1.0,
            freq_spectrum: [0.0; FREQ_SPECTRUM_BINS],
            waveform_left_full: [0.0; WAVEFORM_SAMPLES_FULL],
            waveform_right_full: [0.0; WAVEFORM_SAMPLES_FULL],
        }
    }
}

/// Errors that can occur while starting capture / DSP.
#[derive(Debug)]
pub enum AudioError {
    /// No default input device is available on this host.
    NoInputDevice,
    /// The device's configuration could not be queried.
    Config(String),
    /// The capture stream could not be built or started.
    Stream(String),
}

impl std::fmt::Display for AudioError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AudioError::NoInputDevice => write!(f, "no default audio input device available"),
            AudioError::Config(e) => write!(f, "audio device config error: {e}"),
            AudioError::Stream(e) => write!(f, "audio stream error: {e}"),
        }
    }
}

impl std::error::Error for AudioError {}

/// Ring capacity in samples. ~1 s at 48 kHz mono — generous headroom so a brief
/// DSP-thread stall never overflows the realtime capture callback.
#[cfg(feature = "capture")]
const RING_CAPACITY: usize = 48_000;

/// Real-time audio analysis engine.
///
/// [`AudioEngine::new`] starts capture + DSP immediately on background threads.
/// Dropping the engine stops capture and joins the worker.
#[cfg(feature = "capture")]
pub struct AudioEngine {
    reader: triple_buffer::Reader<Features>,
    /// Scrolling-spectrogram trail snapshot, published by the DSP worker once per
    /// hop and read by the render thread for GPU upload. Separate from the
    /// lock-free `Features` triple buffer because the ring (frames × bins floats)
    /// is far too large to copy into the per-frame `Copy` snapshot; the critical
    /// section here is a single memcpy of a preallocated buffer.
    spectrogram: Arc<Mutex<SpectrogramSnapshot>>,
    running: Arc<AtomicBool>,
    worker: Option<JoinHandle<()>>,
    // Capture stream must stay alive for capture to continue (cpal stops on drop).
    _capture: capture::Capture,
    device_name: String,
    sample_rate: u32,
    source: CaptureSource,
}

#[cfg(feature = "capture")]
impl AudioEngine {
    /// Start capture + DSP on the default input device with default settings.
    pub fn new() -> Result<Self, AudioError> {
        Self::with_config(CaptureConfig::default(), 1.0)
    }

    /// Start capture + DSP with explicit capture preferences and onset sensitivity.
    /// `sensitivity` ~`0.1..3`; higher = onsets fire more readily.
    pub fn with_config(cfg: CaptureConfig, sensitivity: f32) -> Result<Self, AudioError> {
        let (producer, consumer) = rtrb::RingBuffer::<CaptureFrame>::new(RING_CAPACITY);
        let running = Arc::new(AtomicBool::new(true));

        // Start capture first so we know the actual device sample rate (spec §2).
        let cap = capture::start(producer, cfg, running.clone())?;
        let sample_rate = cap.sample_rate;
        let device_name = cap.device_name.clone();
        let source = cap.source;

        // Clear, single startup line stating which path reactivity is driven by,
        // e.g. "audio: system loopback via 'BlackHole 2ch' @ 48000 Hz" vs
        // "audio: mic 'MacBook Pro Microphone' @ 44100 Hz".
        log::info!(
            "audio: {} via '{}' @ {} Hz",
            source.label(),
            device_name,
            sample_rate
        );

        let (writer, reader) = triple_buffer::triple_buffer(Features::default());
        // Shared spectrogram-trail snapshot. Sized to the worker's STFT so the
        // render thread always reads a fully-formed (frames × bins) ring.
        let spectrogram = Arc::new(Mutex::new(
            SpectrogramTrail::new(FFT_LEN, sample_rate as f32).snapshot(),
        ));
        let worker = {
            let running = running.clone();
            let spectrogram = spectrogram.clone();
            let params = dsp::DspParams {
                sample_rate,
                sensitivity,
            };
            std::thread::Builder::new()
                .name("particle-audio-dsp".to_string())
                .spawn(move || dsp::run(consumer, writer, spectrogram, params, running))
                .map_err(|e| AudioError::Stream(format!("failed to spawn DSP thread: {e}")))?
        };

        Ok(Self {
            reader,
            spectrogram,
            running,
            worker: Some(worker),
            _capture: cap,
            device_name,
            sample_rate,
            source,
        })
    }

    /// Latest analyzed features (lock-free snapshot). Never blocks the caller.
    pub fn latest(&self) -> Features {
        self.reader.read()
    }

    /// Clone the most recent scrolling-spectrogram trail snapshot for GPU upload.
    ///
    /// The DSP worker publishes the trail's `frames × bins` ring once per hop; the
    /// render thread calls this once per frame and uploads
    /// [`SpectrogramSnapshot::raw_ring`] as an audio texture (unwrap the ring on
    /// the GPU using the returned write cursor). The lock is held only for the
    /// clone of a preallocated buffer, so the audio thread is never blocked for
    /// more than a memcpy. Returns the last good snapshot if the lock is poisoned.
    pub fn spectrogram(&self) -> SpectrogramSnapshot {
        match self.spectrogram.lock() {
            Ok(snap) => snap.clone(),
            Err(poisoned) => poisoned.into_inner().clone(),
        }
    }

    /// Human-readable name of the capture device in use.
    pub fn device_name(&self) -> &str {
        &self.device_name
    }

    /// Actual capture sample rate in Hz (read from the device, may be 44.1k).
    pub fn sample_rate(&self) -> u32 {
        self.sample_rate
    }

    /// Which path capture ended up on: native system loopback, a virtual loopback
    /// device, or the microphone. Lets the app surface whether reactivity follows
    /// the system mix or the room.
    pub fn source(&self) -> CaptureSource {
        self.source
    }

    /// Convenience: `true` when capturing the post-volume system mix (native
    /// loopback or a virtual cable) rather than the room mic.
    pub fn is_loopback(&self) -> bool {
        self.source.is_loopback()
    }

    /// Whether the engine is still running (capture + DSP active).
    pub fn is_running(&self) -> bool {
        self.running.load(Ordering::Relaxed)
    }
}

#[cfg(feature = "capture")]
impl Drop for AudioEngine {
    fn drop(&mut self) {
        // Signal the worker to stop, then drop the capture stream (stops capture
        // and abandons the ring producer so the worker's loop exits), then join.
        self.running.store(false, Ordering::Relaxed);
        if let Some(handle) = self.worker.take() {
            let _ = handle.join();
        }
    }
}

//! Real-time harmonic/percussive dual-bus separation (public modulation source).
//!
//! Implements the median-filtering HPSS of Fitzgerald, *"Harmonic/Percussive
//! Separation using Median Filtering"* (DAFx 2010): on the STFT magnitude
//! spectrogram, a **horizontal** (time-axis) median filter estimates the
//! harmonic component (sustained partials are smooth across time), while a
//! **vertical** (frequency-axis) median filter estimates the percussive
//! component (transients are smooth across frequency). A soft Wiener-style mask
//! (or a binary mask) then routes each STFT bin's energy onto the harmonic or
//! percussive bus.
//!
//! Where the offline algorithm filters the whole spectrogram both ways, this is
//! a **causal, per-frame** variant suitable for a live visualizer: the
//! time-axis median runs over the last `K` frames held in a small rolling
//! history, so it never looks ahead. It outputs two scalar levels per frame —
//! [`HpssLevels::harmonic_level`] and [`HpssLevels::percussive_level`] — that
//! are normalized energies (`0..1`) ready to drive audio-reactive modulation.
//!
//! This is a standalone, dependency-free DSP block: feed it the same linear
//! STFT magnitude frame the rest of the analyzer already computes (`FFT_LEN/2+1`
//! bins of `Complex::norm()`), one frame per hop. It owns its own smoothing, so
//! the integrator only has to forward the magnitude slice and read the levels.

/// One frame of harmonic/percussive separation output.
///
/// All fields are normalized `0..1` and smoothed, ready to use directly as
/// modulation sources. `harmonic_level` / `percussive_level` are the two primary
/// rails; the `*_ratio` rails express the *balance* between the two buses
/// (independent of overall loudness) and are handy for cross-fades.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct HpssLevels {
    /// Normalized harmonic-bus energy (sustained tone / pads / vocals). `0..1`.
    pub harmonic_level: f32,
    /// Normalized percussive-bus energy (drums / transients / clicks). `0..1`.
    pub percussive_level: f32,
    /// Harmonic share of the separated energy, `harm / (harm + perc)`, `0..1`.
    /// Loudness-independent: ~1.0 for pure tones, ~0.0 for pure percussion.
    pub harmonic_ratio: f32,
    /// Percussive share of the separated energy, `perc / (harm + perc)`, `0..1`.
    pub percussive_ratio: f32,
}

/// Causal median-filtering harmonic/percussive separator.
///
/// Construct once per stream with the analyzer's bin count and hop period, then
/// call [`HpssBus::process`] once per STFT frame.
pub struct HpssBus {
    n_bins: usize,
    /// Number of frames in the time-axis (harmonic) median window.
    time_frames: usize,
    /// Half-width of the frequency-axis (percussive) median window, in bins.
    freq_radius: usize,
    /// Exponent for the soft mask (Wiener-like). 1.0 = magnitude ratio,
    /// 2.0 = power/Wiener ratio (sharper separation). Fitzgerald uses 2.0.
    mask_power: f32,

    /// Rolling STFT magnitude history, `time_frames * n_bins`, laid out frame-major.
    history: Vec<f32>,
    /// Ring write cursor (next frame slot to overwrite).
    write: usize,
    /// Number of frames written so far (saturates at `time_frames`).
    filled: usize,

    /// Scratch reused for the time-axis median (length `time_frames`).
    time_scratch: Vec<f32>,
    /// Scratch reused for the frequency-axis median (length `2*freq_radius + 1`).
    freq_scratch: Vec<f32>,

    /// Output smoothing — light low-pass so the rails read clean for visuals.
    harm_lp: OnePole,
    perc_lp: OnePole,
    harm_ratio_lp: OnePole,
    perc_ratio_lp: OnePole,
    /// Per-bus AGC so the absolute levels map their recent dynamic range to 0..1
    /// regardless of input gain (matches the rest of the analyzer's rails).
    harm_agc: Agc,
    perc_agc: Agc,
}

use crate::smoothing::{Agc, OnePole};

/// Time-axis (harmonic) median window in seconds. Long enough to smooth across
/// transients yet short enough to stay responsive; ~150 ms is a common choice.
const TIME_MEDIAN_SECONDS: f32 = 0.15;
/// Frequency-axis (percussive) median half-width, in bins. A 17-bin window
/// (`2*8+1`) spans broadband transients without smearing tonal peaks.
const FREQ_MEDIAN_RADIUS: usize = 8;
/// Wiener-style soft-mask exponent (power-domain ratio), per Fitzgerald 2010.
const MASK_POWER: f32 = 2.0;

impl HpssBus {
    /// Build a separator for an analyzer producing `n_bins` (= `FFT_LEN/2 + 1`)
    /// magnitude bins at a hop period of `hop_dt` seconds. The time-median window
    /// length is derived from `hop_dt` so the harmonic estimate spans a fixed
    /// wall-clock duration regardless of hop size / sample rate.
    pub fn new(n_bins: usize, hop_dt: f32) -> Self {
        let n_bins = n_bins.max(1);
        // Odd frame count keeps the median well-defined and symmetric.
        let mut time_frames = (TIME_MEDIAN_SECONDS / hop_dt.max(1e-6)).round() as usize;
        time_frames = time_frames.max(3);
        if time_frames % 2 == 0 {
            time_frames += 1;
        }
        let freq_radius = FREQ_MEDIAN_RADIUS.min(n_bins.saturating_sub(1)).max(1);
        Self {
            n_bins,
            time_frames,
            freq_radius,
            mask_power: MASK_POWER,
            history: vec![0.0; time_frames * n_bins],
            write: 0,
            filled: 0,
            time_scratch: vec![0.0; time_frames],
            freq_scratch: vec![0.0; freq_radius * 2 + 1],
            harm_lp: OnePole::new(0.6),
            perc_lp: OnePole::new(0.4),
            harm_ratio_lp: OnePole::new(0.55),
            perc_ratio_lp: OnePole::new(0.35),
            harm_agc: Agc::new(0.9998, 1e-3),
            perc_agc: Agc::new(0.9996, 1e-3),
        }
    }

    /// Number of bins this separator expects per frame.
    pub fn n_bins(&self) -> usize {
        self.n_bins
    }

    /// Number of STFT frames in the causal time-axis (harmonic) median window.
    pub fn time_window_frames(&self) -> usize {
        self.time_frames
    }

    /// Process one STFT magnitude frame (linear magnitudes, `n_bins` long) and
    /// return the smoothed dual-bus levels.
    ///
    /// `is_silent` lets the caller force the rails to drain during gated silence
    /// (the analyzer already computes a hysteresis silence gate); pass `false`
    /// if you do not gate.
    pub fn process(&mut self, mag: &[f32], is_silent: bool) -> HpssLevels {
        debug_assert_eq!(mag.len(), self.n_bins, "frame length must equal n_bins");
        self.push_history(mag);

        // Accumulate masked energy on each bus over the current frame.
        let mut harm_energy = 0.0f32;
        let mut perc_energy = 0.0f32;

        for bin in 0..self.n_bins {
            // Harmonic estimate: median across time at this frequency.
            let harmonic_ref = self.time_median(bin);
            // Percussive estimate: median across frequency in this frame.
            let percussive_ref = self.frequency_median(mag, bin);

            // Soft (Wiener-style) mask from the relative power of the two
            // medians, per Fitzgerald 2010 eq. for the soft mask:
            //   M_h = H^p / (H^p + P^p),  M_p = P^p / (H^p + P^p)
            let h = harmonic_ref.powf(self.mask_power);
            let p = percussive_ref.powf(self.mask_power);
            let denom = h + p + 1e-12;
            let harm_mask = h / denom;
            let perc_mask = p / denom;

            let m = mag[bin];
            let hm = m * harm_mask;
            let pm = m * perc_mask;
            harm_energy += hm * hm;
            perc_energy += pm * pm;
        }

        // RMS over bins keeps the scale independent of FFT size.
        let inv_bins = 1.0 / self.n_bins as f32;
        let harm_rms = (harm_energy * inv_bins).sqrt();
        let perc_rms = (perc_energy * inv_bins).sqrt();

        // Loudness-independent balance between the buses.
        let total = harm_rms + perc_rms;
        let (raw_harm_ratio, raw_perc_ratio) = if is_silent || total <= 1e-9 {
            (0.0, 0.0)
        } else {
            (harm_rms / total, perc_rms / total)
        };

        // Normalize each bus to its own recent dynamic range, then low-pass.
        let harm_norm = if is_silent {
            0.0
        } else {
            self.harm_agc.process(harm_rms)
        };
        let perc_norm = if is_silent {
            0.0
        } else {
            self.perc_agc.process(perc_rms)
        };

        HpssLevels {
            harmonic_level: self.harm_lp.process(harm_norm),
            percussive_level: self.perc_lp.process(perc_norm),
            harmonic_ratio: self.harm_ratio_lp.process(raw_harm_ratio),
            percussive_ratio: self.perc_ratio_lp.process(raw_perc_ratio),
        }
    }

    /// Overwrite the oldest history frame with the newest magnitude frame.
    fn push_history(&mut self, mag: &[f32]) {
        let offset = self.write * self.n_bins;
        self.history[offset..offset + self.n_bins].copy_from_slice(mag);
        self.write = (self.write + 1) % self.time_frames;
        self.filled = (self.filled + 1).min(self.time_frames);
    }

    /// Median of one frequency bin across the causal time history. Only the
    /// frames written so far participate, so the harmonic estimate is sane during
    /// warm-up instead of being dragged down by zero-initialized slots.
    fn time_median(&mut self, bin: usize) -> f32 {
        let count = self.filled;
        if count == 0 {
            return 0.0;
        }
        for frame in 0..count {
            self.time_scratch[frame] = self.history[frame * self.n_bins + bin];
        }
        median(&mut self.time_scratch[..count])
    }

    /// Median of the current frame across a frequency window centered on `bin`.
    fn frequency_median(&mut self, mag: &[f32], bin: usize) -> f32 {
        let lo = bin.saturating_sub(self.freq_radius);
        let hi = (bin + self.freq_radius + 1).min(mag.len());
        let count = hi - lo;
        self.freq_scratch[..count].copy_from_slice(&mag[lo..hi]);
        median(&mut self.freq_scratch[..count])
    }
}

/// Median of a scratch slice (mutates it). Uses the upper-middle element for
/// even counts — matching the existing analyzer convention.
fn median(values: &mut [f32]) -> f32 {
    if values.is_empty() {
        return 0.0;
    }
    values.sort_by(|a, b| a.total_cmp(b));
    values[values.len() / 2]
}

#[cfg(test)]
mod tests {
    use super::*;

    const SAMPLE_RATE: f32 = 48_000.0;
    const FFT_LEN: usize = 2048;
    const HOP: usize = 512;
    const N_BINS: usize = FFT_LEN / 2 + 1;

    fn bus() -> HpssBus {
        HpssBus::new(N_BINS, HOP as f32 / SAMPLE_RATE)
    }

    /// A few narrow, sustained spectral peaks (a chord) → harmonic bus dominates.
    #[test]
    fn sustained_sinusoid_prefers_harmonic_bus() {
        let mut hpss = bus();
        let mut mag = vec![0.0f32; N_BINS];
        // Three stable tonal peaks with a little skirt energy.
        for &center in &[64usize, 128, 192] {
            mag[center] = 1.0;
            mag[center - 1] = 0.3;
            mag[center + 1] = 0.3;
        }

        let mut out = HpssLevels::default();
        for _ in 0..64 {
            out = hpss.process(&mag, false);
        }

        assert!(
            out.harmonic_ratio > 0.75,
            "sustained tones should land on the harmonic bus: {out:?}"
        );
        assert!(
            out.harmonic_level > out.percussive_level,
            "harmonic level should dominate for a sustained tone: {out:?}"
        );
    }

    /// A broadband click against a quiet tonal background → percussive bus spikes.
    #[test]
    fn broadband_click_prefers_percussive_bus() {
        let mut hpss = bus();

        // Establish a quiet, stable tonal background so the time-median has a
        // sensible harmonic reference before the transient lands.
        let mut tone = vec![0.0f32; N_BINS];
        tone[128] = 0.2;
        tone[127] = 0.05;
        tone[129] = 0.05;
        for _ in 0..48 {
            hpss.process(&tone, false);
        }

        // Broadband energy flat across the whole spectrum (a click/burst). Drive
        // it for a few frames so the smoothed ratio rail settles — a real
        // transient spans more than a single hop at this hop rate. The harmonic
        // time-median lags by design, so the percussive bus keeps the energy.
        let click = vec![1.0f32; N_BINS];
        let mut out = HpssLevels::default();
        for _ in 0..5 {
            out = hpss.process(&click, false);
        }

        assert!(
            out.percussive_ratio > out.harmonic_ratio,
            "broadband click should land on the percussive bus: {out:?}"
        );
        assert!(
            out.percussive_ratio > 0.6,
            "percussive ratio should clearly dominate on a click: {out:?}"
        );
        assert!(
            out.percussive_level > out.harmonic_level,
            "percussive level should dominate on a click: {out:?}"
        );
    }

    /// An impulse train (broadband transients every few frames) keeps the
    /// percussive ratio above the harmonic ratio on average.
    #[test]
    fn impulse_train_is_percussive_on_average() {
        let mut hpss = bus();
        let silence = vec![0.0f32; N_BINS];
        let click = vec![1.0f32; N_BINS];

        let mut perc_sum = 0.0f32;
        let mut harm_sum = 0.0f32;
        for frame in 0..120 {
            let out = if frame % 6 == 0 {
                hpss.process(&click, false)
            } else {
                hpss.process(&silence, false)
            };
            perc_sum += out.percussive_ratio;
            harm_sum += out.harmonic_ratio;
        }

        assert!(
            perc_sum > harm_sum,
            "an impulse train should be percussive on average: perc={perc_sum}, harm={harm_sum}"
        );
    }

    /// Silence drains the absolute rails to ~0 even after loud audio.
    #[test]
    fn silence_drains_levels() {
        let mut hpss = bus();
        let loud = vec![1.0f32; N_BINS];
        for _ in 0..32 {
            hpss.process(&loud, false);
        }
        let mut out = HpssLevels::default();
        for _ in 0..32 {
            out = hpss.process(&loud, true);
        }
        assert!(
            out.harmonic_level < 0.05 && out.percussive_level < 0.05,
            "gated silence should drain both rails: {out:?}"
        );
        // The ratio rails are fed raw 0.0 during gated silence and low-pass
        // toward zero (asymptotic, so allow a tiny epsilon).
        assert!(
            out.harmonic_ratio < 1e-4,
            "harmonic ratio should drain: {out:?}"
        );
        assert!(
            out.percussive_ratio < 1e-4,
            "percussive ratio should drain: {out:?}"
        );
    }

    /// The time-median window length scales with the hop period to span a fixed
    /// wall-clock duration, and is always an odd count ≥ 3.
    #[test]
    fn time_window_scales_with_hop() {
        let fast = HpssBus::new(N_BINS, HOP as f32 / SAMPLE_RATE);
        let frames = fast.time_window_frames();
        assert!(frames >= 3 && frames % 2 == 1, "odd, ≥3: {frames}");

        // Larger hop → fewer frames for the same wall-clock window.
        let slow = HpssBus::new(N_BINS, 2048.0 / SAMPLE_RATE);
        assert!(
            slow.time_window_frames() < frames,
            "a larger hop should need fewer frames: slow={}, fast={}",
            slow.time_window_frames(),
            frames
        );
    }
}

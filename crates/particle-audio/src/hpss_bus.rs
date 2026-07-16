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
//! This is a dependency-free DSP block: feed it the same linear STFT magnitude
//! frame the rest of the analyzer already computes (`FFT_LEN/2+1` bins of
//! `Complex::norm()`) plus the shared per-bin harmonic reference from an
//! [`crate::HpssHistory`] (advanced once per hop), one frame per hop. The
//! time-axis median history is stored and sorted a single time and shared across
//! consumers rather than duplicated here; the bus owns its own output smoothing.

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
/// Construct once per stream with the analyzer's bin count, then call
/// [`HpssBus::process`] once per STFT frame with the shared harmonic reference
/// from an [`crate::HpssHistory`].
pub struct HpssBus {
    n_bins: usize,
    /// Half-width of the frequency-axis (percussive) median window, in bins.
    freq_radius: usize,
    /// Exponent for the soft mask (Wiener-like). 1.0 = magnitude ratio,
    /// 2.0 = power/Wiener ratio (sharper separation). Fitzgerald uses 2.0.
    mask_power: f32,

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

/// Frequency-axis (percussive) median half-width, in bins. A 17-bin window
/// (`2*8+1`) spans broadband transients without smearing tonal peaks.
const FREQ_MEDIAN_RADIUS: usize = 8;
/// Wiener-style soft-mask exponent (power-domain ratio), per Fitzgerald 2010.
const MASK_POWER: f32 = 2.0;

#[inline]
fn finite_or_zero(value: f32) -> f32 {
    if value.is_finite() {
        value
    } else {
        0.0
    }
}

impl HpssBus {
    /// Build a separator for an analyzer producing `n_bins` (= `FFT_LEN/2 + 1`)
    /// magnitude bins. The harmonic (time-axis) reference is supplied per hop by a
    /// shared [`crate::HpssHistory`], so the median-filtered magnitude history is
    /// stored and sorted once for every consumer rather than duplicated here.
    pub fn new(n_bins: usize) -> Self {
        let n_bins = n_bins.max(1);
        let freq_radius = FREQ_MEDIAN_RADIUS.min(n_bins.saturating_sub(1)).max(1);
        Self {
            n_bins,
            freq_radius,
            mask_power: MASK_POWER,
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

    /// Process one STFT magnitude frame (linear magnitudes, `n_bins` long) and
    /// return the smoothed dual-bus levels. `harm_ref` is the shared per-bin
    /// harmonic reference (time median) from [`crate::HpssHistory`], already
    /// advanced for this hop.
    ///
    /// `is_silent` lets the caller force the rails to drain during gated silence
    /// (the analyzer already computes a hysteresis silence gate); pass `false`
    /// if you do not gate.
    pub fn process(&mut self, mag: &[f32], harm_ref: &[f32], is_silent: bool) -> HpssLevels {
        debug_assert_eq!(mag.len(), self.n_bins, "frame length must equal n_bins");
        debug_assert_eq!(
            harm_ref.len(),
            self.n_bins,
            "harm_ref length must equal n_bins"
        );

        // Accumulate masked energy on each bus over the current frame.
        let mut harm_energy = 0.0f32;
        let mut perc_energy = 0.0f32;

        for bin in 0..self.n_bins {
            // Harmonic estimate: shared median across time at this frequency.
            let harmonic_ref = finite_or_zero(harm_ref[bin]).max(0.0);
            // Percussive estimate: median across frequency in this frame.
            let percussive_ref = finite_or_zero(self.frequency_median(mag, bin)).max(0.0);

            // Soft (Wiener-style) mask from the relative power of the two
            // medians, per Fitzgerald 2010 eq. for the soft mask:
            //   M_h = H^p / (H^p + P^p),  M_p = P^p / (H^p + P^p)
            let h = harmonic_ref.powf(self.mask_power);
            let p = percussive_ref.powf(self.mask_power);
            let denom = finite_or_zero(h + p).max(0.0) + 1e-12;
            let harm_mask = finite_or_zero(h / denom).clamp(0.0, 1.0);
            let perc_mask = finite_or_zero(p / denom).clamp(0.0, 1.0);

            let m = finite_or_zero(mag[bin]).max(0.0);
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

    /// Median of the current frame across a frequency window centered on `bin`.
    fn frequency_median(&mut self, mag: &[f32], bin: usize) -> f32 {
        let lo = bin.saturating_sub(self.freq_radius);
        let hi = (bin + self.freq_radius + 1).min(mag.len());
        let count = hi - lo;
        for (dst, &src) in self.freq_scratch[..count]
            .iter_mut()
            .zip(mag[lo..hi].iter())
        {
            *dst = finite_or_zero(src).max(0.0);
        }
        median(&mut self.freq_scratch[..count])
    }
}

/// Median of a scratch slice (reorders it). Uses the upper-middle element for
/// even counts via partial selection — the same value a full sort would place at
/// `len / 2`, matching the existing analyzer convention without sorting the whole
/// slice.
fn median(values: &mut [f32]) -> f32 {
    if values.is_empty() {
        return 0.0;
    }
    let mid = values.len() / 2;
    let (_, m, _) = values.select_nth_unstable_by(mid, |a, b| a.total_cmp(b));
    *m
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::hpss::HpssHistory;

    const SAMPLE_RATE: f32 = 48_000.0;
    const FFT_LEN: usize = 2048;
    const HOP: usize = 512;
    const N_BINS: usize = FFT_LEN / 2 + 1;

    fn bus() -> HpssBus {
        HpssBus::new(N_BINS)
    }

    fn history() -> HpssHistory {
        HpssHistory::new(N_BINS, HOP as f32 / SAMPLE_RATE)
    }

    /// A few narrow, sustained spectral peaks (a chord) → harmonic bus dominates.
    #[test]
    fn sustained_sinusoid_prefers_harmonic_bus() {
        let mut hist = history();
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
            hist.advance(&mag);
            out = hpss.process(&mag, hist.harm_ref(), false);
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
        let mut hist = history();
        let mut hpss = bus();

        // Establish a quiet, stable tonal background so the time-median has a
        // sensible harmonic reference before the transient lands.
        let mut tone = vec![0.0f32; N_BINS];
        tone[128] = 0.2;
        tone[127] = 0.05;
        tone[129] = 0.05;
        for _ in 0..48 {
            hist.advance(&tone);
            hpss.process(&tone, hist.harm_ref(), false);
        }

        // Broadband energy flat across the whole spectrum (a click/burst). Drive
        // it for a few frames so the smoothed ratio rail settles — a real
        // transient spans more than a single hop at this hop rate. The harmonic
        // time-median lags by design, so the percussive bus keeps the energy.
        let click = vec![1.0f32; N_BINS];
        let mut out = HpssLevels::default();
        for _ in 0..5 {
            hist.advance(&click);
            out = hpss.process(&click, hist.harm_ref(), false);
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
        let mut hist = history();
        let mut hpss = bus();
        let silence = vec![0.0f32; N_BINS];
        let click = vec![1.0f32; N_BINS];

        let mut perc_sum = 0.0f32;
        let mut harm_sum = 0.0f32;
        for frame in 0..120 {
            let mag = if frame % 6 == 0 { &click } else { &silence };
            hist.advance(mag);
            let out = hpss.process(mag, hist.harm_ref(), false);
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
        let mut hist = history();
        let mut hpss = bus();
        let loud = vec![1.0f32; N_BINS];
        for _ in 0..32 {
            hist.advance(&loud);
            hpss.process(&loud, hist.harm_ref(), false);
        }
        let mut out = HpssLevels::default();
        for _ in 0..32 {
            hist.advance(&loud);
            out = hpss.process(&loud, hist.harm_ref(), true);
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
}

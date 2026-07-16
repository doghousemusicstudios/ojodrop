//! Linkwitz-Riley 4th-order crossover filterbank.
//!
//! A 4th-order Linkwitz-Riley (LR4) crossover is two cascaded identical
//! 2nd-order Butterworth sections (a Butterworth-squared response). At the
//! crossover frequency each LR4 low/high pair is exactly -6 dB and the two sum
//! back to unity gain with a flat magnitude response (the defining LR property),
//! with a steep 24 dB/oct skirt — much cleaner band isolation than the simple
//! FFT triangular macro bands.
//!
//! We split the broadband signal into five bands with four crossover points:
//!
//! ```text
//!   sub  | low  | mid   | high   | air
//!      f0=120  f1=500  f2=2k   f3=6k   (Hz)
//! ```
//!
//! using the standard recursive crossover tree: at each split a LR4 low-pass
//! keeps the band below the crossover and a LR4 high-pass passes the remainder
//! on to the next split. Per hop we measure each band's RMS and normalize every
//! band against a single *shared* running reference (the loudest band's tracked
//! peak), then shape it with the same dB mapping the macro bands use, so the rails
//! read as clean 0..1 per-band energies that preserve inter-band isolation.
//!
//! The shared reference matters: an earlier design ran an independent AGC per band
//! that re-normalized each band to full scale, so faint cross-band leakage (bass
//! bleeding into a mid band, ~-24 dB/oct down but non-zero) got amplified to full
//! scale, destroying isolation. Normalizing all bands by one reference keeps quiet
//! bands quiet relative to the dominant one (P2-AUD-020).
//!
//! Textbook cascaded-Butterworth math, implemented from scratch — no license
//! encumbrance.

use crate::smoothing::{flush_denormal, OnePole};

/// Crossover frequencies (Hz) separating the five bands.
const CROSSOVERS_HZ: [f32; 4] = [120.0, 500.0, 2000.0, 6000.0];
/// Number of output bands (crossovers + 1).
pub const LR_BANDS: usize = 5;

/// Per-hop multiplicative decay of the shared reference peak (slow forget).
const REF_DECAY: f32 = 0.9995;
/// Minimum shared reference level; prevents dividing near-silence up to full scale.
const REF_FLOOR: f32 = 1e-4;

/// A direct-form-II transposed biquad (matches the one in `dsp.rs`, kept local so
/// the multiband filterbank is self-contained).
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
    fn from_coeffs(b0: f32, b1: f32, b2: f32, a0: f32, a1: f32, a2: f32) -> Self {
        let inv = if a0.abs() > 1e-12 { 1.0 / a0 } else { 1.0 };
        Self {
            b0: b0 * inv,
            b1: b1 * inv,
            b2: b2 * inv,
            a1: a1 * inv,
            a2: a2 * inv,
            z1: 0.0,
            z2: 0.0,
        }
    }

    /// 2nd-order Butterworth low-pass (Q = 1/sqrt(2)) via the RBJ cookbook.
    fn butterworth_low_pass(sample_rate: f32, f0: f32) -> Self {
        let (sin_w0, cos_w0) = sin_cos(sample_rate, f0);
        // A Butterworth section has Q = 1/sqrt(2), so alpha = sin/(2Q) = sin*sqrt(2)/2.
        let alpha = sin_w0 * std::f32::consts::SQRT_2 * 0.5;
        let b1 = 1.0 - cos_w0;
        let b0 = b1 * 0.5;
        let b2 = b0;
        let a0 = 1.0 + alpha;
        let a1 = -2.0 * cos_w0;
        let a2 = 1.0 - alpha;
        Self::from_coeffs(b0, b1, b2, a0, a1, a2)
    }

    /// 2nd-order Butterworth high-pass (Q = 1/sqrt(2)).
    fn butterworth_high_pass(sample_rate: f32, f0: f32) -> Self {
        let (sin_w0, cos_w0) = sin_cos(sample_rate, f0);
        let alpha = sin_w0 * std::f32::consts::SQRT_2 * 0.5;
        let b1 = -(1.0 + cos_w0);
        let b0 = (1.0 + cos_w0) * 0.5;
        let b2 = b0;
        let a0 = 1.0 + alpha;
        let a1 = -2.0 * cos_w0;
        let a2 = 1.0 - alpha;
        Self::from_coeffs(b0, b1, b2, a0, a1, a2)
    }

    #[inline]
    fn process(&mut self, x: f32) -> f32 {
        let x = if x.is_finite() { x } else { 0.0 };
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
fn sin_cos(sample_rate: f32, f0: f32) -> (f32, f32) {
    let nyquist_safe = (sample_rate * 0.45).max(10.0);
    let f0 = f0.clamp(1.0, nyquist_safe);
    let w0 = std::f32::consts::TAU * f0 / sample_rate.max(1.0);
    w0.sin_cos()
}

/// A 4th-order Linkwitz-Riley low-pass: two cascaded identical Butterworth LPs.
#[derive(Clone, Copy, Debug)]
struct Lr4Low {
    a: Biquad,
    b: Biquad,
}

impl Lr4Low {
    fn new(sample_rate: f32, f0: f32) -> Self {
        Self {
            a: Biquad::butterworth_low_pass(sample_rate, f0),
            b: Biquad::butterworth_low_pass(sample_rate, f0),
        }
    }
    #[inline]
    fn process(&mut self, x: f32) -> f32 {
        self.b.process(self.a.process(x))
    }
}

/// A 4th-order Linkwitz-Riley high-pass: two cascaded identical Butterworth HPs.
#[derive(Clone, Copy, Debug)]
struct Lr4High {
    a: Biquad,
    b: Biquad,
}

impl Lr4High {
    fn new(sample_rate: f32, f0: f32) -> Self {
        Self {
            a: Biquad::butterworth_high_pass(sample_rate, f0),
            b: Biquad::butterworth_high_pass(sample_rate, f0),
        }
    }
    #[inline]
    fn process(&mut self, x: f32) -> f32 {
        self.b.process(self.a.process(x))
    }
}

/// Five-band Linkwitz-Riley crossover filterbank with per-band level rails.
///
/// The filter sections are stateful and run sample-by-sample; per hop we feed the
/// raw mono samples through the crossover tree, accumulate each band's mean
/// square, take the RMS, then AGC + dB-map to a normalized 0..1 rail.
pub struct LinkwitzRileyBank {
    // Crossover tree filters. At split `i` a LR4 low keeps the band below
    // CROSSOVERS_HZ[i]; a LR4 high passes the remainder to the next split.
    low: [Lr4Low; 4],
    high: [Lr4High; 4],
    // Shared reference peak (linear RMS units): a single slowly-decaying running
    // maximum across all bands. Every band is normalized against this one value so
    // relative band levels — and thus band isolation — are preserved.
    ref_peak: f32,
    // Per-band output smoothing.
    lp: [OnePole; LR_BANDS],
    /// Number of crossover splits that sit safely below Nyquist. Splits at/above
    /// Nyquist are skipped, so the bands above the highest valid crossover are
    /// reported as unavailable (zero) instead of aliasing garbage.
    n_splits: usize,
    /// False for a non-finite / non-positive sample rate: every band is
    /// unavailable (all rails read zero) rather than dividing by zero.
    valid: bool,
}

/// The five normalized band rails (0..1): sub / low / mid / high / air.
#[derive(Clone, Copy, Debug, Default)]
pub struct LrBands {
    pub sub: f32,
    pub low: f32,
    pub mid: f32,
    pub high: f32,
    pub air: f32,
}

impl LinkwitzRileyBank {
    pub fn new(sample_rate: f32) -> Self {
        // Reject a zero / non-finite rate: no valid Nyquist, so no valid bands.
        let valid = sample_rate.is_finite() && sample_rate > 0.0;
        let nyquist = if valid { sample_rate * 0.5 } else { 0.0 };
        // A crossover is usable only if it sits comfortably below Nyquist; because
        // CROSSOVERS_HZ is ascending, this counts the leading valid splits.
        let n_splits = CROSSOVERS_HZ
            .iter()
            .take_while(|&&c| valid && c < nyquist * 0.95)
            .count();
        Self {
            low: std::array::from_fn(|i| Lr4Low::new(sample_rate, CROSSOVERS_HZ[i])),
            high: std::array::from_fn(|i| Lr4High::new(sample_rate, CROSSOVERS_HZ[i])),
            ref_peak: REF_FLOOR,
            lp: std::array::from_fn(|_| OnePole::new(0.4)),
            n_splits,
            valid,
        }
    }

    /// Number of output bands that are meaningful at this sample rate (the valid
    /// crossover splits plus the top remainder band). Bands beyond this always
    /// read zero because their frequency range lies above Nyquist.
    pub fn available_bands(&self) -> usize {
        if self.valid {
            (self.n_splits + 1).min(LR_BANDS)
        } else {
            0
        }
    }

    /// Run one hop of raw mono samples through the crossover tree and return the
    /// five normalized band rails. `is_silent` still advances filter state (so
    /// the IIR memory stays valid) but zeroes the output rails.
    pub fn process(&mut self, samples: &[f32], is_silent: bool) -> LrBands {
        // Number of output bands with valid spectral support (see `available_bands`).
        // The remainder band lands at index `n_splits`; bands above it are above
        // Nyquist and stay zero (unavailable) rather than aliasing garbage.
        let produce = self.available_bands();
        let mut sumsq = [0.0f32; LR_BANDS];
        for &x in samples {
            // Recursive crossover tree: `rest` carries the not-yet-assigned high
            // portion down through each valid split.
            let mut rest = x;
            for i in 0..self.n_splits {
                let band = self.low[i].process(rest);
                rest = self.high[i].process(rest);
                sumsq[i] += band * band;
            }
            // Whatever survives every valid high-pass is the top remainder band.
            if produce > 0 {
                sumsq[self.n_splits] += rest * rest;
            }
        }

        let n = samples.len().max(1) as f32;
        let mut rms = [0.0f32; LR_BANDS];
        let mut hop_peak = 0.0f32;
        for i in 0..produce {
            let r = (sumsq[i] / n).sqrt();
            let r = if r.is_finite() { r } else { 0.0 };
            rms[i] = r;
            hop_peak = hop_peak.max(r);
        }

        // Advance the single shared reference: a slow running peak across all
        // bands. Normalizing every band by this one value preserves inter-band
        // ratios, so leakage in a quiet band stays quiet (P2-AUD-020). Denormal +
        // finite guards keep the follower cheap and un-poisonable (P2-AUD-024).
        self.ref_peak = flush_denormal((self.ref_peak * REF_DECAY).max(REF_FLOOR));
        if hop_peak > self.ref_peak {
            self.ref_peak = hop_peak;
        }
        if !self.ref_peak.is_finite() {
            self.ref_peak = REF_FLOOR;
        }
        let ref_peak = self.ref_peak.max(REF_FLOOR);

        let mut out = [0.0f32; LR_BANDS];
        for i in 0..produce {
            // Ratio of this band to the shared reference (dominant band ≈ 1), then
            // the same dB shaping the macro bands use. A band far below the
            // reference maps toward 0 — true isolation — instead of being AGC'd up.
            let level = if is_silent {
                0.0
            } else {
                lin_to_db_norm(rms[i] / ref_peak)
            };
            out[i] = self.lp[i].process(level);
        }

        LrBands {
            sub: out[0],
            low: out[1],
            mid: out[2],
            high: out[3],
            air: out[4],
        }
    }
}

/// Map a linear RMS magnitude to a normalized 0..1 loudness over a fixed -80 dB
/// floor (same shape `dsp.rs` uses for the macro bands).
#[inline]
fn lin_to_db_norm(lin: f32) -> f32 {
    const FLOOR_DB: f32 = -80.0;
    if lin <= 1e-9 {
        return 0.0;
    }
    let db = 20.0 * lin.log10();
    ((db - FLOOR_DB) / -FLOOR_DB).clamp(0.0, 1.0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f32::consts::TAU;

    fn sine(freq: f32, sample_rate: f32, n: usize) -> Vec<f32> {
        (0..n)
            .map(|i| (TAU * freq * i as f32 / sample_rate).sin())
            .collect()
    }

    /// Raw per-band RMS of one steady block, via a standalone warmed crossover
    /// tree — the same band levels the bank's normalizer sees at steady state.
    fn steady_band_rms(bank: &mut LinkwitzRileyBank, block: &[f32]) -> [f32; LR_BANDS] {
        let sr = 48_000.0;
        let _ = bank; // sample rate is fixed for this helper's callers
        let mut low: [Lr4Low; 4] = std::array::from_fn(|i| Lr4Low::new(sr, CROSSOVERS_HZ[i]));
        let mut high: [Lr4High; 4] = std::array::from_fn(|i| Lr4High::new(sr, CROSSOVERS_HZ[i]));
        for _ in 0..100 {
            for &x in block {
                let mut rest = x;
                for i in 0..4 {
                    let _ = low[i].process(rest);
                    rest = high[i].process(rest);
                }
            }
        }
        let mut sumsq = [0.0f32; LR_BANDS];
        for &x in block {
            let mut rest = x;
            for i in 0..4 {
                let b = low[i].process(rest);
                rest = high[i].process(rest);
                sumsq[i] += b * b;
            }
            sumsq[4] += rest * rest;
        }
        let n = block.len().max(1) as f32;
        std::array::from_fn(|i| (sumsq[i] / n).sqrt())
    }

    /// Run a long stationary sine through the bank and return the steady band rails.
    fn steady_bands(freq: f32, sample_rate: f32) -> LrBands {
        let mut bank = LinkwitzRileyBank::new(sample_rate);
        let hop = 512;
        let block = sine(freq, sample_rate, hop);
        let mut bands = LrBands::default();
        // Warm the filters + AGC over ~1 s of audio fed hop-by-hop.
        let total = (sample_rate as usize / hop) + 4;
        for _ in 0..total {
            bands = bank.process(&block, false);
        }
        bands
    }

    #[test]
    fn low_tone_lands_in_low_bands() {
        let sr = 48_000.0;
        let bands = steady_bands(60.0, sr); // below 120 Hz crossover -> sub band
        assert!(
            bands.sub >= bands.mid && bands.sub >= bands.high && bands.sub >= bands.air,
            "60 Hz should dominate the sub band: {bands:?}"
        );
    }

    /// P2-AUD-020: under a dominant low tone the higher bands must stay clearly
    /// below full scale, in proportion to their (much smaller) real energy — band
    /// isolation. The pre-fix independent per-band AGC re-normalized *every* band to
    /// its own running peak, pinning them all to ~1.0 and destroying isolation; the
    /// shared reference keeps quiet bands quiet relative to the loud one.
    ///
    /// (The LR crossover tree itself leaks a low tone up into higher bands at a
    /// modest floor — a separate filter-precision matter — so "near zero" here means
    /// "clearly below the dominant band", not literally 0.)
    #[test]
    fn dominant_low_tone_preserves_band_isolation() {
        let sr = 48_000.0;
        let mut bank = LinkwitzRileyBank::new(sr);
        let hop = 512;
        let block: Vec<f32> = (0..hop)
            .map(|i| 0.6 * (TAU * 60.0 * i as f32 / sr).sin())
            .collect();

        let mut bands = LrBands::default();
        // Warm the filters + shared reference over ~1 s.
        for _ in 0..(sr as usize / hop + 8) {
            bands = bank.process(&block, false);
        }

        // Reproduce the pre-fix independent per-band AGC on the SAME steady band
        // levels to show it collapses every band to full scale (no isolation).
        let rms = steady_band_rms(&mut bank, &block);
        let mut old_out = [0.0f32; LR_BANDS];
        {
            let mut agc: [crate::smoothing::Agc; LR_BANDS] =
                std::array::from_fn(|_| crate::smoothing::Agc::new(0.9995, 1e-4));
            let mut lp: [OnePole; LR_BANDS] = std::array::from_fn(|_| OnePole::new(0.4));
            for _ in 0..400 {
                for i in 0..LR_BANDS {
                    let level = lin_to_db_norm(rms[i]);
                    old_out[i] = lp[i].process(agc[i].process(level));
                }
            }
        }
        for (i, &v) in old_out.iter().enumerate() {
            assert!(
                v > 0.9,
                "pre-fix independent AGC should pin band {i} to full scale, got {v} (old {old_out:?})"
            );
        }

        // The shared-reference path: dominant sub near full, higher bands clearly
        // attenuated and monotonically decreasing (isolation preserved).
        assert!(
            bands.sub > 0.9,
            "dominant low band should be near full: {bands:?}"
        );
        let ordered = [bands.sub, bands.low, bands.mid, bands.high, bands.air];
        for pair in ordered.windows(2) {
            assert!(
                pair[0] + 1e-4 >= pair[1],
                "bands should decrease from loud to quiet (isolation): {bands:?}"
            );
        }
        assert!(
            bands.sub - bands.air > 0.25,
            "quiet bands must stay well below the dominant band (independent AGC \
             collapses this gap to ~0): {bands:?}"
        );
        assert!(
            bands.air < 0.75,
            "far band must not be pinned near full scale: {} ({bands:?})",
            bands.air
        );
    }

    /// P2-AUD-024: the LR4 biquad state flushes to a hard zero as it decays,
    /// without lingering in the (CPU-expensive) f32 subnormal range.
    #[test]
    fn biquad_state_flushes_denormals_to_zero() {
        let mut bq = Biquad::butterworth_low_pass(48_000.0, 1_000.0);
        bq.process(1.0); // impulse
        let mut settled = false;
        for _ in 0..200_000 {
            bq.process(0.0);
            assert!(
                !bq.z1.is_subnormal() && !bq.z2.is_subnormal(),
                "LR4 biquad state entered the denormal range: z1={:e} z2={:e}",
                bq.z1,
                bq.z2
            );
            if bq.z1 == 0.0 && bq.z2 == 0.0 {
                settled = true;
                break;
            }
        }
        assert!(
            settled,
            "decaying LR4 biquad state should flush to exactly 0"
        );
    }

    #[test]
    fn nan_input_does_not_poison_lr4_state() {
        let sr = 48_000.0;
        let mut bank = LinkwitzRileyBank::new(sr);
        let poisoned = bank.process(&[f32::NAN, f32::INFINITY, -f32::INFINITY], false);
        for value in [
            poisoned.sub,
            poisoned.low,
            poisoned.mid,
            poisoned.high,
            poisoned.air,
        ] {
            assert!(
                value.is_finite(),
                "poisoned output should be finite: {poisoned:?}"
            );
        }

        let recovered = bank.process(&sine(1_000.0, sr, 512), false);
        for value in [
            recovered.sub,
            recovered.low,
            recovered.mid,
            recovered.high,
            recovered.air,
        ] {
            assert!(
                value.is_finite(),
                "recovered output should be finite: {recovered:?}"
            );
        }
    }

    #[test]
    fn high_tone_lands_in_high_bands() {
        let sr = 48_000.0;
        let bands = steady_bands(10_000.0, sr); // above 6 kHz crossover -> air band
        assert!(
            bands.air >= bands.sub && bands.air >= bands.low && bands.air >= bands.mid,
            "10 kHz should dominate the air band: {bands:?}"
        );
    }

    #[test]
    fn mid_tone_lands_in_mid_band() {
        let sr = 48_000.0;
        let bands = steady_bands(1_000.0, sr); // between 500 and 2k -> mid band
        assert!(
            bands.mid >= bands.sub && bands.mid >= bands.air,
            "1 kHz should land in the mid band: {bands:?}"
        );
    }

    #[test]
    fn crossover_sums_back_to_flat_energy() {
        // The defining LR property: summing the band outputs reconstructs the
        // input with a flat magnitude response (low+high of each LR4 split sums
        // to an allpass). An allpass preserves signal energy, so the summed
        // reconstruction has the same RMS as the input even though it is
        // phase-shifted sample-by-sample. We check that energy conservation —
        // the meaningful "sums back to flat" statement for a serial crossover
        // tree — on a broadband (white) input.
        let sr = 48_000.0;
        let n = 32_768;
        // Deterministic pseudo-noise.
        let mut seed = 0x1234_5678u32;
        let mut noise = vec![0.0f32; n];
        for s in noise.iter_mut() {
            seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            *s = (seed >> 8) as f32 / (1u32 << 24) as f32 * 2.0 - 1.0;
        }

        let mut low: [Lr4Low; 4] = std::array::from_fn(|i| Lr4Low::new(sr, CROSSOVERS_HZ[i]));
        let mut high: [Lr4High; 4] = std::array::from_fn(|i| Lr4High::new(sr, CROSSOVERS_HZ[i]));
        let mut recon_sq = 0.0f32;
        let mut sig_sq = 0.0f32;
        let warm = 2048;
        for (idx, &x) in noise.iter().enumerate() {
            let mut rest = x;
            let mut recon = 0.0f32;
            for i in 0..4 {
                let band = low[i].process(rest);
                rest = high[i].process(rest);
                recon += band;
            }
            recon += rest;
            if idx >= warm {
                recon_sq += recon * recon;
                sig_sq += x * x;
            }
        }
        let ratio = (recon_sq / sig_sq.max(1e-12)).sqrt();
        // Energy should be preserved through the allpass-summing crossover tree.
        assert!(
            (ratio - 1.0).abs() < 0.1,
            "LR crossover tree should preserve broadband energy (RMS ratio {ratio})"
        );
    }

    /// Deterministic broadband pseudo-noise (energy up to Nyquist).
    fn noise(n: usize) -> Vec<f32> {
        let mut seed = 0x9e37_79b9u32;
        (0..n)
            .map(|_| {
                seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                (seed >> 8) as f32 / (1u32 << 24) as f32 * 2.0 - 1.0
            })
            .collect()
    }

    /// P2-AUD-015: at a low sample rate the upper crossovers exceed Nyquist, so
    /// those bands must report zero (unavailable) instead of aliasing garbage from
    /// a crossover silently clamped below Nyquist.
    #[test]
    fn bands_above_nyquist_are_unavailable() {
        // 8 kHz → Nyquist 4 kHz. Only the 120/500/2000 Hz crossovers are usable;
        // the 6 kHz crossover is above Nyquist, so the top "air" band is dropped.
        let sr = 8_000.0;
        let mut bank = LinkwitzRileyBank::new(sr);
        assert_eq!(
            bank.available_bands(),
            4,
            "3 usable crossovers + remainder = 4 available bands"
        );

        let block = noise(2048);
        let mut bands = LrBands::default();
        for _ in 0..40 {
            bands = bank.process(&block, false);
        }
        // Broadband energy would leak into a clamped 'air' band pre-fix; now it is
        // zero because its range is entirely above Nyquist.
        assert_eq!(
            bands.air, 0.0,
            "air band above Nyquist must be zero: {bands:?}"
        );
        for v in [bands.sub, bands.low, bands.mid, bands.high, bands.air] {
            assert!(v.is_finite(), "all rails must stay finite: {bands:?}");
        }
        // The remainder ('high') band should still carry the sub-Nyquist top energy.
        assert!(
            bands.high > 0.0,
            "remainder band should carry energy: {bands:?}"
        );
    }

    /// P2-AUD-015: a zero / non-finite sample rate is rejected — every band is
    /// unavailable and the rails stay a safe, finite zero (no divide-by-zero).
    #[test]
    fn zero_rate_yields_no_bands() {
        for sr in [0.0f32, -48_000.0, f32::NAN, f32::INFINITY] {
            let mut bank = LinkwitzRileyBank::new(sr);
            assert_eq!(bank.available_bands(), 0, "no valid bands at rate {sr}");
            let bands = bank.process(&noise(1024), false);
            for v in [bands.sub, bands.low, bands.mid, bands.high, bands.air] {
                assert_eq!(v, 0.0, "rate {sr} must yield zero rails: {bands:?}");
            }
        }
    }

    #[test]
    fn silence_drains_rails_toward_zero() {
        // `is_silent` feeds 0 into each band's AGC + smoothing one-pole, so the
        // rails drain toward (but don't snap to) zero — the same convention the
        // macro bands use. After a short silent run they should be near zero.
        let sr = 48_000.0;
        let mut bank = LinkwitzRileyBank::new(sr);
        let block = sine(1_000.0, sr, 512);
        let audible = bank.process(&block, false);
        let mut bands = LrBands::default();
        for _ in 0..40 {
            bands = bank.process(&vec![0.0f32; 512], true);
        }
        assert!(
            audible.mid > 0.05,
            "should have been audible first: {audible:?}"
        );
        assert!(bands.sub < 1e-3, "sub should drain: {}", bands.sub);
        assert!(bands.low < 1e-3, "low should drain: {}", bands.low);
        assert!(bands.mid < 1e-3, "mid should drain: {}", bands.mid);
        assert!(bands.high < 1e-3, "high should drain: {}", bands.high);
        assert!(bands.air < 1e-3, "air should drain: {}", bands.air);
    }
}

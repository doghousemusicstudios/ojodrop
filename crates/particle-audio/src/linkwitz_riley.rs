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
//! on to the next split. Per hop we measure each band's RMS, AGC-normalize it,
//! and shape it with the same dB mapping the macro bands use, so the rails read
//! as clean 0..1 per-band energies.
//!
//! Textbook cascaded-Butterworth math, implemented from scratch — no license
//! encumbrance.

use crate::smoothing::{Agc, OnePole};

/// Crossover frequencies (Hz) separating the five bands.
const CROSSOVERS_HZ: [f32; 4] = [120.0, 500.0, 2000.0, 6000.0];
/// Number of output bands (crossovers + 1).
pub const LR_BANDS: usize = 5;

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
        let y = self.b0 * x + self.z1;
        self.z1 = self.b1 * x - self.a1 * y + self.z2;
        self.z2 = self.b2 * x - self.a2 * y;
        y
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
    // Per-band post-processing.
    agc: [Agc; LR_BANDS],
    lp: [OnePole; LR_BANDS],
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
        Self {
            low: std::array::from_fn(|i| Lr4Low::new(sample_rate, CROSSOVERS_HZ[i])),
            high: std::array::from_fn(|i| Lr4High::new(sample_rate, CROSSOVERS_HZ[i])),
            agc: std::array::from_fn(|_| Agc::new(0.9995, 1e-4)),
            lp: std::array::from_fn(|_| OnePole::new(0.4)),
        }
    }

    /// Run one hop of raw mono samples through the crossover tree and return the
    /// five normalized band rails. `is_silent` still advances filter state (so
    /// the IIR memory stays valid) but zeroes the output rails.
    pub fn process(&mut self, samples: &[f32], is_silent: bool) -> LrBands {
        let mut sumsq = [0.0f32; LR_BANDS];
        for &x in samples {
            // Recursive crossover tree: `rest` carries the not-yet-assigned high
            // portion down through each split.
            let mut rest = x;
            for i in 0..4 {
                let band = self.low[i].process(rest);
                rest = self.high[i].process(rest);
                sumsq[i] += band * band;
            }
            // Whatever survives all four high-passes is the top "air" band.
            sumsq[LR_BANDS - 1] += rest * rest;
        }

        let n = samples.len().max(1) as f32;
        let mut out = [0.0f32; LR_BANDS];
        for i in 0..LR_BANDS {
            let rms = (sumsq[i] / n).sqrt();
            let level = if is_silent { 0.0 } else { lin_to_db_norm(rms) };
            let agc = self.agc[i].process(level);
            out[i] = self.lp[i].process(agc);
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

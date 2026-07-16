//! Canonical analysis-rate decimation.
//!
//! The whole DSP chain (FFT geometry, hop cadence, per-hop smoothing/median
//! windows) is tuned for a ~44.1–48 kHz analysis rate. At high native capture
//! rates (88.2/96/176.4/192 kHz) the *fixed* [`crate::FFT_LEN`]/[`crate::HOP`]
//! geometry loses low-frequency resolution — `bin_hz = rate / FFT_LEN` grows, so
//! the sub/bass macro bands collapse into a couple of bins and the time-domain
//! pitch window shrinks below one period of a bass note — while the per-hop time
//! constants (expressed per hop) drift because the hop period shrinks with rate.
//!
//! Rather than scale the FFT size/hop per rate (which would break the frozen
//! public `FFT_LEN`/`HOP` contract and every history sized from them), we
//! anti-alias filter and integer-decimate the incoming stream down to a canonical
//! analysis rate near 48 kHz. Downstream everything sees a rate-stable signal, so
//! bass/sub bins, low-pitch detection, and every seconds-derived envelope behave
//! identically regardless of the device's native rate.
//!
//! At the common 44.1/48 kHz rates the decimation factor is 1 and the stream is
//! passed through untouched (bit-identical to the pre-decimation path).

use crate::CaptureFrame;

/// Canonical analysis sample rate (Hz). Native rates are decimated to the nearest
/// integer division of this so the analysis rate always lands in ~[32k, 64k].
pub(crate) const CANONICAL_ANALYSIS_HZ: f32 = 48_000.0;

/// Lowest native capture rate we will build an analyzer for. Below this the
/// Nyquist limit is too low for the band split to be meaningful.
pub(crate) const MIN_NATIVE_HZ: u32 = 4_000;
/// Highest native capture rate we will build an analyzer for. Anything higher is
/// rejected as pathological rather than allocating (bounded) histories for it.
pub(crate) const MAX_NATIVE_HZ: u32 = 768_000;

/// Integer decimation factor bringing `native_rate` closest to the canonical
/// analysis rate. Always ≥ 1; a non-finite or non-positive rate yields 1.
#[inline]
pub(crate) fn decimation_factor(native_rate: f32) -> usize {
    if !native_rate.is_finite() || native_rate <= 0.0 {
        return 1;
    }
    ((native_rate / CANONICAL_ANALYSIS_HZ).round() as i64).max(1) as usize
}

/// Effective analysis rate after decimation, i.e. `native_rate / factor`.
#[inline]
pub(crate) fn effective_rate(native_rate: f32) -> f32 {
    native_rate / decimation_factor(native_rate) as f32
}

/// Validate a native capture rate before building an analyzer. Rejects `0` and
/// rates outside `[MIN_NATIVE_HZ, MAX_NATIVE_HZ]` so construction never allocates
/// (bounded, but arbitrarily large) history buffers for a pathological rate.
pub(crate) fn validate_native_rate(rate: u32) -> Result<(), crate::AudioError> {
    if !(MIN_NATIVE_HZ..=MAX_NATIVE_HZ).contains(&rate) {
        return Err(crate::AudioError::Config(format!(
            "unsupported audio sample rate {rate} Hz (expected {MIN_NATIVE_HZ}..={MAX_NATIVE_HZ})"
        )));
    }
    Ok(())
}

/// Clamp a native rate into the supported band for the infallible constructor, so
/// a pathological rate degrades to a safe analysis rate instead of dividing by
/// zero or allocating unbounded histories. Prefer the fallible path to detect it.
#[inline]
pub(crate) fn clamp_native_rate(rate: u32) -> f32 {
    rate.clamp(MIN_NATIVE_HZ, MAX_NATIVE_HZ) as f32
}

/// A direct-form-II transposed biquad with non-finite guards (mirrors the ones in
/// `dsp.rs` / `linkwitz_riley.rs`, kept local so the decimator is self-contained).
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
    /// 2nd-order Butterworth low-pass (Q = 1/√2) via the RBJ cookbook.
    fn butterworth_low_pass(sample_rate: f32, f0: f32) -> Self {
        let nyquist_safe = (sample_rate * 0.49).max(1.0);
        let f0 = f0.clamp(1.0, nyquist_safe);
        let w0 = std::f32::consts::TAU * f0 / sample_rate.max(1.0);
        let (sin_w0, cos_w0) = w0.sin_cos();
        let alpha = sin_w0 * std::f32::consts::SQRT_2 * 0.5;
        let b1 = 1.0 - cos_w0;
        let b0 = b1 * 0.5;
        let b2 = b0;
        let a0 = 1.0 + alpha;
        let a1 = -2.0 * cos_w0;
        let a2 = 1.0 - alpha;
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

    #[inline]
    fn process(&mut self, x: f32) -> f32 {
        let x = if x.is_finite() { x } else { 0.0 };
        if !(self.z1.is_finite() && self.z2.is_finite()) {
            self.z1 = 0.0;
            self.z2 = 0.0;
        }
        let y = self.b0 * x + self.z1;
        self.z1 = self.b1 * x - self.a1 * y + self.z2;
        self.z2 = self.b2 * x - self.a2 * y;
        if y.is_finite() && self.z1.is_finite() && self.z2.is_finite() {
            y
        } else {
            self.z1 = 0.0;
            self.z2 = 0.0;
            0.0
        }
    }
}

/// A 4th-order Butterworth-squared (Linkwitz-Riley) low-pass: two cascaded
/// identical Butterworth sections, 24 dB/oct — the anti-alias filter run at the
/// native rate before downsampling.
#[derive(Clone, Copy, Debug)]
struct AntiAlias {
    a: Biquad,
    b: Biquad,
}

impl AntiAlias {
    fn new(sample_rate: f32, cutoff: f32) -> Self {
        Self {
            a: Biquad::butterworth_low_pass(sample_rate, cutoff),
            b: Biquad::butterworth_low_pass(sample_rate, cutoff),
        }
    }
    #[inline]
    fn process(&mut self, x: f32) -> f32 {
        self.b.process(self.a.process(x))
    }
}

/// Anti-aliasing integer decimator for the interleaved [`CaptureFrame`] stream.
///
/// Every native input sample is low-passed (per channel) below the post-decimation
/// Nyquist; one of every `factor` filtered samples is emitted. At `factor == 1`
/// the stream is passed through with no filtering, so 44.1/48 kHz capture is
/// bit-identical to the pre-decimation path.
pub(crate) struct FrameDecimator {
    factor: usize,
    lp_mono: AntiAlias,
    lp_left: AntiAlias,
    lp_right: AntiAlias,
    /// Counts filtered input samples since the last emission.
    phase: usize,
}

impl FrameDecimator {
    pub(crate) fn new(native_rate: f32) -> Self {
        let factor = decimation_factor(native_rate);
        let eff = native_rate / factor as f32;
        // Guard band a touch below the post-decimation Nyquist.
        let cutoff = 0.45 * eff;
        Self {
            factor,
            lp_mono: AntiAlias::new(native_rate, cutoff),
            lp_left: AntiAlias::new(native_rate, cutoff),
            lp_right: AntiAlias::new(native_rate, cutoff),
            phase: 0,
        }
    }

    /// Decimation factor in force (1 = passthrough).
    #[cfg(test)]
    #[inline]
    pub(crate) fn factor(&self) -> usize {
        self.factor
    }

    /// Feed one native-rate frame; returns the decimated frame on the samples that
    /// survive downsampling, `None` on the ones dropped between them.
    #[inline]
    pub(crate) fn push(&mut self, frame: CaptureFrame) -> Option<CaptureFrame> {
        if self.factor <= 1 {
            return Some(frame);
        }
        // Filter *every* sample so aliases fold out before we drop samples.
        let mono = self.lp_mono.process(frame.mono);
        let left = self.lp_left.process(frame.left);
        let right = self.lp_right.process(frame.right);
        self.phase += 1;
        if self.phase >= self.factor {
            self.phase = 0;
            Some(CaptureFrame { mono, left, right })
        } else {
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f32::consts::TAU;

    #[test]
    fn decimation_factor_maps_common_rates_to_canonical_band() {
        // Standard rates pass through untouched.
        assert_eq!(decimation_factor(44_100.0), 1);
        assert_eq!(decimation_factor(48_000.0), 1);
        // High rates decimate to ~44.1/48 kHz.
        assert_eq!(decimation_factor(88_200.0), 2);
        assert_eq!(decimation_factor(96_000.0), 2);
        assert_eq!(decimation_factor(176_400.0), 4);
        assert_eq!(decimation_factor(192_000.0), 4);

        for &rate in &[
            44_100.0f32,
            48_000.0,
            88_200.0,
            96_000.0,
            176_400.0,
            192_000.0,
        ] {
            let eff = effective_rate(rate);
            assert!(
                (32_000.0..=64_000.0).contains(&eff),
                "effective rate {eff} for native {rate} left the canonical band"
            );
        }
    }

    #[test]
    fn factor_one_is_bit_identical_passthrough() {
        let mut d = FrameDecimator::new(48_000.0);
        assert_eq!(d.factor(), 1);
        for i in 0..64 {
            let f = CaptureFrame {
                mono: (i as f32 * 0.13).sin(),
                left: i as f32,
                right: -(i as f32),
            };
            let out = d.push(f).expect("factor-1 decimator emits every frame");
            assert_eq!(out.mono, f.mono);
            assert_eq!(out.left, f.left);
            assert_eq!(out.right, f.right);
        }
    }

    #[test]
    fn decimator_passes_low_tone_and_rejects_ultrasonic() {
        // At 192 kHz, decimating by 4 gives a 48 kHz analysis rate (Nyquist 24 kHz).
        let native = 192_000.0f32;
        let factor = decimation_factor(native);
        assert_eq!(factor, 4);

        // A 100 Hz tone is far below the post-decimation Nyquist → preserved.
        // A 40 kHz tone is above it → must be strongly attenuated (would otherwise
        // alias down into the audible band and corrupt the low bins).
        let run = |freq: f32| -> f32 {
            let mut d = FrameDecimator::new(native);
            let mut sumsq = 0.0f32;
            let mut count = 0usize;
            // ~0.2 s of audio; ignore the filter warm-up transient.
            let total = (native * 0.2) as usize;
            let warm = total / 4;
            for i in 0..total {
                let s = (TAU * freq * i as f32 / native).sin();
                if let Some(out) = d.push(CaptureFrame {
                    mono: s,
                    left: s,
                    right: s,
                }) {
                    if i >= warm {
                        sumsq += out.mono * out.mono;
                        count += 1;
                    }
                }
            }
            (sumsq / count.max(1) as f32).sqrt()
        };

        let low_rms = run(100.0);
        let ultrasonic_rms = run(40_000.0);
        // Full-scale sine RMS ≈ 0.707; the low tone should survive nearly intact.
        assert!(
            low_rms > 0.6,
            "100 Hz tone should survive decimation: {low_rms}"
        );
        // The 40 kHz tone must be crushed well below the passband.
        assert!(
            ultrasonic_rms < 0.05,
            "40 kHz tone should be anti-alias filtered out: {ultrasonic_rms}"
        );
    }
}

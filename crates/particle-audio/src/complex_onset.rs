//! Complex-domain onset detection function (Bello et al. 2004, "On the Use of
//! Phase and Energy for Musical Onset Detection in the Complex Domain").
//!
//! The classic energy/spectral-flux ODF only sees magnitude growth; it misses
//! soft onsets (tonal note changes) where the magnitude barely moves but the
//! phase jumps. The complex-domain ODF predicts, for every bin, the *expected*
//! complex value of the current frame from the previous two frames — magnitude
//! held from frame `n-1`, phase extrapolated linearly from the recent phase
//! trajectory (a steady sinusoid advances its phase by a constant amount per
//! hop). The deviation is the Euclidean distance between the predicted and the
//! observed complex spectrum, summed across bins:
//!
//! ```text
//!   target_k = |X_{n-1,k}| * exp(j * (2*phi_{n-1,k} - phi_{n-2,k}))
//!   Gamma_n  = sum_k | X_{n,k} - target_k |
//! ```
//!
//! A stationary tone predicts itself perfectly (deviation ~0); any transient —
//! magnitude burst *or* phase discontinuity — spikes `Gamma`. We then run the
//! novelty through the project's adaptive-median onset detector to get a clean
//! normalized `0..1` transient rail.
//!
//! This is a from-scratch implementation derived from the paper's equations. It
//! takes no GPL code (no aubio); it shares only the pure-Rust onset/smoothing
//! primitives already in this crate.

use realfft::num_complex::Complex;

use crate::onset::OnsetDetector;
use crate::smoothing::OnePole;

/// Wrap a phase value into `(-pi, pi]` (the "princarg" of the phase-vocoder
/// literature) so a linear phase prediction doesn't blow up across wraps.
#[inline]
fn princarg(phase: f32) -> f32 {
    let tau = std::f32::consts::TAU;
    let mut p = phase % tau;
    if p > std::f32::consts::PI {
        p -= tau;
    } else if p <= -std::f32::consts::PI {
        p += tau;
    }
    p
}

/// Stateful complex-domain onset detector. Keeps the previous two frames'
/// magnitude + (wrapped) phase per bin so it can form the stationary prediction
/// for the current frame.
pub struct ComplexOnsetDetector {
    prev_mag: Vec<f32>,
    prev_phase: Vec<f32>,
    prev2_phase: Vec<f32>,
    /// Whether we have seen enough frames to make a valid prediction.
    primed: u8,
    /// Adaptive-median onset gate turning raw novelty into a clean impulse.
    detector: OnsetDetector,
    /// Output envelope smoothing of the detector impulse (light).
    out_lp: OnePole,
}

impl ComplexOnsetDetector {
    /// `n_bins` is the real-FFT output length (`fft_len/2 + 1`); `hop_dt` is the
    /// hop period in seconds (used to tune the median window / refractory time).
    pub fn new(n_bins: usize, hop_dt: f32) -> Self {
        // Short adaptive-median window (~0.2 s) + fast attack and a snappy decay
        // (~140 ms) — the complex-domain ODF is sharper than energy flux so we
        // can afford a tighter envelope. Refractory ~60 ms suppresses re-fires.
        let med = ((0.2 / hop_dt) as usize).max(8);
        Self {
            prev_mag: vec![0.0; n_bins],
            prev_phase: vec![0.0; n_bins],
            prev2_phase: vec![0.0; n_bins],
            primed: 0,
            detector: OnsetDetector::new(med, 1.5, 1e-3, 60.0, 5.0, 140.0, hop_dt),
            out_lp: OnePole::new(0.25),
        }
    }

    /// Process one hop's complex spectrum. Returns the smoothed, normalized
    /// `0..1` complex-domain onset rail (a transient spike). `is_silent` forces
    /// the rail to relax to 0 and keeps the predictor warm without false fires.
    /// `sensitivity` (~0.1..3) raises detection likelihood as it grows.
    pub fn process(&mut self, spectrum: &[Complex<f32>], is_silent: bool, sensitivity: f32) -> f32 {
        let n = spectrum.len().min(self.prev_mag.len());

        // --- raw complex-domain novelty Gamma_n + current-frame magnitude ---
        let mut gamma = 0.0f32;
        let mut mag_sum = 0.0f32;
        if self.primed >= 2 && !is_silent {
            for k in 0..n {
                let cur = spectrum[k];
                let mag_prev = self.prev_mag[k];
                // Linear phase prediction: a steady partial advances its phase by
                // a constant delta per hop, so the predicted phase is
                // phi_{n-1} + (phi_{n-1} - phi_{n-2}). Use princarg on the delta
                // so wraps don't explode the extrapolation.
                let dphi = princarg(self.prev_phase[k] - self.prev2_phase[k]);
                let pred_phase = self.prev_phase[k] + dphi;
                let (sin_p, cos_p) = pred_phase.sin_cos();
                let target = Complex::new(mag_prev * cos_p, mag_prev * sin_p);
                let dev = cur - target;
                gamma += dev.norm();
                mag_sum += cur.norm();
            }
        }

        // --- scale-relative novelty, then gate through the adaptive-median
        //     onset detector to get a clean impulse ---
        // Dividing the summed deviation by the current frame's total magnitude
        // makes the novelty level-invariant (loud and quiet passages read the
        // same) and immune to float-noise amplification: a perfectly predicted
        // (stationary) spectrum has gamma == 0 -> novelty == 0 regardless of the
        // absolute level. A fully-unpredicted spectrum approaches ~1.
        let novelty = if mag_sum > 1e-4 {
            (gamma / mag_sum).clamp(0.0, 4.0)
        } else {
            0.0
        };
        let (impulse, _fired) = if is_silent {
            self.detector.process(0.0, sensitivity);
            (0.0, false)
        } else {
            self.detector.process(novelty, sensitivity)
        };
        let out = self.out_lp.process(impulse).clamp(0.0, 1.0);

        // --- roll the frame history forward ---
        for k in 0..n {
            self.prev2_phase[k] = self.prev_phase[k];
            self.prev_phase[k] = spectrum[k].arg();
            self.prev_mag[k] = spectrum[k].norm();
        }
        if self.primed < 2 {
            self.primed += 1;
        }

        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f32::consts::TAU;

    /// Build a fake complex spectrum: one tonal bin at `bin` with magnitude
    /// `mag` and the given phase, everything else zero.
    fn tone(n: usize, bin: usize, mag: f32, phase: f32) -> Vec<Complex<f32>> {
        let mut s = vec![Complex::new(0.0, 0.0); n];
        if bin < n {
            let (sin_p, cos_p) = phase.sin_cos();
            s[bin] = Complex::new(mag * cos_p, mag * sin_p);
        }
        s
    }

    #[test]
    fn princarg_wraps_into_principal_range() {
        assert!((princarg(0.0)).abs() < 1e-6);
        assert!((princarg(TAU)).abs() < 1e-5);
        let p = princarg(3.0 * std::f32::consts::PI);
        assert!(p.abs() <= std::f32::consts::PI + 1e-5);
    }

    #[test]
    fn steady_tone_produces_low_novelty() {
        // A partial advancing its phase by a constant amount each hop is exactly
        // what the predictor expects -> deviation should be ~0, so no onset.
        let n = 64;
        let hop_dt = 512.0 / 48_000.0;
        let mut det = ComplexOnsetDetector::new(n, hop_dt);
        let dphi = 0.37f32; // constant phase advance
        let mut max_out = 0.0f32;
        for i in 0..120 {
            let phase = dphi * i as f32;
            let s = tone(n, 8, 1.0, phase);
            let out = det.process(&s, false, 1.0);
            if i > 5 {
                max_out = max_out.max(out);
            }
        }
        assert!(
            max_out < 0.2,
            "a steady sinusoid should not fire the complex-domain ODF, got {max_out}"
        );
    }

    #[test]
    fn transient_spikes_the_rail() {
        // Run a steady tone, then inject a sudden broadband magnitude burst with
        // scrambled phase: the complex-domain ODF must spike.
        let n = 64;
        let hop_dt = 512.0 / 48_000.0;
        let mut det = ComplexOnsetDetector::new(n, hop_dt);
        let dphi = 0.37f32;
        let mut baseline = 0.0f32;
        for i in 0..60 {
            let s = tone(n, 8, 0.3, dphi * i as f32);
            baseline = det.process(&s, false, 1.0);
        }

        // Sudden transient: every bin lights up with a fresh (unpredicted) phase.
        let mut burst = vec![Complex::new(0.0, 0.0); n];
        for (k, b) in burst.iter_mut().enumerate() {
            let phase = (k as f32 * 1.7).sin() * TAU; // arbitrary, unpredicted
            let (sin_p, cos_p) = phase.sin_cos();
            *b = Complex::new(cos_p, sin_p);
        }
        let mut peak = 0.0f32;
        for _ in 0..4 {
            peak = peak.max(det.process(&burst, false, 1.0));
        }
        assert!(
            peak > baseline + 0.2,
            "transient should spike the complex onset rail: baseline={baseline}, peak={peak}"
        );
        assert!(peak > 0.25, "transient onset rail too weak: {peak}");
    }

    #[test]
    fn silence_holds_rail_at_zero() {
        let n = 32;
        let hop_dt = 512.0 / 48_000.0;
        let mut det = ComplexOnsetDetector::new(n, hop_dt);
        // Prime with a tone, then go silent: the rail must relax to 0.
        for i in 0..40 {
            det.process(&tone(n, 4, 1.0, 0.2 * i as f32), false, 1.0);
        }
        let mut last = 1.0f32;
        for _ in 0..40 {
            last = det.process(&tone(n, 4, 1.0, 0.0), true, 1.0);
        }
        assert!(last < 1e-3, "silence should hold the rail at 0, got {last}");
    }
}

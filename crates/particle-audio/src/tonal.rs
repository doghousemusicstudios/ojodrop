//! Tonal analysis helpers.
//!
//! This is the first harmony slice for the audio engine: an FFT-derived 12-bin
//! chromagram plus key/chord estimates. It is intentionally not the full backlog
//! CQT tier yet: the DSP feeds it from the causal HPSS harmonic mask, but it still
//! runs on FFT bins. Key output is smoothed with a small online 24-state
//! major/minor tracker so one-frame chroma glitches do not whip the palette.

use std::f32::consts::PI;
use std::sync::Arc;

use realfft::num_complex::Complex;
use realfft::{ComplexToReal, RealFftPlanner, RealToComplex};

pub const CHROMA_BINS: usize = 12;

const CHROMA_LO_HZ: f32 = 40.0;
const CHROMA_HI_HZ: f32 = 5_000.0;
const PITCH_NORM_LO_HZ: f32 = 40.0;
const PITCH_NORM_HI_HZ: f32 = 2_000.0;

// Krumhansl-Schmuckler pitch-class profiles, C-rooted.
const KS_MAJOR: [f32; CHROMA_BINS] = [
    6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88,
];
const KS_MINOR: [f32; CHROMA_BINS] = [
    6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17,
];

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct TonalEstimate {
    /// Pitch class: C=0, C#/Db=1, ..., B=11.
    pub root: usize,
    /// `true` for minor, `false` for major.
    pub is_minor: bool,
    /// Best-vs-runner-up separation, normalized to 0..1.
    pub confidence: f32,
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct PitchEstimate {
    /// Estimated fundamental in Hz. `0.0` means unvoiced/unknown.
    pub hz: f32,
    /// Log-frequency rail normalized between 40 Hz and 2 kHz.
    pub normalized: f32,
    /// Voicing confidence, 0..1.
    pub confidence: f32,
}

/// Online 24-state key smoother (12 roots × major/minor).
///
/// This is a compact Viterbi-style dynamic program over pitch-class profile
/// scores. It favors the previous key unless a new key stays better for several
/// frames, which keeps harmony-driven palettes from flickering on transient
/// chroma errors while still following sustained modulations.
#[derive(Clone, Debug)]
pub struct KeySmoother {
    scores: [f32; KEY_STATE_COUNT],
    warmed: bool,
}

const KEY_STATE_COUNT: usize = CHROMA_BINS * 2;
const KEY_MEMORY: f32 = 0.92;
const KEY_EMISSION_WEIGHT: f32 = 0.35;
const KEY_ROOT_CHANGE_PENALTY: f32 = 0.18;
const KEY_MODE_CHANGE_PENALTY: f32 = 0.12;

impl Default for KeySmoother {
    fn default() -> Self {
        Self {
            scores: [0.0; KEY_STATE_COUNT],
            warmed: false,
        }
    }
}

impl KeySmoother {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn reset(&mut self) {
        self.scores.fill(0.0);
        self.warmed = false;
    }

    pub fn update(&mut self, chroma: &[f32; CHROMA_BINS]) -> TonalEstimate {
        if chroma.iter().copied().fold(0.0f32, f32::max) <= 1e-6 {
            self.reset();
            return TonalEstimate::default();
        }

        let emissions = key_emissions(chroma);
        if !self.warmed {
            self.scores = emissions;
            normalize_scores(&mut self.scores);
            self.warmed = true;
        } else {
            let prev = self.scores;
            let mut next = [0.0f32; KEY_STATE_COUNT];
            for to in 0..KEY_STATE_COUNT {
                let mut best_prev = f32::NEG_INFINITY;
                for from in 0..KEY_STATE_COUNT {
                    best_prev = best_prev.max(prev[from] - transition_penalty(from, to));
                }
                next[to] = best_prev * KEY_MEMORY + emissions[to] * KEY_EMISSION_WEIGHT;
            }
            normalize_scores(&mut next);
            self.scores = next;
        }

        estimate_from_state_scores(&self.scores)
    }
}

/// Clean-room mono pitch detector using FFT-based normalized autocorrelation.
///
/// This is McLeod/YIN-shaped without pulling a GPL dependency into the realtime
/// DSP: it computes the NSDF over the requested f0 range, prefers the first strong
/// local maximum to avoid octave/subharmonic jumps, and refines the selected lag
/// with a parabolic fit. Unlike the naive lag-scan (which is O(N²) per hop — the
/// audited hot path, P2-AUD-002), the autocorrelation numerator is computed once
/// per hop with a zero-padded real FFT (Wiener-Khinchin), so work is
/// O(N log N) and bounded by a fixed window cap while the pitch — including the
/// bass register — is preserved bit-for-bit within FFT rounding.
///
/// STATEFUL only for buffer reuse (FFT plan + scratch): the estimate itself is a
/// pure function of the current window. Polyphonic/percussive rejection still
/// happens in the caller by gating this confidence with the harmonic HPSS rails.
pub struct NsdfPitchDetector {
    /// Analysis window cap (samples). Longer inputs use the most recent
    /// `max_window` samples so per-hop work stays bounded.
    max_window: usize,
    /// FFT length: the smallest power of two ≥ `2 * max_window`, so the circular
    /// autocorrelation from the FFT equals the linear autocorrelation for every
    /// lag the detector inspects (`tau ≤ n/2`).
    fft_len: usize,
    fft: Arc<dyn RealToComplex<f32>>,
    ifft: Arc<dyn ComplexToReal<f32>>,
    /// Mean-subtracted, zero-padded window (also consumed as FFT scratch).
    time: Vec<f32>,
    /// Forward FFT output, reused as the power spectrum / inverse FFT input.
    freq: Vec<Complex<f32>>,
    /// Shared FFT scratch (sized for both directions).
    scratch: Vec<Complex<f32>>,
    /// Inverse FFT output — the (unnormalized) autocorrelation.
    acf: Vec<f32>,
    /// Cumulative sum of squares of the mean-subtracted window (`n + 1` entries)
    /// for the NSDF denominator in O(1) per lag.
    cumsq: Vec<f32>,
    /// NSDF values for lags `0..=max_tau`, reused each hop.
    nsdf: Vec<f32>,
}

impl NsdfPitchDetector {
    /// Build a detector for windows up to `max_window` samples.
    pub fn new(max_window: usize) -> Self {
        let max_window = max_window.max(32);
        let fft_len = (2 * max_window).next_power_of_two();
        let mut planner = RealFftPlanner::<f32>::new();
        let fft = planner.plan_fft_forward(fft_len);
        let ifft = planner.plan_fft_inverse(fft_len);
        let scratch_len = fft.get_scratch_len().max(ifft.get_scratch_len());
        Self {
            max_window,
            fft_len,
            time: vec![0.0; fft_len],
            freq: vec![Complex::new(0.0, 0.0); fft_len / 2 + 1],
            scratch: vec![Complex::new(0.0, 0.0); scratch_len],
            acf: vec![0.0; fft_len],
            cumsq: vec![0.0; max_window + 1],
            nsdf: vec![0.0; max_window / 2 + 1],
            fft,
            ifft,
        }
    }

    /// The analysis window cap in samples (per-hop work is bounded by this).
    #[cfg(test)]
    pub(crate) fn max_window(&self) -> usize {
        self.max_window
    }

    /// Estimate the monophonic fundamental over `samples` (the most recent
    /// `max_window` are used). Mirrors the reference NSDF estimator but computes
    /// the autocorrelation with one FFT instead of an O(N²) lag scan.
    pub fn estimate(
        &mut self,
        samples: &[f32],
        sample_rate: f32,
        lo_hz: f32,
        hi_hz: f32,
    ) -> PitchEstimate {
        if sample_rate <= 0.0 || hi_hz <= lo_hz {
            return PitchEstimate::default();
        }
        // Bound the analysis window (P2-AUD-002): never inspect more than the cap.
        let n = samples.len().min(self.max_window);
        if n < 32 {
            return PitchEstimate::default();
        }
        let samples = &samples[samples.len() - n..];

        let min_tau = (sample_rate / hi_hz.max(1.0)).round().max(2.0) as usize;
        let max_tau = (sample_rate / lo_hz.max(1.0)).round().min((n / 2) as f32) as usize;
        if min_tau + 2 >= max_tau {
            return PitchEstimate::default();
        }

        let mean = samples.iter().copied().sum::<f32>() / n as f32;

        // Mean-subtract into the zero-padded FFT input and accumulate the
        // cumulative sum of squares in the same pass (used for the NSDF
        // denominator and the RMS gate) — done BEFORE the FFT consumes `time`.
        for (dst, &s) in self.time[..n].iter_mut().zip(samples) {
            *dst = s - mean;
        }
        for slot in self.time[n..].iter_mut() {
            *slot = 0.0;
        }
        let mut sumsq = 0.0f32;
        self.cumsq[0] = 0.0;
        for i in 0..n {
            let y = self.time[i];
            sumsq += y * y;
            self.cumsq[i + 1] = sumsq;
        }
        let rms = (sumsq / n as f32).sqrt();
        if rms < 1e-4 {
            return PitchEstimate::default();
        }

        // Autocorrelation via FFT (Wiener-Khinchin): r(tau) = IFFT(|FFT(y)|²) /
        // fft_len. The zero-padding to ≥ 2n makes the circular result equal the
        // linear autocorrelation for all lags we inspect.
        self.fft
            .process_with_scratch(&mut self.time, &mut self.freq, &mut self.scratch)
            .expect("pitch forward FFT length invariant");
        for c in self.freq.iter_mut() {
            let power = c.re * c.re + c.im * c.im;
            c.re = power;
            c.im = 0.0;
        }
        self.ifft
            .process_with_scratch(&mut self.freq, &mut self.acf, &mut self.scratch)
            .expect("pitch inverse FFT length invariant");

        let inv_fft = 1.0 / self.fft_len as f32;
        let total = self.cumsq[n];
        for tau in 0..=max_tau {
            let r = self.acf[tau] * inv_fft;
            // energy(tau) = Σ y[i]² + Σ y[i+tau]² over the overlap i∈[0, n-tau).
            let energy = self.cumsq[n - tau] + total - self.cumsq[tau];
            self.nsdf[tau] = if energy <= 1e-12 {
                0.0
            } else {
                (2.0 * r / energy).clamp(-1.0, 1.0)
            };
        }

        nsdf_peak_pick(
            &self.nsdf[..=max_tau],
            min_tau,
            max_tau,
            rms,
            sample_rate,
            lo_hz,
            hi_hz,
        )
    }
}

/// Pick the pitch from a precomputed NSDF curve (`nsdf[tau]` for `tau` in
/// `0..=max_tau`). Shared by the FFT detector and the reference lag-scan so both
/// resolve the fundamental identically; only the NSDF *source* differs.
fn nsdf_peak_pick(
    nsdf: &[f32],
    min_tau: usize,
    max_tau: usize,
    rms: f32,
    sample_rate: f32,
    lo_hz: f32,
    hi_hz: f32,
) -> PitchEstimate {
    let mut prev_prev = nsdf[min_tau - 1];
    let mut prev = nsdf[min_tau];
    let mut best_tau = 0usize;
    let mut best_peak = 0.0f32;
    let mut first_strong_tau = 0usize;
    let mut first_strong_peak = 0.0f32;

    for tau in (min_tau + 1)..=max_tau {
        let curr = nsdf[tau];
        let peak_tau = tau - 1;
        if prev > prev_prev && prev >= curr && prev > 0.0 {
            if prev > best_peak {
                best_peak = prev;
                best_tau = peak_tau;
            }
            if first_strong_tau == 0 && prev >= 0.72 {
                first_strong_tau = peak_tau;
                first_strong_peak = prev;
            }
        }
        prev_prev = prev;
        prev = curr;
    }

    if best_tau == 0 || best_peak < 0.45 {
        return PitchEstimate::default();
    }

    let chosen_tau = if first_strong_tau != 0 && first_strong_peak >= best_peak * 0.88 {
        first_strong_tau
    } else {
        best_tau
    };

    let left = nsdf[chosen_tau.saturating_sub(1)];
    let center = nsdf[chosen_tau];
    let right = nsdf[(chosen_tau + 1).min(max_tau)];
    let denom = left - 2.0 * center + right;
    let offset = if denom.abs() > 1e-6 {
        (0.5 * (left - right) / denom).clamp(-0.5, 0.5)
    } else {
        0.0
    };
    let tau = (chosen_tau as f32 + offset).max(1.0);
    let hz = sample_rate / tau;
    if !hz.is_finite() || hz < lo_hz * 0.8 || hz > hi_hz * 1.2 {
        return PitchEstimate::default();
    }

    let confidence = (((center.max(best_peak) - 0.45) / 0.5).clamp(0.0, 1.0)
        * (rms / 0.01).clamp(0.0, 1.0))
    .clamp(0.0, 1.0);
    PitchEstimate {
        hz,
        normalized: pitch_norm_from_hz(hz),
        confidence,
    }
}

pub fn pitch_norm_from_hz(hz: f32) -> f32 {
    if hz <= 0.0 || !hz.is_finite() {
        return 0.0;
    }
    ((hz / PITCH_NORM_LO_HZ).log2() / (PITCH_NORM_HI_HZ / PITCH_NORM_LO_HZ).log2()).clamp(0.0, 1.0)
}

/// Reference lag-scan NSDF estimator (the pre-P2-AUD-002 O(N²) implementation),
/// retained as the correctness oracle for [`NsdfPitchDetector`]. Compiled only in
/// test builds — the realtime path uses the FFT detector.
#[cfg(test)]
pub(crate) fn estimate_mono_pitch(
    samples: &[f32],
    sample_rate: f32,
    lo_hz: f32,
    hi_hz: f32,
) -> PitchEstimate {
    if sample_rate <= 0.0 || samples.len() < 32 || hi_hz <= lo_hz {
        return PitchEstimate::default();
    }

    let n = samples.len();
    let min_tau = (sample_rate / hi_hz.max(1.0)).round().max(2.0) as usize;
    let max_tau = (sample_rate / lo_hz.max(1.0)).round().min((n / 2) as f32) as usize;
    if min_tau + 2 >= max_tau {
        return PitchEstimate::default();
    }

    let mean = samples.iter().copied().sum::<f32>() / n as f32;
    let rms = (samples
        .iter()
        .map(|&s| {
            let x = s - mean;
            x * x
        })
        .sum::<f32>()
        / n as f32)
        .sqrt();
    if rms < 1e-4 {
        return PitchEstimate::default();
    }

    // Build the NSDF curve with the direct lag scan, then share the peak picker.
    let mut nsdf = vec![0.0f32; max_tau + 1];
    for (tau, slot) in nsdf.iter_mut().enumerate() {
        *slot = nsdf_at(samples, mean, tau);
    }
    nsdf_peak_pick(&nsdf, min_tau, max_tau, rms, sample_rate, lo_hz, hi_hz)
}

#[cfg(test)]
fn nsdf_at(samples: &[f32], mean: f32, tau: usize) -> f32 {
    if tau == 0 || tau >= samples.len() {
        return 0.0;
    }
    let mut ac = 0.0f32;
    let mut energy = 0.0f32;
    for i in 0..(samples.len() - tau) {
        let a = samples[i] - mean;
        let b = samples[i + tau] - mean;
        ac += a * b;
        energy += a * a + b * b;
    }
    if energy <= 1e-12 {
        0.0
    } else {
        (2.0 * ac / energy).clamp(-1.0, 1.0)
    }
}

/// Fold an FFT magnitude spectrum into 12 pitch classes.
///
/// Frequencies outside a useful musical range are ignored, each FFT bin is
/// assigned to the nearest equal-tempered pitch class, and the output is max
/// normalized. This gives the render side stable color rails without adding a
/// CQT kernel or HPSS dependency yet.
pub fn chroma_from_spectrum(mag: &[f32], bin_hz: f32) -> [f32; CHROMA_BINS] {
    let mut chroma = [0.0f32; CHROMA_BINS];
    if bin_hz <= 0.0 {
        return chroma;
    }

    for (bin, &m) in mag.iter().enumerate().skip(1) {
        if m <= 0.0 {
            continue;
        }
        let hz = bin as f32 * bin_hz;
        if !(CHROMA_LO_HZ..=CHROMA_HI_HZ).contains(&hz) {
            continue;
        }

        let midi = 69.0 + 12.0 * (hz / 440.0).log2();
        let nearest = midi.round();
        let semitone_dist = (midi - nearest).abs();
        if semitone_dist > 0.5 {
            continue;
        }
        let pc = (nearest as i32).rem_euclid(CHROMA_BINS as i32) as usize;
        let semitone_window = 0.5 + 0.5 * (2.0 * PI * semitone_dist).cos();
        let power = m * m;
        chroma[pc] += power * semitone_window;
    }

    max_normalize(&mut chroma);
    chroma
}

pub fn estimate_key(chroma: &[f32; CHROMA_BINS]) -> TonalEstimate {
    estimate_from_profiles(chroma, &KS_MAJOR, &KS_MINOR)
}

pub fn estimate_chord(chroma: &[f32; CHROMA_BINS]) -> TonalEstimate {
    let major = triad_profile(4);
    let minor = triad_profile(3);
    estimate_from_profiles(chroma, &major, &minor)
}

/// Palette-oriented rails derived from a key estimate.
///
/// `hue` is the detected key root around the 12-tone wheel, while `mood` gates
/// toward neutral when the estimate is uncertain: major trends bright/open,
/// minor trends dark/closed.
pub fn palette_from_key(key: TonalEstimate) -> (f32, f32) {
    let confidence = key.confidence.clamp(0.0, 1.0);
    if confidence <= 0.0 {
        return (0.0, 0.0);
    }
    let hue = key.root as f32 / CHROMA_BINS as f32;
    let target_mood = if key.is_minor { 0.0 } else { 1.0 };
    let mood = 0.5 + (target_mood - 0.5) * confidence;
    (hue, mood.clamp(0.0, 1.0))
}

fn triad_profile(third: usize) -> [f32; CHROMA_BINS] {
    let mut out = [0.0f32; CHROMA_BINS];
    out[0] = 1.0;
    out[third] = 0.92;
    out[7] = 0.88;
    out
}

fn estimate_from_profiles(
    chroma: &[f32; CHROMA_BINS],
    major_profile: &[f32; CHROMA_BINS],
    minor_profile: &[f32; CHROMA_BINS],
) -> TonalEstimate {
    if chroma.iter().copied().fold(0.0f32, f32::max) <= 1e-6 {
        return TonalEstimate::default();
    }

    let mut best = (f32::NEG_INFINITY, 0usize, false);
    let mut second = f32::NEG_INFINITY;

    for root in 0..CHROMA_BINS {
        for &(is_minor, profile) in &[(false, major_profile), (true, minor_profile)] {
            let score = pearson_profile_score(chroma, profile, root);
            if score > best.0 {
                second = best.0;
                best = (score, root, is_minor);
            } else if score > second {
                second = score;
            }
        }
    }

    let separation = (best.0 - second).max(0.0);
    TonalEstimate {
        root: best.1,
        is_minor: best.2,
        confidence: (separation * 2.5).clamp(0.0, 1.0),
    }
}

fn key_emissions(chroma: &[f32; CHROMA_BINS]) -> [f32; KEY_STATE_COUNT] {
    let mut scores = [0.0f32; KEY_STATE_COUNT];
    for root in 0..CHROMA_BINS {
        scores[root] = pearson_profile_score(chroma, &KS_MAJOR, root);
        scores[CHROMA_BINS + root] = pearson_profile_score(chroma, &KS_MINOR, root);
    }
    scores
}

fn estimate_from_state_scores(scores: &[f32; KEY_STATE_COUNT]) -> TonalEstimate {
    let mut best = (f32::NEG_INFINITY, 0usize);
    let mut second = f32::NEG_INFINITY;
    for (state, &score) in scores.iter().enumerate() {
        if score > best.0 {
            second = best.0;
            best = (score, state);
        } else if score > second {
            second = score;
        }
    }
    let separation = (best.0 - second).max(0.0);
    TonalEstimate {
        root: best.1 % CHROMA_BINS,
        is_minor: best.1 >= CHROMA_BINS,
        confidence: (separation * 1.6).clamp(0.0, 1.0),
    }
}

fn transition_penalty(from: usize, to: usize) -> f32 {
    if from == to {
        return 0.0;
    }
    let from_root = from % CHROMA_BINS;
    let to_root = to % CHROMA_BINS;
    let clockwise = from_root.abs_diff(to_root);
    let root_steps = clockwise.min(CHROMA_BINS - clockwise) as f32;
    let mode_penalty = if (from >= CHROMA_BINS) == (to >= CHROMA_BINS) {
        0.0
    } else {
        KEY_MODE_CHANGE_PENALTY
    };
    root_steps * KEY_ROOT_CHANGE_PENALTY + mode_penalty
}

fn normalize_scores(scores: &mut [f32; KEY_STATE_COUNT]) {
    let max = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    if !max.is_finite() {
        scores.fill(0.0);
        return;
    }
    for score in scores {
        *score = (*score - max).clamp(-4.0, 0.0);
    }
}

fn pearson_profile_score(
    chroma: &[f32; CHROMA_BINS],
    profile: &[f32; CHROMA_BINS],
    root: usize,
) -> f32 {
    let chroma_mean = chroma.iter().sum::<f32>() / CHROMA_BINS as f32;
    let profile_mean = profile.iter().sum::<f32>() / CHROMA_BINS as f32;
    let mut num = 0.0f32;
    let mut den_a = 0.0f32;
    let mut den_b = 0.0f32;

    for i in 0..CHROMA_BINS {
        let a = chroma[(root + i) % CHROMA_BINS] - chroma_mean;
        let b = profile[i] - profile_mean;
        num += a * b;
        den_a += a * a;
        den_b += b * b;
    }

    if den_a <= 1e-12 || den_b <= 1e-12 {
        return 0.0;
    }
    (num / (den_a * den_b).sqrt()).clamp(-1.0, 1.0)
}

fn max_normalize(values: &mut [f32; CHROMA_BINS]) {
    let max = values.iter().copied().fold(0.0f32, f32::max);
    if max <= 1e-12 {
        values.fill(0.0);
        return;
    }
    for v in values {
        *v = (*v / max).clamp(0.0, 1.0);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn bin_for(hz: f32, bin_hz: f32) -> usize {
        (hz / bin_hz).round() as usize
    }

    fn sine(hz: f32, sample_rate: f32, n: usize) -> Vec<f32> {
        (0..n)
            .map(|i| (2.0 * PI * hz * i as f32 / sample_rate).sin() * 0.75)
            .collect()
    }

    /// P2-AUD-002: the FFT-autocorrelation detector resolves the same pitch as the
    /// reference O(N²) lag scan (within FFT rounding) across the whole range,
    /// including the bass register, and caps its analysis window so per-hop work is
    /// bounded regardless of input length.
    #[test]
    fn fft_detector_matches_reference_across_range_with_bounded_window() {
        let sample_rate = 48_000.0;
        let mut det = NsdfPitchDetector::new(2048);
        assert_eq!(det.max_window(), 2048, "analysis window must be capped");

        // Bass through upper-mid — the bass tones exercise the low-f0 range the
        // audit requires the detector to keep resolving.
        for &hz in &[55.0f32, 82.5, 110.0, 220.0, 440.0, 660.0] {
            let sig = sine(hz, sample_rate, 2048);
            let reference = estimate_mono_pitch(&sig, sample_rate, 45.0, 1600.0);
            let got = det.estimate(&sig, sample_rate, 45.0, 1600.0);
            assert!(
                (got.hz - reference.hz).abs() < 0.5,
                "FFT pitch {} Hz diverged from reference {} Hz at {hz} Hz",
                got.hz,
                reference.hz
            );
            assert!(
                (got.confidence - reference.confidence).abs() < 0.02,
                "confidence diverged at {hz} Hz: {} vs {}",
                got.confidence,
                reference.confidence
            );
            assert!(
                (got.hz - hz).abs() < (hz * 0.01).max(1.0),
                "detected pitch {} strayed from true {hz} Hz",
                got.hz
            );
        }

        // A window longer than the cap is accepted (recent samples only) — work
        // stays bounded by max_window, and the estimate still tracks the tone.
        let long = sine(220.0, sample_rate, 8192);
        let got = det.estimate(&long, sample_rate, 45.0, 1600.0);
        assert!(
            (got.hz - 220.0).abs() < 2.0,
            "capped-window estimate should still resolve 220 Hz, got {}",
            got.hz
        );
    }

    #[test]
    fn mono_pitch_estimates_sine_fundamental() {
        let sample_rate = 48_000.0;
        let pitch = estimate_mono_pitch(&sine(220.0, sample_rate, 2048), sample_rate, 45.0, 1600.0);
        assert!(
            (pitch.hz - 220.0).abs() < 2.0,
            "expected ~220 Hz, got {:?}",
            pitch
        );
        assert!(pitch.confidence > 0.85, "confidence={}", pitch.confidence);
        assert!((pitch.normalized - pitch_norm_from_hz(220.0)).abs() < 0.02);
    }

    #[test]
    fn mono_pitch_prefers_first_strong_peak_over_octave() {
        let sample_rate = 48_000.0;
        let pitch = estimate_mono_pitch(&sine(440.0, sample_rate, 2048), sample_rate, 45.0, 1600.0);
        assert!(
            (pitch.hz - 440.0).abs() < 4.0,
            "expected ~440 Hz, got {:?}",
            pitch
        );
    }

    #[test]
    fn mono_pitch_rejects_silence() {
        let pitch = estimate_mono_pitch(&[0.0; 2048], 48_000.0, 45.0, 1600.0);
        assert_eq!(pitch, PitchEstimate::default());
    }

    #[test]
    fn chroma_folds_octaves_to_pitch_class() {
        let fft_len = 4096;
        let sample_rate = 48_000.0;
        let bin_hz = sample_rate / fft_len as f32;
        let mut mag = vec![0.0f32; fft_len / 2 + 1];
        mag[bin_for(440.0, bin_hz)] = 1.0; // A4
        mag[bin_for(880.0, bin_hz)] = 0.7; // A5

        let chroma = chroma_from_spectrum(&mag, bin_hz);
        let a = chroma[9];
        for (i, &v) in chroma.iter().enumerate() {
            if i != 9 {
                assert!(a >= v, "pitch class {i}={v} exceeded A={a}");
            }
        }
        assert!(a > 0.9);
    }

    #[test]
    fn key_estimate_separates_major_and_minor_triads() {
        let mut c_major = [0.0f32; CHROMA_BINS];
        c_major[0] = 1.0;
        c_major[4] = 0.85;
        c_major[7] = 0.8;
        let key = estimate_key(&c_major);
        assert_eq!(key.root, 0);
        assert!(!key.is_minor);
        assert!(key.confidence > 0.1, "confidence={}", key.confidence);

        let mut a_minor = [0.0f32; CHROMA_BINS];
        a_minor[9] = 1.0;
        a_minor[0] = 0.85;
        a_minor[4] = 0.8;
        let key = estimate_key(&a_minor);
        assert_eq!(key.root, 9);
        assert!(key.is_minor);
        assert!(key.confidence > 0.1, "confidence={}", key.confidence);
    }

    fn major_triad(root: usize) -> [f32; CHROMA_BINS] {
        let mut chroma = [0.0f32; CHROMA_BINS];
        chroma[root % CHROMA_BINS] = 1.0;
        chroma[(root + 4) % CHROMA_BINS] = 0.85;
        chroma[(root + 7) % CHROMA_BINS] = 0.8;
        chroma
    }

    #[test]
    fn key_smoother_rejects_one_frame_glitch() {
        let mut smoother = KeySmoother::new();
        let c_major = major_triad(0);
        let g_major = major_triad(7);

        for _ in 0..6 {
            let key = smoother.update(&c_major);
            assert_eq!(key.root, 0);
            assert!(!key.is_minor);
        }

        let glitch = smoother.update(&g_major);
        assert_eq!(glitch.root, 0, "one-frame glitch should not flip key");
        assert!(!glitch.is_minor);
    }

    #[test]
    fn key_smoother_adapts_after_sustained_modulation() {
        let mut smoother = KeySmoother::new();
        let c_major = major_triad(0);
        let g_major = major_triad(7);

        for _ in 0..6 {
            smoother.update(&c_major);
        }

        let mut key = smoother.update(&g_major);
        for _ in 0..10 {
            key = smoother.update(&g_major);
        }

        assert_eq!(key.root, 7);
        assert!(!key.is_minor);
        assert!(key.confidence > 0.05, "confidence={}", key.confidence);
    }

    #[test]
    fn chord_estimate_tracks_triad_root_and_quality() {
        let mut g_major = [0.0f32; CHROMA_BINS];
        g_major[7] = 1.0;
        g_major[11] = 0.9;
        g_major[2] = 0.85;
        let chord = estimate_chord(&g_major);
        assert_eq!(chord.root, 7);
        assert!(!chord.is_minor);
        assert!(chord.confidence > 0.2, "confidence={}", chord.confidence);

        let mut d_minor = [0.0f32; CHROMA_BINS];
        d_minor[2] = 1.0;
        d_minor[5] = 0.9;
        d_minor[9] = 0.85;
        let chord = estimate_chord(&d_minor);
        assert_eq!(chord.root, 2);
        assert!(chord.is_minor);
        assert!(chord.confidence > 0.2, "confidence={}", chord.confidence);
    }

    #[test]
    fn palette_mood_gates_toward_neutral_when_uncertain() {
        let (hue, mood) = palette_from_key(TonalEstimate {
            root: 4,
            is_minor: false,
            confidence: 0.8,
        });
        assert!((hue - 4.0 / 12.0).abs() < 1e-6);
        assert!(mood > 0.5);

        let (_, mood) = palette_from_key(TonalEstimate {
            root: 4,
            is_minor: true,
            confidence: 0.8,
        });
        assert!(mood < 0.5);

        assert_eq!(palette_from_key(TonalEstimate::default()), (0.0, 0.0));
    }
}

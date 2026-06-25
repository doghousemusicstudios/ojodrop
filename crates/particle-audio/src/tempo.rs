//! Pure-Rust tempo estimation + phase-locked beat tracking (spec §4/§5, step 6).
//!
//! NO aubio (GPL). The pipeline is:
//!   1. Accumulate a broadband onset-strength value per hop into a ring buffer
//!      (the "onset envelope").
//!   2. Periodically autocorrelate that envelope over the lag range corresponding
//!      to 60-200 BPM, pick the strongest periodic lag, and apply octave/half-time
//!      correction (a tempo and its double/half often score similarly; prefer the
//!      one nearest a comfortable 100-150 BPM center).
//!   3. Run a lightweight phase-locked loop: advance a continuous `beat_phase`
//!      (0..1) at the estimated BPM and nudge its phase toward observed onsets so
//!      it locks to the actual kicks. Emit `beat_impulse` when phase wraps,
//!      `bar_phase` over 4 beats, and a `beat_confidence` from the autocorrelation
//!      peak sharpness.
//!   4. Track a compact four-state bar HMM over beat accents so `bar_phase` locks
//!      to likely downbeats instead of assuming `(beat_count % 4)`.

/// Tempo/beat tracker. Fed one onset-strength sample per DSP hop.
pub struct TempoTracker {
    hop_dt: f32,

    // --- onset envelope ring ---
    env: Vec<f32>,
    env_write: usize,
    env_filled: usize,

    // --- tempo estimate ---
    bpm: f32,
    confidence: f32,
    hops_since_estimate: u32,
    estimate_period_hops: u32,

    // --- beat phase loop ---
    beat_phase: f32, // 0..1, wraps each beat
    beat_count: u32, // beats elapsed (for bar phase)
    prev_onset: f32, // for local-peak detection in phase correction
    prev_prev_onset: f32,
    beat_impulse: f32, // decays after a wrap
    impulse_decay: f32,

    // --- downbeat / bar-state tracker ---
    beat_peak: f32,
    beat_level: f32,
    bar_probs: [f32; 4],
    bar_pos: u32,
    bar_confidence: f32,
}

/// Autocorrelation lag search bounds, derived from the BPM range.
const MIN_BPM: f32 = 60.0;
const MAX_BPM: f32 = 200.0;
/// Preferred tempo center for octave disambiguation.
const PREFERRED_BPM: f32 = 125.0;

impl TempoTracker {
    pub fn new(hop_dt: f32) -> Self {
        // Envelope long enough to hold a few seconds (autocorr needs > one period
        // at the slowest tempo: 60 BPM = 1 s/beat; a few beats gives a clear peak).
        let env_len = ((4.0 / hop_dt) as usize).max(256);
        // Re-estimate tempo a few times per second (cheap, but no need every hop).
        let estimate_period_hops = ((0.25 / hop_dt) as u32).max(1);
        Self {
            hop_dt,
            env: vec![0.0; env_len],
            env_write: 0,
            env_filled: 0,
            bpm: 0.0,
            confidence: 0.0,
            hops_since_estimate: 0,
            estimate_period_hops,
            beat_phase: 0.0,
            beat_count: 0,
            prev_onset: 0.0,
            prev_prev_onset: 0.0,
            beat_impulse: 0.0,
            impulse_decay: (-hop_dt / 0.150).exp(), // ~150 ms decay (spec)
            beat_peak: 0.0,
            beat_level: 1e-4,
            bar_probs: [0.25; 4],
            bar_pos: 0,
            bar_confidence: 0.0,
        }
    }

    /// One DSP hop. `onset_strength` is a broadband positive value (e.g. total
    /// spectral flux) ≥ 0. Returns the current beat-tracking outputs.
    pub fn process(&mut self, onset_strength: f32) -> TempoOut {
        let onset_strength = onset_strength.max(0.0);
        self.beat_peak = self.beat_peak.max(onset_strength);

        // --- push onset into the envelope ring ---
        self.env[self.env_write] = onset_strength;
        self.env_write = (self.env_write + 1) % self.env.len();
        self.env_filled = (self.env_filled + 1).min(self.env.len());

        // --- periodically re-estimate tempo via autocorrelation ---
        self.hops_since_estimate += 1;
        if self.hops_since_estimate >= self.estimate_period_hops
            && self.env_filled >= self.env.len() / 2
        {
            self.hops_since_estimate = 0;
            self.estimate_tempo();
        }

        // --- advance the phase loop ---
        if self.bpm > 0.0 {
            let beats_per_hop = self.bpm / 60.0 * self.hop_dt;
            self.beat_phase += beats_per_hop;

            // Phase correction: if a strong local onset peak occurs, pull phase
            // toward the nearest beat (0 or 1). This is the "lock" step.
            let is_peak = self.prev_onset > self.prev_prev_onset
                && self.prev_onset >= onset_strength
                && self.prev_onset > 1e-4;
            if is_peak && self.confidence > 0.1 {
                // Distance from the nearest integer beat boundary.
                let frac = self.beat_phase.fract();
                let err = if frac > 0.5 { frac - 1.0 } else { frac };
                // Gentle correction proportional to confidence; avoids jitter.
                let gain = 0.05 + 0.15 * self.confidence;
                self.beat_phase -= err * gain;
            }

            // Wrap → a beat happened.
            if self.beat_phase >= 1.0 {
                self.beat_phase -= self.beat_phase.floor();
                self.update_bar_state();
                self.beat_count = self.beat_count.wrapping_add(1);
                self.beat_impulse = 1.0;
            }
        }

        // Decay the impulse.
        self.beat_impulse *= self.impulse_decay;

        self.prev_prev_onset = self.prev_onset;
        self.prev_onset = onset_strength;

        let fallback_pos = self.beat_count % 4;
        let metrical_pos = if self.bar_confidence > 0.18 {
            self.bar_pos
        } else {
            fallback_pos
        };
        let bar_phase = (metrical_pos as f32 + self.beat_phase.clamp(0.0, 1.0)) / 4.0;

        TempoOut {
            bpm: self.bpm,
            beat_phase: self.beat_phase.clamp(0.0, 1.0),
            bar_phase: bar_phase.clamp(0.0, 1.0),
            beat_impulse: self.beat_impulse.clamp(0.0, 1.0),
            confidence: self.confidence.clamp(0.0, 1.0),
        }
    }

    /// Advance the four-state downbeat HMM using the beat that just completed.
    ///
    /// States are metrical positions 0..3 for the completed beat, where state 0
    /// is the downbeat. Transitions rotate deterministically by one beat with a
    /// small floor to recover from bad locks. Emissions are accent based: a beat
    /// whose peak onset rises above the adaptive beat-level is more likely to be
    /// state 0, while ordinary beats prefer states 1..3. This is deliberately
    /// model-free and deterministic; it gives us bar-1 alignment without adding
    /// new `AudioInput` fields or ML dependencies.
    fn update_bar_state(&mut self) {
        let peak = self.beat_peak.max(0.0);
        self.beat_peak = 0.0;

        if peak <= 1e-6 {
            self.bar_confidence *= 0.96;
            self.bar_pos = (self.bar_pos + 1) % 4;
            return;
        }

        if self.beat_level <= 1e-4 {
            self.beat_level = peak;
            self.bar_pos = (self.bar_pos + 1) % 4;
            return;
        }

        let level = self.beat_level.max(1e-5);
        let accent = (peak / level).clamp(0.0, 4.0);
        self.beat_level += 0.08 * (peak - self.beat_level);

        let accent_excess = ((accent - 1.0) / 2.0).clamp(0.0, 1.0);
        let weak_accent = (1.0 - accent_excess).clamp(0.0, 1.0);
        let emissions = [
            0.30 + 1.70 * accent_excess,
            0.72 + 0.34 * weak_accent,
            0.64 + 0.28 * weak_accent,
            0.78 + 0.40 * weak_accent,
        ];

        let mut next = [0.0f32; 4];
        for pos in 0..4 {
            let predicted = self.bar_probs[(pos + 3) % 4] * 0.94 + 0.015;
            next[pos] = predicted * emissions[pos];
        }
        let sum = next.iter().sum::<f32>().max(1e-9);
        for p in &mut next {
            *p /= sum;
        }

        let mut best_pos = 0usize;
        let mut best = next[0];
        let mut second = 0.0f32;
        for (pos, &prob) in next.iter().enumerate().skip(1) {
            if prob > best {
                second = best;
                best = prob;
                best_pos = pos;
            } else if prob > second {
                second = prob;
            }
        }

        self.bar_probs = next;
        self.bar_confidence += 0.25 * ((best - second).clamp(0.0, 1.0) - self.bar_confidence);
        self.bar_pos = ((best_pos as u32) + 1) % 4;
    }

    /// Autocorrelate the onset envelope and pick the dominant tempo lag.
    fn estimate_tempo(&mut self) {
        let n = self.env_filled;
        if n < 16 {
            return;
        }
        // Linearize the ring into oldest→newest order.
        let mut sig = Vec::with_capacity(n);
        let start = (self.env_write + self.env.len() - n) % self.env.len();
        for i in 0..n {
            sig.push(self.env[(start + i) % self.env.len()]);
        }
        // Remove DC (mean) so sustained energy doesn't dominate the correlation.
        let mean = sig.iter().sum::<f32>() / n as f32;
        for s in &mut sig {
            *s -= mean;
        }

        let min_lag = (60.0 / MAX_BPM / self.hop_dt).round() as usize;
        let max_lag = (60.0 / MIN_BPM / self.hop_dt).round() as usize;
        let min_lag = min_lag.max(1);
        let max_lag = max_lag.min(n / 2).max(min_lag + 1);

        // Energy at lag 0 for normalization.
        let energy: f32 = sig.iter().map(|s| s * s).sum::<f32>().max(1e-9);

        let mut best_lag = 0usize;
        let mut best_score = 0.0f32;
        let mut scores = vec![0.0f32; max_lag + 1];
        for lag in min_lag..=max_lag {
            let mut acc = 0.0f32;
            for i in lag..n {
                acc += sig[i] * sig[i - lag];
            }
            let score = acc / energy;
            scores[lag] = score;
            if score > best_score {
                best_score = score;
                best_lag = lag;
            }
        }

        if best_lag == 0 || best_score <= 0.0 {
            // No clear periodicity → bleed off confidence.
            self.confidence *= 0.9;
            return;
        }

        let mut bpm = 60.0 / (best_lag as f32 * self.hop_dt);
        // Octave / half-time correction: compare the chosen lag against its
        // double and half; prefer whichever variant lands closer to PREFERRED_BPM
        // while retaining comparable correlation strength.
        bpm = correct_octave(bpm, best_lag, &scores, max_lag);

        // Confidence: peak score, lightly sharpened by how much it beats the mean
        // of the score curve.
        let valid: Vec<f32> = scores[min_lag..=max_lag].to_vec();
        let smean = valid.iter().sum::<f32>() / valid.len().max(1) as f32;
        let sharpness = (best_score - smean).max(0.0);
        let raw_conf = (best_score * 0.5 + sharpness).clamp(0.0, 1.0);

        // Heavy smoothing on bpm (spec) and moderate on confidence.
        if self.bpm <= 0.0 {
            self.bpm = bpm;
        } else {
            self.bpm += 0.1 * (bpm - self.bpm);
        }
        self.confidence += 0.2 * (raw_conf - self.confidence);
    }
}

/// Choose between a tempo and its octave variants (×2, ÷2) by combining
/// correlation strength with proximity to a comfortable preferred tempo.
fn correct_octave(bpm: f32, lag: usize, scores: &[f32], max_lag: usize) -> f32 {
    let candidates = [
        (bpm, lag, scores.get(lag).copied().unwrap_or(0.0)),
        (
            bpm * 2.0,
            lag / 2,
            scores.get(lag / 2).copied().unwrap_or(0.0),
        ),
        (
            bpm / 2.0,
            (lag * 2).min(max_lag),
            scores.get((lag * 2).min(max_lag)).copied().unwrap_or(0.0),
        ),
    ];
    let mut best = bpm;
    let mut best_metric = f32::MIN;
    for &(cand_bpm, _, cand_score) in &candidates {
        if !(MIN_BPM..=MAX_BPM).contains(&cand_bpm) {
            continue;
        }
        // Reward correlation strength, penalize distance from the preferred center
        // (log distance so 2x feels symmetric to 0.5x).
        let dist = (cand_bpm / PREFERRED_BPM).ln().abs();
        let metric = cand_score - 0.25 * dist;
        if metric > best_metric {
            best_metric = metric;
            best = cand_bpm;
        }
    }
    best
}

/// Outputs of one tempo-tracking hop.
#[derive(Clone, Copy, Debug, Default)]
pub struct TempoOut {
    pub bpm: f32,
    pub beat_phase: f32,
    pub bar_phase: f32,
    pub beat_impulse: f32,
    pub confidence: f32,
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f32::consts::PI;

    /// Drive the tracker with a synthetic onset train at a known BPM and confirm
    /// the estimate locks near it. This is a self-contained DSP unit test — it
    /// uses no audio device and no dummy *feature* data, only a deterministic
    /// impulse train to validate the autocorrelation math (allowed: it tests the
    /// algorithm, it is not a stand-in for real capture).
    #[test]
    fn locks_onto_synthetic_120_bpm() {
        let hop_dt = 256.0 / 48_000.0; // ~5.33 ms
        let mut t = TempoTracker::new(hop_dt);
        let bpm = 120.0;
        let beat_period_s = 60.0 / bpm;
        let total_s = 8.0;
        let hops = (total_s / hop_dt) as usize;
        let mut last = TempoOut::default();
        for h in 0..hops {
            let time = h as f32 * hop_dt;
            // Onset strength: a sharp pulse near each beat boundary.
            let phase = (time % beat_period_s) / beat_period_s;
            let pulse = (-((phase) * 30.0).powi(2)).exp() + (-((phase - 1.0) * 30.0).powi(2)).exp();
            last = t.process(pulse.max(0.0));
        }
        // Should land within ~8% of 120 (octave-corrected).
        assert!(
            (last.bpm - bpm).abs() < 10.0,
            "estimated bpm {} not near {}",
            last.bpm,
            bpm
        );
    }

    #[test]
    fn beat_phase_advances_monotonically_between_wraps() {
        let hop_dt = 256.0 / 48_000.0;
        let mut t = TempoTracker::new(hop_dt);
        // Warm up with a steady tempo so bpm becomes nonzero.
        let bpm = 128.0;
        let beat_period_s = 60.0 / bpm;
        for h in 0..((6.0 / hop_dt) as usize) {
            let time = h as f32 * hop_dt;
            let phase = (time % beat_period_s) / beat_period_s;
            let pulse = (-((phase) * 25.0).powi(2)).exp();
            t.process(pulse);
        }
        assert!(t.bpm > 0.0);
        // Now collect phase over a short window; it must increase then wrap, never
        // jump backward by a large amount except at a wrap.
        let mut prev = t.process(0.0).beat_phase;
        let mut saw_wrap = false;
        for _ in 0..400 {
            let p = t.process(0.0).beat_phase;
            if p < prev - 0.5 {
                saw_wrap = true; // legitimate wrap
            } else {
                assert!(p + 1e-3 >= prev, "phase went backward: {prev} -> {p}");
            }
            prev = p;
        }
        assert!(saw_wrap, "phase never wrapped — it should advance and loop");
    }

    #[test]
    fn silence_keeps_confidence_low() {
        let hop_dt = 512.0 / 48_000.0;
        let mut t = TempoTracker::new(hop_dt);
        let mut out = TempoOut::default();
        for _ in 0..2000 {
            out = t.process(0.0);
        }
        assert!(out.confidence < 0.2, "silence should not be confident");
        // sanity: a sine-shaped non-impulsive feed shouldn't NaN anything.
        for i in 0..100 {
            out = t.process((i as f32 * 0.1 * PI).sin().abs());
        }
        assert!(out.bpm.is_finite());
    }

    #[test]
    fn bar_phase_locks_to_accented_downbeat_offset() {
        let hop_dt = 256.0 / 48_000.0;
        let mut t = TempoTracker::new(hop_dt);
        t.bpm = 120.0;
        t.confidence = 0.8;

        let bpm = 120.0;
        let beat_period_s = 60.0 / bpm;
        let total_s = 18.0;
        let true_offset = 2u32;
        let mut last = TempoOut::default();
        for h in 0..((total_s / hop_dt) as usize) {
            let time = h as f32 * hop_dt;
            let beat_index = (time / beat_period_s).floor() as u32;
            let true_pos = (beat_index + true_offset) % 4;
            let phase = (time % beat_period_s) / beat_period_s;
            let pulse = (-((phase) * 38.0).powi(2)).exp() + (-((phase - 1.0) * 38.0).powi(2)).exp();
            let accent = if true_pos == 0 { 3.0 } else { 0.8 };
            last = t.process((pulse * accent).max(0.0));
        }

        let expected_pos = (t.beat_count + true_offset) % 4;
        assert!(
            t.bar_confidence > 0.2,
            "downbeat HMM should gain confidence, got {}",
            t.bar_confidence
        );
        assert_eq!(
            t.bar_pos, expected_pos,
            "bar position should follow the accented one"
        );
        let expected_bar_phase = (expected_pos as f32 + last.beat_phase) / 4.0;
        assert!(
            (last.bar_phase - expected_bar_phase).abs() < 0.02,
            "bar_phase {} not aligned to expected {}",
            last.bar_phase,
            expected_bar_phase
        );
    }
}

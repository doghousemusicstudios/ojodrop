//! Live structure / novelty tracking.
//!
//! This is a compact clean-room Foote-style checkerboard novelty detector over a
//! rolling audio embedding. It is deliberately small enough for the realtime DSP
//! worker: fixed-size arrays, no per-hop allocation, and a bounded history.

use crate::CHROMA_BINS;

const EMBED_DIM: usize = 24;
const SPEC_GROUPS: usize = 8;
const HISTORY: usize = 48;
const WINDOW: usize = 8;

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct StructureFeatures {
    /// Checkerboard self-similarity novelty, 0..1.
    pub novelty: f32,
    /// Decaying section-change impulse derived from novelty above its baseline.
    pub change: f32,
    /// Confidence that enough non-silent history exists for the estimate.
    pub confidence: f32,
}

#[derive(Clone, Debug)]
pub struct StructureTracker {
    history: [[f32; EMBED_DIM]; HISTORY],
    len: usize,
    write: usize,
    novelty_lp: f32,
    baseline: f32,
    change_env: f32,
}

impl Default for StructureTracker {
    fn default() -> Self {
        Self {
            history: [[0.0; EMBED_DIM]; HISTORY],
            len: 0,
            write: 0,
            novelty_lp: 0.0,
            baseline: 0.0,
            change_env: 0.0,
        }
    }
}

impl StructureTracker {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn process(
        &mut self,
        spectrum: &[f32; 32],
        chroma: &[f32; CHROMA_BINS],
        rms: f32,
        brightness: f32,
        flux: f32,
        harmonic_ratio: f32,
        is_silent: bool,
    ) -> StructureFeatures {
        let gate = if is_silent { 0.0 } else { 1.0 };
        let embedding = make_embedding(spectrum, chroma, rms, brightness, flux, harmonic_ratio);
        let raw = finite_clamp(self.checkerboard_novelty(&embedding), 0.0, 1.0, 0.0) * gate;
        self.push(embedding);

        let novelty = finite_clamp(raw * 2.4, 0.0, 1.0, 0.0);
        self.novelty_lp = finite_or_zero(self.novelty_lp) * 0.72 + novelty * 0.28;
        self.baseline = finite_or_zero(self.baseline) * 0.985 + self.novelty_lp * 0.015;

        let hit = finite_clamp(
            (self.novelty_lp - self.baseline - 0.055) * 5.0,
            0.0,
            1.0,
            0.0,
        ) * gate;
        self.change_env = (finite_or_zero(self.change_env) * 0.82).max(hit);
        if is_silent {
            self.change_env *= 0.8;
        }

        let history_conf = (self.len as f32 / (WINDOW * 2) as f32).clamp(0.0, 1.0);
        let energy_conf = finite_clamp(rms * 2.0, 0.0, 1.0, 0.0);
        StructureFeatures {
            novelty: finite_clamp(self.novelty_lp * gate, 0.0, 1.0, 0.0),
            change: finite_clamp(self.change_env, 0.0, 1.0, 0.0),
            confidence: history_conf * energy_conf * gate,
        }
    }

    fn push(&mut self, embedding: [f32; EMBED_DIM]) {
        self.history[self.write] = embedding;
        self.write = (self.write + 1) % HISTORY;
        self.len = (self.len + 1).min(HISTORY);
    }

    fn checkerboard_novelty(&self, current: &[f32; EMBED_DIM]) -> f32 {
        if self.len + 1 < WINDOW * 2 {
            return 0.0;
        }

        let mut frames = [[0.0; EMBED_DIM]; WINDOW * 2];
        let needed_prev = WINDOW * 2 - 1;
        let start = (self.write + HISTORY - needed_prev) % HISTORY;
        for (i, frame) in frames.iter_mut().take(needed_prev).enumerate() {
            *frame = self.history[(start + i) % HISTORY];
        }
        frames[WINDOW * 2 - 1] = *current;

        let mut left = 0.0;
        let mut right = 0.0;
        let mut cross = 0.0;
        let mut same_count = 0.0f32;
        let mut cross_count = 0.0f32;
        for i in 0..WINDOW {
            for j in 0..WINDOW {
                left += dot(&frames[i], &frames[j]);
                right += dot(&frames[WINDOW + i], &frames[WINDOW + j]);
                cross += dot(&frames[i], &frames[WINDOW + j]);
                same_count += 2.0;
                cross_count += 1.0;
            }
        }

        let same = (left + right) / same_count.max(1.0);
        let cross = cross / cross_count.max(1.0);
        finite_clamp(same - cross, 0.0, 1.0, 0.0)
    }
}

fn make_embedding(
    spectrum: &[f32; 32],
    chroma: &[f32; CHROMA_BINS],
    rms: f32,
    brightness: f32,
    flux: f32,
    harmonic_ratio: f32,
) -> [f32; EMBED_DIM] {
    let mut out = [0.0f32; EMBED_DIM];
    for group in 0..SPEC_GROUPS {
        let mut sum = 0.0;
        for i in 0..4 {
            sum += finite_clamp(spectrum[group * 4 + i], 0.0, 2.0, 0.0);
        }
        out[group] = sum * 0.25;
    }
    for i in 0..CHROMA_BINS {
        out[SPEC_GROUPS + i] = finite_clamp(chroma[i], 0.0, 1.0, 0.0);
    }
    out[20] = finite_clamp(rms, 0.0, 1.0, 0.0);
    out[21] = finite_clamp(brightness, 0.0, 1.0, 0.0);
    out[22] = finite_clamp(flux, 0.0, 1.0, 0.0);
    out[23] = finite_clamp(harmonic_ratio, 0.0, 1.0, 0.0);

    let energy = out.iter().map(|v| v * v).sum::<f32>().sqrt();
    if energy.is_finite() && energy > 1e-5 {
        for v in &mut out {
            *v /= energy;
        }
    }
    out
}

fn dot(a: &[f32; EMBED_DIM], b: &[f32; EMBED_DIM]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| finite_or_zero(*x) * finite_or_zero(*y))
        .sum()
}

fn finite_or_zero(value: f32) -> f32 {
    if value.is_finite() {
        value
    } else {
        0.0
    }
}

fn finite_clamp(value: f32, min: f32, max: f32, fallback: f32) -> f32 {
    if value.is_finite() {
        value.clamp(min, max)
    } else {
        fallback.clamp(min, max)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn frame(low: bool) -> ([f32; 32], [f32; CHROMA_BINS]) {
        let mut spectrum = [0.02; 32];
        let mut chroma = [0.0; CHROMA_BINS];
        if low {
            for v in &mut spectrum[..8] {
                *v = 1.0;
            }
            chroma[0] = 1.0;
            chroma[7] = 0.6;
        } else {
            for v in &mut spectrum[22..] {
                *v = 1.0;
            }
            chroma[4] = 1.0;
            chroma[11] = 0.6;
        }
        (spectrum, chroma)
    }

    #[test]
    fn novelty_spikes_when_embedding_changes() {
        let mut tracker = StructureTracker::new();
        let mut peak_before = 0.0f32;
        let mut peak_after = 0.0f32;
        for i in 0..40 {
            let (spectrum, chroma) = frame(i < 24);
            let out = tracker.process(&spectrum, &chroma, 0.75, 0.4, 0.2, 0.8, false);
            if i < 22 {
                peak_before = peak_before.max(out.change);
            } else {
                peak_after = peak_after.max(out.change);
            }
        }
        assert!(
            peak_after > peak_before + 0.20,
            "expected structure-change impulse after section switch; before={peak_before} after={peak_after}"
        );
    }

    #[test]
    fn silence_gates_confidence_and_novelty() {
        let mut tracker = StructureTracker::new();
        let (spectrum, chroma) = frame(true);
        for _ in 0..20 {
            tracker.process(&spectrum, &chroma, 0.8, 0.2, 0.1, 0.9, false);
        }
        let out = tracker.process(&spectrum, &chroma, 0.8, 0.2, 0.1, 0.9, true);
        assert_eq!(out.novelty, 0.0);
        assert_eq!(out.confidence, 0.0);
    }

    #[test]
    fn non_finite_inputs_do_not_poison_structure_state() {
        let mut tracker = StructureTracker::new();
        let mut spectrum = [0.1; 32];
        let mut chroma = [0.1; CHROMA_BINS];
        spectrum[3] = f32::NAN;
        spectrum[7] = f32::INFINITY;
        chroma[2] = f32::NAN;
        chroma[6] = f32::NEG_INFINITY;

        for _ in 0..24 {
            let out = tracker.process(
                &spectrum,
                &chroma,
                f32::NAN,
                f32::INFINITY,
                f32::NAN,
                f32::NEG_INFINITY,
                false,
            );
            assert!(out.novelty.is_finite());
            assert!(out.change.is_finite());
            assert!(out.confidence.is_finite());
        }

        let (clean_spectrum, clean_chroma) = frame(false);
        let out = tracker.process(&clean_spectrum, &clean_chroma, 0.8, 0.5, 0.2, 0.9, false);
        assert!(out.novelty.is_finite());
        assert!(out.change.is_finite());
        assert!(out.confidence.is_finite());
    }

    #[test]
    fn poisoned_history_is_sanitized_by_similarity_dot_product() {
        let mut tracker = StructureTracker::new();
        tracker.len = HISTORY;
        tracker.write = 0;
        tracker.history[0][0] = f32::NAN;
        tracker.history[1][1] = f32::INFINITY;
        tracker.novelty_lp = f32::NAN;
        tracker.baseline = f32::NAN;
        tracker.change_env = f32::NAN;

        let (spectrum, chroma) = frame(true);
        let out = tracker.process(&spectrum, &chroma, 0.8, 0.5, 0.2, 0.9, false);
        assert!(out.novelty.is_finite());
        assert!(out.change.is_finite());
        assert!(out.confidence.is_finite());
        assert!(tracker.novelty_lp.is_finite());
        assert!(tracker.baseline.is_finite());
        assert!(tracker.change_env.is_finite());
    }
}

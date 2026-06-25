//! Predictive "drop" anticipation rail (original implementation).
//!
//! In electronic/dance music a *drop* is preceded by a recognizable build-up:
//! energy ramps upward, the spectral centroid sweeps up (the classic rising
//! low-pass "filter sweep" / white-noise riser), percussive activity intensifies
//! (snare rolls, accelerating hats), and the low end is often momentarily
//! *removed* right before the drop hits. This module fuses those cues over a
//! short rolling history window into a single `0..1` `drop_anticipation` rail
//! that rises as a drop approaches and collapses once the drop lands.
//!
//! It is a heuristic anticipator, not a classifier: there is no training data and
//! no GPL code. The design:
//!
//! - **Energy ramp** — slope of the broadband loudness over the window. A steady
//!   positive slope (getting louder) is the strongest build cue.
//! - **Flux sustain** — sustained high spectral flux means lots of activity
//!   (rolls / risers) rather than a static section.
//! - **Tension sweep** — a rising spectral centroid combined with rising high-band
//!   energy is the filter-sweep / riser signature.
//! - **Bass scoop** — when the build is hot but the sub energy suddenly drops out,
//!   that "calm before the storm" briefly boosts anticipation (the pre-drop gap).
//!
//! The fused score is run through an asymmetric envelope: it ramps up smoothly
//! while cues persist and decays quickly once they stop, and it is hard-reset
//! toward 0 when a large transient lands (the drop itself), so the rail reads as
//! "about to drop" rather than "dropped".

use crate::smoothing::AsymEnv;

/// Output-envelope time constants (ms): a smooth ramp up, a quick relax.
const ENV_ATTACK_MS: f32 = 450.0;
const ENV_DECAY_MS: f32 = 220.0;

/// Rolling-history drop anticipator. Fed per-hop scalar features; returns the
/// smoothed `0..1` anticipation rail.
pub struct DropPredictor {
    // Rolling histories (ring buffers).
    energy: Vec<f32>,
    centroid: Vec<f32>,
    high: Vec<f32>,
    write: usize,
    filled: usize,
    cap: usize,
    // Previous broadband energy for transient (drop-hit) detection.
    prev_energy: f32,
    // Output envelope: smooth ramp up, quick relax.
    env: AsymEnv,
    hop_dt: f32,
    last: f32,
}

impl DropPredictor {
    pub fn new(hop_dt: f32) -> Self {
        // ~1.5 s history window: long enough to see a build-up's slope without
        // lagging into the next section.
        let cap = ((1.5 / hop_dt).round() as usize).max(16);
        Self {
            energy: vec![0.0; cap],
            centroid: vec![0.0; cap],
            high: vec![0.0; cap],
            write: 0,
            filled: 0,
            cap,
            prev_energy: 0.0,
            env: AsymEnv::new(ENV_ATTACK_MS, ENV_DECAY_MS, hop_dt),
            hop_dt,
            last: 0.0,
        }
    }

    /// Process one hop. Inputs (all normalized 0..1):
    /// - `energy`: broadband loudness (e.g. RMS level / short-term loudness).
    /// - `flux`: spectral flux / novelty (activity).
    /// - `centroid`: spectral brightness (0..1, rises on a filter sweep).
    /// - `high`: high-band energy (presence/air), rises on risers.
    /// - `sub`: sub-bass energy (used to spot the pre-drop bass scoop).
    /// - `is_silent`: gate; relaxes the rail and avoids false builds in silence.
    ///
    /// Returns the `0..1` `drop_anticipation` rail.
    pub fn process(
        &mut self,
        energy: f32,
        flux: f32,
        centroid: f32,
        high: f32,
        sub: f32,
        is_silent: bool,
    ) -> f32 {
        if is_silent {
            // Drain the window toward 0 and relax the rail.
            self.push(0.0, 0.0, 0.0);
            self.prev_energy = 0.0;
            self.last = self.env.process(0.0);
            return self.last;
        }

        // --- drop-hit detection: a big sudden energy jump means the drop has
        //     landed; reset anticipation so the rail reads "about to" not "did". ---
        let energy_jump = (energy - self.prev_energy).max(0.0);
        self.prev_energy = energy;

        // --- window slopes (recent-half mean minus older-half mean) ---
        let energy_slope = self.window_slope(&self.energy);
        let centroid_slope = self.window_slope(&self.centroid);
        let high_slope = self.window_slope(&self.high);

        // --- fuse cues into a raw build score (each 0..~1) ---
        // The score is dominated by *change* cues (slopes), not absolute level,
        // so a steady loud section (the drop itself, where slopes flatten) reads
        // low while a build (everything ramping) reads high.
        //
        // Energy ramp: a sustained positive loudness slope is the dominant cue.
        let ramp = (energy_slope * 7.0).clamp(0.0, 1.0);
        // Tension sweep: rising centroid AND rising high energy = filter sweep / riser.
        let sweep =
            ((centroid_slope * 6.0).clamp(0.0, 1.0) + (high_slope * 6.0).clamp(0.0, 1.0)) * 0.5;
        // Flux sustain: ongoing activity (rolls/risers) supports a build but only
        // matters while something is actually ramping, so weight it lightly.
        let activity = flux.clamp(0.0, 1.0);
        // Bass scoop: build is hot but the sub just dropped out -> pre-drop gap.
        let scoop = if ramp > 0.3 {
            ((0.4 - sub).max(0.0) * 1.5).clamp(0.0, 0.4)
        } else {
            0.0
        };

        // Gate the whole build score by absolute energy so a rising-from-silence
        // section doesn't read as a "build" until there is real signal present.
        let energy_gate = energy.clamp(0.0, 1.0);
        let raw =
            ((ramp * 0.55 + sweep * 0.3 + activity * 0.1 + scoop) * energy_gate).clamp(0.0, 1.0);

        // --- envelope, with a hard reset when the drop lands ---
        let mut out = self.env.process(raw);
        // The transient of the drop: a sudden energy jump (or the build plateauing
        // at a loud, no-longer-rising level) after a hot build. Collapse the
        // anticipation so the rail reads "about to drop", not "dropped".
        let plateaued = self.last > 0.45 && ramp < 0.1 && energy > 0.7;
        if (energy_jump > 0.1 && self.last > 0.45) || plateaued {
            self.env = AsymEnv::new(ENV_ATTACK_MS, ENV_DECAY_MS, self.hop_dt);
            out = self.env.process(0.0);
        }

        // Push current frame into the history for next hop's slopes.
        self.push(energy, centroid, high);
        self.last = out.clamp(0.0, 1.0);
        self.last
    }

    fn push(&mut self, energy: f32, centroid: f32, high: f32) {
        self.energy[self.write] = energy;
        self.centroid[self.write] = centroid;
        self.high[self.write] = high;
        self.write = (self.write + 1) % self.cap;
        self.filled = (self.filled + 1).min(self.cap);
    }

    /// Coarse slope estimate over the window: mean of the recent half minus mean
    /// of the older half, normalized by the window span so units stay comparable.
    fn window_slope(&self, ring: &[f32]) -> f32 {
        if self.filled < 4 {
            return 0.0;
        }
        let n = self.filled;
        let half = n / 2;
        let mut recent = 0.0f32;
        let mut older = 0.0f32;
        for offset in 0..half {
            let idx = (self.write + self.cap - 1 - offset) % self.cap;
            recent += ring[idx];
        }
        for offset in half..n {
            let idx = (self.write + self.cap - 1 - offset) % self.cap;
            older += ring[idx];
        }
        let recent = recent / half.max(1) as f32;
        let older = older / (n - half).max(1) as f32;
        recent - older
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn hop_dt() -> f32 {
        512.0 / 48_000.0
    }

    #[test]
    fn rises_on_energy_ramp() {
        let mut p = DropPredictor::new(hop_dt());
        // Flat quiet section first.
        let mut low_phase = 0.0f32;
        for _ in 0..40 {
            low_phase = p.process(0.2, 0.1, 0.2, 0.1, 0.3, false);
        }
        // Then a sustained build: energy, centroid, and high all ramp up.
        let mut peak = 0.0f32;
        for i in 0..80 {
            let f = i as f32 / 80.0;
            let v = p.process(0.2 + 0.7 * f, 0.5, 0.2 + 0.7 * f, 0.2 + 0.6 * f, 0.4, false);
            peak = peak.max(v);
        }
        assert!(
            peak > low_phase + 0.2,
            "anticipation should rise during a build: flat={low_phase}, build_peak={peak}"
        );
        assert!(peak > 0.3, "build peak too weak: {peak}");
    }

    #[test]
    fn stays_low_on_steady_section() {
        let mut p = DropPredictor::new(hop_dt());
        let mut last = 1.0f32;
        // Steady mid-energy section, no ramp.
        for _ in 0..150 {
            last = p.process(0.5, 0.15, 0.5, 0.4, 0.5, false);
        }
        assert!(
            last < 0.4,
            "a steady section should not read as a build: {last}"
        );
    }

    #[test]
    fn collapses_after_the_drop() {
        let mut p = DropPredictor::new(hop_dt());
        // Build up.
        let mut build_peak = 0.0f32;
        for i in 0..90 {
            let f = i as f32 / 90.0;
            let v = p.process(0.2 + 0.7 * f, 0.6, 0.2 + 0.7 * f, 0.2 + 0.7 * f, 0.4, false);
            build_peak = build_peak.max(v);
        }
        // The drop lands: a big energy jump.
        for _ in 0..30 {
            p.process(1.0, 0.9, 0.9, 0.9, 1.0, false);
        }
        let after = p.process(1.0, 0.9, 0.9, 0.9, 1.0, false);
        assert!(build_peak > 0.3, "should have built first: {build_peak}");
        assert!(
            after < build_peak,
            "anticipation should collapse once the drop lands: peak={build_peak}, after={after}"
        );
    }

    #[test]
    fn silence_relaxes_rail_to_zero() {
        let mut p = DropPredictor::new(hop_dt());
        for i in 0..60 {
            let f = i as f32 / 60.0;
            p.process(0.2 + 0.7 * f, 0.6, 0.2 + 0.7 * f, 0.2 + 0.7 * f, 0.4, false);
        }
        // The output envelope relaxes (decays) under silence rather than snapping;
        // after a couple of seconds of silence it should be essentially zero.
        let mut last = 1.0f32;
        for _ in 0..200 {
            last = p.process(0.0, 0.0, 0.0, 0.0, 0.0, true);
        }
        assert!(last < 1e-3, "silence should relax the rail to ~0: {last}");
    }
}

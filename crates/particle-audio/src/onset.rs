//! Per-band spectral-flux onset detection (kick / snare / hat), spec §4/§5.
//!
//! Each detector consumes a band-limited positive-flux value per hop and decides
//! whether an onset fired. The decision uses an *adaptive median threshold* over
//! a short sliding history (robust to sustained loud passages), scaled by a
//! user `sensitivity`, plus a refractory window so a single hit can't retrigger
//! for ~50-80 ms. The fired impulse is then shaped by an asymmetric attack/decay
//! envelope so downstream visuals get a clean 0..1 pop that decays smoothly.

use crate::smoothing::AsymEnv;

/// A single onset detector for one frequency band.
#[derive(Clone, Debug)]
pub struct OnsetDetector {
    /// Recent raw flux values for adaptive-median thresholding.
    history: Vec<f32>,
    /// Preallocated scratch reused by [`OnsetDetector::median`] so the per-hop
    /// median never allocates (P2-AUD-013). Length tracks the copied window.
    median_scratch: Vec<f32>,
    capacity: usize,
    write: usize,
    filled: usize,
    /// Median multiplier (base). Effective threshold = median * mult / sensitivity.
    mult: f32,
    /// Small absolute floor so silence doesn't trigger on float noise.
    floor: f32,
    /// Refractory countdown in frames; while > 0 no new onset may fire.
    refractory_frames: u32,
    refractory_left: u32,
    /// Output envelope (fast attack / slow decay).
    env: AsymEnv,
}

impl OnsetDetector {
    /// `hist_len`: median window length (frames). `mult`: threshold = median*mult.
    /// `refractory_ms` / `attack_ms` / `decay_ms` shape retrigger + envelope.
    /// `hop_dt`: seconds per hop.
    pub fn new(
        hist_len: usize,
        mult: f32,
        floor: f32,
        refractory_ms: f32,
        attack_ms: f32,
        decay_ms: f32,
        hop_dt: f32,
    ) -> Self {
        let cap = hist_len.max(1);
        Self {
            history: vec![0.0; cap],
            median_scratch: Vec::with_capacity(cap),
            capacity: cap,
            write: 0,
            filled: 0,
            mult,
            floor,
            refractory_frames: ((refractory_ms * 1e-3) / hop_dt).round().max(1.0) as u32,
            refractory_left: 0,
            env: AsymEnv::new(attack_ms, decay_ms, hop_dt),
        }
    }

    /// Process one hop's band flux. `sensitivity` in ~`0.1..3` raises detection
    /// likelihood as it grows (it divides the threshold). Returns the shaped
    /// onset envelope value in `0..1` and whether a *new* onset fired this hop.
    pub fn process(&mut self, flux: f32, sensitivity: f32) -> (f32, bool) {
        let sens = sensitivity.clamp(0.05, 5.0);
        let threshold = (self.median() * self.mult / sens).max(self.floor);

        // Push current flux into history AFTER computing the threshold so the
        // threshold reflects the recent past, not the value under test.
        self.history[self.write] = flux;
        self.write = (self.write + 1) % self.capacity;
        self.filled = (self.filled + 1).min(self.capacity);

        let mut fired = false;
        if self.refractory_left > 0 {
            self.refractory_left -= 1;
        } else if flux > threshold {
            fired = true;
            self.refractory_left = self.refractory_frames;
        }

        // Drive the envelope: a fired onset injects a normalized strength (how far
        // over threshold, softly clamped); otherwise it feeds 0 so it decays.
        let drive = if fired {
            (flux / threshold.max(1e-6)).min(4.0) / 4.0
        } else {
            0.0
        };
        let out = self.env.process(drive);
        (out.clamp(0.0, 1.0), fired)
    }

    /// Median of the current history window. Copies the written frames into a
    /// reused scratch buffer (no per-hop allocation, P2-AUD-013) and takes the
    /// center order statistic with a partial selection instead of a full sort.
    /// The selected element is the sorted-order midpoint, so the value is
    /// identical to the previous copy-and-sort implementation.
    fn median(&mut self) -> f32 {
        if self.filled == 0 {
            return 0.0;
        }
        self.median_scratch.clear();
        self.median_scratch
            .extend_from_slice(&self.history[..self.filled]);
        let mid = self.median_scratch.len() / 2;
        let (_, median, _) = self
            .median_scratch
            .select_nth_unstable_by(mid, |a, b| a.total_cmp(b));
        *median
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn detector() -> OnsetDetector {
        let hop_dt = 256.0 / 48_000.0;
        OnsetDetector::new(43, 1.6, 1e-4, 60.0, 5.0, 200.0, hop_dt)
    }

    #[test]
    fn fires_on_spike_above_quiet_baseline() {
        let mut d = detector();
        // Feed a quiet baseline.
        for _ in 0..50 {
            d.process(0.01, 1.0);
        }
        // A large spike should fire.
        let (env, fired) = d.process(1.0, 1.0);
        assert!(fired, "expected onset on a 100x spike");
        assert!(env > 0.0);
    }

    #[test]
    fn refractory_suppresses_double_trigger() {
        let mut d = detector();
        for _ in 0..50 {
            d.process(0.01, 1.0);
        }
        let (_, first) = d.process(1.0, 1.0);
        assert!(first);
        // Immediately following hop, even with high flux, must be suppressed.
        let (_, second) = d.process(1.0, 1.0);
        assert!(!second, "refractory window must block immediate retrigger");
    }

    #[test]
    fn steady_loud_signal_does_not_keep_firing() {
        let mut d = detector();
        let mut fires = 0;
        // Steady high flux: the adaptive median rises, so it should NOT fire every hop.
        for _ in 0..200 {
            let (_, f) = d.process(0.8, 1.0);
            if f {
                fires += 1;
            }
        }
        assert!(fires < 50, "steady signal fired too often: {fires}");
    }

    /// P2-AUD-013: the reused-scratch partial-selection median returns exactly the
    /// same value as the previous allocate-and-full-sort implementation across a
    /// range of odd/even fills, and reuses its scratch buffer (no per-hop
    /// allocation — capacity never grows once warmed).
    #[test]
    fn median_matches_sort_reference_without_reallocating() {
        // Reference: the old copy + full-sort + midpoint element.
        fn sort_reference(history: &[f32]) -> f32 {
            if history.is_empty() {
                return 0.0;
            }
            let mut buf = history.to_vec();
            buf.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            buf[buf.len() / 2]
        }

        let hop_dt = 256.0 / 48_000.0;
        let mut d = OnsetDetector::new(37, 1.6, 1e-4, 60.0, 5.0, 200.0, hop_dt);
        let warm_capacity = d.median_scratch.capacity();

        // A pseudo-random-ish, non-monotonic flux stream so the median is a real
        // order statistic, not a trivially-sorted input.
        let mut acc = 0.123_f32;
        for step in 0..400 {
            acc = (acc * 7.0 + 0.31).fract();
            let flux = acc * (1.0 + (step % 5) as f32);

            // Capture the window the detector will median BEFORE it mutates it,
            // then assert the detector's internal median equals the sort reference.
            let expected = sort_reference(&d.history[..d.filled]);
            let got = d.median();
            assert_eq!(
                got, expected,
                "select-based median diverged from the sort reference at step {step}"
            );

            d.process(flux, 1.0);
            assert_eq!(
                d.median_scratch.capacity(),
                warm_capacity,
                "median scratch reallocated at step {step} — per-hop allocation regressed"
            );
        }
    }

    #[test]
    fn higher_sensitivity_fires_more_easily() {
        let mut low = detector();
        let mut high = detector();
        for _ in 0..50 {
            low.process(0.1, 1.0);
            high.process(0.1, 1.0);
        }
        let (_, low_fire) = low.process(0.16, 0.3);
        let (_, high_fire) = high.process(0.16, 3.0);
        // With a marginal spike, high sensitivity should fire where low does not.
        assert!(high_fire);
        assert!(!low_fire);
    }
}

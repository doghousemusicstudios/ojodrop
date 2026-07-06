//! CPU-side smoothing & normalization primitives.
//!
//! These run in the DSP worker after raw features are computed, so the consumer
//! reads ready-to-use `0..1` values (spec §5 / §7).
//!
//! - [`OnePole`]: first-order low-pass for continuous features (bands, brightness…).
//! - [`Agc`]: per-channel automatic gain control via a slowly-tracking running
//!   peak (a cheap, windup-safe stand-in for a running percentile). Maps the
//!   recent dynamic range of a feature onto `0..1`.
//! - [`SilenceGate`]: RMS floor gate with hysteresis so visuals idle cleanly when
//!   the system is muted without chattering at the threshold.
//! - [`AsymEnv`]: asymmetric attack/decay envelope follower — fast attack, slow
//!   decay — that turns twitchy energy/onset signals into impulses that "pop"
//!   then fall (the single biggest factor in visuals feeling on-beat, spec §4).

#[inline]
fn finite_or_zero(value: f32) -> f32 {
    if value.is_finite() {
        value
    } else {
        0.0
    }
}

/// First-order one-pole low-pass filter. `coeff` in `0..1`: higher = smoother
/// (more inertia). `y[n] = y[n-1] + (1-coeff) * (x[n] - y[n-1])`.
#[derive(Clone, Copy, Debug)]
pub struct OnePole {
    coeff: f32,
    state: f32,
}

impl OnePole {
    /// `smoothing` in `0..1`; 0 = passthrough, →1 = heavy inertia.
    pub fn new(smoothing: f32) -> Self {
        Self {
            coeff: smoothing.clamp(0.0, 0.9999),
            state: 0.0,
        }
    }

    #[inline]
    pub fn process(&mut self, x: f32) -> f32 {
        let x = finite_or_zero(x);
        if !self.state.is_finite() {
            self.state = 0.0;
        }
        self.state += (1.0 - self.coeff) * (x - self.state);
        if !self.state.is_finite() {
            self.state = 0.0;
        }
        self.state
    }

    #[inline]
    #[cfg(test)]
    pub fn value(&self) -> f32 {
        self.state
    }
}

/// Automatic gain control. Tracks a running peak that rises instantly to new
/// maxima and decays slowly, then normalizes the input against it. A noise floor
/// and a clamp keep it from winding up to absurd gains on near-silence.
#[derive(Clone, Copy, Debug)]
pub struct Agc {
    /// Current tracked peak (in the same units as the input).
    peak: f32,
    /// Multiplicative decay per frame applied to the peak (≈ how fast it forgets).
    decay: f32,
    /// Lowest peak we allow — prevents division by ~0 amplifying noise.
    floor: f32,
}

impl Agc {
    /// `decay` in `0..1` per frame (e.g. 0.999 ≈ slow). `floor` is the minimum
    /// reference level (in input units) to avoid windup on silence.
    pub fn new(decay: f32, floor: f32) -> Self {
        Self {
            peak: floor.max(1e-6),
            decay: decay.clamp(0.0, 0.99999),
            floor: floor.max(1e-6),
        }
    }

    /// Normalize `x` to `0..1` against the running peak, updating the peak.
    #[inline]
    pub fn process(&mut self, x: f32) -> f32 {
        let x = finite_or_zero(x).max(0.0);
        if !self.peak.is_finite() {
            self.peak = self.floor;
        }
        // Decay the peak toward the floor, then let a new maximum push it up.
        self.peak = (self.peak * self.decay).max(self.floor);
        if x > self.peak {
            self.peak = x;
        }
        (x / self.peak).clamp(0.0, 1.0)
    }
}

/// RMS-floor silence gate with hysteresis. Returns `true` (silent) when the
/// smoothed level stays below `enter` and only flips back to audible once it
/// rises above `exit` (`exit > enter`).
#[derive(Clone, Copy, Debug)]
pub struct SilenceGate {
    enter: f32,
    exit: f32,
    silent: bool,
}

impl SilenceGate {
    /// `enter`: go-silent threshold. `exit`: become-audible threshold (must be ≥ enter).
    pub fn new(enter: f32, exit: f32) -> Self {
        Self {
            enter,
            exit: exit.max(enter),
            silent: true,
        }
    }

    /// Feed a level (e.g. linear RMS). Returns the gate state (true = silent).
    #[inline]
    pub fn update(&mut self, level: f32) -> bool {
        let level = finite_or_zero(level);
        if self.silent {
            if level > self.exit {
                self.silent = false;
            }
        } else if level < self.enter {
            self.silent = true;
        }
        self.silent
    }
}

/// Asymmetric attack/decay envelope follower. On a rising input it jumps quickly
/// (attack), on a falling input it eases down slowly (decay), producing an
/// impulse that pops then falls. Coefficients are per-frame one-pole rates.
#[derive(Clone, Copy, Debug)]
pub struct AsymEnv {
    attack: f32,
    decay: f32,
    state: f32,
}

impl AsymEnv {
    /// `attack_ms` / `decay_ms` are time constants; `hop_dt` is seconds per frame.
    pub fn new(attack_ms: f32, decay_ms: f32, hop_dt: f32) -> Self {
        Self {
            attack: time_constant_coeff(attack_ms, hop_dt),
            decay: time_constant_coeff(decay_ms, hop_dt),
            state: 0.0,
        }
    }

    #[inline]
    pub fn process(&mut self, x: f32) -> f32 {
        let x = finite_or_zero(x);
        if !self.state.is_finite() {
            self.state = 0.0;
        }
        let coeff = if x > self.state {
            self.attack
        } else {
            self.decay
        };
        // coeff is the per-frame retention of the OLD value; (1-coeff) lets the new in.
        self.state += (1.0 - coeff) * (x - self.state);
        if !self.state.is_finite() {
            self.state = 0.0;
        }
        self.state
    }
}

/// Butterchurn-faithful volume-independent reactivity level (audioLevels.js).
///
/// Tracks a band's *immediate* energy against two adaptive rolling averages and
/// returns the ratio `imm / longAvg` (≈1 at the band's recent average, >1 on a
/// hit, <1 when quieter than recent). Because both numerator and denominator are
/// in the same units, the value is independent of absolute level — only relative
/// change matters, which is exactly MilkDrop/Butterchurn's reactivity convention.
///
/// - `avg` is a fast-attack / slower-release short average (rates 0.2 / 0.5).
/// - `long_avg` is a slow long-term average (rate 0.9 for the first 50 frames,
///   then 0.992) used as the normalization reference.
/// - The returned `att` value (`avg / longAvg`) is the smoother "attenuated"
///   envelope that lags peaks — the MilkDrop `*_att` inputs.
///
/// Both rates are FPS-adjusted (`rate ** (baseFPS/FPS)`, baseFPS=30) so behavior
/// is frame-rate independent, matching `AudioLevels.adjustRateToFPS`. Init avg /
/// long_avg to 1.0 (constructor in butterchurn) so early frames never divide by
/// a tiny reference and spike.
#[derive(Clone, Copy, Debug)]
pub struct ReactiveLevel {
    avg: f32,
    long_avg: f32,
    frame: u64,
}

impl Default for ReactiveLevel {
    fn default() -> Self {
        Self::new()
    }
}

impl ReactiveLevel {
    pub fn new() -> Self {
        Self {
            avg: 1.0,
            long_avg: 1.0,
            frame: 0,
        }
    }

    /// Feed this hop's immediate band sum `imm` (a raw sum of magnitude bins, NOT
    /// a mean) and the per-hop period in seconds. Returns `(val, att)` where
    /// `val = imm/longAvg` and `att = avg/longAvg`.
    #[inline]
    pub fn process(&mut self, imm: f32, hop_dt: f32) -> (f32, f32) {
        const BASE_FPS: f32 = 30.0;
        let fps = (1.0 / hop_dt.max(1e-4)).clamp(15.0, 144.0);
        let adjust = |rate: f32| rate.powf(BASE_FPS / fps);
        let imm = finite_or_zero(imm).max(0.0);
        if !self.avg.is_finite() {
            self.avg = 1.0;
        }
        if !self.long_avg.is_finite() {
            self.long_avg = 1.0;
        }

        // Short rolling average: rises fast (0.2 retention), falls slower (0.5).
        let rate = adjust(if imm > self.avg { 0.2 } else { 0.5 });
        self.avg = self.avg * rate + imm * (1.0 - rate);

        // Long rolling average: very slow once warmed up.
        let rate = adjust(if self.frame < 50 { 0.9 } else { 0.992 });
        self.long_avg = self.long_avg * rate + imm * (1.0 - rate);

        self.frame += 1;

        if self.long_avg < 0.001 {
            (1.0, 1.0)
        } else {
            (
                finite_or_zero(imm / self.long_avg),
                finite_or_zero(self.avg / self.long_avg),
            )
        }
    }
}

/// Convert a time constant (ms) and frame period (s) to a one-pole retention
/// coefficient `exp(-dt / tau)`. Smaller tau → smaller coeff → faster response.
#[inline]
pub fn time_constant_coeff(tau_ms: f32, hop_dt: f32) -> f32 {
    if tau_ms <= 0.0 {
        return 0.0;
    }
    (-hop_dt / (tau_ms * 1e-3)).exp().clamp(0.0, 0.99999)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn one_pole_converges() {
        let mut lp = OnePole::new(0.5);
        for _ in 0..200 {
            lp.process(1.0);
        }
        assert!((lp.value() - 1.0).abs() < 1e-3);
    }

    #[test]
    fn one_pole_and_asym_env_recover_from_nan() {
        let mut lp = OnePole::new(0.5);
        assert_eq!(lp.process(f32::NAN), 0.0);
        lp.state = f32::NAN;
        assert!(lp.process(1.0).is_finite());
        assert!(lp.process(1.0).is_finite());

        let mut env = AsymEnv::new(8.0, 250.0, 256.0 / 48_000.0);
        assert_eq!(env.process(f32::NAN), 0.0);
        env.state = f32::NAN;
        assert!(env.process(1.0).is_finite());
        assert!(env.process(0.0).is_finite());
    }

    #[test]
    fn agc_normalizes_to_unity_at_peak() {
        let mut agc = Agc::new(0.999, 1e-4);
        // Feed a steady level; output should approach 1.0 (level == its own peak).
        let mut out = 0.0;
        for _ in 0..10 {
            out = agc.process(0.3);
        }
        assert!(out > 0.99, "expected ~1.0 at the running peak, got {out}");
    }

    #[test]
    fn agc_floor_prevents_windup() {
        let mut agc = Agc::new(0.9, 1e-3);
        // Run silence long enough for the peak to bottom out at the floor.
        for _ in 0..10_000 {
            agc.process(0.0);
        }
        // A tiny noise sample must NOT be blown up to ~1.0.
        let out = agc.process(1e-5);
        assert!(out < 0.1, "noise floor should suppress windup, got {out}");
    }

    #[test]
    fn silence_gate_has_hysteresis() {
        let mut gate = SilenceGate::new(0.01, 0.05);
        assert!(gate.update(0.0)); // starts silent
        assert!(gate.update(0.02)); // between thresholds → stays silent
        assert!(!gate.update(0.06)); // above exit → audible
        assert!(!gate.update(0.02)); // between thresholds → stays audible
        assert!(gate.update(0.005)); // below enter → silent again
    }

    #[test]
    fn reactive_level_is_volume_independent() {
        let hop_dt = 512.0 / 48_000.0;
        // Two streams identical in shape but scaled by 1000x must converge to the
        // same reactivity ratio — the whole point of imm/longAvg normalization.
        let mut quiet = ReactiveLevel::new();
        let mut loud = ReactiveLevel::new();
        let (mut vq, mut vl) = (0.0, 0.0);
        for _ in 0..2000 {
            vq = quiet.process(0.5, hop_dt).0;
            vl = loud.process(500.0, hop_dt).0;
        }
        // Steady input → ratio settles near 1.0 regardless of absolute level.
        assert!((vq - 1.0).abs() < 0.05, "quiet steady ratio ~1.0, got {vq}");
        assert!((vl - 1.0).abs() < 0.05, "loud steady ratio ~1.0, got {vl}");
        assert!((vq - vl).abs() < 0.02, "scale-invariant: {vq} vs {vl}");
    }

    #[test]
    fn reactive_level_spikes_above_one_on_a_hit() {
        let hop_dt = 512.0 / 48_000.0;
        let mut r = ReactiveLevel::new();
        // Warm up at a steady baseline so longAvg learns it.
        for _ in 0..400 {
            r.process(1.0, hop_dt);
        }
        // A sudden 5x transient must read >1 (louder than recent average).
        let (val, att) = r.process(5.0, hop_dt);
        assert!(val > 1.5, "hit should exceed recent average, got {val}");
        // att (avg/longAvg) lags the raw ratio — smoother envelope.
        assert!(
            att < val,
            "att should lag the immediate ratio: {att} vs {val}"
        );
    }

    #[test]
    fn reactive_level_starts_at_unity() {
        let hop_dt = 512.0 / 48_000.0;
        let mut r = ReactiveLevel::new();
        // First frame against the 1.0-initialized references should not divide-by-tiny.
        let (val, att) = r.process(1.0, hop_dt);
        assert!(val.is_finite() && att.is_finite());
        assert!((0.5..2.0).contains(&val), "no first-frame spike, got {val}");
    }

    #[test]
    fn asym_env_fast_attack_slow_decay() {
        let dt = 256.0 / 48_000.0;
        let mut env = AsymEnv::new(8.0, 250.0, dt);
        let after_attack = env.process(1.0);
        let after_decay = env.process(0.0);
        // One attack frame should rise meaningfully; one decay frame should fall
        // far less (slow release).
        assert!(after_attack > 0.2);
        let attack_rise = after_attack;
        let decay_drop = after_attack - after_decay;
        assert!(
            decay_drop < attack_rise,
            "decay should be slower than attack"
        );
    }
}

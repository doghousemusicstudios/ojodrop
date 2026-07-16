//! Causal harmonic/percussive source separation approximation.
//!
//! This is deliberately lighter than offline HPSS: it reuses the existing STFT
//! magnitude frame, keeps a short causal history for time medians, and compares
//! that against a frequency-axis median from the current frame. The resulting
//! soft masks are good enough for visual-control buses without adding lookahead
//! or reconstructing audio stems.

use crate::analysis;
use crate::onset::OnsetDetector;
use crate::smoothing::{Agc, OnePole};

/// Time-axis (harmonic) median window length, expressed in seconds and converted
/// to a hop-frame count from the live hop period so the window spans the same wall
/// time regardless of sample rate. 0.16 s reproduces the historical 15-frame
/// window at the canonical 48 kHz / 512-hop cadence.
const TIME_MEDIAN_SECONDS: f32 = 0.16;
const FREQ_MEDIAN_RADIUS: usize = 5;
const MASK_POWER: f32 = 2.0;

#[inline]
fn finite_or_zero(value: f32) -> f32 {
    if value.is_finite() {
        value
    } else {
        0.0
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct HpssFeatures {
    pub perc_rms: f32,
    pub perc_flux: f32,
    pub perc_onset: f32,
    pub perc_ratio: f32,
    pub harm_rms: f32,
    pub harm_flux: f32,
    pub harm_brightness: f32,
    pub harm_ratio: f32,
}

/// Shared rolling STFT-magnitude history + time-axis (harmonic) median filter.
///
/// Both the internal [`HpssSeparator`] and the public [`crate::hpss_bus::HpssBus`]
/// previously kept their *own* copy of the STFT magnitude history and ran the same
/// per-bin time-median sort over the same spectrum every hop (P2-AUD-003). They now
/// share this one component: the magnitude frame is stored once and the harmonic
/// reference (the per-bin time median) is computed once per hop, then consumed by
/// both. Each consumer still applies its own frequency-axis (percussive) median,
/// which uses a different radius by design.
pub struct HpssHistory {
    n_bins: usize,
    /// Number of hop frames in the shared time-axis median window.
    time_frames: usize,
    /// Rolling magnitude history, `time_frames * n_bins`, laid out frame-major.
    history: Vec<f32>,
    write: usize,
    filled: usize,
    /// Reused per-bin column scratch for the median (never reallocated per hop).
    scratch: Vec<f32>,
    /// Per-bin time median for the current frame — the harmonic reference.
    harm_ref: Vec<f32>,
}

impl HpssHistory {
    /// Build a shared history for an analyzer producing `n_bins` (= `FFT_LEN/2 + 1`)
    /// magnitude bins at a hop period of `hop_dt` seconds. The time-median window
    /// length is derived from `hop_dt` so the harmonic estimate spans a fixed
    /// wall-clock duration regardless of hop size / sample rate.
    pub fn new(n_bins: usize, hop_dt: f32) -> Self {
        let n_bins = n_bins.max(1);
        // Fixed-duration window in hop frames, kept odd for a true median center.
        // 0.16 s reproduces the historical 15-frame window at the canonical
        // 48 kHz / 512-hop analysis cadence the resample layer pins.
        let mut time_frames = (TIME_MEDIAN_SECONDS / hop_dt.max(1e-6)).round() as usize;
        time_frames = time_frames.max(3);
        if time_frames % 2 == 0 {
            time_frames += 1;
        }
        Self {
            n_bins,
            time_frames,
            history: vec![0.0; n_bins * time_frames],
            write: 0,
            filled: 0,
            scratch: vec![0.0; time_frames],
            harm_ref: vec![0.0; n_bins],
        }
    }

    /// Number of hop frames in the shared time-axis (harmonic) median window.
    pub fn time_window_frames(&self) -> usize {
        self.time_frames
    }

    /// Push the newest magnitude frame and recompute every per-bin time median
    /// (the harmonic reference) for the current frame. Call once per hop before
    /// either consumer runs.
    pub fn advance(&mut self, mag: &[f32]) {
        debug_assert_eq!(mag.len(), self.n_bins);
        let offset = self.write * self.n_bins;
        for (dst, &src) in self.history[offset..offset + self.n_bins]
            .iter_mut()
            .zip(mag.iter())
        {
            *dst = finite_or_zero(src).max(0.0);
        }
        self.write = (self.write + 1) % self.time_frames;
        self.filled = (self.filled + 1).min(self.time_frames);

        // Median only over written frames so warm-up isn't biased toward zero-init
        // slots — matching the pre-share time_median exactly.
        let count = self.filled;
        for bin in 0..self.n_bins {
            if count == 0 {
                self.harm_ref[bin] = 0.0;
                continue;
            }
            for frame in 0..count {
                self.scratch[frame] = self.history[frame * self.n_bins + bin];
            }
            self.harm_ref[bin] = median(&mut self.scratch[..count]);
        }
    }

    /// Per-bin harmonic reference (time median) for the current frame.
    pub fn harm_ref(&self) -> &[f32] {
        &self.harm_ref
    }
}

pub(crate) struct HpssSeparator {
    n_bins: usize,
    sample_rate: f32,
    bin_hz: f32,
    freq_scratch: Vec<f32>,
    harmonic_mag: Vec<f32>,
    percussive_mag: Vec<f32>,
    prev_harmonic_mag: Vec<f32>,
    prev_percussive_mag: Vec<f32>,
    perc_rms_agc: Agc,
    harm_rms_agc: Agc,
    perc_rms_lp: OnePole,
    harm_rms_lp: OnePole,
    perc_flux_lp: OnePole,
    harm_flux_lp: OnePole,
    perc_ratio_lp: OnePole,
    harm_ratio_lp: OnePole,
    harm_brightness_lp: OnePole,
    perc_onset: OnsetDetector,
}

impl HpssSeparator {
    pub(crate) fn new(n_bins: usize, sample_rate: f32, bin_hz: f32, hop_dt: f32) -> Self {
        let med = ((0.22 / hop_dt) as usize).max(8);
        Self {
            n_bins,
            sample_rate,
            bin_hz,
            freq_scratch: vec![0.0; FREQ_MEDIAN_RADIUS * 2 + 1],
            harmonic_mag: vec![0.0; n_bins],
            percussive_mag: vec![0.0; n_bins],
            prev_harmonic_mag: vec![0.0; n_bins],
            prev_percussive_mag: vec![0.0; n_bins],
            perc_rms_agc: Agc::new(0.9996, 1e-3),
            harm_rms_agc: Agc::new(0.9998, 1e-3),
            perc_rms_lp: OnePole::new(0.45),
            harm_rms_lp: OnePole::new(0.65),
            perc_flux_lp: OnePole::new(0.35),
            harm_flux_lp: OnePole::new(0.55),
            perc_ratio_lp: OnePole::new(0.25),
            harm_ratio_lp: OnePole::new(0.55),
            harm_brightness_lp: OnePole::new(0.55),
            perc_onset: OnsetDetector::new(med, 1.45, 1e-4, 55.0, 6.0, 180.0, hop_dt),
        }
    }

    /// Analyze one STFT magnitude frame. `harm_ref` is the shared per-bin harmonic
    /// reference (time median) from [`HpssHistory`], already advanced for this hop;
    /// this separator contributes only its own frequency-axis (percussive) median.
    pub(crate) fn analyze(
        &mut self,
        mag: &[f32],
        harm_ref: &[f32],
        is_silent: bool,
        sensitivity: f32,
    ) -> HpssFeatures {
        debug_assert_eq!(mag.len(), self.n_bins);
        debug_assert_eq!(harm_ref.len(), self.n_bins);

        let mut perc_energy = 0.0f32;
        let mut harm_energy = 0.0f32;

        for bin in 0..self.n_bins {
            let harmonic_ref = finite_or_zero(harm_ref[bin]).max(0.0);
            let percussive_ref = finite_or_zero(self.frequency_median(mag, bin)).max(0.0);
            let h = harmonic_ref.powf(MASK_POWER);
            let p = percussive_ref.powf(MASK_POWER);
            let denom = finite_or_zero(h + p).max(0.0) + 1e-12;
            let harm_mask = finite_or_zero(h / denom).clamp(0.0, 1.0);
            let perc_mask = finite_or_zero(p / denom).clamp(0.0, 1.0);

            let m = finite_or_zero(mag[bin]).max(0.0);
            let hm = m * harm_mask;
            let pm = m * perc_mask;
            self.harmonic_mag[bin] = hm;
            self.percussive_mag[bin] = pm;
            harm_energy += hm * hm;
            perc_energy += pm * pm;
        }

        let harm_energy = (harm_energy / self.n_bins.max(1) as f32).sqrt();
        let perc_energy = (perc_energy / self.n_bins.max(1) as f32).sqrt();
        let total_energy = harm_energy + perc_energy;
        let (raw_perc_ratio, raw_harm_ratio) = if is_silent || total_energy <= 1e-9 {
            (0.0, 0.0)
        } else {
            (perc_energy / total_energy, harm_energy / total_energy)
        };

        let raw_perc_flux = if is_silent {
            0.0
        } else {
            analysis::superflux(&self.prev_percussive_mag, &self.percussive_mag, 2)
        };
        let raw_harm_flux = if is_silent {
            0.0
        } else {
            analysis::superflux(&self.prev_harmonic_mag, &self.harmonic_mag, 3)
        };

        let perc_flux = (raw_perc_flux / (self.n_bins as f32 * 0.010)).clamp(0.0, 1.0);
        let harm_flux = (raw_harm_flux / (self.n_bins as f32 * 0.010)).clamp(0.0, 1.0);
        let (perc_onset, _) = self.perc_onset.process(raw_perc_flux, sensitivity);

        let perc_rms = if is_silent {
            self.perc_rms_lp.process(0.0)
        } else {
            self.perc_rms_lp.process(
                self.perc_rms_agc
                    .process(crate::dsp::lin_to_db_norm(perc_energy)),
            )
        };
        let harm_rms = if is_silent {
            self.harm_rms_lp.process(0.0)
        } else {
            self.harm_rms_lp.process(
                self.harm_rms_agc
                    .process(crate::dsp::lin_to_db_norm(harm_energy)),
            )
        };

        HpssFeatures {
            perc_rms,
            perc_flux: self.perc_flux_lp.process(perc_flux),
            perc_onset,
            perc_ratio: self.perc_ratio_lp.process(raw_perc_ratio),
            harm_rms,
            harm_flux: self.harm_flux_lp.process(harm_flux),
            harm_brightness: self.harm_brightness_lp.process(analysis::spectral_centroid(
                &self.harmonic_mag,
                self.bin_hz,
                self.sample_rate,
            )),
            harm_ratio: self.harm_ratio_lp.process(raw_harm_ratio),
        }
    }

    pub(crate) fn percussive_mag(&self) -> &[f32] {
        &self.percussive_mag
    }

    pub(crate) fn harmonic_mag(&self) -> &[f32] {
        &self.harmonic_mag
    }

    pub(crate) fn prev_percussive_mag(&self) -> &[f32] {
        &self.prev_percussive_mag
    }

    pub(crate) fn finish_frame(&mut self) {
        self.prev_harmonic_mag.copy_from_slice(&self.harmonic_mag);
        self.prev_percussive_mag
            .copy_from_slice(&self.percussive_mag);
    }

    fn frequency_median(&mut self, mag: &[f32], bin: usize) -> f32 {
        let lo = bin.saturating_sub(FREQ_MEDIAN_RADIUS);
        let hi = (bin + FREQ_MEDIAN_RADIUS + 1).min(mag.len());
        let count = hi - lo;
        for (dst, &src) in self.freq_scratch[..count]
            .iter_mut()
            .zip(mag[lo..hi].iter())
        {
            *dst = finite_or_zero(src).max(0.0);
        }
        median(&mut self.freq_scratch[..count])
    }
}

/// Median of a scratch slice (reorders it). Takes the upper-middle element for
/// even counts via partial selection — the same value a full sort would place at
/// `len / 2`, but without sorting the whole slice.
fn median(values: &mut [f32]) -> f32 {
    if values.is_empty() {
        return 0.0;
    }
    let mid = values.len() / 2;
    let (_, m, _) = values.select_nth_unstable_by(mid, |a, b| a.total_cmp(b));
    *m
}

#[cfg(test)]
mod tests {
    use super::*;

    fn history(n_bins: usize) -> HpssHistory {
        HpssHistory::new(n_bins, 512.0 / 48_000.0)
    }

    fn separator(n_bins: usize) -> HpssSeparator {
        let sample_rate = 48_000.0;
        let fft_len = (n_bins - 1) * 2;
        let bin_hz = sample_rate / fft_len as f32;
        HpssSeparator::new(n_bins, sample_rate, bin_hz, 512.0 / sample_rate)
    }

    /// P2-AUD-014: the shared time-median window is expressed in seconds, so it
    /// spans a fixed wall-clock duration regardless of the hop period (sample
    /// rate). With the old fixed 15-frame window the duration drifted with rate.
    #[test]
    fn time_median_window_is_fixed_duration_across_rates() {
        let n_bins = 128;
        // Canonical 48 kHz / 512-hop reproduces the historical 15-frame window.
        let canonical = HpssHistory::new(n_bins, 512.0 / 48_000.0);
        assert_eq!(canonical.time_window_frames(), 15);

        for &hop_dt in &[512.0 / 96_000.0, 512.0 / 48_000.0, 512.0 / 24_000.0, 0.02] {
            let hist = HpssHistory::new(n_bins, hop_dt);
            let duration = hist.time_window_frames() as f32 * hop_dt;
            assert!(
                (duration - TIME_MEDIAN_SECONDS).abs() < 0.05,
                "window duration {duration} s drifted from {TIME_MEDIAN_SECONDS} s at hop_dt {hop_dt}"
            );
            assert_eq!(hist.time_window_frames() % 2, 1, "window must stay odd");
        }
    }

    /// P2-AUD-003: the shared harmonic reference bit-equals an independent
    /// last-N-frame median over the same spectrum (so routing both consumers
    /// through it preserves their separated output), and the median scratch is
    /// reused — no per-hop allocation.
    #[test]
    fn shared_harm_ref_matches_naive_median_without_reallocating() {
        let n_bins = 64;
        let mut hist = history(n_bins);
        let window = hist.time_window_frames();
        let warm_capacity = hist.scratch.capacity();

        // A rolling buffer of the raw pushed frames, for an independent reference.
        let mut frames: Vec<Vec<f32>> = Vec::new();

        for step in 0..80 {
            // Deterministic, per-bin-varying, non-monotonic magnitudes.
            let frame: Vec<f32> = (0..n_bins)
                .map(|b| ((step * 7 + b * 13) % 17) as f32 * 0.05 + (b as f32 * 0.001))
                .collect();
            hist.advance(&frame);
            frames.push(frame);

            // Reference: per-bin median of the most recent `window` frames, using
            // the same non-negative transform and upper-middle even convention.
            let start = frames.len().saturating_sub(window);
            let recent = &frames[start..];
            for bin in 0..n_bins {
                let mut col: Vec<f32> = recent
                    .iter()
                    .map(|f| finite_or_zero(f[bin]).max(0.0))
                    .collect();
                col.sort_by(|a, b| a.total_cmp(b));
                let expected = col[col.len() / 2];
                assert_eq!(
                    hist.harm_ref()[bin],
                    expected,
                    "shared harm_ref diverged at step {step}, bin {bin}"
                );
            }
            assert_eq!(
                hist.scratch.capacity(),
                warm_capacity,
                "time-median scratch reallocated at step {step}"
            );
        }
    }

    #[test]
    fn sustained_isolated_tone_prefers_harmonic_bus() {
        let mut hist = history(128);
        let mut hpss = separator(128);
        let mut mag = vec![0.0f32; 128];
        mag[42] = 1.0;
        mag[43] = 0.35;
        mag[41] = 0.35;

        let mut out = HpssFeatures::default();
        for _ in 0..24 {
            hist.advance(&mag);
            out = hpss.analyze(&mag, hist.harm_ref(), false, 1.0);
            hpss.finish_frame();
        }

        assert!(
            out.harm_ratio > 0.75,
            "isolated sustained tone should land on harmonic bus: {out:?}"
        );
        assert!(
            out.harm_rms > out.perc_rms,
            "harmonic RMS should dominate percussive RMS: {out:?}"
        );
    }

    #[test]
    fn broadband_spike_prefers_percussive_bus_and_fires_onset() {
        let mut hist = history(128);
        let mut hpss = separator(128);
        let silence = vec![0.0f32; 128];
        for _ in 0..24 {
            hist.advance(&silence);
            hpss.analyze(&silence, hist.harm_ref(), true, 1.0);
            hpss.finish_frame();
        }

        let mut hit = vec![0.0f32; 128];
        for bin in 8..100 {
            hit[bin] = 1.0;
        }
        hist.advance(&hit);
        let out = hpss.analyze(&hit, hist.harm_ref(), false, 1.0);

        assert!(
            out.perc_ratio > 0.60,
            "broadband spike should land on percussive bus: {out:?}"
        );
        assert!(
            out.perc_onset > 0.5,
            "percussive bus should produce a shaped onset: {out:?}"
        );
    }
}

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

const TIME_MEDIAN_FRAMES: usize = 15;
const FREQ_MEDIAN_RADIUS: usize = 5;
const MASK_POWER: f32 = 2.0;

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

pub(crate) struct HpssSeparator {
    n_bins: usize,
    sample_rate: f32,
    bin_hz: f32,
    history: Vec<f32>,
    history_write: usize,
    time_scratch: Vec<f32>,
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
            history: vec![0.0; n_bins * TIME_MEDIAN_FRAMES],
            history_write: 0,
            time_scratch: vec![0.0; TIME_MEDIAN_FRAMES],
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

    pub(crate) fn analyze(
        &mut self,
        mag: &[f32],
        is_silent: bool,
        sensitivity: f32,
    ) -> HpssFeatures {
        debug_assert_eq!(mag.len(), self.n_bins);
        self.push_history(mag);

        let mut perc_energy = 0.0f32;
        let mut harm_energy = 0.0f32;

        for bin in 0..self.n_bins {
            let harmonic_ref = self.time_median(bin);
            let percussive_ref = self.frequency_median(mag, bin);
            let h = harmonic_ref.powf(MASK_POWER);
            let p = percussive_ref.powf(MASK_POWER);
            let denom = h + p + 1e-12;
            let harm_mask = h / denom;
            let perc_mask = p / denom;

            let hm = mag[bin] * harm_mask;
            let pm = mag[bin] * perc_mask;
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

    fn push_history(&mut self, mag: &[f32]) {
        let offset = self.history_write * self.n_bins;
        self.history[offset..offset + self.n_bins].copy_from_slice(mag);
        self.history_write = (self.history_write + 1) % TIME_MEDIAN_FRAMES;
    }

    fn time_median(&mut self, bin: usize) -> f32 {
        for frame in 0..TIME_MEDIAN_FRAMES {
            self.time_scratch[frame] = self.history[frame * self.n_bins + bin];
        }
        median(&mut self.time_scratch)
    }

    fn frequency_median(&mut self, mag: &[f32], bin: usize) -> f32 {
        let lo = bin.saturating_sub(FREQ_MEDIAN_RADIUS);
        let hi = (bin + FREQ_MEDIAN_RADIUS + 1).min(mag.len());
        let count = hi - lo;
        self.freq_scratch[..count].copy_from_slice(&mag[lo..hi]);
        median(&mut self.freq_scratch[..count])
    }
}

fn median(values: &mut [f32]) -> f32 {
    if values.is_empty() {
        return 0.0;
    }
    values.sort_by(|a, b| a.total_cmp(b));
    values[values.len() / 2]
}

#[cfg(test)]
mod tests {
    use super::*;

    fn separator(n_bins: usize) -> HpssSeparator {
        let sample_rate = 48_000.0;
        let fft_len = (n_bins - 1) * 2;
        let bin_hz = sample_rate / fft_len as f32;
        HpssSeparator::new(n_bins, sample_rate, bin_hz, 512.0 / sample_rate)
    }

    #[test]
    fn sustained_isolated_tone_prefers_harmonic_bus() {
        let mut hpss = separator(128);
        let mut mag = vec![0.0f32; 128];
        mag[42] = 1.0;
        mag[43] = 0.35;
        mag[41] = 0.35;

        let mut out = HpssFeatures::default();
        for _ in 0..24 {
            out = hpss.analyze(&mag, false, 1.0);
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
        let mut hpss = separator(128);
        let silence = vec![0.0f32; 128];
        for _ in 0..24 {
            hpss.analyze(&silence, true, 1.0);
            hpss.finish_frame();
        }

        let mut hit = vec![0.0f32; 128];
        for bin in 8..100 {
            hit[bin] = 1.0;
        }
        let out = hpss.analyze(&hit, false, 1.0);

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

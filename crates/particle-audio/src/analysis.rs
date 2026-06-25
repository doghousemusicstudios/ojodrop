//! Pure spectral-analysis helpers: window function, log/triangular filterbanks,
//! and the scalar spectral descriptors (centroid, flux, rolloff, flatness,
//! spread, contrast, SuperFlux).
//!
//! All of these are stateless aside from the precomputed weights / the one-frame
//! magnitude history kept by the caller for flux. They contain no I/O and no
//! locks, so they are trivially unit-testable.

use std::f32::consts::PI;

/// The six macro bands, by [low, high] edge in Hz (spec §4).
pub const MACRO_BANDS_HZ: [(f32, f32); 6] = [
    (20.0, 60.0),      // sub_bass
    (60.0, 150.0),     // bass
    (150.0, 400.0),    // low_mid
    (400.0, 2000.0),   // mid
    (2000.0, 6000.0),  // presence
    (6000.0, 20000.0), // air
];

/// Number of coarse log spectrum bands exposed to the visuals.
pub const SPECTRUM_BANDS: usize = 32;
/// Lowest / highest frequency covered by the coarse log spectrum.
pub const SPECTRUM_LO_HZ: f32 = 30.0;
pub const SPECTRUM_HI_HZ: f32 = 18_000.0;

/// Precompute a periodic Hann window of length `n` (the form used for STFT, where
/// the window is treated as one period of a periodic signal).
pub fn hann_window(n: usize) -> Vec<f32> {
    if n <= 1 {
        return vec![1.0; n];
    }
    (0..n)
        .map(|i| 0.5 - 0.5 * (2.0 * PI * i as f32 / n as f32).cos())
        .collect()
}

/// A precomputed triangular filterbank mapping FFT magnitude bins to a fixed set
/// of bands. Each band stores `(start_bin, weights)` so application is a tight,
/// allocation-free dot product over a contiguous slice.
#[derive(Clone, Debug)]
pub struct FilterBank {
    bands: Vec<Band>,
}

#[derive(Clone, Debug)]
struct Band {
    start_bin: usize,
    weights: Vec<f32>,
}

impl FilterBank {
    /// Build a triangular filterbank for arbitrary `[lo, hi]` band edges (Hz).
    /// `fft_len` is the real FFT size; `sample_rate` the actual device rate (so
    /// the Hz↔bin mapping is correct for 44.1k loopback devices, spec §2).
    pub fn from_edges(edges: &[(f32, f32)], fft_len: usize, sample_rate: f32) -> Self {
        let n_bins = fft_len / 2 + 1;
        let bin_hz = sample_rate / fft_len as f32;
        let bands = edges
            .iter()
            .map(|&(lo, hi)| triangular_band(lo, hi, bin_hz, n_bins))
            .collect();
        Self { bands }
    }

    /// Build a `count`-band log-spaced filterbank spanning `[lo, hi]` Hz with
    /// triangles in log-frequency space. Adjacent bands overlap between their
    /// log centers, which makes these rails behave like an equalizer/emitter
    /// bank instead of linear-frequency display buckets.
    pub fn log_spaced(count: usize, lo: f32, hi: f32, fft_len: usize, sample_rate: f32) -> Self {
        let n_bins = fft_len / 2 + 1;
        let bin_hz = sample_rate / fft_len as f32;
        // count+2 log-spaced points: p[i]..p[i+2] are a band's lower edge,
        // center, and upper edge. This gives `count` overlapping triangular
        // bands whose centers are evenly spaced on the log-frequency axis.
        let log_lo = lo.max(1.0).ln();
        let log_hi = hi.max(lo + 1.0).ln();
        let mut bands = Vec::with_capacity(count);
        let denom = (count + 1).max(1) as f32;
        for b in 0..count {
            let f_lo = (log_lo + (log_hi - log_lo) * b as f32 / denom).exp();
            let f_center = (log_lo + (log_hi - log_lo) * (b + 1) as f32 / denom).exp();
            let f_hi = (log_lo + (log_hi - log_lo) * (b + 2) as f32 / denom).exp();
            bands.push(log_triangular_band(f_lo, f_center, f_hi, bin_hz, n_bins));
        }
        Self { bands }
    }

    #[cfg(test)]
    pub fn len(&self) -> usize {
        self.bands.len()
    }

    /// Apply the bank to a magnitude spectrum, writing one weighted-average value
    /// per band into `out`. `out.len()` must equal `self.len()`.
    pub fn apply(&self, mag: &[f32], out: &mut [f32]) {
        debug_assert_eq!(out.len(), self.bands.len());
        for (band, o) in self.bands.iter().zip(out.iter_mut()) {
            let mut acc = 0.0f32;
            let mut wsum = 0.0f32;
            for (k, &w) in band.weights.iter().enumerate() {
                let bin = band.start_bin + k;
                if bin < mag.len() {
                    acc += w * mag[bin];
                    wsum += w;
                }
            }
            *o = if wsum > 0.0 { acc / wsum } else { 0.0 };
        }
    }
}

/// Build a single triangular band: rises 0→1 from `lo` to the center, falls 1→0
/// to `hi`. Always covers at least one bin so narrow low-frequency bands aren't
/// empty.
fn triangular_band(lo: f32, hi: f32, bin_hz: f32, n_bins: usize) -> Band {
    let center = 0.5 * (lo + hi);
    let lo_bin = (lo / bin_hz).floor().max(0.0) as usize;
    let mut hi_bin = (hi / bin_hz).ceil() as usize;
    if hi_bin <= lo_bin {
        hi_bin = lo_bin + 1;
    }
    let hi_bin = hi_bin.min(n_bins.saturating_sub(1)).max(lo_bin);
    let start_bin = lo_bin.min(n_bins.saturating_sub(1));
    let mut weights = Vec::with_capacity(hi_bin - start_bin + 1);
    for bin in start_bin..=hi_bin {
        let f = bin as f32 * bin_hz;
        let w = if f <= center {
            // rising edge
            if center > lo {
                ((f - lo) / (center - lo)).clamp(0.0, 1.0)
            } else {
                1.0
            }
        } else if hi > center {
            // falling edge
            ((hi - f) / (hi - center)).clamp(0.0, 1.0)
        } else {
            1.0
        };
        // Guarantee at least a sliver of weight inside the band.
        weights.push(w.max(if bin == start_bin { 1e-3 } else { 0.0 }));
    }
    Band { start_bin, weights }
}

/// Build a log-frequency triangular band with an explicit center frequency.
fn log_triangular_band(lo: f32, center: f32, hi: f32, bin_hz: f32, n_bins: usize) -> Band {
    let lo = lo.max(1.0);
    let center = center.max(lo + f32::EPSILON);
    let hi = hi.max(center + f32::EPSILON);
    let lo_bin = (lo / bin_hz).floor().max(0.0) as usize;
    let mut hi_bin = (hi / bin_hz).ceil() as usize;
    if hi_bin <= lo_bin {
        hi_bin = lo_bin + 1;
    }
    let hi_bin = hi_bin.min(n_bins.saturating_sub(1)).max(lo_bin);
    let start_bin = lo_bin.min(n_bins.saturating_sub(1));
    let log_lo = lo.ln();
    let log_center = center.ln();
    let log_hi = hi.ln();
    let mut weights = Vec::with_capacity(hi_bin - start_bin + 1);
    for bin in start_bin..=hi_bin {
        let f = (bin as f32 * bin_hz).max(1.0);
        let lf = f.ln();
        let w = if lf <= log_center {
            ((lf - log_lo) / (log_center - log_lo)).clamp(0.0, 1.0)
        } else {
            ((log_hi - lf) / (log_hi - log_center)).clamp(0.0, 1.0)
        };
        weights.push(w.max(if bin == start_bin { 1e-3 } else { 0.0 }));
    }
    Band { start_bin, weights }
}

/// Spectral centroid normalized to `0..1` against the Nyquist-spanning bin range.
/// Returns 0 when the spectrum has no energy.
pub fn spectral_centroid(mag: &[f32], bin_hz: f32, sample_rate: f32) -> f32 {
    let mut num = 0.0f32;
    let mut den = 0.0f32;
    for (i, &m) in mag.iter().enumerate() {
        let f = i as f32 * bin_hz;
        num += f * m;
        den += m;
    }
    if den <= 0.0 {
        return 0.0;
    }
    let centroid_hz = num / den;
    // Normalize against Nyquist; centroid rarely exceeds it but clamp for safety.
    (centroid_hz / (sample_rate * 0.5)).clamp(0.0, 1.0)
}

/// Spectral flux: sum of positive bin-to-bin magnitude increases between the
/// previous and current spectra. The caller keeps `prev`; this updates nothing.
/// Result is the *raw* flux (un-normalized) used to drive onset detectors.
pub fn spectral_flux(prev: &[f32], cur: &[f32]) -> f32 {
    let n = prev.len().min(cur.len());
    let mut flux = 0.0f32;
    for i in 0..n {
        let d = cur[i] - prev[i];
        if d > 0.0 {
            flux += d;
        }
    }
    flux
}

/// SuperFlux novelty: positive spectral change after a small frequency-axis max
/// filter over the previous frame. The max filter suppresses vibrato/warble
/// side-to-side energy shifts while preserving genuine attacks.
pub fn superflux(prev: &[f32], cur: &[f32], max_bins: usize) -> f32 {
    let n = prev.len().min(cur.len());
    let mut flux = 0.0f32;
    for i in 0..n {
        let lo = i.saturating_sub(max_bins);
        let hi = (i + max_bins + 1).min(n);
        let mut prev_max = 0.0f32;
        for &v in &prev[lo..hi] {
            prev_max = prev_max.max(v);
        }
        let d = cur[i].ln_1p() - prev_max.ln_1p();
        if d > 0.0 {
            flux += d;
        }
    }
    flux
}

/// Band-limited positive spectral flux over `[lo_bin, hi_bin)` — the per-band
/// onset driver (kick/snare/hat).
#[allow(dead_code)]
pub fn band_flux(prev: &[f32], cur: &[f32], lo_bin: usize, hi_bin: usize) -> f32 {
    let hi = hi_bin.min(prev.len()).min(cur.len());
    let lo = lo_bin.min(hi);
    let mut flux = 0.0f32;
    for i in lo..hi {
        let d = cur[i] - prev[i];
        if d > 0.0 {
            flux += d;
        }
    }
    flux
}

/// Band-limited SuperFlux over `[lo_bin, hi_bin)`, with the max filter still
/// allowed to look just outside the band so narrow vibrato near an edge does not
/// become a false onset.
pub fn band_superflux(
    prev: &[f32],
    cur: &[f32],
    lo_bin: usize,
    hi_bin: usize,
    max_bins: usize,
) -> f32 {
    let n = prev.len().min(cur.len());
    let hi = hi_bin.min(n);
    let lo = lo_bin.min(hi);
    let mut flux = 0.0f32;
    for i in lo..hi {
        let mlo = i.saturating_sub(max_bins);
        let mhi = (i + max_bins + 1).min(n);
        let mut prev_max = 0.0f32;
        for &v in &prev[mlo..mhi] {
            prev_max = prev_max.max(v);
        }
        let d = cur[i].ln_1p() - prev_max.ln_1p();
        if d > 0.0 {
            flux += d;
        }
    }
    flux
}

/// Spectral rolloff: the fraction of the spectrum (0..1, as freq/Nyquist) below
/// which `pct` of the cumulative magnitude energy lies (spec: 85%).
pub fn spectral_rolloff(mag: &[f32], pct: f32) -> f32 {
    let total: f32 = mag.iter().sum();
    if total <= 0.0 {
        return 0.0;
    }
    let threshold = total * pct;
    let mut acc = 0.0f32;
    for (i, &m) in mag.iter().enumerate() {
        acc += m;
        if acc >= threshold {
            return i as f32 / (mag.len().saturating_sub(1).max(1)) as f32;
        }
    }
    1.0
}

/// Spectral flatness (geometric mean / arithmetic mean), normalized `0..1`.
/// Tonal/peaky spectra approach 0; broad noisy spectra approach 1.
pub fn spectral_flatness(mag: &[f32]) -> f32 {
    if mag.is_empty() {
        return 0.0;
    }
    let mut sum = 0.0f32;
    let mut log_sum = 0.0f32;
    for &m in mag {
        sum += m;
        log_sum += m.max(1e-12).ln();
    }
    if sum <= 0.0 {
        return 0.0;
    }
    let n = mag.len() as f32;
    let geo = (log_sum / n).exp();
    let arith = sum / n;
    (geo / arith.max(1e-12)).clamp(0.0, 1.0)
}

/// Spectral spread: the weighted standard deviation around the centroid,
/// normalized against the Nyquist-spanning bin range.
pub fn spectral_spread(mag: &[f32]) -> f32 {
    if mag.len() <= 1 {
        return 0.0;
    }
    let max_i = (mag.len() - 1) as f32;
    let mut den = 0.0f32;
    let mut centroid = 0.0f32;
    for (i, &m) in mag.iter().enumerate() {
        let x = i as f32 / max_i;
        centroid += x * m;
        den += m;
    }
    if den <= 0.0 {
        return 0.0;
    }
    centroid /= den;

    let mut var = 0.0f32;
    for (i, &m) in mag.iter().enumerate() {
        let x = i as f32 / max_i;
        let d = x - centroid;
        var += d * d * m;
    }
    (var / den).sqrt().clamp(0.0, 1.0)
}

/// Spectral contrast: average per-macro-band peak-vs-mean separation. A flat
/// broadband spectrum is near 0; spectra with isolated harmonic peaks are high.
pub fn spectral_contrast(mag: &[f32], bin_hz: f32) -> f32 {
    if mag.is_empty() || bin_hz <= 0.0 {
        return 0.0;
    }

    let mut acc = 0.0f32;
    let mut count = 0.0f32;
    for &(lo_hz, hi_hz) in &MACRO_BANDS_HZ {
        let lo = (lo_hz / bin_hz).floor().max(0.0) as usize;
        let hi = ((hi_hz / bin_hz).ceil() as usize).min(mag.len());
        if hi <= lo {
            continue;
        }
        let mut sum = 0.0f32;
        let mut peak = 0.0f32;
        let mut n = 0usize;
        for &m in &mag[lo..hi] {
            sum += m;
            peak = peak.max(m);
            n += 1;
        }
        if sum <= 0.0 || n == 0 {
            continue;
        }
        let mean = sum / n as f32;
        let ratio = peak / mean.max(1e-12);
        acc += ((ratio - 1.0) / 8.0).clamp(0.0, 1.0);
        count += 1.0;
    }

    if count > 0.0 {
        acc / count
    } else {
        0.0
    }
}

/// Convert a frequency (Hz) to the nearest FFT bin index for `fft_len` @ `sample_rate`.
#[inline]
pub fn hz_to_bin(hz: f32, fft_len: usize, sample_rate: f32) -> usize {
    let bin_hz = sample_rate / fft_len as f32;
    (hz / bin_hz).round().max(0.0) as usize
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hann_window_endpoints_are_zero() {
        let w = hann_window(8);
        assert!(w[0].abs() < 1e-6);
        // Periodic Hann is zero at the start and symmetric-ish, peak near middle.
        let max = w.iter().cloned().fold(0.0f32, f32::max);
        assert!(max > 0.99);
    }

    #[test]
    fn filterbank_isolates_a_tone() {
        // 1024-pt FFT @ 48k → bin_hz = 46.875. Put energy only in the 'bass' band
        // (60-150 Hz, bins ~1-3) and confirm only that macro band lights up.
        let fft_len = 1024;
        let sr = 48_000.0;
        let n_bins = fft_len / 2 + 1;
        let mut mag = vec![0.0f32; n_bins];
        // 100 Hz → bin ≈ 2.13 → bin 2
        mag[2] = 1.0;
        let bank = FilterBank::from_edges(&MACRO_BANDS_HZ, fft_len, sr);
        let mut out = vec![0.0f32; bank.len()];
        bank.apply(&mag, &mut out);
        // bass (index 1) should dominate.
        let bass = out[1];
        for (i, &v) in out.iter().enumerate() {
            if i != 1 {
                assert!(bass >= v, "band {i}={v} exceeded bass={bass}");
            }
        }
        assert!(bass > 0.0);
    }

    #[test]
    fn log_filterbank_has_requested_count() {
        let bank = FilterBank::log_spaced(
            SPECTRUM_BANDS,
            SPECTRUM_LO_HZ,
            SPECTRUM_HI_HZ,
            2048,
            44_100.0,
        );
        assert_eq!(bank.len(), SPECTRUM_BANDS);
    }

    #[test]
    fn centroid_low_for_bass_high_for_treble() {
        let mut lo = vec![0.0f32; 100];
        lo[2] = 1.0;
        let mut hi = vec![0.0f32; 100];
        hi[90] = 1.0;
        let bin_hz = 48_000.0 / 1024.0;
        let c_lo = spectral_centroid(&lo, bin_hz, 48_000.0);
        let c_hi = spectral_centroid(&hi, bin_hz, 48_000.0);
        assert!(c_hi > c_lo);
    }

    #[test]
    fn flux_only_counts_increases() {
        let prev = vec![0.5, 0.5, 0.5];
        let cur = vec![1.0, 0.0, 0.5]; // +0.5, -0.5, 0
        assert!((spectral_flux(&prev, &cur) - 0.5).abs() < 1e-6);
    }

    #[test]
    fn superflux_suppresses_small_frequency_shifts() {
        let mut prev = vec![0.0f32; 16];
        let mut shifted = vec![0.0f32; 16];
        let mut attack = vec![0.0f32; 16];
        prev[6] = 1.0;
        shifted[7] = 1.0;
        attack[7] = 2.0;

        let plain_shift = spectral_flux(&prev, &shifted);
        let super_shift = superflux(&prev, &shifted, 1);
        let super_attack = superflux(&prev, &attack, 1);

        assert!(plain_shift > 0.9);
        assert!(
            super_shift < 1e-6,
            "neighbor shift should be absorbed by max filter: {super_shift}"
        );
        assert!(super_attack > 0.0);
    }

    #[test]
    fn rolloff_monotonic_with_energy_spread() {
        // All energy at low bin → low rolloff.
        let mut m = vec![0.0f32; 100];
        m[1] = 1.0;
        assert!(spectral_rolloff(&m, 0.85) < 0.1);
        // Energy at high bin → high rolloff.
        let mut m2 = vec![0.0f32; 100];
        m2[95] = 1.0;
        assert!(spectral_rolloff(&m2, 0.85) > 0.9);
    }

    #[test]
    fn flatness_spread_and_contrast_separate_noise_from_tones() {
        let flat = vec![1.0f32; 128];
        let mut tone = vec![0.0f32; 128];
        tone[32] = 1.0;
        let mut wide = vec![0.0f32; 128];
        wide[10] = 1.0;
        wide[100] = 1.0;

        assert!(spectral_flatness(&flat) > 0.99);
        assert!(spectral_flatness(&tone) < 0.05);
        assert!(spectral_spread(&wide) > spectral_spread(&tone));
        assert!(spectral_contrast(&tone, 100.0) > spectral_contrast(&flat, 100.0));
    }
}

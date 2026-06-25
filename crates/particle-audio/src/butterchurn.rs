//! Butterchurn-parity bass/mid/treb + attenuated (`_att`) audio envelopes.
//!
//! MilkDrop presets read `bass`, `mid`, `treb` and their slow-follow companions
//! `bass_att`, `mid_att`, `treb_att` (plus `vol`/`vol_att`). For visual parity
//! with the reference WebGL renderer (Butterchurn), this crate must normalize these
//! exactly the way Butterchurn does — values that hover around `1.0` because each
//! frame's band energy is divided by a long-term running average of that band.
//!
//! This module is a faithful, line-for-line transcription of Butterchurn's
//! audio path. The reference sources (vendored at
//! `/private/tmp/butterchurn-compare/src/audio/`) are:
//!
//! * `fft.js`        — the 1024-point decimation-in-time FFT with the `equalize`
//!                     magnitude curve. We port the exact algorithm so band
//!                     magnitudes are bit-comparable, not just "close".
//! * `audioProcessor.js` — `numSamps = 512`, `fftSize = numSamps * 2 = 1024`.
//!                     Time-domain bytes (`getByteTimeDomainData`, unsigned
//!                     `0..255` centered at `128`) are shifted to signed about 0
//!                     (`byte - 128`), and the *full-width* signed samples feed
//!                     the FFT (`fft.timeToFrequencyDomain(timeArray)`).
//! * `audioLevels.js` — the three frequency bands, the immediate band sum, the
//!                     short (`avg`) and long (`longAvg`) followers, and the
//!                     `val = imm / longAvg` / `att = avg / longAvg` normalization.
//!
//! ## Band split (audioLevels.js constructor)
//! With `bucketHz = sampleRate / fftSize`:
//! * bass: bins `[round(20/bucketHz)-1, round(320/bucketHz)-1)` → ~20–320 Hz
//! * mid:  bins `[round(320/bucketHz)-1, round(2800/bucketHz)-1)` → ~320–2800 Hz
//! * treb: bins `[round(2800/bucketHz)-1, round(11025/bucketHz)-1)` → ~2800–11025 Hz
//! At 44.1 kHz / fftSize 1024 this is `starts=[0,6,64]`, `stops=[6,64,255]`.
//!
//! ## Normalization (audioLevels.js updateAudioLevels)
//! Per band `i`:
//! ```text
//!   imm[i] = Σ freqArray[starts[i]..stops[i]]
//!   rate   = (imm[i] > avg[i] ? 0.2 : 0.5) ^ (30 / fps)      // short follower
//!   avg[i] = avg[i]*rate + imm[i]*(1-rate)
//!   rate   = (frame < 50 ? 0.9 : 0.992) ^ (30 / fps)         // long follower
//!   longAvg[i] = longAvg[i]*rate + imm[i]*(1-rate)
//!   if longAvg[i] < 0.001 { val=1; att=1 }
//!   else { val[i] = imm[i]/longAvg[i];  att[i] = avg[i]/longAvg[i] }
//! ```
//! The long-follower decay coefficient is **0.992** (0.9 during the first 50
//! frames of warmup), FPS-adjusted by `rate^(30/fps)`. `att`, `avg`, `longAvg`
//! all start at `1.0`.
//!
//! `vol`/`vol_att` are not in audioLevels.js; MilkDrop derives them as the mean
//! of the three bands: `vol = (bass+mid+treb)/3`, `vol_att = (bass_att+mid_att+treb_att)/3`.

/// Number of time-domain samples consumed per analysis frame (`audioProcessor.js`).
pub const NUM_SAMPS: usize = 512;
/// FFT size (`audioProcessor.js`: `numSamps * 2`).
pub const FFT_SIZE: usize = NUM_SAMPS * 2;

/// Butterchurn's reference sample rate when no Web Audio context is present
/// (`audioLevels.js`: `sampleRate = 44100`).
pub const DEFAULT_SAMPLE_RATE: f32 = 44_100.0;

/// Full set of normalized MilkDrop audio rails for one frame.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct AudioLevels {
    pub bass: f32,
    pub mid: f32,
    pub treb: f32,
    pub bass_att: f32,
    pub mid_att: f32,
    pub treb_att: f32,
    /// `(bass + mid + treb) / 3` — MilkDrop `vol`.
    pub vol: f32,
    /// `(bass_att + mid_att + treb_att) / 3` — MilkDrop `vol_att`.
    pub vol_att: f32,
}

/// 1024-point decimation-in-time FFT with the `equalize` magnitude curve.
///
/// Direct port of Butterchurn's `fft.js` (`equalize = true`, `samplesIn = 1024`,
/// `samplesOut = 512`). Ported rather than delegated to `realfft` so the per-bin
/// magnitudes are bit-for-bit comparable with the reference renderer.
struct Fft {
    /// `samplesIn` — number of input samples actually copied (the rest are zero).
    samples_in: usize,
    /// `samplesOut` — number of output magnitude bins.
    samples_out: usize,
    /// `NFREQ = samplesOut * 2` — the transform length.
    nfreq: usize,
    bitrevtable: Vec<usize>,
    /// `cossintable[0]` = cos, `cossintable[1]` = sin, per DFT stage.
    cos_table: Vec<f32>,
    sin_table: Vec<f32>,
    equalize: Vec<f32>,
    real: Vec<f32>,
    imag: Vec<f32>,
}

impl Fft {
    fn new(samples_in: usize, samples_out: usize) -> Self {
        let nfreq = samples_out * 2;

        // initEqualizeTable: -0.02 * log((samplesOut - i) / samplesOut)
        let inv_half = 1.0 / samples_out as f32;
        let equalize = (0..samples_out)
            .map(|i| -0.02 * (((samples_out - i) as f32) * inv_half).ln())
            .collect();

        // initBitRevTable
        let mut bitrevtable: Vec<usize> = (0..nfreq).collect();
        let mut j = 0usize;
        for i in 0..nfreq {
            if j > i {
                bitrevtable.swap(i, j);
            }
            let mut m = nfreq >> 1;
            while m >= 1 && j >= m {
                j -= m;
                m >>= 1;
            }
            j += m;
        }

        // initCosSinTable
        let mut cos_table = Vec::new();
        let mut sin_table = Vec::new();
        let mut dftsize = 2usize;
        while dftsize <= nfreq {
            let theta = -2.0 * std::f32::consts::PI / dftsize as f32;
            cos_table.push(theta.cos());
            sin_table.push(theta.sin());
            dftsize <<= 1;
        }

        Self {
            samples_in,
            samples_out,
            nfreq,
            bitrevtable,
            cos_table,
            sin_table,
            equalize,
            real: vec![0.0; nfreq],
            imag: vec![0.0; nfreq],
        }
    }

    /// Port of `timeToFrequencyDomain`. `wave_data_in` is the signed time-domain
    /// data; only the first `samples_in` entries (bit-reversed) are loaded, the
    /// rest are zero. Writes `samples_out` equalized magnitudes into `out`.
    fn time_to_frequency_domain(&mut self, wave_data_in: &[f32], out: &mut [f32]) {
        for i in 0..self.nfreq {
            let idx = self.bitrevtable[i];
            self.real[i] = if idx < self.samples_in {
                wave_data_in[idx]
            } else {
                0.0
            };
            self.imag[i] = 0.0;
        }

        let mut dftsize = 2usize;
        let mut t = 0usize;
        while dftsize <= self.nfreq {
            let wpr = self.cos_table[t];
            let wpi = self.sin_table[t];
            let mut wr = 1.0f32;
            let mut wi = 0.0f32;
            let hdftsize = dftsize >> 1;

            for m in 0..hdftsize {
                let mut i = m;
                while i < self.nfreq {
                    let jj = i + hdftsize;
                    let tempr = wr * self.real[jj] - wi * self.imag[jj];
                    let tempi = wr * self.imag[jj] + wi * self.real[jj];
                    self.real[jj] = self.real[i] - tempr;
                    self.imag[jj] = self.imag[i] - tempi;
                    self.real[i] += tempr;
                    self.imag[i] += tempi;
                    i += dftsize;
                }
                let wtemp = wr;
                wr = wtemp * wpr - wi * wpi;
                wi = wi * wpr + wtemp * wpi;
            }

            dftsize <<= 1;
            t += 1;
        }

        for i in 0..self.samples_out {
            out[i] =
                self.equalize[i] * (self.real[i] * self.real[i] + self.imag[i] * self.imag[i]).sqrt();
        }
    }
}

/// Stateful Butterchurn audio-levels follower. One per analysis stream.
///
/// Mirrors `AudioProcessor` (FFT + signed time-domain conversion) and
/// `AudioLevels` (band split + immediate/short/long followers). Feed it the
/// per-frame unsigned `0..255` time-domain bytes (centered at 128) via
/// [`ButterchurnLevels::update_bytes`], or already-signed `-128..127` samples via
/// [`ButterchurnLevels::update_signed`].
pub struct ButterchurnLevels {
    fft: Fft,
    freq_array: Vec<f32>,
    time_signed: Vec<f32>,

    starts: [usize; 3],
    stops: [usize; 3],

    imm: [f32; 3],
    avg: [f32; 3],
    long_avg: [f32; 3],
    val: [f32; 3],
    att: [f32; 3],
}

impl ButterchurnLevels {
    /// Build a follower for the given sample rate (band bins depend on it).
    /// Pass [`DEFAULT_SAMPLE_RATE`] to match Butterchurn's context-less default.
    pub fn new(sample_rate: f32) -> Self {
        let fft = Fft::new(FFT_SIZE, NUM_SAMPS);

        // audioLevels.js band split.
        let bucket_hz = sample_rate / FFT_SIZE as f32;
        let clamp = |v: i64| v.clamp(0, NUM_SAMPS as i64 - 1) as usize;
        let band_bin = |hz: f32| clamp((hz / bucket_hz).round() as i64 - 1);
        let bass_low = band_bin(20.0);
        let bass_high = band_bin(320.0);
        let mid_high = band_bin(2800.0);
        let treb_high = band_bin(11025.0);

        Self {
            fft,
            freq_array: vec![0.0; NUM_SAMPS],
            time_signed: vec![0.0; FFT_SIZE],
            starts: [bass_low, bass_high, mid_high],
            stops: [bass_high, mid_high, treb_high],
            imm: [0.0; 3],
            // att/avg/longAvg start filled with 1.0 (audioLevels.js constructor).
            avg: [1.0; 3],
            long_avg: [1.0; 3],
            val: [0.0; 3],
            att: [1.0; 3],
        }
    }

    /// Band bin ranges actually in use (`starts`, `stops`), for tests/diagnostics.
    pub fn band_ranges(&self) -> ([usize; 3], [usize; 3]) {
        (self.starts, self.stops)
    }

    /// Update from unsigned time-domain bytes (`getByteTimeDomainData`, `0..255`
    /// centered at 128), exactly as Butterchurn consumes Web Audio data.
    pub fn update_bytes(&mut self, time_byte_array: &[u8], fps: f32, frame: u64) -> AudioLevels {
        // processAudio: timeArray[i] = timeByteArray[i] - 128 (signed about 0).
        for (dst, &b) in self.time_signed.iter_mut().zip(time_byte_array.iter()) {
            *dst = b as f32 - 128.0;
        }
        // Zero-fill any tail if the input is shorter than FFT_SIZE.
        for dst in self.time_signed.iter_mut().skip(time_byte_array.len()) {
            *dst = 0.0;
        }
        self.run(fps, frame)
    }

    /// Update from already-signed time-domain samples (`-128..127` scale, the
    /// post-`-128` convention). Useful when the source is `f32` PCM scaled to the
    /// same range as Butterchurn's signed bytes.
    pub fn update_signed(&mut self, time_signed: &[f32], fps: f32, frame: u64) -> AudioLevels {
        let n = time_signed.len().min(FFT_SIZE);
        self.time_signed[..n].copy_from_slice(&time_signed[..n]);
        for dst in self.time_signed.iter_mut().skip(n) {
            *dst = 0.0;
        }
        self.run(fps, frame)
    }

    fn run(&mut self, fps: f32, frame: u64) -> AudioLevels {
        // processAudio: freqArray = fft.timeToFrequencyDomain(timeArray).
        self.fft
            .time_to_frequency_domain(&self.time_signed, &mut self.freq_array);

        // updateAudioLevels: clamp effective FPS to [15, 144].
        let mut effective_fps = fps;
        if !effective_fps.is_finite() || effective_fps < 15.0 {
            effective_fps = 15.0;
        } else if effective_fps > 144.0 {
            effective_fps = 144.0;
        }

        // imm[i] = Σ freqArray[starts[i]..stops[i]]
        self.imm = [0.0; 3];
        for i in 0..3 {
            let mut acc = 0.0f32;
            for j in self.starts[i]..self.stops[i] {
                acc += self.freq_array[j];
            }
            self.imm[i] = acc;
        }

        for i in 0..3 {
            // Short follower.
            let rate = if self.imm[i] > self.avg[i] { 0.2 } else { 0.5 };
            let rate = adjust_rate_to_fps(rate, 30.0, effective_fps);
            self.avg[i] = self.avg[i] * rate + self.imm[i] * (1.0 - rate);

            // Long follower (warmup 0.9 for first 50 frames, then 0.992).
            let rate = if frame < 50 { 0.9 } else { 0.992 };
            let rate = adjust_rate_to_fps(rate, 30.0, effective_fps);
            self.long_avg[i] = self.long_avg[i] * rate + self.imm[i] * (1.0 - rate);

            if self.long_avg[i] < 0.001 {
                self.val[i] = 1.0;
                self.att[i] = 1.0;
            } else {
                self.val[i] = self.imm[i] / self.long_avg[i];
                self.att[i] = self.avg[i] / self.long_avg[i];
            }
        }

        let bass = self.val[0];
        let mid = self.val[1];
        let treb = self.val[2];
        let bass_att = self.att[0];
        let mid_att = self.att[1];
        let treb_att = self.att[2];
        AudioLevels {
            bass,
            mid,
            treb,
            bass_att,
            mid_att,
            treb_att,
            vol: (bass + mid + treb) / 3.0,
            vol_att: (bass_att + mid_att + treb_att) / 3.0,
        }
    }
}

/// `adjustRateToFPS(rate, baseFPS, FPS) = rate ** (baseFPS / FPS)`.
#[inline]
fn adjust_rate_to_fps(rate: f32, base_fps: f32, fps: f32) -> f32 {
    rate.powf(base_fps / fps)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The band split at 44.1 kHz / fftSize 1024 must match Butterchurn's
    /// `starts=[0,6,64]`, `stops=[6,64,255]` (verified against audioLevels.js).
    #[test]
    fn band_ranges_match_butterchurn() {
        let levels = ButterchurnLevels::new(DEFAULT_SAMPLE_RATE);
        let (starts, stops) = levels.band_ranges();
        assert_eq!(starts, [0, 6, 64], "band starts");
        assert_eq!(stops, [6, 64, 255], "band stops");
    }

    /// Initial state: with no input the long average starts at 1.0 and decays;
    /// the very first frame (`imm = 0`, `longAvg = 1`) yields `att = avg/longAvg`.
    /// Reference (Butterchurn JS over silent bytes, fps=60):
    ///   frame 0 → bass_att = 0.745356; frame 1 → 0.555556; frame 5 → 0.171468.
    #[test]
    fn silent_warmup_attenuation_matches_reference() {
        let silent = [128u8; FFT_SIZE];
        let fps = 60.0;
        let expected_att = [(0u64, 0.7453559637f32), (1, 0.5555555224), (5, 0.1714677513)];
        for (frame, want) in expected_att {
            // The follower is stateful, so replay from frame 0 each time.
            let mut l = ButterchurnLevels::new(DEFAULT_SAMPLE_RATE);
            let mut out = AudioLevels::default();
            for f in 0..=frame {
                out = l.update_bytes(&silent, fps, f);
            }
            assert!(out.bass.abs() < 1e-6, "silent bass must be 0");
            assert!(
                (out.bass_att - want).abs() < 1e-5,
                "frame {frame}: bass_att {} != {want}",
                out.bass_att
            );
        }
    }

    /// A steady tone drives the long average toward the immediate energy, so
    /// `val` (bass/mid/treb) converges toward ~1.0 — the whole point of the
    /// "divide by running average" normalization.
    #[test]
    fn steady_tone_normalizes_toward_unity() {
        let mut levels = ButterchurnLevels::new(DEFAULT_SAMPLE_RATE);
        // ~120 Hz sine in the signed -128..127 domain → energy in the bass band.
        let sr = DEFAULT_SAMPLE_RATE;
        let freq = 120.0;
        let mut last = AudioLevels::default();
        for frame in 0..400u64 {
            let mut bytes = [128u8; FFT_SIZE];
            for (i, b) in bytes.iter_mut().enumerate() {
                let phase =
                    (frame as f32 * NUM_SAMPS as f32 + i as f32) / sr * freq * std::f32::consts::TAU;
                let s = (phase.sin() * 100.0).round();
                *b = (128.0 + s).clamp(0.0, 255.0) as u8;
            }
            last = levels.update_bytes(&bytes, 60.0, frame);
        }
        // Once the long average has tracked the steady energy, bass hovers near 1.
        assert!(
            (last.bass - 1.0).abs() < 0.25,
            "steady bass should normalize near 1.0, got {}",
            last.bass
        );
        assert!(last.bass > last.treb, "bass tone should exceed treble");
    }

    /// vol/vol_att are the mean of the three bands.
    #[test]
    fn vol_is_mean_of_bands() {
        let mut levels = ButterchurnLevels::new(DEFAULT_SAMPLE_RATE);
        let silent = [128u8; FFT_SIZE];
        let out = levels.update_bytes(&silent, 60.0, 0);
        let mean_att = (out.bass_att + out.mid_att + out.treb_att) / 3.0;
        assert!((out.vol_att - mean_att).abs() < 1e-6);
        assert!((out.vol - (out.bass + out.mid + out.treb) / 3.0).abs() < 1e-6);
    }
}

//! Rolling STFT spectrogram-trail history (public scrolling-spectrogram source).
//!
//! Keeps the last `N` STFT frames, each reduced to `M` **log-spaced** frequency
//! bins of normalized **log-magnitude** (`0..1`), in a ring buffer. A scrolling
//! spectrogram / waterfall visual reads the ring each frame and scrolls one
//! column. Log-spaced bins (octave-ish spacing) and a dB magnitude axis are what
//! make a spectrogram read musically rather than crowding everything into the
//! lowest few linear bins.
//!
//! The frequency reduction reuses the analyzer's log-spaced triangular
//! [`FilterBank`](crate::analysis::FilterBank), so the bin centers line up with
//! the rest of the spectrum rails. The magnitude axis reuses the same
//! `lin_to_db_norm` mapping the rest of the analyzer uses, so a column here is
//! directly comparable to the other normalized rails.
//!
//! Real-time / causal: one column is appended per hop with no lookahead. Memory
//! is `frames * bins` floats, fixed at construction.

use crate::analysis::FilterBank;
use crate::dsp::lin_to_db_norm;

/// Default number of frames (columns) of scrollback history.
pub const DEFAULT_FRAMES: usize = 256;
/// Default number of log-spaced frequency bins (rows) per column.
pub const DEFAULT_BINS: usize = 128;
/// Low edge of the log-spaced spectrogram band (Hz).
pub const SPECTROGRAM_LO_HZ: f32 = 30.0;
/// High edge of the log-spaced spectrogram band (Hz).
pub const SPECTROGRAM_HI_HZ: f32 = 18_000.0;

/// Rolling log-magnitude spectrogram history.
///
/// Construct once with the FFT size / sample rate (so the log filterbank maps Hz
/// to bins correctly for any device rate), then call [`SpectrogramTrail::push`]
/// once per STFT magnitude frame. Read the history with [`SpectrogramTrail::column`]
/// (`age` back from the newest) or [`SpectrogramTrail::for_each_column`].
pub struct SpectrogramTrail {
    frames: usize,
    bins: usize,
    /// Ring of `frames` columns, each `bins` normalized log-magnitudes, frame-major.
    ring: Vec<f32>,
    /// Index of the slot the NEXT push will write (i.e. one past the newest).
    write: usize,
    /// Columns written so far (saturates at `frames`).
    filled: usize,
    /// Log-spaced triangular filterbank that reduces the linear FFT magnitude
    /// spectrum to `bins` log-frequency rows.
    bank: FilterBank,
    /// Scratch for one reduced column (length `bins`), reused per push.
    scratch: Vec<f32>,
}

impl SpectrogramTrail {
    /// Build a trail with the default frame / bin counts.
    pub fn new(fft_len: usize, sample_rate: f32) -> Self {
        Self::with_size(DEFAULT_FRAMES, DEFAULT_BINS, fft_len, sample_rate)
    }

    /// Build a trail with explicit `frames` (columns) × `bins` (rows).
    pub fn with_size(frames: usize, bins: usize, fft_len: usize, sample_rate: f32) -> Self {
        let frames = frames.max(1);
        let bins = bins.max(1);
        let bank = FilterBank::log_spaced(
            bins,
            SPECTROGRAM_LO_HZ,
            SPECTROGRAM_HI_HZ,
            fft_len,
            sample_rate,
        );
        Self {
            frames,
            bins,
            ring: vec![0.0; frames * bins],
            write: 0,
            filled: 0,
            bank,
            scratch: vec![0.0; bins],
        }
    }

    /// Number of columns (frames) of scrollback.
    pub fn frames(&self) -> usize {
        self.frames
    }

    /// Number of log-spaced frequency bins (rows) per column.
    pub fn bins(&self) -> usize {
        self.bins
    }

    /// Number of columns written so far (≤ [`frames`](Self::frames)).
    pub fn filled(&self) -> usize {
        self.filled
    }

    /// Reduce one linear STFT magnitude frame to a log-spaced log-magnitude
    /// column and append it (advancing the ring). `is_silent` writes a zeroed
    /// column so the trail visibly drains during gated silence.
    pub fn push(&mut self, mag: &[f32], is_silent: bool) {
        let offset = self.write * self.bins;
        if is_silent {
            for v in &mut self.ring[offset..offset + self.bins] {
                *v = 0.0;
            }
        } else {
            self.bank.apply(mag, &mut self.scratch);
            for (dst, &lin) in self.ring[offset..offset + self.bins]
                .iter_mut()
                .zip(self.scratch.iter())
            {
                *dst = lin_to_db_norm(lin);
            }
        }
        self.write = (self.write + 1) % self.frames;
        self.filled = (self.filled + 1).min(self.frames);
    }

    /// Borrow one column by `age`: `age = 0` is the newest column just pushed,
    /// `age = frames-1` the oldest retained. Returns `None` if `age` is out of
    /// range or that column has not been written yet.
    pub fn column(&self, age: usize) -> Option<&[f32]> {
        if age >= self.frames || age >= self.filled {
            return None;
        }
        // `write` points one past the newest, so newest = write - 1.
        let idx = (self.write + self.frames - 1 - age) % self.frames;
        let offset = idx * self.bins;
        Some(&self.ring[offset..offset + self.bins])
    }

    /// Visit every written column newest-first (`age` 0, 1, 2, …). Convenient for
    /// uploading the trail to a texture or drawing a waterfall without exposing
    /// the ring layout.
    pub fn for_each_column<F: FnMut(usize, &[f32])>(&self, mut f: F) {
        for age in 0..self.filled {
            if let Some(col) = self.column(age) {
                f(age, col);
            }
        }
    }

    /// Raw ring storage (frame-major, `frames * bins`) plus the write cursor, for
    /// callers that want to upload the whole buffer once and unwrap the ring on
    /// the GPU. `write` is the index of the slot the next push will overwrite, so
    /// the oldest valid column (once full) is at `write` and the newest at
    /// `(write + frames - 1) % frames`.
    pub fn raw_ring(&self) -> (&[f32], usize) {
        (&self.ring, self.write)
    }

    /// Copy the current ring + cursor into a reusable [`SpectrogramSnapshot`]
    /// without allocating, so the DSP worker can hand a stable frame to the
    /// render thread for GPU upload. The snapshot must already be sized to this
    /// trail (same `frames`/`bins`); callers obtain one via [`Self::snapshot`].
    pub fn fill_snapshot(&self, dst: &mut SpectrogramSnapshot) {
        dst.frames = self.frames;
        dst.bins = self.bins;
        dst.write = self.write;
        dst.filled = self.filled;
        // Ring is fixed-size after construction, so this is a straight memcpy into
        // the preallocated snapshot storage — no allocation on the audio thread.
        dst.ring.copy_from_slice(&self.ring);
    }

    /// Allocate a [`SpectrogramSnapshot`] sized to this trail (call once, then
    /// reuse it across [`Self::fill_snapshot`] calls).
    pub fn snapshot(&self) -> SpectrogramSnapshot {
        SpectrogramSnapshot {
            frames: self.frames,
            bins: self.bins,
            write: self.write,
            filled: self.filled,
            ring: self.ring.clone(),
        }
    }
}

/// An owned, stable copy of a [`SpectrogramTrail`]'s ring buffer, suitable for
/// handing across threads (the DSP worker fills it, the render thread reads it
/// for GPU upload). Storage is allocated once and reused via
/// [`SpectrogramTrail::fill_snapshot`]; reading never allocates.
#[derive(Clone, Debug)]
pub struct SpectrogramSnapshot {
    frames: usize,
    bins: usize,
    write: usize,
    filled: usize,
    /// Ring of `frames` columns × `bins` rows, frame-major (mirrors the trail).
    ring: Vec<f32>,
}

impl SpectrogramSnapshot {
    /// Number of columns (frames) of scrollback.
    pub fn frames(&self) -> usize {
        self.frames
    }

    /// Number of log-spaced frequency bins (rows) per column.
    pub fn bins(&self) -> usize {
        self.bins
    }

    /// Columns written so far (≤ [`frames`](Self::frames)).
    pub fn filled(&self) -> usize {
        self.filled
    }

    /// Raw ring storage (frame-major, `frames * bins`) plus the write cursor.
    /// `write` is the slot the next push would overwrite; the newest column is at
    /// `(write + frames - 1) % frames` and the oldest valid (once full) at `write`.
    /// Upload `ring` as a `frames × bins` texture and unwrap the ring on the GPU
    /// using `write`.
    pub fn raw_ring(&self) -> (&[f32], usize) {
        (&self.ring, self.write)
    }

    /// Borrow one column by `age` (0 = newest just pushed). Mirrors
    /// [`SpectrogramTrail::column`]. Returns `None` if out of range / unwritten.
    pub fn column(&self, age: usize) -> Option<&[f32]> {
        if age >= self.frames || age >= self.filled {
            return None;
        }
        let idx = (self.write + self.frames - 1 - age) % self.frames;
        let offset = idx * self.bins;
        Some(&self.ring[offset..offset + self.bins])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const SAMPLE_RATE: f32 = 48_000.0;
    const FFT_LEN: usize = 2048;
    const N_BINS: usize = FFT_LEN / 2 + 1;

    fn linear_bin_for_hz(hz: f32) -> usize {
        (hz / (SAMPLE_RATE / FFT_LEN as f32)).round() as usize
    }

    /// Pushing advances the ring and `column(0)` returns the most recent frame.
    #[test]
    fn ring_advances_and_newest_is_age_zero() {
        let mut trail = SpectrogramTrail::with_size(8, 16, FFT_LEN, SAMPLE_RATE);
        assert_eq!(trail.filled(), 0);
        assert!(trail.column(0).is_none(), "empty trail has no columns");

        // Push a frame with energy only in the lowest band, then one with energy
        // only high — the newest (age 0) must reflect the last push.
        let mut low = vec![0.0f32; N_BINS];
        low[linear_bin_for_hz(60.0)] = 1.0;
        let mut high = vec![0.0f32; N_BINS];
        high[linear_bin_for_hz(12_000.0)] = 1.0;

        trail.push(&low, false);
        assert_eq!(trail.filled(), 1);
        trail.push(&high, false);
        assert_eq!(trail.filled(), 2);

        let newest = trail.column(0).expect("newest column");
        let prev = trail.column(1).expect("previous column");
        // Newest came from the high-frequency push: its peak is in the upper bins.
        let newest_peak = argmax(newest);
        let prev_peak = argmax(prev);
        assert!(
            newest_peak > prev_peak,
            "newest (high) column should peak above previous (low): newest={newest_peak}, prev={prev_peak}"
        );
    }

    /// A known tone lands in the expected log-spaced bin, and that bin is the
    /// column's peak.
    #[test]
    fn known_tone_lands_in_expected_bin() {
        let bins = 64;
        let mut trail = SpectrogramTrail::with_size(4, bins, FFT_LEN, SAMPLE_RATE);

        // 1 kHz tone. Its expected row is the log-spaced bin whose band contains
        // 1 kHz within [lo, hi].
        let tone_hz = 1_000.0;
        let mut mag = vec![0.0f32; N_BINS];
        mag[linear_bin_for_hz(tone_hz)] = 1.0;
        mag[linear_bin_for_hz(tone_hz) - 1] = 0.4;
        mag[linear_bin_for_hz(tone_hz) + 1] = 0.4;
        trail.push(&mag, false);

        let col = trail.column(0).expect("column");
        let peak = argmax(col);

        // Expected log-spaced row for this Hz (matches FilterBank::log_spaced).
        let expected = expected_log_bin(tone_hz, bins);
        let diff = peak.abs_diff(expected);
        assert!(
            diff <= 1,
            "1 kHz should peak at log bin ~{expected}, got {peak} (col={col:?})"
        );
        // And it should be a real, non-trivial peak.
        assert!(
            col[peak] > 0.5,
            "tone column peak should be strong: {}",
            col[peak]
        );
    }

    /// `is_silent` writes a zeroed column so the waterfall drains.
    #[test]
    fn silence_writes_blank_column() {
        let mut trail = SpectrogramTrail::with_size(4, 16, FFT_LEN, SAMPLE_RATE);
        let mut mag = vec![0.0f32; N_BINS];
        mag[linear_bin_for_hz(1_000.0)] = 1.0;
        trail.push(&mag, false);
        trail.push(&mag, true);

        let blank = trail.column(0).expect("blank column");
        assert!(
            blank.iter().all(|&v| v == 0.0),
            "silent column should be all zero: {blank:?}"
        );
    }

    /// The ring wraps: once full, only `frames` columns are retained and the
    /// oldest pushes are forgotten.
    #[test]
    fn ring_wraps_and_caps_history() {
        let frames = 4;
        let mut trail = SpectrogramTrail::with_size(frames, 8, FFT_LEN, SAMPLE_RATE);
        let mut mag = vec![0.0f32; N_BINS];
        mag[linear_bin_for_hz(1_000.0)] = 1.0;
        for _ in 0..(frames * 3) {
            trail.push(&mag, false);
        }
        assert_eq!(trail.filled(), frames, "history caps at `frames`");
        assert!(trail.column(frames).is_none(), "no column beyond capacity");

        let mut visited = 0;
        trail.for_each_column(|_, _| visited += 1);
        assert_eq!(visited, frames);
    }

    fn argmax(v: &[f32]) -> usize {
        let mut best = 0usize;
        for i in 1..v.len() {
            if v[i] > v[best] {
                best = i;
            }
        }
        best
    }

    /// Mirror of `FilterBank::log_spaced` center spacing: band `b` centers on the
    /// log-spaced point `(b+1)/(count+1)` between lo and hi.
    fn expected_log_bin(hz: f32, count: usize) -> usize {
        let log_lo = SPECTROGRAM_LO_HZ.ln();
        let log_hi = SPECTROGRAM_HI_HZ.ln();
        let frac = (hz.ln() - log_lo) / (log_hi - log_lo);
        // center of band b is at (b+1)/(count+1) → b ≈ frac*(count+1) - 1
        let b = frac * (count as f32 + 1.0) - 1.0;
        b.round().clamp(0.0, (count - 1) as f32) as usize
    }
}

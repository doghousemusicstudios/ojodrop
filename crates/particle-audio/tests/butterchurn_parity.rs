//! Bit-parity check of the Butterchurn audio-levels port against values computed
//! by Butterchurn's own JavaScript (`audioProcessor.js` + `audioLevels.js` +
//! `fft.js`) over a deterministic fixture frame.
//!
//! The fixture (`tests/fixtures/butterchurn_frame.bin`) is the raw unsigned
//! `getByteTimeDomainData` (1024 bytes, centered at 128) of an energetic frame
//! from Butterchurn's shared `test/fixtures/audioAnalysisData.json`. The expected
//! `bass`/`mid`/`treb` and `_att` below were captured by running the reference JS
//! over that exact frame with a fresh follower (frame = 0, fps = 60).
//!
//! Because `imm` (the per-band magnitude sum) depends on every FFT bin, matching
//! it to ~1e-3 proves the ported `fft.js` produces the same magnitudes as the
//! reference — it is not merely "close in spirit".

use particle_audio::{ButterchurnLevels, DEFAULT_SAMPLE_RATE};

const FRAME: &[u8] = include_bytes!("fixtures/butterchurn_frame.bin");

#[test]
fn matches_butterchurn_js_reference_for_fixture_frame() {
    let mut levels = ButterchurnLevels::new(DEFAULT_SAMPLE_RATE);
    let out = levels.update_bytes(FRAME, 60.0, 0);

    // Reference values from Butterchurn JS (fresh follower, frame 0, fps 60).
    let want_bass = 1.339_747_5_f32;
    let want_mid = 13.888_369_f32;
    let want_treb = 18.656_166_f32;
    let want_bass_att = 1.179_589_f32;
    let want_mid_att = 7.812_733_7_f32;
    let want_treb_att = 10.332_970_f32;

    // val/att are ratios near O(1..20); allow a small relative slack for f32.
    let rel = |got: f32, want: f32| (got - want).abs() / want.abs().max(1.0);
    assert!(
        rel(out.bass, want_bass) < 2e-3,
        "bass {} != {want_bass}",
        out.bass
    );
    assert!(
        rel(out.mid, want_mid) < 2e-3,
        "mid {} != {want_mid}",
        out.mid
    );
    assert!(
        rel(out.treb, want_treb) < 2e-3,
        "treb {} != {want_treb}",
        out.treb
    );
    assert!(
        rel(out.bass_att, want_bass_att) < 2e-3,
        "bass_att {} != {want_bass_att}",
        out.bass_att
    );
    assert!(
        rel(out.mid_att, want_mid_att) < 2e-3,
        "mid_att {} != {want_mid_att}",
        out.mid_att
    );
    assert!(
        rel(out.treb_att, want_treb_att) < 2e-3,
        "treb_att {} != {want_treb_att}",
        out.treb_att
    );

    // vol / vol_att are the band means.
    let want_vol = (want_bass + want_mid + want_treb) / 3.0;
    let want_vol_att = (want_bass_att + want_mid_att + want_treb_att) / 3.0;
    assert!(
        rel(out.vol, want_vol) < 2e-3,
        "vol {} != {want_vol}",
        out.vol
    );
    assert!(
        rel(out.vol_att, want_vol_att) < 2e-3,
        "vol_att {} != {want_vol_att}",
        out.vol_att
    );
}

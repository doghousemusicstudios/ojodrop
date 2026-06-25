//! Live audio meter — proves real capture + DSP works.
//!
//! Run it and play music (or speak into the mic):
//!
//! ```sh
//! cargo run --example meter -p particle-audio
//! ```
//!
//! It starts the [`AudioEngine`] (capture + DSP on background threads) and prints
//! ~30 frames of the macro bands, RMS, brightness, onsets, and tempo so a human
//! can confirm the numbers move with the audio. No dummy data: every value comes
//! from the live input device.

use std::io::Write;
use std::thread;
use std::time::Duration;

use particle_audio::AudioEngine;

fn main() {
    // Quiet, opt-in logging so device-selection lines are visible.
    let _ = env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info"))
        .try_init();

    let engine = match AudioEngine::new() {
        Ok(e) => e,
        Err(e) => {
            eprintln!("Failed to start audio engine: {e}");
            eprintln!("(No input device? Try granting microphone permission.)");
            std::process::exit(1);
        }
    };

    println!(
        "Capturing from: {}  @ {} Hz",
        engine.device_name(),
        engine.sample_rate()
    );
    println!("Play some audio. Printing ~30 frames (one every ~150 ms)...\n");

    // Header.
    println!(
        "{:>5} {:>5} {:>5} {:>5} {:>5} {:>5} | {:>5} {:>5} | {:>5} {:>5} {:>5} | {:>6} {:>5} {:>5} {:>4}",
        "sub", "bass", "lmid", "mid", "pres", "air", "rms", "brt", "kick", "snr", "hat", "bpm",
        "phase", "conf", "sil"
    );

    for _ in 0..30 {
        let f = engine.latest();
        println!(
            "{:>5.2} {:>5.2} {:>5.2} {:>5.2} {:>5.2} {:>5.2} | {:>5.2} {:>5.2} | \
             {:>5.2} {:>5.2} {:>5.2} | {:>6.1} {:>5.2} {:>5.2} {:>4.0}",
            f.sub_bass,
            f.bass,
            f.low_mid,
            f.mid,
            f.presence,
            f.air,
            f.rms_level,
            f.brightness,
            f.kick_onset,
            f.snare_onset,
            f.hat_onset,
            f.bpm,
            f.beat_phase,
            f.beat_confidence,
            f.is_silent,
        );
        let _ = std::io::stdout().flush();
        thread::sleep(Duration::from_millis(150));
    }

    println!("\nDone. (Engine stops and joins on drop.)");
}

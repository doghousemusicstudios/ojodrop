//! macOS CoreAudio process-tap loopback (spec §2, the "preferred" macOS path).
//!
//! This is the **system-mix loopback** route that needs no virtual device: on
//! macOS 14.6+ CoreAudio exposes `AudioHardwareCreateProcessTap` plus the
//! aggregate-device tap list (`kAudioAggregateDeviceTapListKey`) so an app can
//! capture the *post-volume* system output mix — the approach popularized by
//! [insidegui/AudioCap](https://github.com/insidegui/AudioCap). It requires the
//! TCC **audio-capture** entitlement at runtime (the app bundle must carry an
//! `Info.plist` `NSAudioCaptureUsageDescription`, and the user must approve the
//! one-time system prompt).
//!
//! Status: **structured stub.** The full process-tap dance (create a
//! system-output process tap, wrap it in a private aggregate device, install an
//! `AudioDeviceIOProc`, and bridge its render callback into the `rtrb` ring) is a
//! large, permission-gated chunk of `objc2-core-audio` FFI that cannot be
//! exercised in CI (no audio device, no TCC grant). Rather than ship an untested
//! native callback path, [`try_open_process_tap`] declines (returns the producer
//! unchanged) after a clear log line, so the caller's fallback chain proceeds to
//! the default input. Everything here is `#[cfg(target_os = "macos")]`-gated and
//! the TODO below marks exactly where the native callback slots in.
//!
//! Why this is safe to gate this way: cpal 0.18 already pulls in the entire
//! `objc2-core-audio` / `objc2-core-audio-types` / `objc2-foundation` stack
//! transitively (all MIT/Apache), so wiring the real tap later needs **no new
//! GPL/proprietary deps** — just the FFI calls listed in the TODO.

use rtrb::Producer;

use crate::CaptureFrame;

/// A live CoreAudio process-tap capture (aggregate device + IOProc).
///
/// When the native tap is implemented this owns the aggregate device's
/// `AudioDeviceIOProcID` + `AudioObjectID` so capture is torn down on `Drop`,
/// mirroring how cpal's `Stream` keeps capture alive. Empty for the stub.
#[allow(dead_code)]
pub struct ProcessTapCapture {
    pub sample_rate: u32,
    pub channels: u16,
    pub device_name: String,
}

/// Outcome of attempting the process-tap path.
///
/// `Opened` is reserved for the native-tap implementation (see the TODO in
/// [`try_open_process_tap`]); the stub only ever returns `Declined`, so
/// `dead_code` is expected here until the tap is wired.
#[allow(dead_code)]
pub enum TapOutcome {
    /// A live tap was opened; capture is running into the ring.
    Opened(ProcessTapCapture),
    /// Tap unavailable / unimplemented — caller should fall back. The producer is
    /// handed back unchanged so the next rung can reuse the same ring.
    Declined(Producer<CaptureFrame>),
}

/// Try to open the system-output process-tap loopback and stream raw frames into
/// `producer`.
///
/// On decline the producer is returned via [`TapOutcome::Declined`] so the caller
/// can open the default input with the same ring (no rebuild needed).
pub fn try_open_process_tap(producer: Producer<CaptureFrame>) -> TapOutcome {
    // TODO(macos-tap): implement the AudioCap-style process tap. The shape is:
    //   1. Build a CATapDescription for the system output (stereo, mixdown) and
    //      call `AudioHardwareCreateProcessTap` → tap AudioObjectID.
    //   2. Create a private aggregate device
    //      (`AudioHardwareCreateAggregateDevice`) whose description sets
    //      `kAudioAggregateDeviceIsPrivateKey = 1` and
    //      `kAudioAggregateDeviceTapListKey = [ <tap UID> ]`.
    //   3. Read the aggregate device's actual nominal sample rate + channel count
    //      (BlackHole/aggregate rates vary: 44.1k/48k/96k — never assume 48k).
    //   4. `AudioDeviceCreateIOProcID` + `AudioDeviceStart`; in the IOProc,
    //      preserve L/R, downmix to mono, and `producer.push(CaptureFrame { .. })`
    //      with NO alloc / NO locks (same realtime discipline as capture.rs).
    //   5. Keep the IOProcID + aggregate AudioObjectID in `ProcessTapCapture` and
    //      tear them down in `Drop` (`AudioDeviceStop`, destroy proc id, destroy
    //      aggregate, destroy tap).
    // All FFI is available via objc2-core-audio (already a transitive dep). Until
    // it is implemented and testable on a real 14.6+ machine with the TCC grant,
    // we decline cleanly so the fallback path takes over.
    log::info!(
        "particle-audio: macOS CoreAudio process-tap loopback not yet wired \
         (needs TCC audio-capture + Info.plist NSAudioCaptureUsageDescription); \
         falling back to default-input selection"
    );
    TapOutcome::Declined(producer)
}

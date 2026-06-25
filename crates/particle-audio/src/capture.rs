//! Real-time audio capture via cpal (spec §1/§2).
//!
//! Builds a realtime cpal **input** stream, preserves the first stereo pair before
//! downmixing to mono, and pushes raw frames into an `rtrb` SPSC ring. The cpal data
//! callback does **no allocation, no FFT, and no locks**: it only converts and pushes.
//! Dropped samples on a full ring are counted, not blocked on (a full ring means
//! the DSP worker stalled — better to drop than to glitch capture).
//!
//! ## Device selection (spec §2 — runtime-selected with fallback)
//!
//! When [`CaptureConfig::prefer_loopback`] is set, [`start`] walks a fallback
//! chain so the engine reacts to the *system mix* (whatever music is playing),
//! not the room mic:
//!
//! 1. **Virtual loopback device by name** — an input device whose name contains a
//!    known virtual-cable marker (`BlackHole`, `Loopback`, `Soundflower`,
//!    `Aggregate`, `Multi-Output`). This is the zero-permission, most-reliable
//!    path for a DJ/VJ rig, so it is tried **first**.
//! 2. **macOS CoreAudio process-tap loopback** (14.6+) — captures the post-volume
//!    system mix with no virtual device, but needs the TCC audio-capture grant.
//!    See [`crate::capture_macos`] (currently a structured stub that declines
//!    cleanly so the chain proceeds).
//! 3. **Default input device** (mic / line-in) — the universally available
//!    fallback, and the only path when `prefer_loopback` is unset.
//!
//! Other platforms have their loopback rungs stubbed with TODOs (Windows WASAPI
//! render-loopback, Linux PipeWire/Pulse `.monitor`) but always fall through to
//! the default input device, which captures real audio today.

use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{FromSample, Sample, SampleFormat, Stream, StreamConfig};
use rtrb::Producer;

use crate::{AudioError, CaptureFrame};

/// Substrings that mark an input device as a system-audio virtual loopback /
/// aggregate cable. Matched case-insensitively against the device name.
///
/// These are the common macOS/cross-platform virtual cables and aggregate
/// devices a DJ/VJ rig routes its master out through, so opening one as a normal
/// cpal input captures the post-volume system mix with **zero permissions**.
const LOOPBACK_NAME_MARKERS: &[&str] = &[
    "blackhole",
    "loopback",
    "soundflower",
    "aggregate",
    "multi-output",
];

/// Which physical/virtual path the capture stream ended up on.
///
/// Exposed so the app can show/log whether reactivity is driven by the system
/// mix (loopback/virtual) or the room mic, and branch UI accordingly.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum CaptureSource {
    /// A native OS system-mix loopback (macOS CoreAudio process-tap, future
    /// WASAPI render-loopback, future PipeWire `.monitor`). Post-volume desktop
    /// audio with no virtual cable.
    SystemLoopback,
    /// A virtual loopback / aggregate device opened by name (BlackHole, Loopback,
    /// Soundflower, an Aggregate or Multi-Output device). Also the system mix, via
    /// a user-installed cable.
    VirtualDevice,
    /// The default input device (microphone / line-in) — the room, not the mix.
    Microphone,
}

impl CaptureSource {
    /// `true` when this source carries the post-volume *system mix* rather than
    /// the room mic (so visuals follow the music even at low room volume).
    pub fn is_loopback(self) -> bool {
        matches!(
            self,
            CaptureSource::SystemLoopback | CaptureSource::VirtualDevice
        )
    }

    /// Short human-readable label for logs / UI (e.g. "system loopback").
    pub fn label(self) -> &'static str {
        match self {
            CaptureSource::SystemLoopback => "system loopback",
            CaptureSource::VirtualDevice => "virtual loopback device",
            CaptureSource::Microphone => "mic",
        }
    }
}

/// Capture configuration / preferences.
#[derive(Clone, Copy, Debug)]
pub struct CaptureConfig {
    /// Prefer a system-loopback source (post-volume desktop mix) over the mic
    /// when one can be found. When set, [`start`] tries, in order: a virtual
    /// loopback device by name → the platform native loopback (macOS process-tap)
    /// → the default input device. When unset, only the default input is used.
    pub prefer_loopback: bool,
}

impl Default for CaptureConfig {
    fn default() -> Self {
        // Default to loopback-preferred: the engine's whole purpose is to react to
        // the music playing, not the room mic. Falls back to the mic automatically
        // when no loopback/virtual source exists, so this is safe as a default.
        Self {
            prefer_loopback: true,
        }
    }
}

/// A live capture stream plus the device facts the DSP worker needs.
///
/// The returned [`Stream`] must be kept alive for capture to continue (cpal stops
/// capture when the stream is dropped), so the engine owns it.
pub struct Capture {
    /// Held only for its RAII lifetime: capture runs until this is dropped. The
    /// engine never reads it again, hence `dead_code` is expected and allowed.
    #[allow(dead_code)]
    pub stream: Stream,
    pub sample_rate: u32,
    /// Native channel count of the source (informational; downmix is to mono).
    #[allow(dead_code)]
    pub channels: u16,
    pub device_name: String,
    /// Which selection path produced this stream (loopback/virtual/mic).
    pub source: CaptureSource,
}

/// Start capturing into `producer`, selecting the source per `cfg` (spec §2).
///
/// On success returns a [`Capture`] whose `stream` is already `play()`-ing.
pub fn start(producer: Producer<CaptureFrame>, cfg: CaptureConfig) -> Result<Capture, AudioError> {
    let host = cpal::default_host();

    if cfg.prefer_loopback {
        // --- Fallback rung 1: virtual loopback device by name (zero-permission) ---
        if let Some((device, name)) = find_virtual_loopback_device(&host) {
            log::info!("particle-audio: found virtual loopback device '{name}'");
            // A virtual device is present and explicitly preferred: if opening it
            // fails, surface the error rather than silently dropping to the mic —
            // the operator chose loopback and should know it broke. (`producer` is
            // moved into the stream on success or consumed on a build failure.)
            return open_input_device(&device, name, CaptureSource::VirtualDevice, producer);
        }
        log::debug!(
            "particle-audio: no virtual loopback device found \
             (markers: {LOOPBACK_NAME_MARKERS:?}); trying native loopback"
        );

        // --- Fallback rung 2: platform native system loopback ---
        // Each platform either opens a native loopback and returns, or hands the
        // producer back so we drop to the default input below.
        let producer = match try_native_loopback(&host, producer) {
            Ok(cap) => return Ok(cap),
            Err(producer) => producer,
        };

        // --- Fallback rung 3: default input device (mic / line-in) ---
        return open_default_input(&host, producer);
    }

    // prefer_loopback unset: original behavior, default input device only.
    open_default_input(&host, producer)
}

/// Attempt the platform-native system-loopback path.
///
/// `Ok(cap)` on success; `Err(producer)` hands the (unconsumed) producer back so
/// the caller can fall through to the default input device with the same ring.
#[allow(unused_variables, unused_mut)]
fn try_native_loopback(
    host: &cpal::Host,
    mut producer: Producer<CaptureFrame>,
) -> Result<Capture, Producer<CaptureFrame>> {
    #[cfg(target_os = "macos")]
    {
        use crate::capture_macos::{try_open_process_tap, TapOutcome};
        match try_open_process_tap(producer) {
            TapOutcome::Opened(tap) => {
                // TODO(macos-tap): the stub never reaches here. When the real tap
                // lands it returns a live ProcessTapCapture; bridge its teardown
                // handle into `Capture` and report CaptureSource::SystemLoopback.
                // For now this arm is unreachable, but kept so the wiring is clear.
                let _ = tap;
                unreachable!("process-tap stub never opens");
            }
            TapOutcome::Declined(producer) => Err(producer),
        }
    }

    #[cfg(target_os = "windows")]
    {
        // TODO(windows-loopback): WASAPI render-loopback. cpal 0.18 can open a
        // render endpoint in loopback mode; alternatively use the `wasapi` crate
        // (MIT). No admin and no virtual device required. Open the default render
        // device's loopback capture stream here, build a `Capture` with
        // CaptureSource::SystemLoopback, and return Ok(cap). Until then, decline.
        log::info!(
            "particle-audio: WASAPI render-loopback not implemented yet; \
             using default input device"
        );
        Err(producer)
    }

    #[cfg(target_os = "linux")]
    {
        // TODO(linux-loopback): PipeWire/PulseAudio default-sink `.monitor` source.
        // Enumerate input devices for a `.monitor` (Pulse) or the sink monitor
        // (PipeWire), open it as a normal cpal input via `open_input_device` with
        // CaptureSource::SystemLoopback, and return Ok(cap). Until then, decline.
        log::info!(
            "particle-audio: PipeWire/Pulse .monitor loopback not implemented yet; \
             using default input device"
        );
        Err(producer)
    }

    #[cfg(not(any(target_os = "macos", target_os = "windows", target_os = "linux")))]
    {
        Err(producer)
    }
}

/// Find the first input device whose name matches a known virtual-loopback marker.
///
/// Case-insensitive substring match against [`LOOPBACK_NAME_MARKERS`]. Returns the
/// device and its resolved name. `None` if enumeration fails or nothing matches.
fn find_virtual_loopback_device(host: &cpal::Host) -> Option<(cpal::Device, String)> {
    let devices = match host.input_devices() {
        Ok(d) => d,
        Err(e) => {
            log::warn!("particle-audio: could not enumerate input devices: {e}");
            return None;
        }
    };

    for device in devices {
        let name = match device_name(&device) {
            Some(n) => n,
            None => continue,
        };
        if is_loopback_name(&name) {
            return Some((device, name));
        }
    }
    None
}

/// `true` if a device name matches a known virtual-loopback marker
/// (case-insensitive substring). Pure function, kept separate so the matching
/// rules are unit-testable without a live device.
fn is_loopback_name(name: &str) -> bool {
    let lower = name.to_lowercase();
    LOOPBACK_NAME_MARKERS.iter().any(|m| lower.contains(m))
}

/// Resolve a device's human-readable name (cpal 0.18 `description().name()`).
fn device_name(device: &cpal::Device) -> Option<String> {
    device.description().ok().map(|d| d.name().to_string())
}

/// Open the default input device (mic / line-in) — the universal fallback.
fn open_default_input(
    host: &cpal::Host,
    producer: Producer<CaptureFrame>,
) -> Result<Capture, AudioError> {
    let device = host
        .default_input_device()
        .ok_or(AudioError::NoInputDevice)?;
    let name = device_name(&device).unwrap_or_else(|| "<unknown input device>".to_string());

    open_input_device(&device, name, CaptureSource::Microphone, producer)
}

/// Build + start a cpal input stream on `device`, tagging it with `source`.
///
/// Consumes `producer` (it is moved into the realtime data callback). On any
/// failure the producer is unrecoverable (cpal owns the closure), so we return a
/// plain [`AudioError`] — by this point the caller has already committed to this
/// device (a named virtual device it preferred, or the default input).
fn open_input_device(
    device: &cpal::Device,
    device_name: String,
    source: CaptureSource,
    producer: Producer<CaptureFrame>,
) -> Result<Capture, AudioError> {
    // Read the *actual* device config (spec §2): many loopback/virtual devices run
    // at 44.1k/96k and have 2+ channels — never assume 48k stereo.
    let supported = device
        .default_input_config()
        .map_err(|e| AudioError::Config(e.to_string()))?;

    let sample_format = supported.sample_format();
    let config: StreamConfig = supported.into();
    let sample_rate = config.sample_rate;
    let channels = config.channels;

    log::info!(
        "particle-audio: capturing from '{device_name}' @ {sample_rate} Hz, \
         {channels} ch, {sample_format:?} [{}]",
        source.label()
    );

    let stream = build_stream(device, &config, sample_format, channels, producer)?;
    stream
        .play()
        .map_err(|e| AudioError::Stream(e.to_string()))?;

    Ok(Capture {
        stream,
        sample_rate,
        channels,
        device_name,
        source,
    })
}

/// Build the typed input stream for the device's native sample format, with a
/// mono-downmixing data callback. Generic over the sample type so we support the
/// common formats (f32 / i16 / u16 / i32 / …) without duplicating logic.
fn build_stream(
    device: &cpal::Device,
    config: &StreamConfig,
    sample_format: SampleFormat,
    channels: u16,
    producer: Producer<CaptureFrame>,
) -> Result<Stream, AudioError> {
    macro_rules! build {
        ($t:ty) => {
            build_typed::<$t>(device, config, channels, producer)
        };
    }
    match sample_format {
        SampleFormat::F32 => build!(f32),
        SampleFormat::I16 => build!(i16),
        SampleFormat::U16 => build!(u16),
        SampleFormat::I32 => build!(i32),
        SampleFormat::I8 => build!(i8),
        SampleFormat::U8 => build!(u8),
        SampleFormat::F64 => build!(f64),
        other => Err(AudioError::Config(format!(
            "unsupported sample format: {other:?}"
        ))),
    }
}

fn build_typed<T>(
    device: &cpal::Device,
    config: &StreamConfig,
    channels: u16,
    mut producer: Producer<CaptureFrame>,
) -> Result<Stream, AudioError>
where
    T: Sample + cpal::SizedSample + Send + 'static,
    f32: FromSample<T>,
{
    let channels = channels.max(1) as usize;
    let inv_ch = 1.0 / channels as f32;

    let data_cb = move |data: &[T], _: &cpal::InputCallbackInfo| {
        // Preserve the first stereo pair, downmix interleaved frames → mono, and push.
        // No allocation, no locks. `push` returns Err when the ring is full; we drop the frame (an xrun on
        // the DSP side, not the capture side) rather than block the audio thread.
        // This is identical for mic and multi-channel virtual/loopback devices —
        // the channel count comes from the device's actual config.
        for frame in data.chunks(channels) {
            let mut sum = 0.0f32;
            let mut left = 0.0f32;
            let mut right = 0.0f32;
            for (i, &s) in frame.iter().enumerate() {
                let sample = f32::from_sample(s);
                if i == 0 {
                    left = sample;
                } else if i == 1 {
                    right = sample;
                }
                sum += sample;
            }
            if channels == 1 {
                right = left;
            }
            let mono = sum * inv_ch;
            let _ = producer.push(CaptureFrame { mono, left, right });
        }
    };

    let err_cb = |err: cpal::Error| {
        log::error!("particle-audio: capture stream error: {err}");
    };

    device
        .build_input_stream(*config, data_cb, err_cb, None)
        .map_err(|e| AudioError::Stream(e.to_string()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detects_common_virtual_loopback_devices() {
        // Real device names from common DJ/VJ rigs (varying case / suffixes).
        for name in [
            "BlackHole 2ch",
            "BlackHole 16ch",
            "Loopback Audio",
            "Soundflower (2ch)",
            "My Aggregate Device",
            "Multi-Output Device",
            "blackhole",     // lowercase
            "ZOOM Loopback", // marker mid-string
        ] {
            assert!(
                is_loopback_name(name),
                "expected '{name}' to match a marker"
            );
        }
    }

    #[test]
    fn does_not_match_real_input_devices() {
        for name in [
            "MacBook Pro Microphone",
            "External Microphone",
            "USB Audio CODEC",
            "Built-in Input",
            "Scarlett 2i2 USB",
        ] {
            assert!(!is_loopback_name(name), "did not expect '{name}' to match");
        }
    }

    #[test]
    fn source_classification_and_labels() {
        assert!(CaptureSource::SystemLoopback.is_loopback());
        assert!(CaptureSource::VirtualDevice.is_loopback());
        assert!(!CaptureSource::Microphone.is_loopback());

        assert_eq!(CaptureSource::SystemLoopback.label(), "system loopback");
        assert_eq!(
            CaptureSource::VirtualDevice.label(),
            "virtual loopback device"
        );
        assert_eq!(CaptureSource::Microphone.label(), "mic");
    }

    #[test]
    fn default_config_prefers_loopback() {
        assert!(CaptureConfig::default().prefer_loopback);
    }
}

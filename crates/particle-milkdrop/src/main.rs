// particle-milkdrop — renders a single .milk preset using wgpu + winit.
//
// Usage:
//   cargo run -p particle-milkdrop -- preset.milk          # windowed (needs display)
//   cargo run -p particle-milkdrop -- preset.milk --headless 300 out.png  # offscreen

use std::path::Path;
use std::sync::Arc;

use particle_audio::{AudioEngine, CaptureConfig};
use winit::{
    application::ApplicationHandler,
    dpi::PhysicalSize,
    event::WindowEvent,
    event_loop::{ActiveEventLoop, EventLoop},
    window::{Window, WindowId},
};

// The engine lives in the library crate (single source of truth). This bin is a
// thin CLI/window shell over it.
use particle_milkdrop::{fallback_preset, load_preset_path, MilkShaders, MilkdropRenderer};

/// Thin path-based wrapper over the library ingest ([`load_preset_path`]) used by
/// the headless / anim CLI paths and the windowed app. `.json` → Butterchurn
/// loader, anything else → raw `.milk` parser (native converter when built in).
fn load_preset(path: &str) -> Result<MilkShaders, String> {
    load_preset_path(Path::new(path))
}

/// Audio reactivity fed to the renderer for one frame, in MilkDrop/Butterchurn
/// convention: each value is a *volume-independent reactivity ratio* (≈1.0 at the
/// band's recent average, 0 when quiet, >1 on a hit). `*_att` are the smoother
/// attenuated envelopes that lag peaks.
struct AudioFrame {
    bass: f32,
    mid: f32,
    treb: f32,
    vol: f32,
    bass_att: f32,
    mid_att: f32,
    treb_att: f32,
    vol_att: f32,
}

/// Map particle-audio's Butterchurn-faithful reactivity ratios onto MilkDrop's
/// bass/mid/treb/vol(+ _att) convention. These ratios are already in the
/// "~1.0 = average, >1 = loud" space the EEL/shaders expect — no 0.5+1.5x remap
/// (that remap was the wrong-signal hack on top of AGC'd EQ levels).
fn map_audio(f: &particle_audio::Features) -> AudioFrame {
    AudioFrame {
        bass: f.bass_react,
        mid: f.mid_react,
        treb: f.treb_react,
        vol: f.vol_react,
        bass_att: f.bass_react_att,
        mid_att: f.mid_react_att,
        treb_att: f.treb_react_att,
        vol_att: f.vol_react_att,
    }
}

// ---------------------------------------------------------------------------
// Headless (offscreen) mode — no display required
// ---------------------------------------------------------------------------

/// Manufactured beat-driven audio (120 BPM) for headless rendering. The Butterchurn
/// oracle (scripts/butterchurn-oracle/render.html) drives an IDENTICAL model so both
/// renderers bloom on the same beats — a fair, audio-driven fidelity comparison
/// instead of the misleading silent-audio one (which leaves reactive presets black).
struct SynthAudio {
    bass: f32, mid: f32, treb: f32, vol: f32,
    bass_att: f32, mid_att: f32, treb_att: f32, vol_att: f32,
    waveform: Vec<f32>, // 512, [-1,1]
    spectrum: Vec<f32>, // 512, [0,1]
}

fn synth_audio(frame: u32, fps: f32) -> SynthAudio {
    use std::f32::consts::PI;
    let t = frame as f32 / fps;
    let bps = 2.0_f32; // 120 BPM
    let beat = t * bps;
    let bp = beat - beat.floor(); // 0..1 within the beat
    let env = (-bp * 5.0).exp(); // kick pulse
    let env_s = (-bp * 2.0).exp(); // smoothed (attenuated band)
    let hp = { let x = beat + 0.5; x - x.floor() };
    let hat = (-hp * 9.0).exp(); // off-beat hi-hat
    // Punchy levels (bass peaks ~3.1) — deliberately HOTTER than Butterchurn's
    // AGC-normalized reaction to the same beat. This vibrant, energetic look is the
    // preferred aesthetic for the engine's output; we do NOT tone it down to match
    // the reference's subtler response.
    // Higher sustained floors (~1.0 = "average" energy) so presets that build content
    // from mid/treb over the run aren't starved between beats — real music (the
    // MilkDrop2 references) has broadband sustain, not just a bass-heavy pulse.
    // Higher between-beat FLOORS (the constant terms) so feedback-buildup presets
    // accumulate over the 0..90 run instead of starving between beats; env/hat
    // coefficients trimmed so the on-beat PEAKS stay at the preferred ~3.1 bass
    // (do not raise the peak — that only adds washed-white blow-outs).
    let bass = 1.3 + 1.8 * env;
    let mid = 1.3 + 0.7 * env + 0.45 * hat;
    let treb = 1.15 + 1.25 * hat + 0.4 * env;
    let vol = (bass + mid + treb) / 3.0;
    let bass_att = 1.3 + 1.2 * env_s;
    let mid_att = 1.2 + 0.55 * env_s;
    let treb_att = 1.1 + 0.85 * env_s;
    let vol_att = (bass_att + mid_att + treb_att) / 3.0;

    let n = 512usize;
    let mut waveform = vec![0.0f32; n];
    for k in 0..n {
        let x = k as f32 / n as f32;
        // Band-correct cycle counts so an FFT (Butterchurn) banks energy into the
        // bass / mid / treb thirds respectively.
        let w = 0.45 * (2.0 * PI * x * 5.0 + t * 5.0).sin() * bass.min(2.5)
            + 0.30 * (2.0 * PI * x * 110.0 + t * 9.0).sin() * mid.min(2.5)
            + 0.18 * (2.0 * PI * x * 210.0 + t * 20.0).sin() * treb.min(2.5);
        waveform[k] = (w * 0.4).clamp(-1.0, 1.0);
    }
    let mut spectrum = vec![0.0f32; n];
    for b in 0..n {
        let f = b as f32 / n as f32;
        let bass_band = (-f * 22.0).exp() * bass;
        let mid_band = (-(f - 0.33).abs() * 10.0).exp() * mid;
        let treb_band = (-(f - 0.7).abs() * 7.0).exp() * treb;
        spectrum[b] = ((bass_band + mid_band + treb_band) * 0.22).clamp(0.0, 1.0);
    }
    SynthAudio { bass, mid, treb, vol, bass_att, mid_att, treb_att, vol_att, waveform, spectrum }
}

fn run_headless(milk_path: &str, frames: u32, out_path: &str, synth: bool) {
    let (w, h) = (1280u32, 720u32);

    let instance = wgpu::Instance::default();
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        compatible_surface: None,
        force_fallback_adapter: false,
    }))
    .or_else(|_| pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::None,
        compatible_surface: None,
        force_fallback_adapter: true,
    })))
    .expect("no wgpu adapter — no GPU/software renderer found");
    eprintln!("adapter: {:?}", adapter.get_info());

    let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        label: Some("milk-headless"),
        required_features: wgpu::Features::empty(),
        required_limits: adapter.limits(),
        memory_hints: wgpu::MemoryHints::Performance,
        trace: wgpu::Trace::Off,
        experimental_features: Default::default(),
    }))
    .expect("no device");
    let device = Arc::new(device);
    let queue  = Arc::new(queue);

    // Offscreen RGBA target — comp pipeline writes here instead of a surface
    let fmt = wgpu::TextureFormat::Rgba8Unorm;
    let target = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("headless-target"),
        size: wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: fmt,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    });
    let target_view = target.create_view(&Default::default());

    // Parse preset (.milk = HLSL bodies / .json = Butterchurn GLSL bodies).
    // Headless/CLI path: fail loudly (a batch run wants a clear non-zero exit, not a
    // silent fallback). The windowed app path uses graceful fallback instead.
    let shaders = load_preset(milk_path).unwrap_or_else(|e| panic!("{e}"));

    let mut rnd = MilkdropRenderer::new(
        device.clone(), queue.clone(), w, h, fmt, &shaders,
    )
    .unwrap_or_else(|e| panic!("renderer: {e}"));

    if synth {
        rnd.set_fixed_fps(30.0); // deterministic time so the beat model lands on frames
    }
    eprintln!("Rendering {frames} frames of {milk_path}… (synth_audio={synth})");
    // Push error scopes to catch silent GPU validation failures
    let scope_oom = device.push_error_scope(wgpu::ErrorFilter::OutOfMemory);
    let scope_val = device.push_error_scope(wgpu::ErrorFilter::Validation);
    for i in 0..frames {
        if synth {
            let a = synth_audio(i, 30.0);
            rnd.set_audio(a.bass, a.mid, a.treb, a.vol);
            rnd.set_audio_att(a.bass_att, a.mid_att, a.treb_att, a.vol_att);
            rnd.set_waveform(&a.waveform, &a.waveform);
            rnd.set_freq_spectrum(&a.spectrum);
        }
        rnd.render(&target_view);
        if (i + 1) % 60 == 0 { eprint!("  frame {}/{frames}\r", i + 1); }
    }
    // Flush and check for GPU errors before reading back
    device.poll(wgpu::PollType::wait_indefinitely()).ok();
    if let Some(e) = pollster::block_on(scope_val.pop()) {
        eprintln!("GPU validation error: {e}");
    }
    if let Some(e) = pollster::block_on(scope_oom.pop()) {
        eprintln!("GPU OOM error: {e}");
    }
    eprintln!("Done. Saving {out_path}…");

    // Read back the texture
    let bytes_per_row = align256(w * 4);
    let readback = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("readback"),
        size: (bytes_per_row * h) as u64,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut enc = device.create_command_encoder(&Default::default());
    enc.copy_texture_to_buffer(
        target.as_image_copy(),
        wgpu::TexelCopyBufferInfo {
            buffer: &readback,
            layout: wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(bytes_per_row),
                rows_per_image: Some(h),
            },
        },
        wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
    );
    queue.submit(std::iter::once(enc.finish()));

    let slice = readback.slice(..);
    slice.map_async(wgpu::MapMode::Read, |_| {});
    device.poll(wgpu::PollType::wait_indefinitely()).ok();

    let data = slice.get_mapped_range();
    // Strip padding: each row is `bytes_per_row` bytes but only `w*4` are pixel data
    let stride = bytes_per_row as usize;
    let row_bytes = (w * 4) as usize;
    let mut pixels: Vec<u8> = Vec::with_capacity((w * h * 4) as usize);
    for row in 0..h as usize {
        pixels.extend_from_slice(&data[row * stride..row * stride + row_bytes]);
    }
    drop(data);
    readback.unmap();

    // Save PNG
    let f = std::fs::File::create(out_path).expect("create png");
    let mut enc = png::Encoder::new(f, w, h);
    enc.set_color(png::ColorType::Rgba);
    enc.set_depth(png::BitDepth::Eight);
    let mut writer = enc.write_header().expect("png header");
    writer.write_image_data(&pixels).expect("png data");
    println!("Saved {out_path}  ({w}x{h})");
}

fn align256(n: u32) -> u32 {
    (n + 255) & !255
}

// ---------------------------------------------------------------------------
// Animation export — render a sequence of PNG frames at a fixed timestep
// ---------------------------------------------------------------------------

fn run_anim(milk_path: &str, frames: u32, out_dir: &str) {
    let (w, h) = (640u32, 360u32); // smaller for quick GIF assembly

    let instance = wgpu::Instance::default();
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        compatible_surface: None,
        force_fallback_adapter: false,
    }))
    .or_else(|_| pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::None,
        compatible_surface: None,
        force_fallback_adapter: true,
    })))
    .expect("no wgpu adapter");
    eprintln!("adapter: {:?}", adapter.get_info());

    let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        label: Some("milk-anim"),
        required_features: wgpu::Features::empty(),
        required_limits: adapter.limits(),
        memory_hints: wgpu::MemoryHints::Performance,
        trace: wgpu::Trace::Off,
        experimental_features: Default::default(),
    }))
    .expect("no device");
    let device = Arc::new(device);
    let queue  = Arc::new(queue);

    let fmt = wgpu::TextureFormat::Rgba8Unorm;
    let target = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("anim-target"),
        size: wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: fmt,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    });
    let target_view = target.create_view(&Default::default());

    let shaders = load_preset(milk_path).unwrap_or_else(|e| panic!("{e}"));

    let mut rnd = MilkdropRenderer::new(device.clone(), queue.clone(), w, h, fmt, &shaders)
        .unwrap_or_else(|e| panic!("renderer: {e}"));
    rnd.set_fixed_fps(30.0); // deterministic time so synthetic audio animates

    std::fs::create_dir_all(out_dir).expect("create out dir");
    eprintln!("Rendering {frames} frames of {milk_path} → {out_dir}/ …");

    let bytes_per_row = align256(w * 4);
    for i in 0..frames {
        rnd.render(&target_view);

        // Read back this frame
        let readback = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("readback"),
            size: (bytes_per_row * h) as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut enc = device.create_command_encoder(&Default::default());
        enc.copy_texture_to_buffer(
            target.as_image_copy(),
            wgpu::TexelCopyBufferInfo {
                buffer: &readback,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(bytes_per_row),
                    rows_per_image: Some(h),
                },
            },
            wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
        );
        queue.submit(std::iter::once(enc.finish()));

        let slice = readback.slice(..);
        slice.map_async(wgpu::MapMode::Read, |_| {});
        device.poll(wgpu::PollType::wait_indefinitely()).ok();

        let data = slice.get_mapped_range();
        let stride = bytes_per_row as usize;
        let row_bytes = (w * 4) as usize;
        let mut pixels: Vec<u8> = Vec::with_capacity((w * h * 4) as usize);
        for row in 0..h as usize {
            pixels.extend_from_slice(&data[row * stride..row * stride + row_bytes]);
        }
        drop(data);
        readback.unmap();

        let path = format!("{out_dir}/frame_{i:04}.png");
        let f = std::fs::File::create(&path).expect("create png");
        let mut penc = png::Encoder::new(f, w, h);
        penc.set_color(png::ColorType::Rgba);
        penc.set_depth(png::BitDepth::Eight);
        let mut writer = penc.write_header().expect("png header");
        writer.write_image_data(&pixels).expect("png data");

        if (i + 1) % 30 == 0 { eprint!("  frame {}/{frames}\r", i + 1); }
    }
    eprintln!("\nDone. {frames} frames in {out_dir}/");
}

// ---------------------------------------------------------------------------
// Windowed mode
// ---------------------------------------------------------------------------

struct App {
    /// Preset to load on startup, or `None` to open in the idle "drop a file"
    /// empty-state. After launch, presets arrive by drag-and-drop.
    initial_path: Option<String>,
    window: Option<Arc<Window>>,
    state: Option<GpuState>,
}

const APP_NAME: &str = "OjoDrop";
const IDLE_TITLE: &str = "OjoDrop — drag a .milk or .json preset here";

struct GpuState {
    surface: wgpu::Surface<'static>,
    device: Arc<wgpu::Device>,
    /// Kept alive so dropped presets can rebuild the renderer in place (and so mic
    /// capture stays funded). The renderer holds its own clone.
    queue: Arc<wgpu::Queue>,
    config: wgpu::SurfaceConfiguration,
    renderer: MilkdropRenderer,
    /// Live mic capture + DSP. None if no input device was available; the
    /// renderer then falls back to its synthetic audio. Must stay alive for
    /// capture to continue (cpal stops on drop).
    audio: Option<AudioEngine>,
}

impl GpuState {
    fn new(window: Arc<Window>, initial_path: Option<&str>) -> Self {
        let size = window.inner_size();
        let (w, h) = (size.width.max(1), size.height.max(1));

        let instance = wgpu::Instance::default();
        let surface = instance.create_surface(window).expect("create surface");

        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: Some(&surface),
            force_fallback_adapter: false,
        }))
        .expect("no adapter");
        log::info!("adapter: {:?}", adapter.get_info());

        let (device, queue) =
            pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
                label: Some("milk-device"),
                required_features: wgpu::Features::empty(),
                required_limits: adapter.limits(),
                memory_hints: wgpu::MemoryHints::Performance,
                trace: wgpu::Trace::Off,
                experimental_features: Default::default(),
            }))
            .expect("no device");
        let device = Arc::new(device);
        let queue  = Arc::new(queue);

        let caps = surface.get_capabilities(&adapter);
        let format = caps
            .formats
            .iter()
            .copied()
            .find(|f| !f.is_srgb())
            .unwrap_or(caps.formats[0]);
        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format,
            width: w,
            height: h,
            present_mode: wgpu::PresentMode::Fifo,
            alpha_mode: caps.alpha_modes[0],
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
        };
        surface.configure(&device, &config);

        // Windowed app path: an untrusted drag-dropped file must never crash the app.
        // No initial file → idle empty-state (passthrough). On a load error
        // (unreadable / non-UTF-8 / malformed JSON) fall back to passthrough too.
        let shaders = match initial_path {
            Some(p) => {
                let s = load_preset(p).unwrap_or_else(|e| {
                    log::error!("{e} — using passthrough fallback");
                    fallback_preset()
                });
                if s.warp.is_none() { log::warn!("no warp shader found in {p}"); }
                if s.comp.is_none() { log::warn!("no comp shader found in {p}"); }
                s
            }
            None => fallback_preset(),
        };
        let (renderer, _compiled) = Self::build_renderer(&device, &queue, w, h, format, &shaders);

        // Start live mic capture (prefer the room mic, not system loopback).
        let audio = match AudioEngine::with_config(
            CaptureConfig { prefer_loopback: false }, 1.0,
        ) {
            Ok(eng) => {
                log::info!("audio: capturing '{}' @ {} Hz", eng.device_name(), eng.sample_rate());
                Some(eng)
            }
            Err(e) => {
                log::warn!("audio: {e} — falling back to synthetic reactivity");
                None
            }
        };

        Self { surface, device, queue, config, renderer, audio }
    }

    /// Build a renderer for `shaders`, falling back to the passthrough preset on a
    /// shader-compile error so an unsupported or hostile preset never crashes the
    /// app. The passthrough is a known-good internal asset; if it *also* fails that
    /// is a genuine engine bug worth surfacing loudly.
    ///
    /// Returns `(renderer, compiled)` where `compiled` is `true` when the preset's
    /// OWN shaders built, and `false` when we fell back to passthrough — so the UI
    /// can tell the user a preset is unsupported instead of silently showing blank.
    fn build_renderer(
        device: &Arc<wgpu::Device>,
        queue: &Arc<wgpu::Queue>,
        w: u32,
        h: u32,
        format: wgpu::TextureFormat,
        shaders: &MilkShaders,
    ) -> (MilkdropRenderer, bool) {
        match MilkdropRenderer::new(device.clone(), queue.clone(), w, h, format, shaders) {
            Ok(r) => (r, true),
            Err(e) => {
                log::error!("shader compile failed ({e}) — using passthrough fallback preset");
                let fallback = fallback_preset();
                let r = MilkdropRenderer::new(device.clone(), queue.clone(), w, h, format, &fallback)
                    .unwrap_or_else(|e2| panic!("fallback renderer also failed: {e2}"));
                (r, false)
            }
        }
    }

    /// Load a dropped preset, rebuilding the renderer in place. Never panics on
    /// bad input: a file/parse error keeps the current visuals; a shader-compile
    /// error falls back to passthrough. Returns the human-readable outcome for the
    /// window title bar.
    fn load_path(&mut self, path: &str) -> String {
        let name = std::path::Path::new(path)
            .file_name()
            .map(|n| n.to_string_lossy().into_owned())
            .unwrap_or_else(|| path.to_string());
        let is_json = path.to_ascii_lowercase().ends_with(".json");
        if !is_json && !particle_milkdrop::native_converter_available() {
            log::warn!(
                "{name}: this build has no native .milk converter linked; rendering may be \
                 degraded. Drop a .json preset, or use a build with the converter."
            );
        }
        match load_preset(path) {
            Ok(shaders) => {
                if shaders.warp.is_none() { log::warn!("no warp shader found in {name}"); }
                if shaders.comp.is_none() { log::warn!("no comp shader found in {name}"); }
                let (w, h) = (self.config.width, self.config.height);
                let (renderer, compiled) =
                    Self::build_renderer(&self.device, &self.queue, w, h, self.config.format, &shaders);
                self.renderer = renderer;
                if compiled {
                    log::info!("loaded {name}");
                    format!("{APP_NAME} — {name}")
                } else {
                    log::warn!("{name}: preset shader did not compile — showing passthrough");
                    format!("{APP_NAME} — {name}  ·  shader unsupported")
                }
            }
            Err(e) => {
                log::error!("{e} — keeping current preset");
                format!("{APP_NAME} — could not load {name}")
            }
        }
    }

    fn resize(&mut self, size: PhysicalSize<u32>) {
        let (w, h) = (size.width.max(1), size.height.max(1));
        self.config.width  = w;
        self.config.height = h;
        self.surface.configure(&self.device, &self.config);
    }

    fn render(&mut self) {
        let frame = match self.surface.get_current_texture() {
            wgpu::CurrentSurfaceTexture::Success(t)
            | wgpu::CurrentSurfaceTexture::Suboptimal(t) => t,
            wgpu::CurrentSurfaceTexture::Outdated | wgpu::CurrentSurfaceTexture::Lost => {
                self.surface.configure(&self.device, &self.config);
                return;
            }
            other => { log::warn!("surface: {other:?}"); return; }
        };
        // Pull the latest mic analysis and drive reactivity.
        if let Some(eng) = &self.audio {
            let f = eng.latest();
            let a = map_audio(&f);
            self.renderer.set_audio(a.bass, a.mid, a.treb, a.vol);
            self.renderer
                .set_audio_att(a.bass_att, a.mid_att, a.treb_att, a.vol_att);
            // Feed the full-resolution 512-sample waveform (time-domain) and the
            // 512-bin freq_spectrum so `bSpectrum` custom waveforms read real FFT bins.
            self.renderer
                .set_waveform(&f.waveform_left_full, &f.waveform_right_full);
            self.renderer.set_freq_spectrum(&f.freq_spectrum);
        }

        let view = frame.texture.create_view(&Default::default());
        self.renderer.render(&view);
        frame.present();
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.window.is_some() { return; }
        // Title reflects whether we boot into a preset or the idle empty-state.
        let title = match self.initial_path.as_deref() {
            Some(p) => {
                let name = std::path::Path::new(p)
                    .file_name()
                    .map(|n| n.to_string_lossy().into_owned())
                    .unwrap_or_else(|| p.to_string());
                format!("{APP_NAME} — {name}")
            }
            None => IDLE_TITLE.to_string(),
        };
        let attrs = Window::default_attributes()
            .with_title(&title)
            .with_inner_size(PhysicalSize::new(1280u32, 720u32));
        let win = Arc::new(event_loop.create_window(attrs).expect("create window"));
        let state = GpuState::new(win.clone(), self.initial_path.as_deref());
        self.window = Some(win);
        self.state  = Some(state);
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        _id: WindowId,
        event: WindowEvent,
    ) {
        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::KeyboardInput { event, .. }
                if event.physical_key
                    == winit::keyboard::PhysicalKey::Code(winit::keyboard::KeyCode::Escape) =>
            {
                event_loop.exit();
            }
            WindowEvent::Resized(size) => {
                if let Some(s) = &mut self.state { s.resize(size); }
            }
            // Drag-and-drop: a hovered file previews intent in the title bar; the
            // actual drop loads it. Loading is crash-safe (see GpuState::load_path).
            WindowEvent::HoveredFile(path) => {
                if let Some(w) = &self.window {
                    let name = path
                        .file_name()
                        .map(|n| n.to_string_lossy().into_owned())
                        .unwrap_or_default();
                    w.set_title(&format!("{APP_NAME} — release to load {name}"));
                }
            }
            WindowEvent::HoveredFileCancelled => {
                if let Some(w) = &self.window {
                    let loaded = self.state.is_some();
                    w.set_title(if loaded { APP_NAME } else { IDLE_TITLE });
                }
            }
            WindowEvent::DroppedFile(path) => {
                if let (Some(s), Some(w)) = (&mut self.state, &self.window) {
                    let title = s.load_path(&path.to_string_lossy());
                    w.set_title(&title);
                }
            }
            WindowEvent::RedrawRequested => {
                if let Some(s) = &mut self.state { s.render(); }
                if let Some(w) = &self.window { w.request_redraw(); }
            }
            _ => {}
        }
    }

    fn about_to_wait(&mut self, _event_loop: &ActiveEventLoop) {
        if let Some(w) = &self.window { w.request_redraw(); }
    }
}

/// Credits + license summary, surfaced in-app via `--about` and a one-line
/// startup banner. OjoDrop stands on MilkDrop / Butterchurn and the open shader
/// toolchain — keep the thank-you visible.
fn print_about() {
    println!(
        "\
OjoDrop — a native Rust + wgpu MilkDrop / Butterchurn preset player.

Built on the work of:
  • Ryan Geiss        — MilkDrop, the visualizer this engine reimplements
  • Jordan 'jberg' Berg — Butterchurn + milkdrop-shader-converter (HLSL→GLSL)
  • Nullsoft / Winamp  — MilkDrop's home and preset ecosystem
  • hlsl2glslfork / glsl-optimizer / Mesa / MojoShader authors

License: MIT (see LICENSE). Bundled converter components keep their own
permissive licenses (BSD-3 / zlib / MIT) — see THIRD_PARTY_NOTICES.md.
Native .milk converter in this build: {}",
        if particle_milkdrop::native_converter_available() { "yes" } else { "no (JSON-only)" }
    );
}

fn main() {
    env_logger::init();

    let args: Vec<String> = std::env::args().collect();

    if args.iter().any(|a| a == "--about" || a == "--credits") {
        print_about();
        return;
    }

    // Detect --anim FRAMES OUT_DIR  (dumps a PNG sequence at fixed 30fps)
    if let Some(pos) = args.iter().position(|a| a == "--anim") {
        let frames: u32 = args.get(pos + 1).and_then(|s| s.parse().ok()).unwrap_or(90);
        let out_dir = args.get(pos + 2).map(|s| s.as_str()).unwrap_or("frames");
        let Some(milk_path) = args.iter()
            .find(|a| a.ends_with(".milk") || a.ends_with(".json"))
            .map(|s| s.as_str())
        else {
            eprintln!("--anim requires a .milk or .json preset path");
            std::process::exit(2);
        };
        run_anim(milk_path, frames, out_dir);
        return;
    }

    // Detect --headless FRAMES OUTPUT.png  [--synth-audio]
    if let Some(pos) = args.iter().position(|a| a == "--headless") {
        let frames: u32 = args.get(pos + 1)
            .and_then(|s| s.parse().ok())
            .unwrap_or(300);
        let out = args.get(pos + 2).map(|s| s.as_str()).unwrap_or("milkdrop.png");
        // Opt-in manufactured beat audio (120 BPM) so audio-reactive presets bloom,
        // matching the Butterchurn oracle's identical synth model for a fair compare.
        let synth = args.iter().any(|a| a == "--synth-audio");
        let Some(milk_path) = args.iter()
            .find(|a| a.ends_with(".milk") || a.ends_with(".json"))
            .map(|s| s.as_str())
        else {
            eprintln!("--headless requires a .milk or .json preset path");
            std::process::exit(2);
        };
        run_headless(milk_path, frames, out, synth);
        return;
    }

    // Windowed: an optional file arg boots straight into a preset; otherwise the
    // app opens in the idle "drop a file" empty-state. Presets then arrive by
    // drag-and-drop (WindowEvent::DroppedFile).
    let initial_path = args.get(1)
        .filter(|a| a.ends_with(".milk") || a.ends_with(".json"))
        .cloned();

    println!("OjoDrop — MilkDrop/Butterchurn player. Credits: run with --about.");
    match &initial_path {
        Some(p) => println!("Loading: {p}"),
        None => println!("{IDLE_TITLE}"),
    }
    if !particle_milkdrop::native_converter_available() {
        println!(
            "(note: native .milk converter not linked in this build — .json presets \
             load fully; raw .milk may render degraded)"
        );
    }
    let event_loop = EventLoop::new().expect("event loop");
    let mut app = App { initial_path, window: None, state: None };
    event_loop.run_app(&mut app).expect("run");
}

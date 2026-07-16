//! Native MilkDrop named-texture discovery, decode, caching, and array planning.
//!
//! Custom shaders commonly declare samplers such as `sampler_fw_worms` or
//! `sampler_rose`. Historically OjoDrop discarded those identities and sampled
//! `sampler_noise_lq` instead. This module keeps the shader name, resolves the
//! corresponding image from a configurable texture pack, and prepares uniform-size
//! RGBA layers suitable for one `texture_2d_array` GPU binding. A texture array is
//! deliberate: the existing MilkDrop shader layout already sits at Metal's common
//! per-stage sampled-texture limit, so adding one binding per custom image is not a
//! viable corpus-wide design.

use std::collections::{BTreeMap, HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use image::imageops::FilterType;

use crate::preprocess::custom_sampler_names;

/// Hard cap on unique custom images referenced by one preset. Real presets use far
/// fewer; the cap bounds directory-to-GPU work for hostile shader text.
pub const MAX_NAMED_TEXTURE_LAYERS: usize = 16;
pub const NAMED_TEXTURE_ATLAS_GRID: u32 = 4;
pub const NAMED_TEXTURE_ATLAS_GUTTER: u32 = 2;
pub const DEFAULT_NAMED_TEXTURE_LAYER_SIZE: u32 = 256;

const MAX_SCAN_DEPTH: usize = 8;
const MAX_INDEXED_FILES: usize = 50_000;
const TEXTURE_PATH_ENV_VARS: &[&str] = &["OJODROP_TEXTURE_PATH", "MILKDROP_TEXTURE_PATH"];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SamplerFilterMode {
    Linear,
    Point,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SamplerAddressMode {
    Wrap,
    Clamp,
}

/// One shader identifier mapped to a texture-array layer. Multiple identifiers
/// (for example `sampler_fw_rose` and `sampler_fc_rose`) may share the same layer
/// while retaining different sampling modes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NamedSamplerBinding {
    pub shader_name: String,
    pub asset_name: String,
    pub layer: u32,
    pub filter: SamplerFilterMode,
    pub address: SamplerAddressMode,
}

/// Deterministic, bounded custom-texture manifest for a preset's warp+comp source.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct NamedTexturePlan {
    pub bindings: Vec<NamedSamplerBinding>,
    pub unique_assets: Vec<String>,
    /// Unique assets omitted after [`MAX_NAMED_TEXTURE_LAYERS`]. Their shader
    /// identifiers remain eligible for the existing noise fallback.
    pub truncated_assets: Vec<String>,
}

impl NamedTexturePlan {
    pub fn from_sources<'a>(sources: impl IntoIterator<Item = &'a str>) -> Self {
        Self::from_sources_with_limit(sources, MAX_NAMED_TEXTURE_LAYERS)
    }

    pub fn from_sources_with_limit<'a>(
        sources: impl IntoIterator<Item = &'a str>,
        max_layers: usize,
    ) -> Self {
        let max_layers = max_layers.min(MAX_NAMED_TEXTURE_LAYERS);
        let mut names: Vec<(String, Option<String>)> = Vec::new();
        let mut seen_names = HashSet::new();
        for source in sources {
            let aliases = sampler_macro_aliases(source);
            for name in custom_sampler_names(source) {
                if seen_names.insert(name.to_ascii_lowercase()) {
                    let target = aliases.get(&name.to_ascii_lowercase()).cloned();
                    names.push((name, target));
                }
            }
        }

        let mut unique_assets = Vec::new();
        let mut asset_layers: HashMap<String, u32> = HashMap::new();
        let mut truncated_assets = Vec::new();
        let mut bindings = Vec::new();

        for (shader_name, alias_target) in names {
            let descriptor = sampler_descriptor(alias_target.as_deref().unwrap_or(&shader_name));
            if descriptor.asset_name.is_empty() {
                continue;
            }
            let layer = if let Some(layer) = asset_layers.get(&descriptor.asset_name) {
                *layer
            } else if unique_assets.len() < max_layers {
                let layer = unique_assets.len() as u32;
                unique_assets.push(descriptor.asset_name.clone());
                asset_layers.insert(descriptor.asset_name.clone(), layer);
                layer
            } else {
                if !truncated_assets.contains(&descriptor.asset_name) {
                    truncated_assets.push(descriptor.asset_name);
                }
                continue;
            };
            bindings.push(NamedSamplerBinding {
                shader_name,
                asset_name: descriptor.asset_name,
                layer,
                filter: descriptor.filter,
                address: descriptor.address,
            });
        }

        Self {
            bindings,
            unique_assets,
            truncated_assets,
        }
    }

    pub fn is_empty(&self) -> bool {
        self.bindings.is_empty()
    }

    pub fn shader_rewrite_bindings(&self) -> Vec<crate::preprocess::NamedTextureRewriteBinding> {
        self.bindings
            .iter()
            .map(|binding| crate::preprocess::NamedTextureRewriteBinding {
                sampler_name: binding.shader_name.clone(),
                layer: binding.layer,
                point_filter: binding.filter == SamplerFilterMode::Point,
                clamp: binding.address == SamplerAddressMode::Clamp,
            })
            .collect()
    }
}

fn sampler_macro_aliases(source: &str) -> HashMap<String, String> {
    let mut aliases = HashMap::new();
    for line in source.lines() {
        let line = line.trim();
        let Some(rest) = line.strip_prefix("#define ") else {
            continue;
        };
        let mut parts = rest.split_whitespace();
        let Some(alias) = parts.next() else {
            continue;
        };
        let Some(target) = parts.next() else {
            continue;
        };
        if target.to_ascii_lowercase().starts_with("sampler_") {
            aliases.insert(alias.to_ascii_lowercase(), target.to_string());
        }
    }
    aliases
}

struct SamplerDescriptor {
    asset_name: String,
    filter: SamplerFilterMode,
    address: SamplerAddressMode,
}

fn sampler_descriptor(shader_name: &str) -> SamplerDescriptor {
    let lower = shader_name.to_ascii_lowercase();
    let mut asset = lower.strip_prefix("sampler_").unwrap_or(&lower);
    let (filter, address) = if let Some(rest) = asset.strip_prefix("fw_") {
        asset = rest;
        (SamplerFilterMode::Linear, SamplerAddressMode::Wrap)
    } else if let Some(rest) = asset.strip_prefix("fc_") {
        asset = rest;
        (SamplerFilterMode::Linear, SamplerAddressMode::Clamp)
    } else if let Some(rest) = asset.strip_prefix("pw_") {
        asset = rest;
        (SamplerFilterMode::Point, SamplerAddressMode::Wrap)
    } else if let Some(rest) = asset.strip_prefix("pc_") {
        asset = rest;
        (SamplerFilterMode::Point, SamplerAddressMode::Clamp)
    } else {
        (SamplerFilterMode::Linear, SamplerAddressMode::Wrap)
    };

    let asset_name = Path::new(asset)
        .file_stem()
        .and_then(|v| v.to_str())
        .unwrap_or(asset)
        .to_ascii_lowercase();
    SamplerDescriptor {
        asset_name,
        filter,
        address,
    }
}

#[derive(Debug, Clone)]
pub struct NamedTextureConfig {
    pub roots: Vec<PathBuf>,
    pub layer_size: u32,
}

impl Default for NamedTextureConfig {
    fn default() -> Self {
        let mut roots = Vec::new();
        for variable in TEXTURE_PATH_ENV_VARS {
            if let Some(value) = std::env::var_os(variable) {
                roots.extend(std::env::split_paths(&value));
            }
        }
        dedup_paths(&mut roots);
        Self {
            roots,
            layer_size: DEFAULT_NAMED_TEXTURE_LAYER_SIZE,
        }
    }
}

impl NamedTextureConfig {
    /// Adds the preset directory (and its conventional `textures/` child) ahead of
    /// global roots, matching MilkDrop texture-pack lookup expectations.
    pub fn with_preset_path(mut self, preset_path: &Path) -> Self {
        if let Some(parent) = preset_path.parent() {
            self.roots.insert(0, parent.to_path_buf());
            self.roots.insert(0, parent.join("textures"));
        }
        dedup_paths(&mut self.roots);
        self
    }
}

fn dedup_paths(paths: &mut Vec<PathBuf>) {
    let mut seen = HashSet::new();
    paths.retain(|path| seen.insert(path.clone()));
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum NamedTextureSource {
    File(PathBuf),
    DeterministicFallback { seed: u64 },
}

#[derive(Debug, Clone)]
pub struct ResolvedNamedTexture {
    pub asset_name: String,
    pub width: u32,
    pub height: u32,
    pub rgba8: Vec<u8>,
    pub source: NamedTextureSource,
}

/// CPU-side payload ready for upload as a single 2D-array texture. Layer order is
/// exactly `plan.unique_assets`; `bindings[*].layer` indexes this byte vector.
#[derive(Debug, Clone)]
pub struct NamedTextureArray {
    pub width: u32,
    pub height: u32,
    pub layers_rgba8: Vec<u8>,
    pub bindings: Vec<NamedSamplerBinding>,
    pub sources: Vec<NamedTextureSource>,
}

/// A fixed 4×4, guttered atlas that fits the 16-layer manifest into the two
/// reserved `sampler_named_{linear,point}` binding slots without increasing the
/// renderer's sampled-texture count. Both samplers bind this same texture view.
#[derive(Debug, Clone)]
pub struct NamedTextureAtlas {
    pub width: u32,
    pub height: u32,
    pub rgba8: Vec<u8>,
    pub bindings: Vec<NamedSamplerBinding>,
    pub sources: Vec<NamedTextureSource>,
}

impl NamedTextureArray {
    pub fn layer_count(&self) -> u32 {
        self.sources.len() as u32
    }

    pub fn bytes_per_layer(&self) -> usize {
        self.width as usize * self.height as usize * 4
    }
}

/// Reusable process/runtime texture library. Build it once, share it between clean
/// per-preset engines, and retain decoded images in the cache.
pub struct NamedTextureResolver {
    config: NamedTextureConfig,
    index: BTreeMap<String, Vec<PathBuf>>,
    cache: Mutex<HashMap<String, Arc<ResolvedNamedTexture>>>,
}

impl std::fmt::Debug for NamedTextureResolver {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("NamedTextureResolver")
            .field("config", &self.config)
            .field("indexed_names", &self.index.len())
            .finish_non_exhaustive()
    }
}

impl NamedTextureResolver {
    pub fn new(mut config: NamedTextureConfig) -> Self {
        config.layer_size = config.layer_size.clamp(1, 4096);
        dedup_paths(&mut config.roots);
        let index = build_index(&config.roots);
        Self {
            config,
            index,
            cache: Mutex::new(HashMap::new()),
        }
    }

    pub fn for_preset(preset_path: &Path) -> Self {
        Self::new(NamedTextureConfig::default().with_preset_path(preset_path))
    }

    pub fn config(&self) -> &NamedTextureConfig {
        &self.config
    }

    pub fn resolve(&self, asset_name: &str) -> Arc<ResolvedNamedTexture> {
        let key = normalized_asset_key(asset_name);
        if let Some(cached) = self
            .cache
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .get(&key)
            .cloned()
        {
            return cached;
        }

        let resolved = self
            .index
            .get(&key)
            .and_then(|candidates| decode_first(&key, candidates))
            .unwrap_or_else(|| Arc::new(fallback_texture(&key, self.config.layer_size)));
        self.cache
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .insert(key, resolved.clone());
        resolved
    }

    pub fn resolve_plan(&self, plan: &NamedTexturePlan) -> NamedTextureArray {
        let size = self.config.layer_size;
        let bytes_per_layer = size as usize * size as usize * 4;
        let mut layers_rgba8 = Vec::with_capacity(bytes_per_layer * plan.unique_assets.len());
        let mut sources = Vec::with_capacity(plan.unique_assets.len());
        for asset in &plan.unique_assets {
            let texture = self.resolve(asset);
            let rgba =
                image::RgbaImage::from_raw(texture.width, texture.height, texture.rgba8.clone())
                    .expect("resolved texture byte length is validated at construction");
            let layer = if texture.width == size && texture.height == size {
                rgba
            } else {
                image::imageops::resize(&rgba, size, size, FilterType::Triangle)
            };
            layers_rgba8.extend_from_slice(layer.as_raw());
            sources.push(texture.source.clone());
        }
        NamedTextureArray {
            width: size,
            height: size,
            layers_rgba8,
            bindings: plan.bindings.clone(),
            sources,
        }
    }

    pub fn resolve_plan_atlas(&self, plan: &NamedTexturePlan) -> NamedTextureAtlas {
        let layer_size = self.config.layer_size;
        let stride = layer_size + NAMED_TEXTURE_ATLAS_GUTTER * 2;
        let atlas_size = stride * NAMED_TEXTURE_ATLAS_GRID;
        let mut rgba8 = vec![0u8; atlas_size as usize * atlas_size as usize * 4];
        let mut sources = Vec::with_capacity(plan.unique_assets.len());

        for (layer_index, asset) in plan.unique_assets.iter().enumerate() {
            let texture = self.resolve(asset);
            let source =
                image::RgbaImage::from_raw(texture.width, texture.height, texture.rgba8.clone())
                    .expect("resolved texture byte length is validated at construction");
            let layer = if texture.width == layer_size && texture.height == layer_size {
                source
            } else {
                image::imageops::resize(&source, layer_size, layer_size, FilterType::Triangle)
            };
            let cell_x = layer_index as u32 % NAMED_TEXTURE_ATLAS_GRID;
            let cell_y = layer_index as u32 / NAMED_TEXTURE_ATLAS_GRID;
            copy_layer_with_gutter(
                &mut rgba8,
                atlas_size,
                &layer,
                cell_x * stride,
                cell_y * stride,
            );
            sources.push(texture.source.clone());
        }

        NamedTextureAtlas {
            width: atlas_size,
            height: atlas_size,
            rgba8,
            bindings: plan.bindings.clone(),
            sources,
        }
    }
}

fn copy_layer_with_gutter(
    atlas: &mut [u8],
    atlas_width: u32,
    layer: &image::RgbaImage,
    cell_x: u32,
    cell_y: u32,
) {
    let gutter = NAMED_TEXTURE_ATLAS_GUTTER;
    let layer_width = layer.width();
    let layer_height = layer.height();
    let stride = layer_width + gutter * 2;
    for local_y in 0..stride {
        for local_x in 0..stride {
            let source_x = local_x.saturating_sub(gutter).min(layer_width - 1);
            let source_y = local_y.saturating_sub(gutter).min(layer_height - 1);
            let pixel = layer.get_pixel(source_x, source_y).0;
            let atlas_x = cell_x + local_x;
            let atlas_y = cell_y + local_y;
            let offset = (atlas_y as usize * atlas_width as usize + atlas_x as usize) * 4;
            atlas[offset..offset + 4].copy_from_slice(&pixel);
        }
    }
}

fn normalized_asset_key(asset_name: &str) -> String {
    let lower = asset_name.to_ascii_lowercase();
    Path::new(&lower)
        .file_stem()
        .and_then(|v| v.to_str())
        .unwrap_or(&lower)
        .to_string()
}

fn supported_image(path: &Path) -> bool {
    path.extension()
        .and_then(|v| v.to_str())
        .is_some_and(|ext| matches!(ext.to_ascii_lowercase().as_str(), "png" | "jpg" | "jpeg"))
}

fn build_index(roots: &[PathBuf]) -> BTreeMap<String, Vec<PathBuf>> {
    let mut index: BTreeMap<String, Vec<PathBuf>> = BTreeMap::new();
    let mut remaining = MAX_INDEXED_FILES;
    for root in roots {
        if remaining == 0 {
            break;
        }
        index_directory(root, 0, &mut remaining, &mut index);
    }
    index
}

fn index_directory(
    directory: &Path,
    depth: usize,
    remaining: &mut usize,
    index: &mut BTreeMap<String, Vec<PathBuf>>,
) {
    if depth > MAX_SCAN_DEPTH || *remaining == 0 {
        return;
    }
    let Ok(read_dir) = std::fs::read_dir(directory) else {
        return;
    };
    let mut entries: Vec<_> = read_dir.filter_map(Result::ok).collect();
    entries.sort_by_key(|entry| entry.path());
    for entry in entries {
        if *remaining == 0 {
            return;
        }
        let Ok(file_type) = entry.file_type() else {
            continue;
        };
        // Do not follow directory symlinks out of a user-selected texture root.
        if file_type.is_symlink() {
            continue;
        }
        let path = entry.path();
        if file_type.is_dir() {
            index_directory(&path, depth + 1, remaining, index);
        } else if file_type.is_file() && supported_image(&path) {
            *remaining -= 1;
            if let Some(stem) = path.file_stem().and_then(|v| v.to_str()) {
                let candidates = index.entry(stem.to_ascii_lowercase()).or_default();
                if !candidates.contains(&path) {
                    candidates.push(path);
                }
            }
        }
    }
}

fn decode_first(asset_name: &str, candidates: &[PathBuf]) -> Option<Arc<ResolvedNamedTexture>> {
    for path in candidates {
        let decoded = std::fs::read(path)
            .map_err(|err| err.to_string())
            .and_then(|bytes| image::load_from_memory(&bytes).map_err(|err| err.to_string()));
        match decoded {
            Ok(image) => {
                let rgba = image.to_rgba8();
                return Some(Arc::new(ResolvedNamedTexture {
                    asset_name: asset_name.to_string(),
                    width: rgba.width(),
                    height: rgba.height(),
                    rgba8: rgba.into_raw(),
                    source: NamedTextureSource::File(path.clone()),
                }));
            }
            Err(err) => log::warn!("cannot decode MilkDrop texture {}: {err}", path.display()),
        }
    }
    None
}

fn stable_hash64(value: &str) -> u64 {
    let mut hash = 0xcbf29ce484222325u64;
    for byte in value.bytes() {
        hash ^= byte as u64;
        hash = hash.wrapping_mul(0x100000001b3);
    }
    hash
}

fn mix64(mut value: u64) -> u64 {
    value ^= value >> 30;
    value = value.wrapping_mul(0xbf58476d1ce4e5b9);
    value ^= value >> 27;
    value = value.wrapping_mul(0x94d049bb133111eb);
    value ^ (value >> 31)
}

fn lattice(seed: u64, x: u32, y: u32) -> u8 {
    mix64(seed ^ (x as u64).wrapping_mul(0x9e3779b185ebca87) ^ (y as u64).rotate_left(29)) as u8
}

fn fallback_texture(asset_name: &str, size: u32) -> ResolvedNamedTexture {
    let seed = stable_hash64(asset_name);
    let mut rgba8 = vec![0u8; size as usize * size as usize * 4];
    for y in 0..size {
        for x in 0..size {
            // Multi-scale value-noise-like field plus a name-specific interference
            // pattern. It is intentionally contentful and colorful so a missing pack
            // does not collapse every named texture to the same gray noise image.
            let n0 = lattice(seed, x / 4, y / 4) as u32;
            let n1 = lattice(seed.rotate_left(17), x / 16, y / 16) as u32;
            let n2 = lattice(seed.rotate_left(37), x / 48, y / 48) as u32;
            let field = (n0 * 2 + n1 * 3 + n2 * 5) / 10;
            let dx = x as f32 / size as f32 - 0.5;
            let dy = y as f32 / size as f32 - 0.5;
            let phase = ((dx * dx + dy * dy).sqrt() * 42.0
                + dx.atan2(dy) * 3.0
                + (seed as u32 & 255) as f32 * 0.03)
                .sin();
            let ripple = ((phase * 0.5 + 0.5) * 96.0) as u32;
            let base = (field + ripple).min(255) as u8;
            let index = (y as usize * size as usize + x as usize) * 4;
            rgba8[index] = base.wrapping_add((seed >> 8) as u8 / 3);
            rgba8[index + 1] = base.rotate_left(1).wrapping_add((seed >> 24) as u8 / 4);
            rgba8[index + 2] = 255u8
                .wrapping_sub(base / 2)
                .wrapping_add((seed >> 40) as u8 / 5);
            rgba8[index + 3] = 255;
        }
    }
    ResolvedNamedTexture {
        asset_name: asset_name.to_string(),
        width: size,
        height: size,
        rgba8,
        source: NamedTextureSource::DeterministicFallback { seed },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp_dir(label: &str) -> PathBuf {
        let path = std::env::temp_dir().join(format!(
            "ojodrop-named-textures-{label}-{}-{}",
            std::process::id(),
            stable_hash64(std::thread::current().name().unwrap_or("test"))
        ));
        let _ = std::fs::remove_dir_all(&path);
        std::fs::create_dir_all(&path).unwrap();
        path
    }

    #[test]
    fn extracts_names_and_shares_layers_across_sampling_modes() {
        let source = r#"
sampler sampler_fw_worms;
sampler sampler_fc_worms;
uniform sampler2D sampler_rose;
ret = tex2D(sampler_rand00, uv).rgb;
ret += tex2D(sampler_noise_lq, uv).rgb;
"#;
        let plan = NamedTexturePlan::from_sources([source]);
        assert_eq!(plan.unique_assets, ["worms", "rose", "rand00"]);
        assert_eq!(plan.bindings.len(), 4);
        assert_eq!(plan.bindings[0].layer, plan.bindings[1].layer);
        assert_eq!(plan.bindings[0].address, SamplerAddressMode::Wrap);
        assert_eq!(plan.bindings[1].address, SamplerAddressMode::Clamp);
    }

    #[test]
    fn sampler_macros_keep_shader_alias_but_resolve_target_asset() {
        let source = "#define MYSAMP sampler_fw_devboxb\nret = tex2D(MYSAMP, uv).rgb;";
        let plan = NamedTexturePlan::from_sources([source]);
        assert_eq!(plan.unique_assets, ["devboxb"]);
        let alias = plan
            .bindings
            .iter()
            .find(|binding| binding.shader_name == "MYSAMP")
            .unwrap();
        assert_eq!(alias.asset_name, "devboxb");
        assert_eq!(alias.filter, SamplerFilterMode::Linear);
        assert_eq!(alias.address, SamplerAddressMode::Wrap);
    }

    #[test]
    fn file_resolution_is_case_insensitive_and_cached() {
        let root = temp_dir("decode");
        let path = root.join("WoRmS.PNG");
        let pixels = vec![255u8, 0, 0, 255, 0, 255, 0, 255];
        image::save_buffer_with_format(
            &path,
            &pixels,
            2,
            1,
            image::ColorType::Rgba8,
            image::ImageFormat::Png,
        )
        .unwrap();
        let resolver = NamedTextureResolver::new(NamedTextureConfig {
            roots: vec![root.clone()],
            layer_size: 4,
        });
        let first = resolver.resolve("worms");
        let second = resolver.resolve("WORMS.png");
        assert!(Arc::ptr_eq(&first, &second));
        assert_eq!(first.width, 2);
        assert_eq!(first.height, 1);
        assert_eq!(first.source, NamedTextureSource::File(path));
        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn missing_names_get_stable_but_distinct_fallbacks() {
        let resolver = NamedTextureResolver::new(NamedTextureConfig {
            roots: vec![],
            layer_size: 16,
        });
        let a1 = resolver.resolve("missing-a");
        let a2 = resolver.resolve("missing-a");
        let b = resolver.resolve("missing-b");
        assert!(Arc::ptr_eq(&a1, &a2));
        assert_eq!(a1.rgba8, a2.rgba8);
        assert_ne!(a1.rgba8, b.rgba8);
        assert!(matches!(
            a1.source,
            NamedTextureSource::DeterministicFallback { .. }
        ));
    }

    #[test]
    fn resolved_plan_is_a_tightly_packed_array() {
        let resolver = NamedTextureResolver::new(NamedTextureConfig {
            roots: vec![],
            layer_size: 8,
        });
        let plan =
            NamedTexturePlan::from_sources(["sampler sampler_worms;\nsampler sampler_rose;\n"]);
        let array = resolver.resolve_plan(&plan);
        assert_eq!(array.layer_count(), 2);
        assert_eq!(array.bytes_per_layer(), 8 * 8 * 4);
        assert_eq!(array.layers_rgba8.len(), 2 * 8 * 8 * 4);
    }

    #[test]
    fn atlas_has_fixed_grid_and_edge_padding() {
        let resolver = NamedTextureResolver::new(NamedTextureConfig {
            roots: vec![],
            layer_size: 8,
        });
        let plan = NamedTexturePlan::from_sources(["sampler sampler_worms;\n"]);
        let atlas = resolver.resolve_plan_atlas(&plan);
        let expected = (8 + NAMED_TEXTURE_ATLAS_GUTTER * 2) * NAMED_TEXTURE_ATLAS_GRID;
        assert_eq!(atlas.width, expected);
        assert_eq!(atlas.height, expected);
        assert_eq!(atlas.rgba8.len(), expected as usize * expected as usize * 4);
        // Left gutter repeats the first content texel.
        let gutter = NAMED_TEXTURE_ATLAS_GUTTER as usize;
        let row = gutter;
        let left = (row * expected as usize) * 4;
        let content = (row * expected as usize + gutter) * 4;
        assert_eq!(
            &atlas.rgba8[left..left + 4],
            &atlas.rgba8[content..content + 4]
        );
    }
}

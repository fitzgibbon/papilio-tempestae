//! Per-platform rendering budgets, shared by the CPU code and (via shader defs) the WGSL shaders.

use bevy::shader::ShaderDefVal;

/// Octaves the heightmap needs regardless of profile: trench shaping samples up to `f0 * 32`.
pub const MIN_NOISE_OCTAVES: u32 = 6;
pub const MAX_NOISE_OCTAVES: u32 = 9;

/// WebGPU's default `maxStorageBufferBindingSize`.
const WEBGPU_MAX_STORAGE_BINDING: u64 = 128 << 20;

pub const VERTEX_STRIDE: u64 = 32;
pub const TRIANGLE_STRIDE: u64 = 48;

pub trait QualityProfile {
    /// Heightmap octaves sampled at `f0 * 2^i`, in `MIN_NOISE_OCTAVES..=MAX_NOISE_OCTAVES`.
    const NOISE_OCTAVES: u32;
    /// Deepest subdivision level; the compute shader runs `MAX_LOD_DEPTH + 1` passes.
    const MAX_LOD_DEPTH: u32;
    const LOD_SPLIT_FACTOR: f32;
    const MAX_VERTICES: u32;
    /// Triangle capacity of each ping-pong subdivision queue.
    const MAX_QUEUE_SIZE: u32;
    /// Ray-march steps for terrain self-shadowing over a fixed ray length.
    const SHADOW_STEPS: u32;
    /// Overrides the window's device pixel ratio, trading resolution for fragment cost.
    const SCALE_FACTOR_OVERRIDE: Option<f32>;

    fn shader_defs() -> Vec<ShaderDefVal> {
        vec![
            ShaderDefVal::UInt("NOISE_OCTAVES".into(), Self::NOISE_OCTAVES),
            ShaderDefVal::UInt("MAX_LOD_DEPTH".into(), Self::MAX_LOD_DEPTH),
            ShaderDefVal::UInt("MAX_VERTICES".into(), Self::MAX_VERTICES),
            ShaderDefVal::UInt("MAX_QUEUE_SIZE".into(), Self::MAX_QUEUE_SIZE),
            ShaderDefVal::UInt("SHADOW_STEPS".into(), Self::SHADOW_STEPS),
        ]
    }
}

pub struct Native;

impl QualityProfile for Native {
    const NOISE_OCTAVES: u32 = 9;
    const MAX_LOD_DEPTH: u32 = 10;
    const LOD_SPLIT_FACTOR: f32 = 4500.0;
    const MAX_VERTICES: u32 = 65536 * 128;
    const MAX_QUEUE_SIZE: u32 = 2097152;
    const SHADOW_STEPS: u32 = 10;
    const SCALE_FACTOR_OVERRIDE: Option<f32> = None;
}

/// Roughly a quarter of the native triangle count, a third fewer noise samples per height
/// lookup, half the shadow steps, and rendering at CSS-pixel resolution.
pub struct Web;

impl QualityProfile for Web {
    const NOISE_OCTAVES: u32 = 6;
    const MAX_LOD_DEPTH: u32 = 8;
    const LOD_SPLIT_FACTOR: f32 = 2500.0;
    const MAX_VERTICES: u32 = 1 << 18;
    const MAX_QUEUE_SIZE: u32 = 1 << 16;
    const SHADOW_STEPS: u32 = 5;
    const SCALE_FACTOR_OVERRIDE: Option<f32> = Some(1.0);
}

#[cfg(not(target_arch = "wasm32"))]
pub type Active = Native;
#[cfg(target_arch = "wasm32")]
pub type Active = Web;

const fn validate<Q: QualityProfile>() {
    assert!(Q::NOISE_OCTAVES >= MIN_NOISE_OCTAVES && Q::NOISE_OCTAVES <= MAX_NOISE_OCTAVES);
    assert!(Q::MAX_LOD_DEPTH >= 1);
    assert!(Q::MAX_QUEUE_SIZE % 4 == 0);
    assert!(Q::SHADOW_STEPS >= 1);
}

const _: () = validate::<Native>();
const _: () = validate::<Web>();
const _: () = assert!(Web::MAX_VERTICES as u64 * VERTEX_STRIDE <= WEBGPU_MAX_STORAGE_BINDING);
const _: () = assert!(Web::MAX_QUEUE_SIZE as u64 * TRIANGLE_STRIDE <= WEBGPU_MAX_STORAGE_BINDING);

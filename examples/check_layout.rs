use bevy::prelude::*;
use bevy::render::render_resource::ShaderType;

#[derive(ShaderType, Clone, Default)]
struct GlobalsUniform {
    camera_pos: Vec3,
    planet_radius: f32,
    planet_center: Vec3,
    noise_frequency: f32,
    noise_amplitude: f32,
    lod_split_factor: f32,
    frustum_planes: [Vec4; 6],
}

fn main() {
    println!("GlobalsUniform ShaderType size: {}", GlobalsUniform::min_size());
    // We can run this to see if it compiles and verify the size.
}

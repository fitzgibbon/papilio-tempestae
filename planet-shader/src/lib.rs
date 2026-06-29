#![cfg_attr(target_arch = "spirv", no_std)]
#![allow(clippy::excessive_precision)]

pub use glam;
use glam::{Vec3, Vec4};

#[cfg(target_arch = "spirv")]
use spirv_std::num_traits::Float;

fn mod289_vec4(x: Vec4) -> Vec4 {
    x - (x * (1.0 / 289.0)).floor() * 289.0
}

fn mod48_vec4(x: Vec4) -> Vec4 {
    x - (x * (1.0 / 48.0)).floor() * 48.0
}

fn permute_opensimplex2(t: Vec4) -> Vec4 {
    let t_mod = mod289_vec4(t);
    mod289_vec4(t_mod * (t_mod * 34.0 + Vec4::splat(133.0)))
}

fn step_vec3(edge: Vec3, x: Vec3) -> Vec3 {
    Vec3::new(
        if x.x < edge.x { 0.0 } else { 1.0 },
        if x.y < edge.y { 0.0 } else { 1.0 },
        if x.z < edge.z { 0.0 } else { 1.0 },
    )
}

fn sign_vec3(v: Vec3) -> Vec3 {
    Vec3::new(
        if v.x < 0.0 {
            -1.0
        } else if v.x > 0.0 {
            1.0
        } else {
            0.0
        },
        if v.y < 0.0 {
            -1.0
        } else if v.y > 0.0 {
            1.0
        } else {
            0.0
        },
        if v.z < 0.0 {
            -1.0
        } else if v.z > 0.0 {
            1.0
        } else {
            0.0
        },
    )
}

fn grad(hash: f32) -> Vec3 {
    let h1 = (hash / 1.0).floor();
    let h2 = (hash / 2.0).floor();
    let h4 = (hash / 4.0).floor();

    let cube = Vec3::new(
        (h1 - (h1 * 0.5).floor() * 2.0) * 2.0 - 1.0,
        (h2 - (h2 * 0.5).floor() * 2.0) * 2.0 - 1.0,
        (h4 - (h4 * 0.5).floor() * 2.0) * 2.0 - 1.0,
    );

    let index = (hash / 16.0).floor() as i32;
    let mut cuboct = cube;
    if index == 0 {
        cuboct.x = 0.0;
    } else if index == 1 {
        cuboct.y = 0.0;
    } else {
        cuboct.z = 0.0;
    }

    let type_val = (hash / 8.0).floor() - ((hash / 8.0).floor() * 0.5).floor() * 2.0;

    let rhomb = (1.0 - type_val) * cube + type_val * (cuboct + cube.cross(cuboct));

    let mut grad_val = cuboct * 1.22474487139 + rhomb;
    grad_val *= (1.0 - 0.042942436724648037 * type_val) * 32.80201376986577;

    grad_val
}

fn open_simplex2_base(x_coord: Vec3) -> f32 {
    // First half-lattice, closest edge
    let v1 = x_coord.round();
    let d1 = x_coord - v1;
    let score1 = d1.abs();

    let score1_yzx = Vec3::new(score1.y, score1.z, score1.x);
    let score1_zxy = Vec3::new(score1.z, score1.x, score1.y);
    let max_score1 = score1_yzx.max(score1_zxy);
    let dir1 = step_vec3(max_score1, score1);

    let sign_d1 = sign_vec3(d1);
    let v2 = v1 + dir1 * sign_d1;
    let d2 = x_coord - v2;

    // Second half-lattice, closest edge
    let x2_coord = x_coord + Vec3::splat(144.5);
    let v3 = x2_coord.round();
    let d3 = x2_coord - v3;
    let score2 = d3.abs();

    let score2_yzx = Vec3::new(score2.y, score2.z, score2.x);
    let score2_zxy = Vec3::new(score2.z, score2.x, score2.y);
    let max_score2 = score2_yzx.max(score2_zxy);
    let dir2 = step_vec3(max_score2, score2);

    let sign_d3 = sign_vec3(d3);
    let v4 = v3 + dir2 * sign_d3;
    let d4 = x2_coord - v4;

    // Gradient hashes
    let mut hashes = permute_opensimplex2(mod289_vec4(Vec4::new(v1.x, v2.x, v3.x, v4.x)));
    hashes = permute_opensimplex2(mod289_vec4(hashes + Vec4::new(v1.y, v2.y, v3.y, v4.y)));
    hashes = mod48_vec4(permute_opensimplex2(mod289_vec4(
        hashes + Vec4::new(v1.z, v2.z, v3.z, v4.z),
    )));

    // Gradient extrapolations & kernel function
    let dots = Vec4::new(d1.dot(d1), d2.dot(d2), d3.dot(d3), d4.dot(d4));
    let a = (Vec4::splat(0.5) - dots).max(Vec4::ZERO);
    let aa = a * a;
    let aaaa = aa * aa;

    let g1 = grad(hashes.x);
    let g2 = grad(hashes.y);
    let g3 = grad(hashes.z);
    let g4 = grad(hashes.w);

    let extrapolations = Vec4::new(d1.dot(g1), d2.dot(g2), d3.dot(g3), d4.dot(g4));

    aaaa.dot(extrapolations)
}

/// 3D OpenSimplex2f (Conventional orientation) Noise function, returning a value in [-1.0, 1.0].
#[inline(never)]
pub fn snoise3(v: Vec3) -> f32 {
    let v_rotated = Vec3::splat(v.dot(Vec3::splat(2.0 / 3.0))) - v;
    open_simplex2_base(v_rotated)
}

#[cfg(target_arch = "spirv")]
#[spirv_std::macros::spirv(compute(threads(1)))]
pub fn dummy_main(
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] in_val: &Vec3,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] out_val: &mut f32,
) {
    *out_val = snoise3(*in_val);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_noise_values() {
        let latitude = 0.1f32;
        let longitude = 0.0f32;
        let y_u = latitude.sin();
        let r_xz_u = latitude.cos();
        let x_u = r_xz_u * longitude.sin();
        let z_u = r_xz_u * longitude.cos();
        let start_pos = Vec3::new(x_u, y_u, z_u);
        let p = start_pos * 1.5;
        println!("Starting pos: {:?}", start_pos);
        println!("Noise at starting pos * 1.5: {}", snoise3(p));

        for &(x, y, z) in &[
            (0.1, 0.2, 0.3),
            (0.5, -0.2, 0.8),
            (-0.7, 0.1, -0.4),
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, 0.0, 1.0),
        ] {
            let v = Vec3::new(x, y, z);
            let n = snoise3(v);
            println!("Noise value at ({}, {}, {}): {}", x, y, z, n);
        }
    }
}

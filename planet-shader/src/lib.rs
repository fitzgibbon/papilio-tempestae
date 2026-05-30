#![cfg_attr(target_arch = "spirv", no_std)]
#![allow(clippy::excessive_precision)]

pub use glam;
use glam::{Vec2, Vec3, Vec4};


#[cfg(target_arch = "spirv")]
use spirv_std::num_traits::Float;



fn mod289_vec3(x: Vec3) -> Vec3 {
    x - (x * (1.0 / 289.0)).floor() * 289.0
}

fn mod289_vec4(x: Vec4) -> Vec4 {
    x - (x * (1.0 / 289.0)).floor() * 289.0
}

fn permute(x: Vec4) -> Vec4 {
    mod289_vec4((x * 34.0 + Vec4::ONE) * x)
}

fn taylor_inv_sqrt(r: Vec4) -> Vec4 {
    Vec4::splat(1.79284291400159) - r * 0.85373472095314
}

fn step_vec3(edge: Vec3, x: Vec3) -> Vec3 {
    Vec3::new(
        if x.x < edge.x { 0.0 } else { 1.0 },
        if x.y < edge.y { 0.0 } else { 1.0 },
        if x.z < edge.z { 0.0 } else { 1.0 },
    )
}

fn step_vec4(edge: Vec4, x: Vec4) -> Vec4 {
    Vec4::new(
        if x.x < edge.x { 0.0 } else { 1.0 },
        if x.y < edge.y { 0.0 } else { 1.0 },
        if x.z < edge.z { 0.0 } else { 1.0 },
        if x.w < edge.w { 0.0 } else { 1.0 },
    )
}

/// 3D Simplex Noise function, returning a value in [-1.0, 1.0].
#[inline(never)]
pub fn snoise3(v: Vec3) -> f32 {
    let c = Vec2::new(1.0 / 6.0, 1.0 / 3.0);
    let d = Vec4::new(0.0, 0.5, 1.0, 2.0);

    // First corner
    let i = (v + Vec3::splat(v.dot(Vec3::new(c.y, c.y, c.y)))).floor();
    let x0 = v - i + Vec3::splat(i.dot(Vec3::new(c.x, c.x, c.x)));

    // Other corners
    let x0_yzx = Vec3::new(x0.y, x0.z, x0.x);
    let g = step_vec3(x0_yzx, x0);
    let l = Vec3::ONE - g;
    let i1 = g.min(Vec3::new(l.z, l.x, l.y));
    let i2 = g.max(Vec3::new(l.z, l.x, l.y));

    let x1 = x0 - i1 + Vec3::new(c.x, c.x, c.x);
    let x2 = x0 - i2 + Vec3::new(c.y, c.y, c.y);
    let x3 = x0 - Vec3::splat(0.5);

    // Permutations
    let i_mod = mod289_vec3(i);
    let p = permute(permute(permute(
        Vec4::splat(i_mod.z) + Vec4::new(0.0, i1.z, i2.z, 1.0))
        + Vec4::splat(i_mod.y) + Vec4::new(0.0, i1.y, i2.y, 1.0))
        + Vec4::splat(i_mod.x) + Vec4::new(0.0, i1.x, i2.x, 1.0));

    // Gradients
    let n_ = 0.142857142857_f32; // 1.0/7.0
    let d_wyz = Vec3::new(d.w, d.y, d.z);
    let d_xzx = Vec3::new(d.x, d.z, d.x);
    let ns = d_wyz * n_ - d_xzx;

    let j = p - (p * ns.z * ns.z).floor() * 49.0;

    let x_ = (j * ns.z).floor();
    let y_ = (j - x_ * 7.0).floor();

    let x = x_ * ns.x + Vec4::splat(ns.y);
    let y = y_ * ns.x + Vec4::splat(ns.y);
    let h = Vec4::ONE - x.abs() - y.abs();

    let b0 = Vec4::new(x.x, x.y, y.x, y.y);
    let b1 = Vec4::new(x.z, x.w, y.z, y.w);

    let s0 = b0.floor() * 2.0 + Vec4::ONE;
    let s1 = b1.floor() * 2.0 + Vec4::ONE;
    let sh = -step_vec4(h, Vec4::ZERO);

    let a0 = Vec4::new(b0.x, b0.z, b0.y, b0.w) + Vec4::new(s0.x, s0.z, s0.y, s0.w) * Vec4::new(sh.x, sh.x, sh.y, sh.y);
    let a1 = Vec4::new(b1.x, b1.z, b1.y, b1.w) + Vec4::new(s1.x, s1.z, s1.y, s1.w) * Vec4::new(sh.z, sh.z, sh.w, sh.w);

    let mut p0 = Vec3::new(a0.x, a0.y, h.x);
    let mut p1 = Vec3::new(a0.z, a0.w, h.y);
    let mut p2 = Vec3::new(a1.x, a1.y, h.z);
    let mut p3 = Vec3::new(a1.z, a1.w, h.w);

    let norm = taylor_inv_sqrt(Vec4::new(p0.dot(p0), p1.dot(p1), p2.dot(p2), p3.dot(p3)));
    p0 *= norm.x;
    p1 *= norm.y;
    p2 *= norm.z;
    p3 *= norm.w;

    let m = (Vec4::splat(0.6) - Vec4::new(x0.dot(x0), x1.dot(x1), x2.dot(x2), x3.dot(x3))).max(Vec4::ZERO);
    let m2 = m * m;
    let m4 = m2 * m2;
    105.0 * m4.dot(Vec4::new(p0.dot(x0), p1.dot(x1), p2.dot(x2), p3.dot(x3)))
}

#[cfg(target_arch = "spirv")]
#[spirv_std::macros::spirv(compute(threads(1)))]
pub fn dummy_main(
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] in_val: &Vec3,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] out_val: &mut f32,
) {
    *out_val = snoise3(*in_val);
}




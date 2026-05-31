// Script to analyze flat triangle mesh height vs curved heightmap height on the sphere.

use planet_shader::glam::Vec3;

// 20 Base icosahedron faces
const X: f32 = 0.5257311121191336_f32;
const Z: f32 = 0.8506508083520399_f32;

fn get_base_vertices() -> [Vec3; 12] {
    [
        Vec3::new(-X, Z, 0.0), Vec3::new(X, Z, 0.0), Vec3::new(-X, -Z, 0.0), Vec3::new(X, -Z, 0.0),
        Vec3::new(0.0, -X, Z), Vec3::new(0.0, X, Z), Vec3::new(0.0, -X, -Z), Vec3::new(0.0, X, -Z),
        Vec3::new(Z, 0.0, -X), Vec3::new(Z, 0.0, X), Vec3::new(-Z, 0.0, -X), Vec3::new(-Z, 0.0, X)
    ]
}

fn get_base_faces() -> [[usize; 3]; 20] {
    [
        [0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11],
        [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8],
        [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9],
        [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1]
    ]
}

// Ray-triangle intersection: returns barycentric coordinates (u, v, w) if ray intersects triangle
fn ray_intersects_triangle(p: Vec3, A: Vec3, B: Vec3, C: Vec3) -> Option<(f32, f32, f32)> {
    let e1 = B - A;
    let e2 = C - A;
    let h = p.cross(e2);
    let a = e1.dot(h);
    if a.abs() < 1e-8 {
        return None;
    }
    
    let normal = e1.cross(e2);
    let det = -p.dot(normal);
    if det.abs() < 1e-8 {
        return None;
    }
    
    // Solve for k, u, v:
    let k = (-A.dot(normal)) / det;
    if k < 0.0 {
        return None;
    }

    // Solve for u:
    let u = p.dot(A.cross(e2)) / det;
    // Solve for v:
    let v = p.dot(e1.cross(A)) / det;
    let w = 1.0 - u - v;

    let eps = -1e-5;
    if u >= eps && v >= eps && w >= eps {
        Some((u.max(0.0), v.max(0.0), w.max(0.0)))
    } else {
        None
    }
}

fn get_noise_height(pos_unit: Vec3) -> f32 {
    let planet_radius = 100.0f32;
    let noise_frequency = 1.5f32;
    let noise_amplitude = 40.0f32;

    let eps = 0.01f32;
    let mut total_disp = 0.0f32;
    let mut accum_grad = Vec3::ZERO;

    let sample_noise_rust = |p: Vec3| -> f32 {
        planet_shader::snoise3(p)
    };

    let f_mask = noise_frequency * 0.4;
    let n_mask = sample_noise_rust(pos_unit * f_mask);
    let mountain_density = ((n_mask * 1.8) + 0.3).clamp(0.0, 1.0);

    // Octave 0
    let f0 = noise_frequency;
    let a0 = noise_amplitude * 0.5;

    let p0_plains = pos_unit * f0;
    let n0_plains = sample_noise_rust(p0_plains) * 0.25;

    let p0_mount = pos_unit * (f0 * 0.8);
    let n0_mount = 1.0 - sample_noise_rust(p0_mount).abs();

    let n0 = n0_plains * (1.0 - mountain_density) + (n0_mount * 1.1 - 0.3) * mountain_density;

    let sample_blended_octave0 = |pos: Vec3| -> f32 {
        let mask = ((sample_noise_rust(pos * f_mask) * 1.8) + 0.3).clamp(0.0, 1.0);
        let plains = sample_noise_rust(pos * f0) * 0.25;
        let mount = 1.0 - sample_noise_rust(pos * (f0 * 0.8)).abs();
        plains * (1.0 - mask) + (mount * 1.1 - 0.3) * mask
    };

    let dx0 = sample_blended_octave0(pos_unit + Vec3::new(eps, 0.0, 0.0)) - n0;
    let dy0 = sample_blended_octave0(pos_unit + Vec3::new(0.0, eps, 0.0)) - n0;
    let dz0 = sample_blended_octave0(pos_unit + Vec3::new(0.0, 0.0, eps)) - n0;
    let g0 = Vec3::new(dx0, dy0, dz0) / eps;

    total_disp += n0 * a0;
    accum_grad += g0 * a0;

    // Octave 1
    let f1 = f0 * 2.0;
    let a1 = a0 * 0.35;
    let w1 = 0.1 + 1.9 * (accum_grad.length() / (a0 * f0)).clamp(0.0, 1.0);
    let p1 = pos_unit * f1;
    let n1 = sample_noise_rust(p1);
    let dx1 = sample_noise_rust(p1 + Vec3::new(eps, 0.0, 0.0)) - n1;
    let dy1 = sample_noise_rust(p1 + Vec3::new(0.0, eps, 0.0)) - n1;
    let dz1 = sample_noise_rust(p1 + Vec3::new(0.0, 0.0, eps)) - n1;
    let g1 = Vec3::new(dx1, dy1, dz1) / eps;
    total_disp += n1 * a1 * w1;
    accum_grad += g1 * a1 * w1;

    // Octave 2
    let f2 = f1 * 2.0;
    let a2 = a1 * 0.35;
    let w2 = 0.1 + 1.9 * (accum_grad.length() / (a0 * f0)).clamp(0.0, 1.0);
    let p2 = pos_unit * f2;
    let n2 = sample_noise_rust(p2);
    let dx2 = sample_noise_rust(p2 + Vec3::new(eps, 0.0, 0.0)) - n2;
    let dy2 = sample_noise_rust(p2 + Vec3::new(0.0, eps, 0.0)) - n2;
    let dz2 = sample_noise_rust(p2 + Vec3::new(0.0, 0.0, eps)) - n2;
    let g2 = Vec3::new(dx2, dy2, dz2) / eps;
    total_disp += n2 * a2 * w2;
    accum_grad += g2 * a2 * w2;

    // Octave 3
    let f3 = f2 * 2.0;
    let a3 = a2 * 0.35;
    let w3 = 0.1 + 1.9 * (accum_grad.length() / (a0 * f0)).clamp(0.0, 1.0);
    let p3 = pos_unit * f3;
    let n3 = sample_noise_rust(p3);
    total_disp += n3 * a3 * w3;

    // Add Sedimentary Terracing Effect on slopes
    let slope = (accum_grad.length() / (a0 * f0)).clamp(0.0, 1.0);
    let terrace_pattern = (total_disp * 1.5).sin();
    let terrace_amp = 0.8 * slope * mountain_density;
    total_disp += terrace_pattern * terrace_amp;

    planet_radius + total_disp
}

fn find_leaf_triangle(p: Vec3, A: Vec3, B: Vec3, C: Vec3, depth: u32) -> (Vec3, Vec3, Vec3) {
    if depth == 8 {
        return (A, B, C);
    }

    let m0 = (A + B).normalize();
    let m1 = (B + C).normalize();
    let m2 = (C + A).normalize();

    let sub_tris = [
        (A, m0, m2),
        (B, m1, m0),
        (C, m2, m1),
        (m0, m1, m2),
    ];

    for (sa, sb, sc) in sub_tris {
        if ray_intersects_triangle(p, sa, sb, sc).is_some() {
            return find_leaf_triangle(p, sa, sb, sc, depth + 1);
        }
    }

    // Fallback if float errors
    (A, B, C)
}

fn main() {
    let vertices = get_base_vertices();
    let faces = get_base_faces();

    println!("Starting flat vs curved terrain comparison...");

    // Generate random test points on unit sphere
    let mut rng = 12345u32;
    let mut next_random = move || -> f32 {
        rng = rng.wrapping_mul(1664525).wrapping_add(1013904223);
        (rng as f32) / (u32::MAX as f32)
    };

    let mut points = Vec::new();
    while points.len() < 5000 {
        let z_coord = next_random() * 2.0 - 1.0;
        let phi = next_random() * 2.0 * std::f32::consts::PI;
        let r = (1.0 - z_coord * z_coord).sqrt();
        let x_coord = r * phi.cos();
        let y_coord = r * phi.sin();
        let p = Vec3::new(x_coord, y_coord, z_coord);
        if p.length() > 0.99 && p.length() < 1.01 {
            points.push(p.normalize());
        }
    }

    let mut total_diff = 0.0f64;
    let mut max_diff_val = 0.0f32;
    let mut min_diff_val = 0.0f32;

    let mut peak_diffs = Vec::new();
    let mut valley_diffs = Vec::new();

    for p in points {
        // Find which base face contains the ray p
        let mut base_tri = None;
        for face in &faces {
            let A = vertices[face[0]];
            let B = vertices[face[1]];
            let C = vertices[face[2]];
            if ray_intersects_triangle(p, A, B, C).is_some() {
                base_tri = Some((A, B, C));
                break;
            }
        }

        let Some((A, B, C)) = base_tri else {
            continue;
        };

        // Find leaf triangle at depth 8
        let (la, lb, lc) = find_leaf_triangle(p, A, B, C, 0);

        // Displace leaf vertices
        let ha = get_noise_height(la);
        let hb = get_noise_height(lb);
        let hc = get_noise_height(lc);

        let da = la * ha;
        let db = lb * hb;
        let dc = lc * hc;

        // Plane equation for flat triangle: (P - da) . N = 0
        let normal = (db - da).cross(dc - da);
        // Intersection along ray: k * p
        // (k * p - da) . normal = 0 => k = (da . normal) / (p . normal)
        let k = da.dot(normal) / p.dot(normal);
        
        let h_flat = k;
        let h_curved = get_noise_height(p);
        let diff = h_flat - h_curved;

        total_diff += diff.abs() as f64;
        if diff > max_diff_val {
            max_diff_val = diff;
        }
        if diff < min_diff_val {
            min_diff_val = diff;
        }

        // Check if we are near a peak or a valley
        let noise_val = planet_shader::snoise3(p * 1.5);
        if noise_val > 0.8 {
            peak_diffs.push(diff);
        } else if noise_val < -0.8 {
            valley_diffs.push(diff);
        }
    }
    
    // Evaluate at the starting position
    let start_pos = Vec3::new(0.0, 0.1f32.sin(), 0.1f32.cos()).normalize();
    let mut base_tri = None;
    for face in &faces {
        let A = vertices[face[0]];
        let B = vertices[face[1]];
        let C = vertices[face[2]];
        if ray_intersects_triangle(start_pos, A, B, C).is_some() {
            base_tri = Some((A, B, C));
            break;
        }
    }
    if let Some((A, B, C)) = base_tri {
        let (la, lb, lc) = find_leaf_triangle(start_pos, A, B, C, 0);
        let ha = get_noise_height(la);
        let hb = get_noise_height(lb);
        let hc = get_noise_height(lc);
        let da = la * ha;
        let db = lb * hb;
        let dc = lc * hc;
        let normal = (db - da).cross(dc - da);
        let k = da.dot(normal) / start_pos.dot(normal);
        println!("==================================================");
        println!("EVALUATION AT CAMERA SPAWN POSITION {:?}", start_pos);
        println!("Curved heightmap:  {:.8}", get_noise_height(start_pos));
        println!("Flat triangle:     {:.8}", k);
        println!("Difference:        {:.8}", k - get_noise_height(start_pos));
        println!("==================================================");
    }

    println!("\n==================================================");
    println!("COMPARISON STATS: FLAT TRIANGLE vs CURVED HEIGHTMAP");
    println!("==================================================");
    println!("Average absolute difference: {:.8}", total_diff / 5000.0);
    println!("Maximum positive difference (Flat > Curved): {:.8}", max_diff_val);
    println!("Maximum negative difference (Flat < Curved): {:.8}", min_diff_val);
    
    let mean_peak_diff: f32 = peak_diffs.iter().sum::<f32>() / (peak_diffs.len() as f32).max(1.0);
    let mean_valley_diff: f32 = valley_diffs.iter().sum::<f32>() / (valley_diffs.len() as f32).max(1.0);
    println!("Mean difference at PEAKS (noise > 0.8):   {:.8}", mean_peak_diff);
    println!("Mean difference at VALLEYS (noise < -0.8): {:.8}", mean_valley_diff);
    println!("==================================================\n");
}

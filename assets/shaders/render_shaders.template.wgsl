struct ViewUniforms {
    view_proj: mat4x4<f32>,
    light_dir: vec3<f32>,
    ambient: f32,
    camera_pos: vec3<f32>,
    show_wireframe: f32,
}

struct Globals {
    camera_pos: vec3<f32>,
    planet_radius: f32,
    planet_center: vec3<f32>,
    noise_frequency: f32,
    noise_amplitude: f32,
    lod_split_factor: f32,
    frustum_planes: array<vec4<f32>, 6>,
}

struct VertexInput {
    position: vec4<f32>,
    normal: vec4<f32>,
}

@group(0) @binding(0) var<uniform> view_uniforms: ViewUniforms;
@group(0) @binding(1) var<storage, read> vertices: array<VertexInput>;
@group(0) @binding(2) var<uniform> globals: Globals;

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) world_position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) barycentric: vec3<f32>,
}

// {{SIMPLEX_NOISE}}

fn sample_noise(p: vec3<f32>) -> f32 {
    return snoise3_shared(Vec3Shared(p.x, p.y, p.z));
}

fn sample_blended_octave0(pos: vec3<f32>) -> f32 {
    let f_mask = globals.noise_frequency * 0.4;
    let f0 = globals.noise_frequency;
    let mask = clamp(sample_noise(pos * f_mask) * 1.8 + 0.3, 0.0, 1.0);
    let plains = sample_noise(pos * f0) * 0.25;
    let mount = 1.0 - abs(sample_noise(pos * (f0 * 0.8)));
    return mix(plains, mount * 1.1 - 0.3, mask * mask);
}

fn get_displacement(pos_unit: vec3<f32>) -> f32 {
    let eps = 0.01;
    var total_disp = 0.0;
    var accum_grad = vec3<f32>(0.0, 0.0, 0.0);

    let f_mask = globals.noise_frequency * 0.4;
    let n_mask = sample_noise(pos_unit * f_mask);
    let mountain_density = clamp(n_mask * 1.8 + 0.3, 0.0, 1.0);
    let mountain_factor = mountain_density * mountain_density;

    // Octave 0
    let f0 = globals.noise_frequency;
    let a0 = globals.noise_amplitude * 0.5; // Base octave has 50% amplitude

    let p0_plains = pos_unit * f0;
    let n0_plains = sample_noise(p0_plains) * 0.25;

    let p0_mount = pos_unit * (f0 * 0.8);
    let n0_mount = 1.0 - abs(sample_noise(p0_mount));

    let n0 = mix(n0_plains, n0_mount * 1.1 - 0.3, mountain_factor);

    let dx0 = sample_blended_octave0(pos_unit + vec3<f32>(eps, 0.0, 0.0)) - n0;
    let dy0 = sample_blended_octave0(pos_unit + vec3<f32>(0.0, eps, 0.0)) - n0;
    let dz0 = sample_blended_octave0(pos_unit + vec3<f32>(0.0, 0.0, eps)) - n0;
    let g0 = vec3<f32>(dx0, dy0, dz0) / eps;

    total_disp += n0 * a0;
    accum_grad += g0 * a0;

    // Octave 1
    let f1 = f0 * 2.0;
    let a1 = a0 * 0.35;
    let w1 = 0.1 + 1.9 * clamp(length(accum_grad) / (a0 * f0), 0.0, 1.0);
    let p1 = pos_unit * f1;
    let n1 = sample_noise(p1);
    let dx1 = sample_noise(p1 + vec3<f32>(eps, 0.0, 0.0)) - n1;
    let dy1 = sample_noise(p1 + vec3<f32>(0.0, eps, 0.0)) - n1;
    let dz1 = sample_noise(p1 + vec3<f32>(0.0, 0.0, eps)) - n1;
    let g1 = vec3<f32>(dx1, dy1, dz1) / eps;
    total_disp += n1 * a1 * w1;
    accum_grad += g1 * a1 * w1;

    // Octave 2
    let f2 = f1 * 2.0;
    let a2 = a1 * 0.35;
    let w2 = 0.1 + 1.9 * clamp(length(accum_grad) / (a0 * f0), 0.0, 1.0);
    let p2 = pos_unit * f2;
    let n2 = sample_noise(p2);
    let dx2 = sample_noise(p2 + vec3<f32>(eps, 0.0, 0.0)) - n2;
    let dy2 = sample_noise(p2 + vec3<f32>(0.0, eps, 0.0)) - n2;
    let dz2 = sample_noise(p2 + vec3<f32>(0.0, 0.0, eps)) - n2;
    let g2 = vec3<f32>(dx2, dy2, dz2) / eps;
    total_disp += n2 * a2 * w2;
    accum_grad += g2 * a2 * w2;

    // Octave 3
    let f3 = f2 * 2.0;
    let a3 = a2 * 0.35;
    let w3 = 0.1 + 1.9 * clamp(length(accum_grad) / (a0 * f0), 0.0, 1.0);
    let p3 = pos_unit * f3;
    let n3 = sample_noise(p3);
    total_disp += n3 * a3 * w3;

    // Add Sedimentary Terracing Effect on slopes
    let slope = clamp(length(accum_grad) / (a0 * f0), 0.0, 1.0);
    let terrace_noise = sample_noise(pos_unit * (f0 * 4.0));
    let terrace_pattern = sin(total_disp * 1.5 + terrace_noise * 0.4);
    let terrace_amp = 0.8 * slope * mountain_density;
    total_disp += terrace_pattern * terrace_amp;

    // Clamp displacement to flat ocean floor
    total_disp = max(total_disp, -2.5);

    return total_disp;
}

fn get_heightmap_normal(pos_unit: vec3<f32>) -> vec3<f32> {
    let eps = 0.005;
    let h0 = get_displacement(pos_unit);
    let h_dx = get_displacement(normalize(pos_unit + vec3<f32>(eps, 0.0, 0.0)));
    let h_dy = get_displacement(normalize(pos_unit + vec3<f32>(0.0, eps, 0.0)));
    let h_dz = get_displacement(normalize(pos_unit + vec3<f32>(0.0, 0.0, eps)));
    let grad = vec3<f32>(h_dx - h0, h_dy - h0, h_dz - h0) / eps;
    let grad_tangent = grad - dot(grad, pos_unit) * pos_unit;
    return normalize(pos_unit - grad_tangent);
}

@vertex
fn vs_main(@builtin(vertex_index) vertex_id: u32) -> VertexOutput {
    var out: VertexOutput;

    // Pull vertex data from the storage buffer
    let v_data = vertices[vertex_id];
    let pos_world = v_data.position.xyz;
    let normal_world = v_data.normal.xyz;

    out.world_position = pos_world;
    out.normal = normal_world;
    out.clip_position = view_uniforms.view_proj * vec4<f32>(pos_world, 1.0);

    // Compute barycentric coordinates based on vertex ID modulo 3
    let mod3 = vertex_id % 3u;
    if (mod3 == 0u) {
        out.barycentric = vec3<f32>(1.0, 0.0, 0.0);
    } else if (mod3 == 1u) {
        out.barycentric = vec3<f32>(0.0, 1.0, 0.0);
    } else {
        out.barycentric = vec3<f32>(0.0, 0.0, 1.0);
    }

    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let radial_dir = normalize(in.world_position);
    let normal = get_heightmap_normal(radial_dir);
    let view_dir = normalize(view_uniforms.camera_pos - in.world_position);
    let light_dir = normalize(view_uniforms.light_dir);

    let diffuse = max(dot(normal, light_dir), 0.0);
    let ambient = view_uniforms.ambient;

    let height = length(in.world_position);
    let displacement = height - 100.0;
    let slope = 1.0 - dot(normal, radial_dir);

    // Multi-biome coloring
    var biome_color = vec3<f32>(0.0);

    // 1. Base biomes based on altitude
    if (displacement <= -2.48) {
        // Deep water
        biome_color = vec3<f32>(0.02, 0.12, 0.28);
    } else if (displacement <= -1.8) {
        // Beach / Sand transition
        let t = (displacement - (-2.48)) / ((-1.8) - (-2.48));
        biome_color = mix(vec3<f32>(0.02, 0.12, 0.28), vec3<f32>(0.76, 0.68, 0.52), t);
    } else if (displacement <= 4.0) {
        // Grass / Plains transition
        let t = (displacement - (-1.8)) / (4.0 - (-1.8));
        biome_color = mix(vec3<f32>(0.76, 0.68, 0.52), vec3<f32>(0.18, 0.34, 0.14), t);
    } else if (displacement <= 14.0) {
        // Highlands / Foothills transition
        let t = (displacement - 4.0) / (14.0 - 4.0);
        biome_color = mix(vec3<f32>(0.18, 0.34, 0.14), vec3<f32>(0.32, 0.32, 0.18), t);
    } else {
        // High peaks
        biome_color = vec3<f32>(0.32, 0.32, 0.18);
    }

    // 2. Steep Rock Cliffs (slope-dependent)
    let rock_weight = clamp((slope - 0.12) / 0.10, 0.0, 1.0);
    let rock_terrace = sin(displacement * 1.5) * 0.04;
    let rock_color = vec3<f32>(0.24, 0.24, 0.24) + vec3<f32>(rock_terrace);
    biome_color = mix(biome_color, rock_color, rock_weight);

    // 3. Snow caps on flat areas at high elevations
    let snow_weight = clamp((displacement - 16.0) / 4.0, 0.0, 1.0) * (1.0 - rock_weight);
    biome_color = mix(biome_color, vec3<f32>(0.95, 0.95, 0.98), snow_weight);

    // Shading / Lighting
    let shading = diffuse + ambient;
    var face_color = biome_color * shading;

    // 4. Specular highlight on water
    if (displacement <= -2.48) {
        let half_dir = normalize(light_dir + view_dir);
        let specular = pow(max(dot(normal, half_dir), 0.0), 64.0) * 0.7;
        face_color = face_color + vec3<f32>(specular);
    }

    // 5. Rayleigh scattering representation (Rim Light / Atmospheric Glow)
    let rim = pow(1.0 - max(dot(view_dir, radial_dir), 0.0), 4.0);
    let rim_color = vec3<f32>(0.35, 0.55, 1.0) * rim * 0.45 * shading;
    face_color = face_color + rim_color;

    // 6. Ground Fog / Atmospheric Haze (fades out as camera goes to space)
    let distance_to_cam = distance(view_uniforms.camera_pos, in.world_position);
    let cam_height = length(view_uniforms.camera_pos);
    let fog_density = 0.002;
    let fog_factor = 1.0 - exp(-distance_to_cam * fog_density);
    let fog_intensity = clamp((150.0 - cam_height) / 50.0, 0.0, 1.0);
    let fog_color = vec3<f32>(0.55, 0.68, 0.85) * shading;
    let final_terrain_color = mix(face_color, fog_color, fog_factor * fog_intensity);

    // 7. Toggleable wireframe
    var final_color = final_terrain_color;
    if (view_uniforms.show_wireframe > 0.5) {
        let d = fwidth(in.barycentric);
        let a3 = smoothstep(vec3<f32>(0.0), d * 1.2, in.barycentric);
        let edge_factor = min(a3.x, min(a3.y, a3.z));
        let wireframe_color = vec3<f32>(0.0, 0.8, 0.5); // Subtle dark cyan-green
        final_color = mix(wireframe_color, final_terrain_color, edge_factor);
    }

    return vec4<f32>(final_color, 1.0);
}

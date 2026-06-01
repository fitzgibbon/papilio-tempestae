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

struct DisplacementData {
    displacement: f32,
    mountain_factor: f32,
    land_mask: f32,
    temp_noise: f32,
    humid_noise: f32,
}

fn get_displacement(pos_unit: vec3<f32>) -> DisplacementData {
    let f0 = globals.noise_frequency;
    
    // Sample exactly 6 shared noise frequencies for a unified heightmap
    let n_f0 = sample_noise(pos_unit * f0);
    let n_f0_2 = sample_noise(pos_unit * (f0 * 2.0));
    let n_f0_4 = sample_noise(pos_unit * (f0 * 4.0));
    let n_f0_8 = sample_noise(pos_unit * (f0 * 8.0));
    let n_f0_16 = sample_noise(pos_unit * (f0 * 16.0));
    let n_f0_32 = sample_noise(pos_unit * (f0 * 32.0));

    // Combine 6 octaves into a single heightmap using heterogeneous multifractal (slope scaling)
    var h = n_f0;
    
    let w1 = 0.15 + 0.85 * (1.0 - abs(n_f0));
    h += n_f0_2 * 0.5 * w1;
    
    let w2 = 0.15 + 0.85 * (1.0 - abs(n_f0_2));
    h += n_f0_4 * 0.25 * w2;
    
    let w3 = 0.15 + 0.85 * (1.0 - abs(n_f0_4));
    h += n_f0_8 * 0.125 * w3;
    
    let w4 = 0.15 + 0.85 * (1.0 - abs(n_f0_8));
    h += n_f0_16 * 0.0625 * w4;
    
    let w5 = 0.15 + 0.85 * (1.0 - abs(n_f0_16));
    h += n_f0_32 * 0.03125 * w5;

    // Define sea level
    let sea_level = 0.0;

    // land_mask: smooth transition at shorelines
    let land_mask = clamp((h - sea_level) * 10.0, 0.0, 1.0);

    // mountain_factor: high altitude areas smoothly become mountains
    let mountain_factor = clamp((h - 0.15) * 3.0, 0.0, 1.0) * land_mask;

    // Continuous elevation: h maps smoothly through sea level
    // Base term is linear (continuous at h=0), mountain amplification uses max(h,0)^2
    // which has both value=0 and derivative=0 at h=0, guaranteeing C1 continuity
    let h_land = max(h, 0.0);
    var elevation = h * 1.5 + h_land * h_land * mountain_factor * 12.0;

    // 6. Terracing in mountains
    let terrace_pattern = sin(elevation * 1.5 + n_f0_4 * 0.4);
    let terrace_amp = 0.5 * mountain_factor;
    elevation += terrace_pattern * terrace_amp;

    // Scale by globals.noise_amplitude
    let disp = elevation * (globals.noise_amplitude * 0.025);

    // Climate perturbations (reuse existing noise samples to avoid extra calls)
    let temp_noise = n_f0_2;
    let humid_noise = n_f0;

    return DisplacementData(disp, mountain_factor, land_mask, temp_noise, humid_noise);
}

fn get_heightmap_normal(pos_unit: vec3<f32>, h0: f32) -> vec3<f32> {
    let eps = 0.005;
    // Find orthogonal tangent vectors
    var tangent_x = vec3<f32>(1.0, 0.0, 0.0);
    if (abs(pos_unit.x) > 0.9) {
        tangent_x = vec3<f32>(0.0, 1.0, 0.0);
    }
    tangent_x = normalize(cross(pos_unit, tangent_x));
    let tangent_y = cross(pos_unit, tangent_x);

    let h1 = get_displacement(normalize(pos_unit + tangent_x * eps)).displacement;
    let h2 = get_displacement(normalize(pos_unit + tangent_y * eps)).displacement;

    let grad = tangent_x * (h1 - h0) / eps + tangent_y * (h2 - h0) / eps;
    return normalize(pos_unit - grad);
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

fn get_biome_color(temp: f32, humid: f32, displacement: f32, slope_weight: f32, mountain_factor: f32) -> vec3<f32> {
    var color = vec3<f32>(0.0);

    // 1. Climate-based land biomes (Whittaker style)
    if (temp < 0.2) {
        // Coldest: Tundra / Ice Sheet
        if (humid < 0.3) {
            color = vec3<f32>(0.95, 0.95, 0.98); // Ice sheet (pure white)
        } else {
            color = vec3<f32>(0.85, 0.85, 0.88); // Snowy tundra (light grey)
        }
    } else if (temp < 0.45) {
        // Cold temperate: Boreal forest (Taiga) vs Cold desert
        if (humid < 0.25) {
            color = vec3<f32>(0.55, 0.50, 0.42); // Cold desert / steppe (pale brownish grey)
        } else if (humid < 0.6) {
            color = vec3<f32>(0.10, 0.28, 0.16); // Boreal forest (crisp pine green)
        } else {
            color = vec3<f32>(0.08, 0.26, 0.20); // Wet taiga (dark blue-green)
        }
    } else if (temp < 0.75) {
        // Warm temperate: Grasslands, woodlands, seasonal forests
        if (humid < 0.2) {
            color = vec3<f32>(0.65, 0.62, 0.46); // Temperate desert / dry grassland (sandy olive)
        } else if (humid < 0.5) {
            color = vec3<f32>(0.24, 0.48, 0.20); // Grassy plains / shrubland (lush light green)
        } else if (humid < 0.8) {
            color = vec3<f32>(0.12, 0.38, 0.14); // Deciduous forest (rich forest green)
        } else {
            color = vec3<f32>(0.08, 0.34, 0.16); // Temperate rainforest (deep green)
        }
    } else {
        // Hot tropical: Deserts, savannas, rainforests
        if (humid < 0.15) {
            color = vec3<f32>(0.82, 0.72, 0.55); // Subtropical desert (Sahara golden-sand)
        } else if (humid < 0.4) {
            color = vec3<f32>(0.50, 0.52, 0.25); // Savanna (dry golden-green)
        } else if (humid < 0.65) {
            color = vec3<f32>(0.14, 0.42, 0.16); // Tropical seasonal forest (bright warm green)
        } else {
            color = vec3<f32>(0.02, 0.50, 0.12); // Tropical rainforest / jungle (emerald green)
        }
    }

    // 2. Steep Rock Cliffs Override (slope-dependent, modulated by mountain factor to protect plains)
    let rock_weight = clamp((slope_weight - 0.22) / 0.15, 0.0, 1.0) * clamp(mountain_factor * 1.5, 0.0, 1.0);
    let rock_terrace = sin(displacement * 1.5) * 0.03;
    let rock_color = vec3<f32>(0.26, 0.25, 0.24) + vec3<f32>(rock_terrace);
    color = mix(color, rock_color, rock_weight);

    return color;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let radial_dir = normalize(in.world_position);
    let disp_data = get_displacement(radial_dir);
    let displacement = disp_data.displacement;
    let mountain_factor = disp_data.mountain_factor;
    let land_mask = disp_data.land_mask;
    let temp_noise = disp_data.temp_noise;
    let humid_noise = disp_data.humid_noise;

    // Compute heightmap normals for ALL terrain (including underwater)
    let normal = get_heightmap_normal(radial_dir, displacement);

    let view_dir = normalize(view_uniforms.camera_pos - in.world_position);
    let light_dir = normalize(view_uniforms.light_dir);

    let diffuse = max(dot(normal, light_dir), 0.0);
    let ambient = view_uniforms.ambient;

    let slope = 1.0 - dot(normal, radial_dir);

    // Climate Calculations
    let perturbed_dir = normalize(radial_dir + temp_noise * 0.20);
    let t_base = 1.0 - abs(perturbed_dir.y);
    let temp = clamp(t_base - max(displacement, 0.0) * 0.04, 0.0, 1.0);
    let humid = clamp(humid_noise * 0.5 + 0.5, 0.0, 1.0);

    // Multi-biome coloring
    var biome_color = vec3<f32>(0.0);

    if (displacement < 0.0) {
        // Underwater seabed: muted sandy/rocky colors
        let depth = -displacement;
        let shallow_seabed = vec3<f32>(0.45, 0.40, 0.30); // sandy
        let deep_seabed = vec3<f32>(0.15, 0.13, 0.10);    // dark sediment
        let depth_t = clamp(depth / 3.0, 0.0, 1.0);
        biome_color = mix(shallow_seabed, deep_seabed, depth_t);
    } else if (displacement < 0.05) {
        // Beach
        let t_beach = displacement / 0.05;

        let cold_beach = vec3<f32>(0.25, 0.24, 0.24);
        let warm_beach = vec3<f32>(0.76, 0.68, 0.52);
        let tropical_beach = vec3<f32>(0.86, 0.83, 0.76);

        var beach_color = vec3<f32>(0.0);
        if (temp < 0.35) {
            beach_color = cold_beach;
        } else if (temp < 0.7) {
            beach_color = warm_beach;
        } else {
            beach_color = tropical_beach;
        }

        let seabed_color = vec3<f32>(0.45, 0.40, 0.30);
        biome_color = mix(seabed_color, beach_color, t_beach);
    } else {
        // Land
        biome_color = get_biome_color(temp, humid, displacement, slope, mountain_factor);
    }

    // Shading / Lighting
    let shading = diffuse + ambient;
    var face_color = biome_color * shading;

    // Rayleigh scattering (atmospheric rim glow)
    let rim = pow(1.0 - max(dot(view_dir, radial_dir), 0.0), 4.0);
    let rim_color = vec3<f32>(0.35, 0.55, 1.0) * rim * 0.45 * shading;
    face_color = face_color + rim_color;

    // Ground Fog / Atmospheric Haze
    let distance_to_cam = distance(view_uniforms.camera_pos, in.world_position);
    let cam_height = length(view_uniforms.camera_pos);
    let fog_density = 0.002;
    let fog_factor = 1.0 - exp(-distance_to_cam * fog_density);
    let fog_intensity = clamp((150.0 - cam_height) / 50.0, 0.0, 1.0);
    let fog_color = vec3<f32>(0.55, 0.68, 0.85) * shading;
    let final_terrain_color = mix(face_color, fog_color, fog_factor * fog_intensity);

    // Toggleable wireframe
    var final_color = final_terrain_color;
    if (view_uniforms.show_wireframe > 0.5) {
        let d = fwidth(in.barycentric);
        let a3 = smoothstep(vec3<f32>(0.0), d * 1.2, in.barycentric);
        let edge_factor = min(a3.x, min(a3.y, a3.z));
        let wireframe_color = vec3<f32>(1.0, 1.0, 1.0);
        let blend_factor = mix(0.5, 1.0, edge_factor);
        final_color = mix(wireframe_color, final_terrain_color, blend_factor);
    }

    return vec4<f32>(final_color, 1.0);
}

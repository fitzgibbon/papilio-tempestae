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
    
    // Sample shared noise frequencies to stay within the 33-call budget
    let n_f0_3 = sample_noise(pos_unit * (f0 * 0.3));
    let n_f0_6 = sample_noise(pos_unit * (f0 * 0.6));
    let n_f0_12 = sample_noise(pos_unit * (f0 * 1.2));
    let n_f0 = sample_noise(pos_unit * f0);
    let n_f0_15 = sample_noise(pos_unit * (f0 * 1.5));
    let n_f0_2 = sample_noise(pos_unit * (f0 * 2.0));
    let n_f0_3_0 = sample_noise(pos_unit * (f0 * 3.0));
    let n_f0_4 = sample_noise(pos_unit * (f0 * 4.0));
    let n_f0_6_0 = sample_noise(pos_unit * (f0 * 6.0));
    let n_f0_8_0 = sample_noise(pos_unit * (f0 * 8.0));
    let n_f0_16_0 = sample_noise(pos_unit * (f0 * 16.0));

    // 1. Continent / Ocean mask (large scale) - 3 Octaves for organic coastlines
    let continent_noise = n_f0_3 + n_f0_6 * 0.4 + n_f0_12 * 0.15;
    let land_mask = clamp(continent_noise * 2.0 + 0.3, 0.0, 1.0);

    // 2. Mountain selector (where mountain ranges form) - 2 Octaves for winding chains
    let mountain_selector = n_f0_6 + n_f0_15 * 0.3;
    let mountain_factor = clamp(mountain_selector * 1.8 - 0.2, 0.0, 1.0) * land_mask;

    // 3. Plains elevation (bumpy hills / plains) - 4 Octaves (boosted detail)
    let plains = n_f0 * 0.25 + 0.25 + n_f0_3_0 * 0.12 + n_f0_6_0 * 0.06 + n_f0_16_0 * 0.02;

    // 4. Mountain elevation (rugged peaks) - 5 Octaves (boosted detail)
    let n0_mount = 1.0 - abs(n_f0);
    let mountain = 1.0 + (n0_mount * 1.3 - 0.3 + n_f0_2 * 0.35 + n_f0_4 * 0.2 + n_f0_8_0 * 0.08 + n_f0_16_0 * 0.03) * 8.0;

    // 5. Ocean elevation (deep basins)
    let ocean_floor = -5.0 + n_f0 * 1.0;

    // Mix land elevation (plains vs mountains)
    let land_elevation = mix(plains, mountain, mountain_factor * mountain_factor);

    // Mix ocean and land
    var elevation = mix(ocean_floor, land_elevation, land_mask);

    // 6. Terracing in mountains
    let terrace_pattern = sin(elevation * 1.5 + n_f0_4 * 0.4);
    let terrace_amp = 0.5 * mountain_factor;
    elevation += terrace_pattern * terrace_amp;

    // Scale by globals.noise_amplitude
    let disp = elevation * (globals.noise_amplitude * 0.025);
    return DisplacementData(disp, mountain_factor, land_mask, n_f0_15, n_f0);
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

    var normal = radial_dir;
    if (displacement >= 0.0) {
        normal = get_heightmap_normal(radial_dir, displacement);
    }

    let view_dir = normalize(view_uniforms.camera_pos - in.world_position);
    let light_dir = normalize(view_uniforms.light_dir);

    let diffuse = max(dot(normal, light_dir), 0.0);
    let ambient = view_uniforms.ambient;

    let slope = 1.0 - dot(normal, radial_dir);

    // Climate Calculations
    // 1. Temperature: perturbed equator-to-pole gradient
    let perturbed_dir = normalize(radial_dir + temp_noise * 0.20);
    let t_base = 1.0 - abs(perturbed_dir.y);
    // Apply altitude cooling lapse rate (peaks are colder)
    let temp = clamp(t_base - max(displacement, 0.0) * 0.04, 0.0, 1.0);

    // 2. Humidity: reuse humid_noise (already sampled n_f0)
    let humid = clamp(humid_noise * 0.5 + 0.5, 0.0, 1.0);

    // Multi-biome coloring
    var biome_color = vec3<f32>(0.0);

    if (displacement < 0.0) {
        // Ocean: color based on temperature and depth
        let cold_ocean = vec3<f32>(0.06, 0.10, 0.18);
        let warm_ocean = vec3<f32>(0.01, 0.38, 0.46);
        let base_ocean = mix(cold_ocean, warm_ocean, temp);
        
        // Deepen the ocean color based on distance from land (smooth land_mask continental shelf)
        let depth_factor = clamp(1.0 - land_mask, 0.0, 1.0);
        let deep_ocean = vec3<f32>(0.005, 0.02, 0.08); // very dark navy
        
        biome_color = mix(base_ocean, deep_ocean, depth_factor * 0.9);
    } else if (displacement < 0.05) {
        // Beach: transition based on temperature
        let t_beach = displacement / 0.05;
        
        let cold_beach = vec3<f32>(0.25, 0.24, 0.24); // volcanic black sand
        let warm_beach = vec3<f32>(0.76, 0.68, 0.52); // golden sand
        let tropical_beach = vec3<f32>(0.86, 0.83, 0.76); // white sand
        
        var beach_color = vec3<f32>(0.0);
        if (temp < 0.35) {
            beach_color = cold_beach;
        } else if (temp < 0.7) {
            beach_color = warm_beach;
        } else {
            beach_color = tropical_beach;
        }
        
        // Blend beach with ocean at the shoreline
        let ocean_color_at_shore = mix(vec3<f32>(0.06, 0.10, 0.18), vec3<f32>(0.01, 0.38, 0.46), temp);
        biome_color = mix(ocean_color_at_shore, beach_color, t_beach);
    } else {
        // Land
        biome_color = get_biome_color(temp, humid, displacement, slope, mountain_factor);
    }

    // Shading / Lighting
    let shading = diffuse + ambient;
    var face_color = biome_color * shading;

    // 4. Specular highlight on water
    if (displacement < 0.0) {
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
        let wireframe_color = vec3<f32>(1.0, 1.0, 1.0); // White
        let blend_factor = mix(0.5, 1.0, edge_factor);
        final_color = mix(wireframe_color, final_terrain_color, blend_factor);
    }

    return vec4<f32>(final_color, 1.0);
}

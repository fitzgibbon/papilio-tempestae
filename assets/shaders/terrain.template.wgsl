// WGSL Compute Shader for dynamic 3D isosphere subdivision, frustum culling, and displacement

struct Globals {
    camera_pos: vec3<f32>,
    planet_radius: f32,
    planet_center: vec3<f32>,
    noise_frequency: f32,
    noise_amplitude: f32,
    lod_split_factor: f32,
    frustum_planes: array<vec4<f32>, 6>,
}

struct VertexOutput {
    position: vec4<f32>,
    normal: vec4<f32>,
}

struct DrawIndirectArgs {
    vertex_count: atomic<u32>,
    instance_count: u32,
    first_vertex: u32,
    first_instance: u32,
}

struct PassUniforms {
    depth: u32,
}

struct Triangle {
    v0: vec4<f32>,
    v1: vec4<f32>,
    v2: vec4<f32>,
}

@group(0) @binding(0) var<uniform> globals: Globals;
@group(0) @binding(1) var<storage, read_write> out_vertices: array<VertexOutput>;
@group(0) @binding(2) var<storage, read_write> indirect_args: DrawIndirectArgs;
@group(0) @binding(3) var<uniform> pass_uniforms: PassUniforms;
@group(0) @binding(4) var<storage, read> base_faces: array<Triangle>;
@group(0) @binding(5) var<storage, read_write> input_queue: array<Triangle>;
@group(0) @binding(6) var<storage, read_write> output_queue: array<Triangle>;
@group(0) @binding(7) var<storage, read> input_counter: u32;
@group(0) @binding(8) var<storage, read_write> output_counter: atomic<u32>;

// {{SIMPLEX_NOISE}}

fn get_barycentric_point(A: vec3<f32>, B: vec3<f32>, C: vec3<f32>, u_val: f32, v_val: f32) -> vec3<f32> {
    let w = 1.0 - u_val - v_val;
    return normalize(A * w + B * u_val + C * v_val);
}

fn sample_noise(p: vec3<f32>) -> f32 {
    return snoise3_shared(Vec3Shared(p.x, p.y, p.z));
}

fn sample_blended_octave0(pos: vec3<f32>) -> f32 {
    let f_mask = globals.noise_frequency * 0.4;
    let f0 = globals.noise_frequency;
    let mask = clamp(sample_noise(pos * f_mask) * 1.8 + 0.3, 0.0, 1.0);
    let plains = sample_noise(pos * f0) * 0.25;
    let mount = 1.0 - abs(sample_noise(pos * (f0 * 0.8)));
    return mix(plains, mount * 1.1 - 0.3, mask);
}

// Displace a normalized sphere coordinate using 4 octaves of 3D Simplex noise with gradient modulation
fn get_displaced_vertex(pos_unit: vec3<f32>) -> vec3<f32> {
    let eps = 0.01;
    var total_disp = 0.0;
    var accum_grad = vec3<f32>(0.0, 0.0, 0.0);

    let f_mask = globals.noise_frequency * 0.4;
    let n_mask = sample_noise(pos_unit * f_mask);
    let mountain_density = clamp(n_mask * 1.8 + 0.3, 0.0, 1.0);

    // Octave 0
    let f0 = globals.noise_frequency;
    let a0 = globals.noise_amplitude * 0.5; // Base octave has 50% amplitude

    let p0_plains = pos_unit * f0;
    let n0_plains = sample_noise(p0_plains) * 0.25;

    let p0_mount = pos_unit * (f0 * 0.8);
    let n0_mount = 1.0 - abs(sample_noise(p0_mount));

    let n0 = mix(n0_plains, n0_mount * 1.1 - 0.3, mountain_density);

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
    let terrace_pattern = sin(total_disp * 1.5);
    let terrace_amp = 0.8 * slope * mountain_density;
    total_disp += terrace_pattern * terrace_amp;

    let height = globals.planet_radius + total_disp;
    return globals.planet_center + pos_unit * height;
}

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let face_id = global_id.x;

    // Determine the active count of input triangles
    var input_count = 20u;
    if (pass_uniforms.depth > 0u) {
        input_count = input_counter;
    }

    // Early exit if this invocation exceeds the input queue
    if (face_id >= input_count) {
        return;
    }

    // Read triangle from appropriate buffer
    var tri: Triangle;
    if (pass_uniforms.depth == 0u) {
        tri = base_faces[face_id];
    } else {
        tri = input_queue[face_id];
    }

    let A = tri.v0.xyz;
    let B = tri.v1.xyz;
    let C = tri.v2.xyz;

    let center = (A + B + C) / 3.0;
    let world_center = get_displaced_vertex(normalize(center));
    let bounding_radius = max(distance(center, A), max(distance(center, B), distance(center, C))) * globals.planet_radius + globals.noise_amplitude;

    // 1. Frustum Culling
    var culled = false;
    for (var i = 0u; i < 6u; i = i + 1u) {
        let plane = globals.frustum_planes[i];
        let dist = dot(plane.xyz, world_center) + plane.w;
        if (dist < -bounding_radius) {
            culled = true;
            break;
        }
    }

    if (culled) {
        return;
    }

    // 2. Dynamic LOD based on distance to camera
    let dist_to_cam = distance(globals.camera_pos, world_center);

    // Split threshold halves at each depth level
    let split_dist = globals.lod_split_factor / pow(2.0, f32(pass_uniforms.depth));

    // Split if we are close enough and haven't hit maximum depth (8)
    let should_split = dist_to_cam < split_dist && pass_uniforms.depth < 8u;

    if (should_split) {
        // Compute edge midpoints projected onto the sphere
        let m0 = vec4<f32>(normalize(A + B), 0.0);
        let m1 = vec4<f32>(normalize(B + C), 0.0);
        let m2 = vec4<f32>(normalize(C + A), 0.0);

        // Allocate slots in the output queue
        let out_idx = atomicAdd(&output_counter, 4u);

        // Prevent queue overflow (buffer capacity MAX_QUEUE_SIZE = 524288)
        if (out_idx + 4u <= 524288u) {
            output_queue[out_idx] = Triangle(tri.v0, m0, m2);
            output_queue[out_idx + 1u] = Triangle(vec4<f32>(B, 0.0), m1, m0);
            output_queue[out_idx + 2u] = Triangle(vec4<f32>(C, 0.0), m2, m1);
            output_queue[out_idx + 3u] = Triangle(m0, m1, m2);
        }
    } else {
        // Output leaf triangle to vertex buffer
        let p1 = get_displaced_vertex(A);
        let p2 = get_displaced_vertex(B);
        let p3 = get_displaced_vertex(C);

        // Flat normal
        let flat_normal = normalize(cross(p2 - p1, p3 - p1));

        // Allocate slots in the vertex buffer
        let v_start = atomicAdd(&indirect_args.vertex_count, 3u);

        // Prevent vertex buffer overflow (MAX_VERTICES = 2097152)
        if (v_start + 3u <= 2097152u) {
            out_vertices[v_start] = VertexOutput(vec4<f32>(p1, 1.0), vec4<f32>(flat_normal, 0.0));
            out_vertices[v_start + 1u] = VertexOutput(vec4<f32>(p2, 1.0), vec4<f32>(flat_normal, 0.0));
            out_vertices[v_start + 2u] = VertexOutput(vec4<f32>(p3, 1.0), vec4<f32>(flat_normal, 0.0));
        }
    }
}

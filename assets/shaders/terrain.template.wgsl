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

// Displace a normalized sphere coordinate using 3D Simplex noise
fn get_displaced_vertex(pos_unit: vec3<f32>) -> vec3<f32> {
    let p = pos_unit * globals.noise_frequency;
    let noise_val = snoise3_shared(Vec3Shared(p.x, p.y, p.z));
    // Displace outwards from center
    let height = globals.planet_radius + noise_val * globals.noise_amplitude;
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

    // Compute bounding sphere of this triangle
    let center = (A + B + C) / 3.0;
    let world_center = globals.planet_center + normalize(center) * globals.planet_radius;
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

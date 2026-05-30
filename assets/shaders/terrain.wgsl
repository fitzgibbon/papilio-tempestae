// WGSL Compute Shader for dynamic 3D isosphere subdivision, frustum culling, and displacement

struct Globals {
    camera_pos: vec3<f32>,
    planet_radius: f32,
    planet_center: vec3<f32>,
    noise_frequency: f32,
    noise_amplitude: f32,
    dummy: f32,
    frustum_planes: array<vec4<f32>, 6>,
}

struct VertexOutput {
    position: vec4<f32>,
    normal: vec4<f32>,
}

struct DrawIndexedIndirectArgs {
    index_count: atomic<u32>,
    instance_count: u32,
    first_index: u32,
    base_vertex: i32,
    first_instance: u32,
}

@group(0) @binding(0) var<uniform> globals: Globals;
@group(0) @binding(1) var<storage, read_write> out_vertices: array<VertexOutput>;
@group(0) @binding(2) var<storage, read_write> out_indices: array<u32>;
@group(0) @binding(3) var<storage, read_write> indirect_args: DrawIndexedIndirectArgs;
@group(0) @binding(4) var<storage, read_write> vertex_counter: atomic<u32>;

// Ashima Arts 3D Perlin Noise Port to WGSL (by munrocket)
fn permute4(x: vec4<f32>) -> vec4<f32> {
    return ((x * 34.0 + 1.0) * x) % vec4<f32>(289.0);
}

fn taylorInvSqrt4(r: vec4<f32>) -> vec4<f32> {
    return 1.79284291400159 - 0.85373472095314 * r;
}

fn fade3(t: vec3<f32>) -> vec3<f32> {
    return t * t * t * (t * (t * 6.0 - 15.0) + 10.0);
}

fn perlinNoise3(P: vec3<f32>) -> f32 {
    var Pi0 = floor(P);
    var Pi1 = Pi0 + vec3<f32>(1.0);
    Pi0 = Pi0 % vec3<f32>(289.0);
    Pi1 = Pi1 % vec3<f32>(289.0);
    let Pf0 = fract(P);
    let Pf1 = Pf0 - vec3<f32>(1.0);
    let ix = vec4<f32>(Pi0.x, Pi1.x, Pi0.x, Pi1.x);
    let iy = vec4<f32>(Pi0.y, Pi0.y, Pi1.y, Pi1.y);
    let iz0 = Pi0.zzzz;
    let iz1 = Pi1.zzzz;

    let ixy = permute4(permute4(ix) + iy);
    let ixy0 = permute4(ixy + iz0);
    let ixy1 = permute4(ixy + iz1);

    var gx0 = ixy0 / 7.0;
    var gy0 = fract(floor(gx0) / 7.0) - 0.5;
    gx0 = fract(gx0);
    var gz0 = vec4<f32>(0.5) - abs(gx0) - abs(gy0);
    var sz0 = step(gz0, vec4<f32>(0.0));
    gx0 = gx0 + sz0 * (step(vec4<f32>(0.0), gx0) - 0.5);
    gy0 = gy0 + sz0 * (step(vec4<f32>(0.0), gy0) - 0.5);

    var gx1 = ixy1 / 7.0;
    var gy1 = fract(floor(gx1) / 7.0) - 0.5;
    gx1 = fract(gx1);
    var gz1 = vec4<f32>(0.5) - abs(gx1) - abs(gy1);
    var sz1 = step(gz1, vec4<f32>(0.0));
    gx1 = gx1 - sz1 * (step(vec4<f32>(0.0), gx1) - 0.5);
    gy1 = gy1 - sz1 * (step(vec4<f32>(0.0), gy1) - 0.5);

    var g000 = vec3<f32>(gx0.x, gy0.x, gz0.x);
    var g100 = vec3<f32>(gx0.y, gy0.y, gz0.y);
    var g010 = vec3<f32>(gx0.z, gy0.z, gz0.z);
    var g110 = vec3<f32>(gx0.w, gy0.w, gz0.w);
    var g001 = vec3<f32>(gx1.x, gy1.x, gz1.x);
    var g101 = vec3<f32>(gx1.y, gy1.y, gz1.y);
    var g011 = vec3<f32>(gx1.z, gy1.z, gz1.z);
    var g111 = vec3<f32>(gx1.w, gy1.w, gz1.w);

    let norm0 = taylorInvSqrt4(
        vec4<f32>(dot(g000, g000), dot(g010, g010), dot(g100, g100), dot(g110, g110))
    );
    g000 = g000 * norm0.x;
    g010 = g010 * norm0.y;
    g100 = g100 * norm0.z;
    g110 = g110 * norm0.w;
    let norm1 = taylorInvSqrt4(
        vec4<f32>(dot(g001, g001), dot(g011, g011), dot(g101, g101), dot(g111, g111))
    );
    g001 = g001 * norm1.x;
    g011 = g011 * norm1.y;
    g101 = g101 * norm1.z;
    g111 = g111 * norm1.w;

    let n000 = dot(g000, Pf0);
    let n100 = dot(g100, vec3<f32>(Pf1.x, Pf0.yz));
    let n010 = dot(g010, vec3<f32>(Pf0.x, Pf1.y, Pf0.z));
    let n110 = dot(g110, vec3<f32>(Pf1.xy, Pf0.z));
    let n001 = dot(g001, vec3<f32>(Pf0.xy, Pf1.z));
    let n101 = dot(g101, vec3<f32>(Pf1.x, Pf0.y, Pf1.z));
    let n011 = dot(g011, vec3<f32>(Pf0.x, Pf1.yz));
    let n111 = dot(g111, Pf1);

    var fade_xyz = fade3(Pf0);
    let n_z = mix(vec4<f32>(n000, n100, n010, n110), vec4<f32>(n001, n101, n011, n111), fade_xyz.z);
    let n_yz = mix(n_z.xy, n_z.zw, vec2<f32>(fade_xyz.y));
    let n_xyz = mix(n_yz.x, n_yz.y, fade_xyz.x);
    return 2.2 * n_xyz;
}

// Normalized icosahedron base geometry
const X: f32 = 0.525731112119133606;
const Z: f32 = 0.850650808352039932;

const BASE_VERTICES = array<vec3<f32>, 12>(
    vec3<f32>(-X, Z, 0.0), vec3<f32>(X, Z, 0.0), vec3<f32>(-X, -Z, 0.0), vec3<f32>(X, -Z, 0.0),
    vec3<f32>(0.0, -X, Z), vec3<f32>(0.0, X, Z), vec3<f32>(0.0, -X, -Z), vec3<f32>(0.0, X, -Z),
    vec3<f32>(Z, 0.0, -X), vec3<f32>(Z, 0.0, X), vec3<f32>(-Z, 0.0, -X), vec3<f32>(-Z, 0.0, X)
);

const BASE_FACES = array<vec3<u32>, 20>(
    vec3<u32>(0, 11, 5), vec3<u32>(0, 5, 1), vec3<u32>(0, 1, 7), vec3<u32>(0, 7, 10), vec3<u32>(0, 10, 11),
    vec3<u32>(1, 5, 9), vec3<u32>(5, 11, 4), vec3<u32>(11, 10, 2), vec3<u32>(10, 7, 6), vec3<u32>(7, 1, 8),
    vec3<u32>(3, 9, 4), vec3<u32>(3, 4, 2), vec3<u32>(3, 2, 6), vec3<u32>(3, 6, 8), vec3<u32>(3, 8, 9),
    vec3<u32>(4, 9, 5), vec3<u32>(2, 4, 11), vec3<u32>(6, 2, 10), vec3<u32>(8, 6, 7), vec3<u32>(9, 8, 1)
);

fn get_barycentric_point(A: vec3<f32>, B: vec3<f32>, C: vec3<f32>, u_val: f32, v_val: f32) -> vec3<f32> {
    let w = 1.0 - u_val - v_val;
    return normalize(A * w + B * u_val + C * v_val);
}

// Displace a normalized sphere coordinate using 3D Perlin noise
fn get_displaced_vertex(pos_unit: vec3<f32>) -> vec3<f32> {
    let noise_val = perlinNoise3(pos_unit * globals.noise_frequency);
    // Displace outwards from center
    let height = globals.planet_radius + noise_val * globals.noise_amplitude;
    return globals.planet_center + pos_unit * height;
}

@compute @workgroup_size(20, 1, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let face_id = global_id.x;
    if (face_id >= 20u) {
        return;
    }

    // Get the base vertices of the icosahedron face
    let A = BASE_VERTICES[BASE_FACES[face_id].x];
    let B = BASE_VERTICES[BASE_FACES[face_id].y];
    let C = BASE_VERTICES[BASE_FACES[face_id].z];

    // Compute bounding sphere of this base face
    let center = (A + B + C) / 3.0;
    let world_center = globals.planet_center + center * globals.planet_radius;
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
    let h_above_surface = max(0.0, dist_to_cam - globals.planet_radius);

    // Calculate subdivision level (12 distinct levels of detail)
    var LOD_SEGMENTS = array<u32, 12>(1u, 2u, 3u, 4u, 6u, 8u, 11u, 15u, 20u, 26u, 32u, 40u);
    let t_val = clamp(1.0 - h_above_surface / 12.0, 0.0, 1.0);
    let index = u32(clamp(t_val * 11.0, 0.0, 11.0));
    let S = LOD_SEGMENTS[index];

    // 3. Dynamic Tessellation
    // Loop over the subdivision grid and output triangles
    for (var j = 0u; j < S; j = j + 1u) {
        for (var i = 0u; i < S - j; i = i + 1u) {
            // Triangle 1: (i, j) -> (i+1, j) -> (i, j+1)
            output_triangle(A, B, C, i, j, i + 1u, j, i, j + 1u, S);

            // Triangle 2: (i+1, j) -> (i+1, j+1) -> (i, j+1)
            if (i + j + 1u < S) {
                output_triangle(A, B, C, i + 1u, j, i + 1u, j + 1u, i, j + 1u, S);
            }
        }
    }
}

// Generate a single flat-shaded triangle, allocate storage space, and write buffers
fn output_triangle(
    A: vec3<f32>, B: vec3<f32>, C: vec3<f32>,
    u1: u32, v1: u32,
    u2: u32, v2: u32,
    u3: u32, v3: u32,
    S: u32
) {
    // Generate normalized points
    let p1_unit = get_barycentric_point(A, B, C, f32(u1)/f32(S), f32(v1)/f32(S));
    let p2_unit = get_barycentric_point(A, B, C, f32(u2)/f32(S), f32(v2)/f32(S));
    let p3_unit = get_barycentric_point(A, B, C, f32(u3)/f32(S), f32(v3)/f32(S));

    // Apply displacement
    let pos1 = get_displaced_vertex(p1_unit);
    let pos2 = get_displaced_vertex(p2_unit);
    let pos3 = get_displaced_vertex(p3_unit);

    // Flat normal calculation
    let flat_normal = normalize(cross(pos2 - pos1, pos3 - pos1));

    // Allocate storage slots
    let v_start = atomicAdd(&vertex_counter, 3u);
    let i_start = atomicAdd(&indirect_args.index_count, 3u);

    // Safety checks against buffer limits
    if (v_start + 3u > 65536u || i_start + 3u > 131072u) {
        return;
    }

    // Write vertex positions & normals
    out_vertices[v_start].position = vec4<f32>(pos1, 1.0);
    out_vertices[v_start].normal = vec4<f32>(flat_normal, 0.0);

    out_vertices[v_start + 1u].position = vec4<f32>(pos2, 1.0);
    out_vertices[v_start + 1u].normal = vec4<f32>(flat_normal, 0.0);

    out_vertices[v_start + 2u].position = vec4<f32>(pos3, 1.0);
    out_vertices[v_start + 2u].normal = vec4<f32>(flat_normal, 0.0);

    // Write index data
    out_indices[i_start] = v_start;
    out_indices[i_start + 1u] = v_start + 1u;
    out_indices[i_start + 2u] = v_start + 2u;
}

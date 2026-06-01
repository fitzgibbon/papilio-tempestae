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

@group(0) @binding(0) var<uniform> view_uniforms: ViewUniforms;
@group(0) @binding(1) var<uniform> globals: Globals;

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) world_position: vec3<f32>,
    @location(1) normal: vec3<f32>,
}

@vertex
fn vs_main(@builtin(vertex_index) vertex_id: u32) -> VertexOutput {
    var out: VertexOutput;

    // Generate icosphere vertices procedurally
    // 20 base icosahedron faces, subdivided 4 times = 5120 triangles = 15360 vertices
    let face_id = vertex_id / 3u;
    let vert_in_face = vertex_id % 3u;

    // Base icosahedron
    let X = 0.5257311121191336;
    let Z = 0.8506508083520399;

    var base_verts = array<vec3<f32>, 12>(
        vec3<f32>(-X, Z, 0.0), vec3<f32>(X, Z, 0.0), vec3<f32>(-X, -Z, 0.0), vec3<f32>(X, -Z, 0.0),
        vec3<f32>(0.0, -X, Z), vec3<f32>(0.0, X, Z), vec3<f32>(0.0, -X, -Z), vec3<f32>(0.0, X, -Z),
        vec3<f32>(Z, 0.0, -X), vec3<f32>(Z, 0.0, X), vec3<f32>(-Z, 0.0, -X), vec3<f32>(-Z, 0.0, X)
    );

    var base_faces = array<vec3<u32>, 20>(
        vec3<u32>(0u, 11u, 5u), vec3<u32>(0u, 5u, 1u), vec3<u32>(0u, 1u, 7u), vec3<u32>(0u, 7u, 10u), vec3<u32>(0u, 10u, 11u),
        vec3<u32>(1u, 5u, 9u), vec3<u32>(5u, 11u, 4u), vec3<u32>(11u, 10u, 2u), vec3<u32>(10u, 7u, 6u), vec3<u32>(7u, 1u, 8u),
        vec3<u32>(3u, 9u, 4u), vec3<u32>(3u, 4u, 2u), vec3<u32>(3u, 2u, 6u), vec3<u32>(3u, 6u, 8u), vec3<u32>(3u, 8u, 9u),
        vec3<u32>(4u, 9u, 5u), vec3<u32>(2u, 4u, 11u), vec3<u32>(6u, 2u, 10u), vec3<u32>(8u, 6u, 7u), vec3<u32>(9u, 8u, 1u)
    );

    // Subdivide: 4 levels = 20 * 4^4 = 5120 faces
    let SUB_DEPTH = 4u;
    let base_face_idx = face_id / 256u; // 4^4 = 256 sub-faces per base face
    var sub_face_idx = face_id % 256u;

    if (base_face_idx >= 20u) {
        out.clip_position = vec4<f32>(0.0, 0.0, 0.0, 1.0);
        out.world_position = vec3<f32>(0.0);
        out.normal = vec3<f32>(0.0, 1.0, 0.0);
        return out;
    }

    let face = base_faces[base_face_idx];
    var a = normalize(base_verts[face.x]);
    var b = normalize(base_verts[face.y]);
    var c = normalize(base_verts[face.z]);

    // Subdivide 4 times
    for (var level = 0u; level < SUB_DEPTH; level++) {
        let m0 = normalize(a + b);
        let m1 = normalize(b + c);
        let m2 = normalize(c + a);

        let quad = sub_face_idx % 4u;
        sub_face_idx = sub_face_idx / 4u;

        if (quad == 0u) {
            b = m0; c = m2;
        } else if (quad == 1u) {
            a = m0; c = m1;
            // a=b, b=m1, c=m0 -> remap
            let tmp_a = b; let tmp_b = m1; let tmp_c = m0;
            a = tmp_a; b = tmp_b; c = tmp_c;
        } else if (quad == 2u) {
            a = c; b = m2;
            let tmp_a = c; let tmp_b = m2; let tmp_c = m1;
            a = tmp_a; b = tmp_b; c = tmp_c;
        } else {
            a = m0; b = m1; c = m2;
        }
    }

    var pos_unit = vec3<f32>(0.0);
    if (vert_in_face == 0u) {
        pos_unit = a;
    } else if (vert_in_face == 1u) {
        pos_unit = b;
    } else {
        pos_unit = c;
    }

    pos_unit = normalize(pos_unit);
    let world_pos = globals.planet_center + pos_unit * globals.planet_radius;

    out.world_position = world_pos;
    out.normal = pos_unit;
    out.clip_position = view_uniforms.view_proj * vec4<f32>(world_pos, 1.0);

    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let normal = normalize(in.normal);
    let view_dir = normalize(view_uniforms.camera_pos - in.world_position);
    let light_dir = normalize(view_uniforms.light_dir);

    // Fresnel: more opaque at grazing angles
    let n_dot_v = max(dot(normal, view_dir), 0.0);
    let fresnel = pow(1.0 - n_dot_v, 3.0);

    // Base water color varies with latitude (temperature proxy)
    let radial_dir = normalize(in.world_position);
    let temp_proxy = 1.0 - abs(radial_dir.y);
    let cold_water = vec3<f32>(0.04, 0.08, 0.18);
    let warm_water = vec3<f32>(0.01, 0.25, 0.38);
    let base_color = mix(cold_water, warm_water, temp_proxy);

    // Diffuse lighting on the water surface
    let diffuse = max(dot(normal, light_dir), 0.0);
    let ambient = view_uniforms.ambient;
    let shading = diffuse + ambient;

    var water_color = base_color * shading;

    // Specular highlight (sun glint)
    let half_dir = normalize(light_dir + view_dir);
    let specular = pow(max(dot(normal, half_dir), 0.0), 128.0) * 1.2;
    water_color += vec3<f32>(specular);

    // Alpha: more transparent when looking straight down, opaque at edges
    let alpha = mix(0.35, 0.85, fresnel);

    return vec4<f32>(water_color, alpha);
}

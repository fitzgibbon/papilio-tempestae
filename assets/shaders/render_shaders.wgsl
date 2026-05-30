// WGSL rendering shaders with pixel-perfect barycentric wireframe overlay

struct ViewUniforms {
    view_proj: mat4x4<f32>,
    light_dir: vec3<f32>,
    ambient: f32,
}

struct VertexInput {
    position: vec4<f32>,
    normal: vec4<f32>,
}

@group(0) @binding(0) var<uniform> view_uniforms: ViewUniforms;
@group(0) @binding(1) var<storage, read> vertices: array<VertexInput>;

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) world_position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) barycentric: vec3<f32>,
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
    // Basic flat-shaded lighting calculation
    let normal = normalize(in.normal);
    let light_dir = normalize(view_uniforms.light_dir);
    let diffuse = max(dot(normal, light_dir), 0.0);
    let face_color = vec3<f32>(0.1, 0.2, 0.4) * (diffuse + view_uniforms.ambient);

    // Pixel-perfect screen-space wireframe using barycentrics and standard derivatives
    let d = fwidth(in.barycentric);
    let a3 = smoothstep(vec3<f32>(0.0), d * 1.2, in.barycentric);
    let edge_factor = min(a3.x, min(a3.y, a3.z));

    // Green wireframe color overlay
    let wireframe_color = vec3<f32>(0.0, 1.0, 0.6);
    let final_color = mix(wireframe_color, face_color, edge_factor);

    return vec4<f32>(final_color, 1.0);
}

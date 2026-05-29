// WGSL Compute Shader template for terrain generation

struct Vertex {
    position: vec3<f32>,
    normal: vec3<f32>,
}

@group(0) @binding(0) var<storage, read_write> vertices: array<Vertex>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.x;
    if (index >= arrayLength(&vertices)) {
        return;
    }

    // Access the vertex position
    let pos = vertices[index].position;
    let norm = vertices[index].normal;

    // Displace vertex along its normal (3D noise would be calculated here)
    // vertices[index].position = pos + norm * displacement;
}

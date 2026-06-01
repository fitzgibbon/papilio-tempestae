// Verification script to run wgpu compute shader on random points and compare GPU vs CPU heightmap/noise.

use planet_shader::glam::Vec3;
use wgpu::util::DeviceExt;

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct Globals {
    camera_pos: [f32; 3],
    planet_radius: f32,
    planet_center: [f32; 3],
    noise_frequency: f32,
    noise_amplitude: f32,
    lod_split_factor: f32,
    _pad: [f32; 2], // padding to align frustum_planes to 16-byte boundary (offset 48)
    frustum_planes: [[f32; 4]; 6],
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct PassUniforms {
    depth: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct GpuTriangle {
    v0: [f32; 4],
    v1: [f32; 4],
    v2: [f32; 4],
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct VertexOutput {
    position: [f32; 4],
    normal: [f32; 4],
}

fn main() {
    bevy::tasks::block_on(run());
}

async fn run() {
    println!("Initializing wgpu for heightmap verification...");
    let instance = wgpu::Instance::default();
    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions::default())
        .await
        .expect("Failed to find wgpu adapter");

    let (device, queue) = adapter
        .request_device(&wgpu::DeviceDescriptor::default())
        .await
        .expect("Failed to create wgpu device");

    println!("Loading WGSL shader from assets/shaders/terrain.wgsl...");
    let shader_source = std::fs::read_to_string("assets/shaders/terrain.wgsl")
        .expect("Failed to read terrain.wgsl");

    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("Terrain Shader"),
        source: wgpu::ShaderSource::Wgsl(shader_source.into()),
    });

    // Generate 1000 random triangles (3000 vertices) on the unit sphere
    let mut rng = 54321u32;
    let mut next_random = move || -> f32 {
        rng = rng.wrapping_mul(1664525).wrapping_add(1013904223);
        (rng as f32) / (u32::MAX as f32)
    };

    let mut random_triangles = Vec::new();
    for _ in 0..1000 {
        let mut generate_unit_vec = || {
            let z = next_random() * 2.0 - 1.0;
            let phi = next_random() * 2.0 * std::f32::consts::PI;
            let r = (1.0 - z * z).sqrt();
            Vec3::new(r * phi.cos(), r * phi.sin(), z).normalize()
        };
        random_triangles.push(GpuTriangle {
            v0: generate_unit_vec().extend(0.0).to_array(),
            v1: generate_unit_vec().extend(0.0).to_array(),
            v2: generate_unit_vec().extend(0.0).to_array(),
        });
    }

    // Set up buffers
    let globals_data = Globals {
        camera_pos: [0.0, 0.0, 10.0],
        planet_radius: 2.0,
        planet_center: [0.0, 0.0, 0.0],
        noise_frequency: 1.5,
        noise_amplitude: 0.20,
        lod_split_factor: 0.0, // Force NO split
        _pad: [0.0, 0.0],
        frustum_planes: [[0.0, 0.0, 0.0, 100.0]; 6], // Large planes to avoid culling
    };

    let pass_data = PassUniforms {
        depth: 0,
        _pad0: 0,
        _pad1: 0,
        _pad2: 0,
    };

    let globals_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Globals Buffer"),
        contents: bytemuck::bytes_of(&globals_data),
        usage: wgpu::BufferUsages::UNIFORM,
    });

    let pass_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Pass Buffer"),
        contents: bytemuck::bytes_of(&pass_data),
        usage: wgpu::BufferUsages::UNIFORM,
    });

    let base_faces_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Base Faces Buffer"),
        contents: bytemuck::cast_slice(&random_triangles),
        usage: wgpu::BufferUsages::STORAGE,
    });

    // Indirect args (unused but bound)
    let indirect_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Indirect Args"),
        size: 16,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    // We need queues and counters. Since we dispatch at depth 0, we bind dummy storage buffers.
    let dummy_queue_a = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Dummy Queue A"),
        size: 256,
        usage: wgpu::BufferUsages::STORAGE,
        mapped_at_creation: false,
    });

    let dummy_queue_b = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Dummy Queue B"),
        size: 256,
        usage: wgpu::BufferUsages::STORAGE,
        mapped_at_creation: false,
    });

    let input_counter_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Input Counter Buffer"),
        size: 4,
        usage: wgpu::BufferUsages::STORAGE,
        mapped_at_creation: false,
    });

    let output_counter_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Output Counter Buffer"),
        size: 4,
        usage: wgpu::BufferUsages::STORAGE,
        mapped_at_creation: false,
    });

    // Vertex Output Buffer (3 vertices per face * 1000 faces = 3000 vertices)
    let vertex_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Vertex Buffer"),
        size: 3000 * 32, // 3000 vertices * 32 bytes each
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    // Bind Group layout matches group(0) @binding(...) in shader
    let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("Bind Group Layout"),
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 2,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 3,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 4,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 5,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 6,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 7,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 8,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
        ],
    });

    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("Bind Group"),
        layout: &bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: globals_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: vertex_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: indirect_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 3, resource: pass_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 4, resource: base_faces_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 5, resource: dummy_queue_a.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 6, resource: dummy_queue_b.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 7, resource: input_counter_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 8, resource: output_counter_buf.as_entire_binding() },
        ],
    });

    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("Pipeline Layout"),
        bind_group_layouts: &[&bind_group_layout],
        push_constant_ranges: &[],
    });

    let compute_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("Compute Pipeline"),
        layout: Some(&pipeline_layout),
        module: &shader,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    });

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    {
        let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("Compute Pass"),
            timestamp_writes: None,
        });
        compute_pass.set_pipeline(&compute_pipeline);
        compute_pass.set_bind_group(0, &bind_group, &[]);
        // Dispatch workgroups: 1000 faces / 64 = 16 workgroups
        compute_pass.dispatch_workgroups(16, 1, 1);
    }

    // Read back buffer
    let read_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Read Buffer"),
        size: 3000 * 32,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    encoder.copy_buffer_to_buffer(&vertex_buf, 0, &read_buf, 0, 3000 * 32);

    queue.submit(Some(encoder.finish()));

    let buffer_slice = read_buf.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    buffer_slice.map_async(wgpu::MapMode::Read, move |result| tx.send(result).unwrap());

    device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
    rx.recv().unwrap().unwrap();

    let data = buffer_slice.get_mapped_range();
    let out_vertices: &[VertexOutput] = bytemuck::cast_slice(&data);

    println!("\n==================================================");
    println!("VERIFYING GPU DISPLACED VERTICES vs CPU CALCULATION ON 3000 RANDOM POINTS");
    println!("==================================================");
    let mut total_diff = 0.0;
    let mut count = 0;
    let mut max_diff = 0.0f32;
    for (i, v) in out_vertices.iter().enumerate().take(3000) {
        let pos_gpu = Vec3::new(v.position[0], v.position[1], v.position[2]);
        if pos_gpu.length() < 0.1 {
            continue;
        }

        let pos_unit = pos_gpu.normalize();
        let total_disp;

        let sample_noise_rust = |p: Vec3| -> f32 {
            planet_shader::snoise3(p)
        };

        let f0 = globals_data.noise_frequency;
        
        // Sample shared noise frequencies
        let n_f0_3 = sample_noise_rust(pos_unit * (f0 * 0.3));
        let n_f0_6 = sample_noise_rust(pos_unit * (f0 * 0.6));
        let n_f0_12 = sample_noise_rust(pos_unit * (f0 * 1.2));
        let n_f0 = sample_noise_rust(pos_unit * f0);
        let n_f0_15 = sample_noise_rust(pos_unit * (f0 * 1.5));
        let n_f0_2 = sample_noise_rust(pos_unit * (f0 * 2.0));
        let n_f0_3_0 = sample_noise_rust(pos_unit * (f0 * 3.0));
        let n_f0_4 = sample_noise_rust(pos_unit * (f0 * 4.0));
        let n_f0_6_0 = sample_noise_rust(pos_unit * (f0 * 6.0));
        let n_f0_8_0 = sample_noise_rust(pos_unit * (f0 * 8.0));
        let n_f0_16_0 = sample_noise_rust(pos_unit * (f0 * 16.0));

        // 1. Continent / Ocean mask (large scale) - 3 Octaves for organic coastlines
        let continent_noise = n_f0_3 + n_f0_6 * 0.4 + n_f0_12 * 0.15;
        let land_mask = (continent_noise * 2.0 + 0.3).clamp(0.0, 1.0);

        // 2. Mountain selector (where mountain ranges form) - 2 Octaves for winding chains
        let mountain_selector = n_f0_6 + n_f0_15 * 0.3;
        let mountain_factor = (mountain_selector * 1.8 - 0.2).clamp(0.0, 1.0) * land_mask;

        // 3. Plains elevation (bumpy hills / plains) - 4 Octaves (boosted detail)
        let plains = n_f0 * 0.25 + 0.25 + n_f0_3_0 * 0.12 + n_f0_6_0 * 0.06 + n_f0_16_0 * 0.02;

        // 4. Mountain elevation (rugged peaks) - 5 Octaves (boosted detail)
        let n0_mount = 1.0 - n_f0.abs();
        let mountain = 1.0 + (n0_mount * 1.3 - 0.3 + n_f0_2 * 0.35 + n_f0_4 * 0.2 + n_f0_8_0 * 0.08 + n_f0_16_0 * 0.03) * 8.0;

        // 5. Ocean elevation (deep basins)
        let ocean_floor = -5.0 + n_f0 * 1.0;

        // Mix land elevation (plains vs mountains)
        let land_elevation = plains * (1.0 - mountain_factor * mountain_factor) + mountain * (mountain_factor * mountain_factor);

        // Mix ocean and land
        let mut elevation = ocean_floor * (1.0 - land_mask) + land_elevation * land_mask;

        // 6. Terracing in mountains
        let terrace_pattern = (elevation * 1.5 + n_f0_4 * 0.4).sin();
        let terrace_amp = 0.5 * mountain_factor;
        elevation += terrace_pattern * terrace_amp;

        // Scale by globals.noise_amplitude
        total_disp = (elevation * (globals_data.noise_amplitude * 0.025)).max(0.0);

        let expected_height = globals_data.planet_radius + total_disp;
        let expected_pos_cpu = pos_unit * expected_height;

        let diff = pos_gpu.distance(expected_pos_cpu);
        total_diff += diff;
        count += 1;
        if diff > max_diff {
            max_diff = diff;
        }

        if i < 10 {
            println!(
                "Point {:2}: GPU Height={:.6} | CPU Height={:.6} | Diff={:.8}",
                i, pos_gpu.length(), expected_height, diff
            );
        }
    }

    println!("==================================================");
    println!("Tested {} random points.", count);
    println!("Mean Deviation (GPU vs CPU expected): {:.8}", total_diff / (count as f32));
    println!("Max Deviation: {:.8}", max_diff);
    println!("==================================================\n");
}

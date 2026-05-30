// Bevy example: Procedural 3D Isosphere planet with GPU-driven dynamic LOD and frustum culling.

use bevy::{
    prelude::*,
    input::mouse::{MouseMotion, MouseWheel},
    render::{
        view::ExtractedView,
        render_graph::{self, RenderGraph, RenderLabel},
        render_resource::*,
        renderer::{RenderContext, RenderDevice, RenderQueue},
        Render, RenderApp, RenderStartup,
    },
    core_pipeline::core_3d::graph::Node3d,
};
use bytemuck::{Pod, Zeroable};

const SHADER_COMPUTE_PATH: &str = "shaders/terrain.wgsl";
const SHADER_RENDER_PATH: &str = "shaders/render_shaders.wgsl";

fn main() {
    App::new()
        .add_plugins((
            DefaultPlugins,
            PlanetRenderPlugin,
        ))
        .insert_resource(ClearColor(Color::BLACK))
        .run();
}

// ---------------------------------------------------------------------------
// 1. CPU State and Input Management
// ---------------------------------------------------------------------------

#[derive(Resource, Clone, Default)]
struct PlanetCameraState {
    latitude: f32,
    longitude: f32,
    distance: f32,
    max_distance: f32,
}

struct PlanetRenderPlugin;

impl Plugin for PlanetRenderPlugin {
    fn build(&self, app: &mut App) {
        app.insert_resource(PlanetCameraState {
            latitude: 0.1,
            longitude: 0.0,
            distance: 12.0,
            max_distance: 15.0,
        })
        .add_systems(Startup, setup_scene)
        .add_systems(Update, update_camera_and_state);

        let Some(render_app) = app.get_sub_app_mut(RenderApp) else {
            return;
        };

        render_app
            .init_resource::<RenderGlobalsUniform>()
            .init_resource::<RenderViewUniform>()
            .add_systems(RenderStartup, init_gpu_resources)
            .add_systems(Render, prepare_uniforms);
    }

    fn finish(&self, app: &mut App) {
        let Some(render_app) = app.get_sub_app_mut(RenderApp) else {
            return;
        };

        // Register the render node inside the Core3d sub-graph
        let mut render_graph = render_app.world_mut().resource_mut::<RenderGraph>();
        if let Some(core_3d_graph) = render_graph.get_sub_graph_mut(bevy::core_pipeline::core_3d::graph::Core3d) {
            core_3d_graph.add_node(PlanetRenderLabel, PlanetRenderNode);
            core_3d_graph.add_node_edge(
                Node3d::EndMainPass,
                PlanetRenderLabel,
            );
            core_3d_graph.add_node_edge(
                PlanetRenderLabel,
                Node3d::StartMainPassPostProcessing,
            );
        }
    }
}

fn setup_scene(mut commands: Commands) {
    // Spawn directional light representing the sun
    commands.spawn((
        DirectionalLight {
            shadows_enabled: false,
            ..default()
        },
        Transform::from_rotation(Quat::from_rotation_x(-0.5)),
    ));

    // Spawn 3D camera
    commands.spawn((
        Camera3d::default(),
        Transform::from_xyz(0.0, 0.0, 12.0).looking_at(Vec3::ZERO, Vec3::Y),
        Msaa::Off,
    ));
}

fn update_camera_and_state(
    mut camera_state: ResMut<PlanetCameraState>,
    mouse_buttons: Res<ButtonInput<MouseButton>>,
    mut mouse_wheel_events: MessageReader<MouseWheel>,
    mut mouse_motion_events: MessageReader<MouseMotion>,
    mut camera_query: Query<&mut Transform, With<Camera3d>>,
) {
    // Zoom with scroll wheel (proportional zoom speed for smooth descent)
    let mut scroll = 0.0;
    for event in mouse_wheel_events.read() {
        scroll += event.y;
    }
    let zoom_speed = 0.08 * (camera_state.distance - 1.95).max(0.15);
    camera_state.distance -= scroll * zoom_speed;

    // Orbit with left click drag
    if mouse_buttons.pressed(MouseButton::Left) {
        for event in mouse_motion_events.read() {
            camera_state.longitude -= event.delta.x * 0.005;
            camera_state.latitude += event.delta.y * 0.005;
            camera_state.latitude = camera_state.latitude.clamp(
                -std::f32::consts::FRAC_PI_2 + 0.05,
                std::f32::consts::FRAC_PI_2 - 0.05,
            );
        }
    } else {
        // Drain events anyway
        let _ = mouse_motion_events.read().count();
    }

    // Dynamic heightmap collision clamping based on 3D Simplex noise
    let y_u = camera_state.latitude.sin();
    let r_xz_u = camera_state.latitude.cos();
    let x_u = r_xz_u * camera_state.longitude.sin();
    let z_u = r_xz_u * camera_state.longitude.cos();
    let pos_unit = Vec3::new(x_u, y_u, z_u);

    let noise_val = planet_shader::snoise3(pos_unit * 1.5);
    let height = 2.0 + noise_val * 0.25;
    let min_allowed = height + 0.15; // Keep camera 0.15 units above displaced surface
    camera_state.distance = camera_state.distance.clamp(min_allowed, camera_state.max_distance);

    let Ok(mut transform) = camera_query.single_mut() else {
        return;
    };

    // Position in spherical coordinates
    let y = camera_state.distance * camera_state.latitude.sin();
    let r_xz = camera_state.distance * camera_state.latitude.cos();
    let x = r_xz * camera_state.longitude.sin();
    let z = r_xz * camera_state.longitude.cos();
    let camera_pos = Vec3::new(x, y, z);

    let local_up = camera_pos.normalize();
    let local_right = Vec3::Y.cross(local_up).normalize();
    let local_forward = local_right.cross(local_up).normalize();

    // Track a point on the surface of the planet that is a fraction of the way to the horizon.
    // The horizon angle from the camera is acos(R / d).
    let r = 2.0; // Planet radius
    let d = camera_state.distance;
    let horizon_angle = (r / d).clamp(0.0, 1.0).acos();

    // Look at a target point on the surface at a fraction of the horizon angle (e.g. 85%)
    let target_angle = 0.85 * horizon_angle;
    let target_pos = (local_up * target_angle.cos() + local_forward * target_angle.sin()) * r;

    // Look direction is from the camera to the target point
    let look_dir = (target_pos - camera_pos).normalize();

    *transform = Transform::from_translation(camera_pos).looking_to(look_dir, local_up);
}

// ---------------------------------------------------------------------------
// 2. GPU Shading and Pipeline Resource Setups
// ---------------------------------------------------------------------------

#[derive(ShaderType, Clone, Default)]
struct GlobalsUniform {
    camera_pos: Vec3,
    planet_radius: f32,
    planet_center: Vec3,
    noise_frequency: f32,
    noise_amplitude: f32,
    dummy: f32,
    frustum_planes: [Vec4; 6],
}

#[derive(ShaderType, Clone, Default)]
struct ViewUniform {
    view_proj: Mat4,
    light_dir: Vec3,
    ambient: f32,
}

#[repr(C)]
#[derive(Copy, Clone, Debug, Default, Zeroable, Pod)]
struct DrawIndexedIndirect {
    index_count: u32,
    instance_count: u32,
    first_index: u32,
    base_vertex: i32,
    first_instance: u32,
}

#[derive(Resource)]
struct PlanetPipeline {
    compute_pipeline_id: CachedComputePipelineId,
    render_pipeline_id: CachedRenderPipelineId,
}

#[derive(Resource)]
struct PlanetGpuResources {
    vertex_buffer: Buffer,
    index_buffer: Buffer,
    indirect_buffer: Buffer,
    vertex_counter_buffer: Buffer,
}

#[derive(Resource, Default)]
struct RenderGlobalsUniform(UniformBuffer<GlobalsUniform>);

#[derive(Resource, Default)]
struct RenderViewUniform(UniformBuffer<ViewUniform>);

fn init_gpu_resources(
    mut commands: Commands,
    render_device: Res<RenderDevice>,
    render_queue: Res<RenderQueue>,
    asset_server: Res<AssetServer>,
    pipeline_cache: Res<PipelineCache>,
    mut render_globals: ResMut<RenderGlobalsUniform>,
    mut render_view: ResMut<RenderViewUniform>,
) {
    // Create base storage buffers
    // Max 65,536 vertices. Each vertex holds position (16 bytes) and normal (16 bytes) = 32 bytes
    let vertex_buffer = render_device.create_buffer(&BufferDescriptor {
        label: Some("Planet Vertex Buffer"),
        size: 65536 * 32,
        usage: BufferUsages::STORAGE,
        mapped_at_creation: false,
    });

    // Max 131,072 indices * 4 bytes per index = 524,288 bytes
    let index_buffer = render_device.create_buffer(&BufferDescriptor {
        label: Some("Planet Index Buffer"),
        size: 131072 * 4,
        usage: BufferUsages::STORAGE | BufferUsages::INDEX,
        mapped_at_creation: false,
    });

    // Indirect Arguments Buffer (20 bytes)
    let indirect_buffer = render_device.create_buffer(&BufferDescriptor {
        label: Some("Planet Indirect Arguments Buffer"),
        size: 20,
        usage: BufferUsages::STORAGE | BufferUsages::INDIRECT | BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    // Vertex Counter Buffer (4 bytes)
    let vertex_counter_buffer = render_device.create_buffer(&BufferDescriptor {
        label: Some("Planet Vertex Counter Buffer"),
        size: 4,
        usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    // Bind group layouts using vectors and static cows
    let compute_layout = BindGroupLayoutDescriptor {
        label: std::borrow::Cow::Borrowed("Planet Compute Bind Group Layout"),
        entries: vec![
            BindGroupLayoutEntry {
                binding: 0,
                visibility: ShaderStages::COMPUTE,
                ty: BindingType::Buffer {
                    ty: BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            BindGroupLayoutEntry {
                binding: 1,
                visibility: ShaderStages::COMPUTE,
                ty: BindingType::Buffer {
                    ty: BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            BindGroupLayoutEntry {
                binding: 2,
                visibility: ShaderStages::COMPUTE,
                ty: BindingType::Buffer {
                    ty: BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            BindGroupLayoutEntry {
                binding: 3,
                visibility: ShaderStages::COMPUTE,
                ty: BindingType::Buffer {
                    ty: BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            BindGroupLayoutEntry {
                binding: 4,
                visibility: ShaderStages::COMPUTE,
                ty: BindingType::Buffer {
                    ty: BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
        ],
    };

    let render_layout = BindGroupLayoutDescriptor {
        label: std::borrow::Cow::Borrowed("Planet Render Bind Group Layout"),
        entries: vec![
            BindGroupLayoutEntry {
                binding: 0,
                visibility: ShaderStages::VERTEX | ShaderStages::FRAGMENT,
                ty: BindingType::Buffer {
                    ty: BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            BindGroupLayoutEntry {
                binding: 1,
                visibility: ShaderStages::VERTEX,
                ty: BindingType::Buffer {
                    ty: BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
        ],
    };

    // Initialize uniforms
    render_globals.0.write_buffer(&render_device, &render_queue);
    render_view.0.write_buffer(&render_device, &render_queue);

    // Compile pipelines
    let compute_shader = asset_server.load(SHADER_COMPUTE_PATH);
    let render_shader = asset_server.load(SHADER_RENDER_PATH);

    let compute_pipeline_id = pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
        label: Some(std::borrow::Cow::Borrowed("Planet Compute Pipeline")),
        layout: vec![compute_layout.clone()],
        shader: compute_shader,
        entry_point: Some(std::borrow::Cow::Borrowed("main")),
        push_constant_ranges: vec![],
        shader_defs: vec![],
        zero_initialize_workgroup_memory: false,
    });

    let render_pipeline_id = pipeline_cache.queue_render_pipeline(RenderPipelineDescriptor {
        label: Some(std::borrow::Cow::Borrowed("Planet Render Pipeline")),
        layout: vec![render_layout.clone()],
        vertex: VertexState {
            shader: render_shader.clone(),
            entry_point: Some(std::borrow::Cow::Borrowed("vs_main")),
            buffers: vec![],
            shader_defs: vec![],
        },
        fragment: Some(FragmentState {
            shader: render_shader,
            entry_point: Some(std::borrow::Cow::Borrowed("fs_main")),
            targets: vec![Some(ColorTargetState {
                format: TextureFormat::Rgba8UnormSrgb,
                blend: Some(BlendState::REPLACE),
                write_mask: ColorWrites::ALL,
            })],
            shader_defs: vec![],
        }),
        primitive: PrimitiveState {
            topology: PrimitiveTopology::TriangleList,
            cull_mode: Some(Face::Back),
            ..default()
        },
        depth_stencil: Some(DepthStencilState {
            format: TextureFormat::Depth32Float,
            depth_write_enabled: true,
            depth_compare: CompareFunction::GreaterEqual,
            stencil: StencilState::default(),
            bias: DepthBiasState::default(),
        }),
        multisample: MultisampleState::default(),
        push_constant_ranges: vec![],
        zero_initialize_workgroup_memory: false,
    });

    commands.insert_resource(PlanetPipeline {
        compute_pipeline_id,
        render_pipeline_id,
    });

    commands.insert_resource(PlanetGpuResources {
        vertex_buffer,
        index_buffer,
        indirect_buffer,
        vertex_counter_buffer,
    });
}

fn extract_frustum_planes(m: Mat4) -> [Vec4; 6] {
    let row0 = m.row(0);
    let row1 = m.row(1);
    let row2 = m.row(2);
    let row3 = m.row(3);

    let mut planes = [
        row3 + row0, // Left
        row3 - row0, // Right
        row3 + row1, // Bottom
        row3 - row1, // Top
        row3 + row2, // Near
        row3 - row2, // Far
    ];

    for plane in &mut planes {
        let normal = Vec3::new(plane.x, plane.y, plane.z);
        let len = normal.length();
        *plane /= len;
    }

    planes
}

fn prepare_uniforms(
    render_device: Res<RenderDevice>,
    render_queue: Res<RenderQueue>,
    mut render_globals: ResMut<RenderGlobalsUniform>,
    mut render_view: ResMut<RenderViewUniform>,
    resources: Option<Res<PlanetGpuResources>>,
    view_query: Query<&ExtractedView>,
) {
    let Some(gpu_resources) = resources else {
        return;
    };

    let Ok(extracted_view) = view_query.single() else {
        return;
    };

    let camera_pos = extracted_view.world_from_view.translation();
    let clip_from_view = extracted_view.clip_from_view;
    let world_from_view = extracted_view.world_from_view.to_matrix();
    let view_from_world = world_from_view.inverse();
    let view_proj = extracted_view.clip_from_world.unwrap_or(clip_from_view * view_from_world);
    let frustum_planes = extract_frustum_planes(view_proj);

    // Update global settings for compute shader
    let globals = GlobalsUniform {
        camera_pos,
        planet_radius: 2.0,
        planet_center: Vec3::ZERO,
        noise_frequency: 1.5,
        noise_amplitude: 0.25,
        dummy: 0.0,
        frustum_planes,
    };
    render_globals.0.set(globals);
    render_globals.0.write_buffer(&render_device, &render_queue);

    // Update view matrix for render pass
    let view = ViewUniform {
        view_proj,
        light_dir: Vec3::new(1.0, 1.0, 1.0).normalize(),
        ambient: 0.15,
    };
    render_view.0.set(view);
    render_view.0.write_buffer(&render_device, &render_queue);

    // Reset dynamic counters on the GPU
    let zero_args = DrawIndexedIndirect {
        index_count: 0,
        instance_count: 1, // Draw exactly 1 instance of the dynamic mesh
        first_index: 0,
        base_vertex: 0,
        first_instance: 0,
    };
    render_queue.write_buffer(&gpu_resources.indirect_buffer, 0, bytemuck::bytes_of(&zero_args));
    render_queue.write_buffer(&gpu_resources.vertex_counter_buffer, 0, bytemuck::bytes_of(&0u32));
}

// ---------------------------------------------------------------------------
// 3. Render Graph Integration
// ---------------------------------------------------------------------------

#[derive(RenderLabel, Debug, Hash, PartialEq, Eq, Clone)]
struct PlanetRenderLabel;


struct PlanetRenderNode;

impl render_graph::Node for PlanetRenderNode {
    fn run<'w>(
        &self,
        graph: &mut render_graph::RenderGraphContext,
        render_context: &mut RenderContext<'w>,
        world: &'w World,
    ) -> Result<(), render_graph::NodeRunError> {
        let pipeline = world.resource::<PlanetPipeline>();
        let resources = world.resource::<PlanetGpuResources>();
        let pipeline_cache = world.resource::<PipelineCache>();
        let render_globals = world.resource::<RenderGlobalsUniform>();
        let render_view = world.resource::<RenderViewUniform>();

        // Ensure shaders are compiled
        let Some(compute_pipeline) = pipeline_cache.get_compute_pipeline(pipeline.compute_pipeline_id) else {
            return Ok(());
        };
        let Some(render_pipeline) = pipeline_cache.get_render_pipeline(pipeline.render_pipeline_id) else {
            return Ok(());
        };

        // Query layouts dynamically from compiled pipelines
        let compute_layout: BindGroupLayout = compute_pipeline.get_bind_group_layout(0).into();
        let render_layout: BindGroupLayout = render_pipeline.get_bind_group_layout(0).into();

        // Create bind groups dynamically when layouts are resolved
        let compute_bind_group = render_context.render_device().create_bind_group(
            None,
            &compute_layout,
            &BindGroupEntries::sequential((
                render_globals.0.binding().unwrap(),
                resources.vertex_buffer.as_entire_buffer_binding(),
                resources.index_buffer.as_entire_buffer_binding(),
                resources.indirect_buffer.as_entire_buffer_binding(),
                resources.vertex_counter_buffer.as_entire_buffer_binding(),
            )),
        );

        let render_bind_group = render_context.render_device().create_bind_group(
            None,
            &render_layout,
            &BindGroupEntries::sequential((
                render_view.0.binding().unwrap(),
                resources.vertex_buffer.as_entire_buffer_binding(),
            )),
        );

        // Query render target and depth views
        let view_entity = graph.view_entity();
        let view_target = world.get::<bevy::render::view::ViewTarget>(view_entity).unwrap();
        let view_depth = world.get::<bevy::render::view::ViewDepthTexture>(view_entity).unwrap();

        // 1. Run Compute Pass to generate LOD subdivision and vertices
        {
            let mut compute_pass = render_context.command_encoder().begin_compute_pass(&ComputePassDescriptor {
                label: Some("Planet Compute Pass"),
                timestamp_writes: None,
            });
            compute_pass.set_pipeline(compute_pipeline);
            compute_pass.set_bind_group(0, &compute_bind_group, &[]);
            compute_pass.dispatch_workgroups(1, 1, 1); // 20 threads (one per icosahedron face)
        }

        // 2. Run Render Pass using indirect drawing
        {
            let mut render_pass = render_context.command_encoder().begin_render_pass(&RenderPassDescriptor {
                label: Some("Planet Render Pass"),
                color_attachments: &[Some(RenderPassColorAttachment {
                    view: view_target.main_texture_view(),
                    resolve_target: None,
                    ops: Operations {
                        load: LoadOp::Load, // Preserve background/cleared pixels
                        store: StoreOp::Store,
                    },
                    depth_slice: None,
                })],
                depth_stencil_attachment: Some(RenderPassDepthStencilAttachment {
                    view: view_depth.view(),
                    depth_ops: Some(Operations {
                        load: LoadOp::Load, // Preserve existing depth
                        store: StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
            });
            render_pass.set_pipeline(render_pipeline);
            render_pass.set_bind_group(0, &render_bind_group, &[]);
            render_pass.set_index_buffer((*resources.index_buffer).slice(..), IndexFormat::Uint32);
            render_pass.draw_indexed_indirect(&resources.indirect_buffer, 0);
        }

        Ok(())
    }
}


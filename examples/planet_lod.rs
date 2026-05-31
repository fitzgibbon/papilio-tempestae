// Bevy example: Procedural 3D Isosphere planet with GPU-driven dynamic LOD and frustum culling.

use bevy::{
    prelude::*,
    input::mouse::{MouseMotion, MouseWheel},
    render::{
        view::ExtractedView,
        render_graph::{self, RenderGraph, RenderLabel},
        render_resource::*,
        renderer::{RenderContext, RenderDevice, RenderQueue},
        Render, RenderApp, RenderStartup, Extract, ExtractSchedule,
    },
    core_pipeline::core_3d::graph::Node3d,
    diagnostic::{DiagnosticsStore, FrameTimeDiagnosticsPlugin},
};
use bytemuck::{Pod, Zeroable};

const SHADER_COMPUTE_PATH: &str = "shaders/terrain.wgsl";
const SHADER_RENDER_PATH: &str = "shaders/render_shaders.wgsl";

// Shader configuration constants
const PLANET_RADIUS: f32 = 100.0;
const EYE_HEIGHT: f32 = 1.0;
const NOISE_FREQUENCY: f32 = 1.5;
const NOISE_AMPLITUDE: f32 = 40.0;
const LOD_SPLIT_FACTOR: f32 = 4500.0; // scaled 100x (45.0 * 100.0)

// Maximum buffer capacities scaled up by 128x to support high LOD levels safely
const MAX_VERTICES: usize = 65536 * 128; // 8,388,608 vertices
const MAX_QUEUE_SIZE: usize = 2097152; // 2,097,152 triangles max queue size

fn main() {
    App::new()
        .add_plugins((
            DefaultPlugins,
            PlanetRenderPlugin,
            FrameTimeDiagnosticsPlugin::default(),
        ))
        .insert_resource(ClearColor(Color::BLACK))
        .run();
}

// ---------------------------------------------------------------------------
// 1. CPU State and Input Management
// ---------------------------------------------------------------------------

#[derive(Resource, Clone)]
struct PlanetCameraState {
    pos_unit: Vec3,
    local_forward: Vec3,
    look_pitch: f32,
    elevation: f32,
    max_distance: f32,
    show_wireframe: bool,
}

impl Default for PlanetCameraState {
    fn default() -> Self {
        let latitude = 0.1f32;
        let longitude = 0.0f32;
        let y_u = latitude.sin();
        let r_xz_u = latitude.cos();
        let x_u = r_xz_u * longitude.sin();
        let z_u = r_xz_u * longitude.cos();
        let pos_unit = Vec3::new(x_u, y_u, z_u);

        let local_up = pos_unit;
        let local_right = if local_up.y.abs() > 0.99 {
            Vec3::X.cross(local_up).normalize()
        } else {
            Vec3::Y.cross(local_up).normalize()
        };
        let local_forward = local_right.cross(local_up).normalize();

        Self {
            pos_unit,
            local_forward,
            look_pitch: -std::f32::consts::FRAC_PI_2,
            elevation: 500.0,
            max_distance: 650.0,
            show_wireframe: true,
        }
    }
}

fn extract_camera_state(
    mut commands: Commands,
    camera_state: Extract<Res<PlanetCameraState>>,
) {
    commands.insert_resource(camera_state.clone());
}

struct PlanetRenderPlugin;

impl Plugin for PlanetRenderPlugin {
    fn build(&self, app: &mut App) {
        app.insert_resource(PlanetCameraState::default())
        .add_systems(Startup, setup_scene)
        .add_systems(Update, (update_camera_and_state, update_ui));

        let Some(render_app) = app.get_sub_app_mut(RenderApp) else {
            return;
        };

        render_app
            .insert_resource(PlanetCameraState::default())
            .add_systems(ExtractSchedule, extract_camera_state)
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

#[derive(Component)]
struct AltitudeText;

fn setup_scene(mut commands: Commands, mut cursor_options: Query<&mut bevy::window::CursorOptions>) {
    // Grab and lock the cursor at startup so mouselook is active immediately
    if let Ok(mut cursor) = cursor_options.single_mut() {
        cursor.visible = false;
        cursor.grab_mode = bevy::window::CursorGrabMode::Locked;
    }

    // Spawn directional light representing the sun
    commands.spawn((
        DirectionalLight {
            shadows_enabled: false,
            ..default()
        },
        Transform::from_rotation(Quat::from_rotation_x(-0.5)),
    ));

    // Spawn 3D camera with default near projection plane and 90 degree FoV
    commands.spawn((
        Camera3d::default(),
        Projection::Perspective(PerspectiveProjection {
            fov: std::f32::consts::FRAC_PI_2,
            ..default()
        }),
        Transform::from_xyz(0.0, 0.0, 12.0).looking_at(Vec3::ZERO, Vec3::Y),
        Msaa::Off,
    ));

    // Spawn UI Text for Altitude Overlay
    commands.spawn((
        Text::new("Intended Altitude: 0.0000\nActual Altitude: 0.0000\nFPS: 0.0"),
        TextFont {
            font_size: 20.0,
            ..default()
        },
        TextColor(Color::WHITE),
        Node {
            position_type: PositionType::Absolute,
            top: Val::Px(15.0),
            left: Val::Px(15.0),
            ..default()
        },
        AltitudeText,
    ));
}

fn update_ui(
    camera_state: Res<PlanetCameraState>,
    camera_query: Query<&Transform, With<Camera3d>>,
    mut text_query: Query<&mut Text, With<AltitudeText>>,
    diagnostics: Res<DiagnosticsStore>,
) {
    let Ok(camera_transform) = camera_query.single() else {
        return;
    };
    let actual_alt = camera_transform.translation.length() - PLANET_RADIUS;
    let intended_alt = camera_state.elevation;
    let fps = diagnostics
        .get(&FrameTimeDiagnosticsPlugin::FPS)
        .and_then(|fps| fps.smoothed())
        .unwrap_or(0.0);

    let wireframe_status = if camera_state.show_wireframe { "ON" } else { "OFF" };

    for mut text in text_query.iter_mut() {
        text.0 = format!(
            "Intended Altitude: {:.4}\nActual Altitude: {:.4}\nFPS: {:.1}\nWireframe (Press V to toggle): {}",
            intended_alt, actual_alt, fps, wireframe_status
        );
    }
}

fn update_camera_and_state(
    mut camera_state: ResMut<PlanetCameraState>,
    mouse_buttons: Res<ButtonInput<MouseButton>>,
    mut mouse_wheel_events: MessageReader<MouseWheel>,
    mut mouse_motion_events: MessageReader<MouseMotion>,
    mut camera_query: Query<&mut Transform, With<Camera3d>>,
    keyboard: Res<ButtonInput<KeyCode>>,
    mut app_exit_events: MessageWriter<AppExit>,
    mut cursor_options: Query<&mut bevy::window::CursorOptions>,
    time: Res<Time>,
) {
    // 1. Handle KeyQ to quit the application
    if keyboard.just_pressed(KeyCode::KeyQ) {
        app_exit_events.write(AppExit::Success);
        return;
    }

    let get_height_at = |dir: Vec3| -> f32 {
        let pos_unit = dir.normalize();
        let eps = 0.01f32;
        let mut total_disp = 0.0f32;
        let mut accum_grad = Vec3::ZERO;

        let sample_noise_rust = |p: Vec3| -> f32 {
            planet_shader::snoise3(planet_shader::glam::Vec3::new(p.x, p.y, p.z))
        };

        let f_mask = NOISE_FREQUENCY * 0.4;
        let n_mask = sample_noise_rust(pos_unit * f_mask);
        let mountain_density = ((n_mask * 1.8) + 0.3).clamp(0.0, 1.0);
        let mountain_factor = mountain_density * mountain_density;

        // Octave 0
        let f0 = NOISE_FREQUENCY;
        let a0 = NOISE_AMPLITUDE * 0.5;

        let p0_plains = pos_unit * f0;
        let n0_plains = sample_noise_rust(p0_plains) * 0.25;

        let p0_mount = pos_unit * (f0 * 0.8);
        let n0_mount = 1.0 - sample_noise_rust(p0_mount).abs();

        let n0 = n0_plains * (1.0 - mountain_factor) + (n0_mount * 1.1 - 0.3) * mountain_factor;

        let sample_blended_octave0 = |pos: Vec3| -> f32 {
            let mask = ((sample_noise_rust(pos * f_mask) * 1.8) + 0.3).clamp(0.0, 1.0);
            let m_factor = mask * mask;
            let plains = sample_noise_rust(pos * f0) * 0.25;
            let mount = 1.0 - sample_noise_rust(pos * (f0 * 0.8)).abs();
            plains * (1.0 - m_factor) + (mount * 1.1 - 0.3) * m_factor
        };

        let dx0 = sample_blended_octave0(pos_unit + Vec3::new(eps, 0.0, 0.0)) - n0;
        let dy0 = sample_blended_octave0(pos_unit + Vec3::new(0.0, eps, 0.0)) - n0;
        let dz0 = sample_blended_octave0(pos_unit + Vec3::new(0.0, 0.0, eps)) - n0;
        let g0 = Vec3::new(dx0, dy0, dz0) / eps;

        total_disp += n0 * a0;
        accum_grad += g0 * a0;

        // Octave 1
        let f1 = f0 * 2.0;
        let a1 = a0 * 0.35;
        let w1 = 0.1 + 1.9 * (accum_grad.length() / (a0 * f0)).clamp(0.0, 1.0);
        let p1 = pos_unit * f1;
        let n1 = sample_noise_rust(p1);
        let dx1 = sample_noise_rust(p1 + Vec3::new(eps, 0.0, 0.0)) - n1;
        let dy1 = sample_noise_rust(p1 + Vec3::new(0.0, eps, 0.0)) - n1;
        let dz1 = sample_noise_rust(p1 + Vec3::new(0.0, 0.0, eps)) - n1;
        let g1 = Vec3::new(dx1, dy1, dz1) / eps;
        total_disp += n1 * a1 * w1;
        accum_grad += g1 * a1 * w1;

        // Octave 2
        let f2 = f1 * 2.0;
        let a2 = a1 * 0.35;
        let w2 = 0.1 + 1.9 * (accum_grad.length() / (a0 * f0)).clamp(0.0, 1.0);
        let p2 = pos_unit * f2;
        let n2 = sample_noise_rust(p2);
        let dx2 = sample_noise_rust(p2 + Vec3::new(eps, 0.0, 0.0)) - n2;
        let dy2 = sample_noise_rust(p2 + Vec3::new(0.0, eps, 0.0)) - n2;
        let dz2 = sample_noise_rust(p2 + Vec3::new(0.0, 0.0, eps)) - n2;
        let g2 = Vec3::new(dx2, dy2, dz2) / eps;
        total_disp += n2 * a2 * w2;
        accum_grad += g2 * a2 * w2;

        // Octave 3
        let f3 = f2 * 2.0;
        let a3 = a2 * 0.35;
        let w3 = 0.1 + 1.9 * (accum_grad.length() / (a0 * f0)).clamp(0.0, 1.0);
        let p3 = pos_unit * f3;
        let n3 = sample_noise_rust(p3);
        total_disp += n3 * a3 * w3;

        // Add Sedimentary Terracing Effect on slopes
        let slope = (accum_grad.length() / (a0 * f0)).clamp(0.0, 1.0);
        let terrace_noise = sample_noise_rust(pos_unit * (f0 * 4.0));
        let terrace_pattern = (total_disp * 1.5 + terrace_noise * 0.4).sin();
        let terrace_amp = 0.8 * slope * mountain_density;
        total_disp += terrace_pattern * terrace_amp;

        // Clamp displacement to flat ocean floor
        total_disp = total_disp.max(-2.5);

        PLANET_RADIUS + total_disp
    };

    // Grab or release the cursor
    if mouse_buttons.just_pressed(MouseButton::Left) {
        if let Ok(mut cursor) = cursor_options.single_mut() {
            cursor.visible = false;
            cursor.grab_mode = bevy::window::CursorGrabMode::Locked;
        }
    }
    if keyboard.just_pressed(KeyCode::Escape) {
        if let Ok(mut cursor) = cursor_options.single_mut() {
            cursor.visible = true;
            cursor.grab_mode = bevy::window::CursorGrabMode::None;
        }
    }
    // Toggle wireframe view
    if keyboard.just_pressed(KeyCode::KeyV) {
        camera_state.show_wireframe = !camera_state.show_wireframe;
    }

    // Accumulate mouse movement for mouselook if cursor is locked
    let mut mouse_dx = 0.0f32;
    let mut mouse_dy = 0.0f32;
    for event in mouse_motion_events.read() {
        mouse_dx += event.delta.x;
        mouse_dy += event.delta.y;
    }

    let is_locked = cursor_options
        .single()
        .map(|c| c.grab_mode == bevy::window::CursorGrabMode::Locked)
        .unwrap_or(false);

    if is_locked {
        let sensitivity = 0.002f32;

        // Mouse horizontal movement directly rotates player heading (local_forward) on the sphere surface
        if mouse_dx != 0.0 {
            let d_yaw = mouse_dx * sensitivity;
            let local_right = camera_state.local_forward.cross(camera_state.pos_unit).normalize();
            camera_state.local_forward = (camera_state.local_forward * d_yaw.cos() + local_right * d_yaw.sin()).normalize();
        }

        // Mouse vertical movement changes look pitch relative to the horizon (reversed so pushing mouse up pitches down)
        camera_state.look_pitch += mouse_dy * sensitivity;
        // Clamp pitch to prevent going past straight down / straight up
        camera_state.look_pitch = camera_state.look_pitch.clamp(
            -std::f32::consts::FRAC_PI_2 + 0.005,
            std::f32::consts::FRAC_PI_2 - 0.005,
        );
    }

    // Zoom (elevation) with scroll wheel
    let mut scroll = 0.0f32;
    for event in mouse_wheel_events.read() {
        scroll += event.y;
    }
    let zoom_speed = 0.08f32 * camera_state.elevation.max(7.5);
    camera_state.elevation -= scroll * zoom_speed;
    camera_state.elevation = camera_state.elevation.clamp(0.0, camera_state.max_distance);

    let dt = time.delta_secs();

    // Scale movement speed with elevation to make traversal comfortable at high altitudes
    let walk_speed = (0.05f32 + camera_state.elevation * 0.004) * dt;

    let mut move_forward = 0.0f32;
    if keyboard.pressed(KeyCode::KeyW) {
        move_forward += 1.0;
    }
    if keyboard.pressed(KeyCode::KeyS) {
        move_forward -= 1.0;
    }

    let mut move_right = 0.0f32;
    if keyboard.pressed(KeyCode::KeyD) {
        move_right += 1.0;
    }
    if keyboard.pressed(KeyCode::KeyA) {
        move_right -= 1.0;
    }

    // Apply W/S movement along player body forward vector (geodesic rotation)
    if move_forward != 0.0 {
        let d_theta = move_forward * walk_speed;
        let new_pos = (camera_state.pos_unit * d_theta.cos() + camera_state.local_forward * d_theta.sin()).normalize();
        let new_forward = (camera_state.local_forward * d_theta.cos() - camera_state.pos_unit * d_theta.sin()).normalize();
        camera_state.pos_unit = new_pos;
        camera_state.local_forward = new_forward;
    }

    // Apply A/D movement along player body right vector (geodesic rotation)
    if move_right != 0.0 {
        let d_theta = move_right * walk_speed;
        let local_right = camera_state.local_forward.cross(camera_state.pos_unit).normalize();
        let new_pos = (camera_state.pos_unit * d_theta.cos() + local_right * d_theta.sin()).normalize();
        camera_state.pos_unit = new_pos;
    }

    // Sanitize frame vectors to guarantee orthogonality and prevent float drift
    camera_state.local_forward = (camera_state.local_forward - camera_state.pos_unit * camera_state.local_forward.dot(camera_state.pos_unit)).normalize();

    // Determine final camera distance: terrain heightmap + elevation + player eye height offset (EYE_HEIGHT)
    let terrain_height = get_height_at(camera_state.pos_unit);
    let actual_distance = terrain_height + camera_state.elevation + EYE_HEIGHT;

    let camera_pos = camera_state.pos_unit * actual_distance;

    // Construct right-handed orthonormal camera orientation frame (Right, Up, -Forward)
    // to avoid looking_to's colinear up/look singularity when looking straight down/up.
    let camera_up = camera_state.pos_unit * camera_state.look_pitch.cos() - camera_state.local_forward * camera_state.look_pitch.sin();
    let camera_forward = (camera_state.local_forward * camera_state.look_pitch.cos() + camera_state.pos_unit * camera_state.look_pitch.sin()).normalize();
    let camera_right = camera_forward.cross(camera_up).normalize();
    let camera_rotation = Quat::from_mat3(&Mat3::from_cols(
        camera_right,
        camera_up,
        -camera_forward,
    ));

    let Ok(mut transform) = camera_query.single_mut() else {
        return;
    };
    *transform = Transform::from_translation(camera_pos).with_rotation(camera_rotation);
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
    lod_split_factor: f32,
    frustum_planes: [Vec4; 6],
}

#[derive(ShaderType, Clone, Default)]
struct ViewUniform {
    view_proj: Mat4,
    light_dir: Vec3,
    ambient: f32,
    camera_pos: Vec3,
    show_wireframe: f32,
}

#[repr(C)]
#[derive(Copy, Clone, Debug, Default, Zeroable, Pod)]
struct DrawIndirect {
    vertex_count: u32,
    instance_count: u32,
    first_vertex: u32,
    first_instance: u32,
}

#[repr(C)]
#[derive(Copy, Clone, Debug, Default, Zeroable, Pod)]
struct PassUniforms {
    depth: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

#[repr(C)]
#[derive(Copy, Clone, Debug, Default, Zeroable, Pod)]
struct GpuTriangle {
    v0: [f32; 4],
    v1: [f32; 4],
    v2: [f32; 4],
}

#[derive(Resource)]
struct PlanetPipeline {
    compute_pipeline_id: CachedComputePipelineId,
    render_pipeline_id: CachedRenderPipelineId,
}

#[derive(Resource)]
struct PlanetGpuResources {
    vertex_buffer: Buffer,
    indirect_buffer: Buffer,
    base_faces_buffer: Buffer,
    queue_a: Buffer,
    queue_b: Buffer,
    counter_a: Buffer,
    counter_b: Buffer,
    pass_buffers: Vec<Buffer>,
}

#[derive(Resource, Default)]
struct RenderGlobalsUniform(UniformBuffer<GlobalsUniform>);

#[derive(Resource, Default)]
struct RenderViewUniform(UniformBuffer<ViewUniform>);

#[allow(clippy::excessive_precision)]
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
    // Each vertex holds position (16 bytes) and normal (16 bytes) = 32 bytes
    let vertex_buffer = render_device.create_buffer(&BufferDescriptor {
        label: Some("Planet Vertex Buffer"),
        size: (MAX_VERTICES * 32) as u64,
        usage: BufferUsages::STORAGE,
        mapped_at_creation: false,
    });

    // Indirect Arguments Buffer (16 bytes)
    let indirect_buffer = render_device.create_buffer(&BufferDescriptor {
        label: Some("Planet Indirect Arguments Buffer"),
        size: 16,
        usage: BufferUsages::STORAGE | BufferUsages::INDIRECT | BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    // 20 Base icosahedron faces
    let x = 0.5257311121191336_f32;
    let z = 0.8506508083520399_f32;
    let base_vertices = [
        Vec3::new(-x, z, 0.0), Vec3::new(x, z, 0.0), Vec3::new(-x, -z, 0.0), Vec3::new(x, -z, 0.0),
        Vec3::new(0.0, -x, z), Vec3::new(0.0, x, z), Vec3::new(0.0, -x, -z), Vec3::new(0.0, x, -z),
        Vec3::new(z, 0.0, -x), Vec3::new(z, 0.0, x), Vec3::new(-z, 0.0, -x), Vec3::new(-z, 0.0, x)
    ];
    let base_faces = [
        [0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11],
        [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8],
        [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9],
        [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1]
    ];

    let mut base_triangles = Vec::new();
    for face in base_faces {
        base_triangles.push(GpuTriangle {
            v0: base_vertices[face[0]].extend(0.0).to_array(),
            v1: base_vertices[face[1]].extend(0.0).to_array(),
            v2: base_vertices[face[2]].extend(0.0).to_array(),
        });
    }

    let base_faces_buffer = render_device.create_buffer(&BufferDescriptor {
        label: Some("Planet Base Faces Buffer"),
        size: 960,
        usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    render_queue.write_buffer(&base_faces_buffer, 0, bytemuck::cast_slice(&base_triangles));

    // Intermediate queues (Ping-Pong buffers)
    let queue_a = render_device.create_buffer(&BufferDescriptor {
        label: Some("Planet Queue A"),
        size: (MAX_QUEUE_SIZE * 48) as u64,
        usage: BufferUsages::STORAGE,
        mapped_at_creation: false,
    });

    let queue_b = render_device.create_buffer(&BufferDescriptor {
        label: Some("Planet Queue B"),
        size: (MAX_QUEUE_SIZE * 48) as u64,
        usage: BufferUsages::STORAGE,
        mapped_at_creation: false,
    });

    // Counters for intermediate queues
    let counter_a = render_device.create_buffer(&BufferDescriptor {
        label: Some("Planet Counter A"),
        size: 4,
        usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let counter_b = render_device.create_buffer(&BufferDescriptor {
        label: Some("Planet Counter B"),
        size: 4,
        usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    // Static uniform buffers for depth 0..11
    let mut pass_buffers = Vec::new();
    for depth in 0..11 {
        let buffer = render_device.create_buffer(&BufferDescriptor {
            label: Some(&format!("Planet Pass Uniforms Depth {}", depth)),
            size: 16,
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let pass_uniforms = PassUniforms {
            depth: depth as u32,
            _pad0: 0,
            _pad1: 0,
            _pad2: 0,
        };
        render_queue.write_buffer(&buffer, 0, bytemuck::bytes_of(&pass_uniforms));
        pass_buffers.push(buffer);
    }

    // Bind group layouts
    let compute_layout = BindGroupLayoutDescriptor {
        label: std::borrow::Cow::Borrowed("Planet Compute Bind Group Layout"),
        entries: vec![
            // 0: Globals
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
            // 1: out_vertices
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
            // 2: indirect_args
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
            // 3: PassUniforms
            BindGroupLayoutEntry {
                binding: 3,
                visibility: ShaderStages::COMPUTE,
                ty: BindingType::Buffer {
                    ty: BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            // 4: base_faces (read-only storage)
            BindGroupLayoutEntry {
                binding: 4,
                visibility: ShaderStages::COMPUTE,
                ty: BindingType::Buffer {
                    ty: BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            // 5: input_queue
            BindGroupLayoutEntry {
                binding: 5,
                visibility: ShaderStages::COMPUTE,
                ty: BindingType::Buffer {
                    ty: BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            // 6: output_queue
            BindGroupLayoutEntry {
                binding: 6,
                visibility: ShaderStages::COMPUTE,
                ty: BindingType::Buffer {
                    ty: BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            // 7: input_counter (read-only storage)
            BindGroupLayoutEntry {
                binding: 7,
                visibility: ShaderStages::COMPUTE,
                ty: BindingType::Buffer {
                    ty: BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            // 8: output_counter
            BindGroupLayoutEntry {
                binding: 8,
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
        indirect_buffer,
        base_faces_buffer,
        queue_a,
        queue_b,
        counter_a,
        counter_b,
        pass_buffers,
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
    camera_state: Option<Res<PlanetCameraState>>,
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
        planet_radius: PLANET_RADIUS,
        planet_center: Vec3::ZERO,
        noise_frequency: NOISE_FREQUENCY,
        noise_amplitude: NOISE_AMPLITUDE,
        lod_split_factor: LOD_SPLIT_FACTOR,
        frustum_planes,
    };
    render_globals.0.set(globals);
    render_globals.0.write_buffer(&render_device, &render_queue);

    let show_wireframe = if let Some(state) = camera_state {
        if state.show_wireframe { 1.0f32 } else { 0.0f32 }
    } else {
        1.0f32
    };

    // Update view matrix for render pass
    let view = ViewUniform {
        view_proj,
        light_dir: Vec3::new(1.0, 1.0, 1.0).normalize(),
        ambient: 0.15,
        camera_pos,
        show_wireframe,
    };
    render_view.0.set(view);
    render_view.0.write_buffer(&render_device, &render_queue);

    // Reset dynamic indirect arguments buffer on the GPU
    let zero_args = DrawIndirect {
        vertex_count: 0,
        instance_count: 1, // Draw exactly 1 instance of the dynamic mesh
        first_vertex: 0,
        first_instance: 0,
    };
    render_queue.write_buffer(&gpu_resources.indirect_buffer, 0, bytemuck::bytes_of(&zero_args));
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

        let render_bind_group = render_context.render_device().create_bind_group(
            None,
            &render_layout,
            &BindGroupEntries::sequential((
                render_view.0.binding().unwrap(),
                resources.vertex_buffer.as_entire_buffer_binding(),
            )),
        );

        // Query render target and depth views safely (avoid panics on window exit)
        let view_entity = graph.view_entity();
        let Some(view_target) = world.get::<bevy::render::view::ViewTarget>(view_entity) else {
            return Ok(());
        };
        let Some(view_depth) = world.get::<bevy::render::view::ViewDepthTexture>(view_entity) else {
            return Ok(());
        };

        // 1. Run 11 sequential compute passes to subdivide dynamically
        for k in 0..11 {
            let (input_queue, output_queue, input_counter, output_counter) = if k % 2 == 0 {
                (
                    &resources.queue_a,
                    &resources.queue_b,
                    &resources.counter_a,
                    &resources.counter_b,
                )
            } else {
                (
                    &resources.queue_b,
                    &resources.queue_a,
                    &resources.counter_b,
                    &resources.counter_a,
                )
            };

            // Clear output counter to 0 on GPU
            render_context.command_encoder().clear_buffer(output_counter, 0, None);

            // Create dynamic bind group for pass k
            let compute_bind_group = render_context.render_device().create_bind_group(
                None,
                &compute_layout,
                &BindGroupEntries::sequential((
                    render_globals.0.binding().unwrap(),
                    resources.vertex_buffer.as_entire_buffer_binding(),
                    resources.indirect_buffer.as_entire_buffer_binding(),
                    resources.pass_buffers[k].as_entire_buffer_binding(),
                    resources.base_faces_buffer.as_entire_buffer_binding(),
                    input_queue.as_entire_buffer_binding(),
                    output_queue.as_entire_buffer_binding(),
                    input_counter.as_entire_buffer_binding(),
                    output_counter.as_entire_buffer_binding(),
                )),
            );

            // Dispatch workgroups (max possible triangles for pass k is 20 * 4^k)
            let max_triangles = 20 * 4u32.pow(k as u32);
            let workgroup_count = max_triangles.div_ceil(64);
            let workgroups_x = workgroup_count.min(65535);
            let workgroups_y = workgroup_count.div_ceil(65535);

            let mut compute_pass = render_context.command_encoder().begin_compute_pass(&ComputePassDescriptor {
                label: Some(&format!("Planet Compute Pass Depth {}", k)),
                timestamp_writes: None,
            });
            compute_pass.set_pipeline(compute_pipeline);
            compute_pass.set_bind_group(0, &compute_bind_group, &[]);
            compute_pass.dispatch_workgroups(workgroups_x, workgroups_y, 1);
        }

        // 2. Run Render Pass using indirect drawing (non-indexed)
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
            render_pass.draw_indirect(&resources.indirect_buffer, 0);
        }

        Ok(())
    }
}

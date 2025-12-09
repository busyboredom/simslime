//! A compute shader that simulates Conway's Game of Life.
//!
//! Compute shaders use the GPU for computing arbitrary information, that may be independent of what
//! is rendered to the screen.

use bevy::{
    DefaultPlugins,
    asset::RenderAssetUsages,
    prelude::{
        App, Assets, Camera2d, ClearColor, Color, Commands, DirectAssetAccessExt, FromWorld,
        Handle, Image, ImagePlugin, IntoScheduleConfigs, Plugin, PluginGroup, Res, ResMut,
        Resource, Single, Sprite, Startup, Transform, Update, Vec2, Vec3, Window, WindowPlugin,
        World, default,
    },
    render::{
        Render, RenderApp, RenderSystems,
        extract_resource::{ExtractResource, ExtractResourcePlugin},
        render_asset::RenderAssets,
        render_graph::{self, RenderGraph, RenderLabel},
        render_resource::{
            BindGroup, BindGroupEntries, BindGroupLayout, BindGroupLayoutEntries, Buffer,
            BufferInitDescriptor, BufferUsages, CachedComputePipelineId, CachedPipelineState,
            ComputePassDescriptor, ComputePipelineDescriptor, Extent3d, PipelineCache,
            ShaderStages, ShaderType, StorageTextureAccess, TextureDimension, TextureFormat,
            TextureUsages,
            binding_types::{storage_buffer, storage_buffer_read_only, texture_storage_2d},
        },
        renderer::{RenderContext, RenderDevice},
        texture::GpuImage,
    },
};
use std::borrow::Cow;

/// This example uses a shader source file from the assets subdirectory
const SHADER_ASSET_PATH: &str = "shaders.wgsl";

const DISPLAY_FACTOR: u32 = 1;
const SIZE: (u32, u32) = (1500 / DISPLAY_FACTOR, 1000 / DISPLAY_FACTOR);
const WORKGROUP_SIZE: u32 = 8;

#[derive(ShaderType, Clone)]
struct CountBuffer {
    count: u32,
}

fn main() {
    App::new()
        .insert_resource(ClearColor(Color::BLACK))
        .add_plugins((
            DefaultPlugins
                .set(WindowPlugin {
                    primary_window: Some(Window {
                        resolution: ((SIZE.0 * DISPLAY_FACTOR), (SIZE.1 * DISPLAY_FACTOR)).into(),
                        // uncomment for unthrottled FPS
                        present_mode: bevy::window::PresentMode::AutoNoVsync,
                        ..default()
                    }),
                    ..default()
                })
                .set(ImagePlugin::default_nearest()),
            GameOfLifeComputePlugin,
        ))
        .add_systems(Startup, setup)
        .add_systems(Update, switch_textures)
        .run();
}

fn setup(mut commands: Commands, mut images: ResMut<Assets<Image>>) {
    let mut image = Image::new_fill(
        Extent3d {
            width: SIZE.0,
            height: SIZE.1,
            depth_or_array_layers: 1,
        },
        TextureDimension::D2,
        &[0, 0, 0, 255],
        TextureFormat::R32Float,
        RenderAssetUsages::RENDER_WORLD,
    );
    image.texture_descriptor.usage =
        TextureUsages::COPY_DST | TextureUsages::STORAGE_BINDING | TextureUsages::TEXTURE_BINDING;
    let image_a = images.add(image.clone());
    let image_b = images.add(image);

    #[expect(clippy::cast_precision_loss)]
    commands.spawn((
        Sprite {
            image: image_a.clone(),
            custom_size: Some(Vec2::new(SIZE.0 as f32, SIZE.1 as f32)),
            ..default()
        },
        Transform::from_scale(Vec3::splat(DISPLAY_FACTOR as f32)),
    ));
    commands.spawn(Camera2d);

    commands.insert_resource(GameOfLifeImages {
        texture_a: image_a,
        texture_b: image_b,
    });
}

// Switch texture to display every frame to show the one that was written to most recently.
#[expect(clippy::needless_pass_by_value)]
fn switch_textures(images: Res<GameOfLifeImages>, mut sprite: Single<&mut Sprite>) {
    if sprite.image == images.texture_a {
        sprite.image = images.texture_b.clone();
    } else {
        sprite.image = images.texture_a.clone();
    }
}

struct GameOfLifeComputePlugin;

#[derive(Debug, Hash, PartialEq, Eq, Clone, RenderLabel)]
struct GameOfLifeLabel;

impl Plugin for GameOfLifeComputePlugin {
    fn build(&self, app: &mut App) {
        // Extract the game of life image resource from the main world into the render world
        // for operation on by the compute shader and display on the sprite.
        app.add_plugins(ExtractResourcePlugin::<GameOfLifeImages>::default());
        let render_app = app.sub_app_mut(RenderApp);
        render_app.add_systems(
            Render,
            prepare_bind_group.in_set(RenderSystems::PrepareBindGroups),
        );

        let mut render_graph = render_app.world_mut().resource_mut::<RenderGraph>();
        render_graph.add_node(GameOfLifeLabel, GameOfLifeNode::default());
        render_graph.add_node_edge(GameOfLifeLabel, bevy::render::graph::CameraDriverLabel);
    }

    fn finish(&self, app: &mut App) {
        let render_app = app.sub_app_mut(RenderApp);
        render_app.init_resource::<GameOfLifePipeline>();
    }
}

#[derive(Resource, Clone, ExtractResource)]
struct GameOfLifeImages {
    texture_a: Handle<Image>,
    texture_b: Handle<Image>,
}

#[derive(Resource)]
struct GameOfLifeBindGroups {
    // Ping-pong bind groups for the main simulation pass
    sim_group_a: BindGroup,
    sim_group_b: BindGroup,
    // Ping-pong bind groups for the counting pass
    count_group_a: BindGroup,
    count_group_b: BindGroup,
}

#[expect(clippy::needless_pass_by_value)]
fn prepare_bind_group(
    mut commands: Commands,
    pipeline: Res<GameOfLifePipeline>,
    gpu_images: Res<RenderAssets<GpuImage>>,
    game_of_life_images: Res<GameOfLifeImages>,
    render_device: Res<RenderDevice>,
) {
    let view_a = gpu_images.get(&game_of_life_images.texture_a).unwrap();
    let view_b = gpu_images.get(&game_of_life_images.texture_b).unwrap();
    let count_buffer_binding = pipeline.count_buffer.as_entire_binding();

    let sim_group_a = render_device.create_bind_group(
        None,
        &pipeline.texture_bind_group_layout,
        &BindGroupEntries::sequential((
            &view_a.texture_view,
            &view_b.texture_view,
            count_buffer_binding.clone(),
        )),
    );
    let sim_group_b = render_device.create_bind_group(
        None,
        &pipeline.texture_bind_group_layout,
        &BindGroupEntries::sequential((
            &view_b.texture_view,
            &view_a.texture_view,
            count_buffer_binding.clone(),
        )),
    );
    let count_group_a = render_device.create_bind_group(
        "count_bind_group_a",
        &pipeline.count_layout,
        &BindGroupEntries::sequential((&view_a.texture_view, count_buffer_binding.clone())),
    );
    let count_group_b = render_device.create_bind_group(
        "count_bind_group_b",
        &pipeline.count_layout,
        &BindGroupEntries::sequential((&view_b.texture_view, count_buffer_binding)),
    );

    commands.insert_resource(GameOfLifeBindGroups {
        sim_group_a,
        sim_group_b,
        count_group_a,
        count_group_b,
    });
}

#[derive(Resource)]
struct GameOfLifePipeline {
    texture_bind_group_layout: BindGroupLayout,
    count_layout: BindGroupLayout,
    count_buffer: Buffer,
    init_pipeline: CachedComputePipelineId,
    update_pipeline: CachedComputePipelineId,
    count_pipeline: CachedComputePipelineId,
}

impl FromWorld for GameOfLifePipeline {
    fn from_world(world: &mut World) -> Self {
        let render_device = world.resource::<RenderDevice>();
        let texture_bind_group_layout = render_device.create_bind_group_layout(
            "GameOfLifeImages",
            &BindGroupLayoutEntries::sequential(
                ShaderStages::COMPUTE,
                (
                    texture_storage_2d(TextureFormat::R32Float, StorageTextureAccess::ReadOnly),
                    texture_storage_2d(TextureFormat::R32Float, StorageTextureAccess::WriteOnly),
                    storage_buffer_read_only::<CountBuffer>(false),
                ),
            ),
        );

        let count_layout = render_device.create_bind_group_layout(
            "count_layout",
            &BindGroupLayoutEntries::sequential(
                ShaderStages::COMPUTE,
                (
                    texture_storage_2d(TextureFormat::R32Float, StorageTextureAccess::ReadOnly),
                    storage_buffer::<CountBuffer>(false),
                ),
            ),
        );

        let count_buffer = render_device.create_buffer_with_data(&BufferInitDescriptor {
            label: Some("count_buffer"),
            contents: &0u32.to_ne_bytes(),
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST | BufferUsages::COPY_SRC,
        });

        let shader = world.load_asset(SHADER_ASSET_PATH);
        let pipeline_cache = world.resource::<PipelineCache>();

        let init_pipeline = pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
            label: None,
            layout: vec![texture_bind_group_layout.clone()],
            push_constant_ranges: Vec::new(),
            shader: shader.clone(),
            shader_defs: vec![],
            entry_point: Some(Cow::from("init")),
            zero_initialize_workgroup_memory: false,
        });
        let update_pipeline = pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
            label: None,
            layout: vec![texture_bind_group_layout.clone()],
            push_constant_ranges: Vec::new(),
            shader: shader.clone(),
            shader_defs: vec![],
            entry_point: Some(Cow::from("update")),
            zero_initialize_workgroup_memory: false,
        });
        let count_pipeline = pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
            label: Some("count_pipeline".into()),
            layout: vec![count_layout.clone()],
            shader,
            shader_defs: Vec::new(),
            entry_point: Some(Cow::from("count_alive_pixels")),
            push_constant_ranges: Vec::new(),
            zero_initialize_workgroup_memory: false,
        });

        GameOfLifePipeline {
            texture_bind_group_layout,
            count_layout,
            count_buffer,
            init_pipeline,
            update_pipeline,
            count_pipeline,
        }
    }
}

enum GameOfLifeState {
    Loading,
    Init,
    Update(usize),
}

struct GameOfLifeNode {
    state: GameOfLifeState,
}

impl Default for GameOfLifeNode {
    fn default() -> Self {
        Self {
            state: GameOfLifeState::Loading,
        }
    }
}

impl render_graph::Node for GameOfLifeNode {
    fn update(&mut self, world: &mut World) {
        let pipeline = world.resource::<GameOfLifePipeline>();
        let pipeline_cache = world.resource::<PipelineCache>();

        // if the corresponding pipeline has loaded, transition to the next stage
        match self.state {
            GameOfLifeState::Loading => {
                match pipeline_cache.get_compute_pipeline_state(pipeline.init_pipeline) {
                    CachedPipelineState::Ok(_) => {
                        self.state = GameOfLifeState::Init;
                    }
                    CachedPipelineState::Err(err) => {
                        panic!("Initializing assets/{SHADER_ASSET_PATH}:\n{err}")
                    }
                    _ => {}
                }
            }
            GameOfLifeState::Init => {
                if let (CachedPipelineState::Ok(_), CachedPipelineState::Ok(_)) = (
                    pipeline_cache.get_compute_pipeline_state(pipeline.update_pipeline),
                    pipeline_cache.get_compute_pipeline_state(pipeline.count_pipeline),
                ) {
                    self.state = GameOfLifeState::Update(1);
                }
            }
            GameOfLifeState::Update(0) => {
                self.state = GameOfLifeState::Update(1);
            }
            GameOfLifeState::Update(1) => {
                self.state = GameOfLifeState::Update(0);
            }
            GameOfLifeState::Update(_) => unreachable!(),
        }
    }

    fn run(
        &self,
        _graph: &mut render_graph::RenderGraphContext,
        render_context: &mut RenderContext,
        world: &World,
    ) -> Result<(), render_graph::NodeRunError> {
        let bind_groups = &world.resource::<GameOfLifeBindGroups>();
        let pipeline_cache = world.resource::<PipelineCache>();
        let pipeline = world.resource::<GameOfLifePipeline>();

        let command_encoder = render_context.command_encoder();

        // select the pipeline based on the current state
        match self.state {
            GameOfLifeState::Loading => {}
            GameOfLifeState::Init => {
                let mut pass =
                    command_encoder.begin_compute_pass(&ComputePassDescriptor::default());
                let init_pipeline = pipeline_cache
                    .get_compute_pipeline(pipeline.init_pipeline)
                    .unwrap();
                pass.set_bind_group(0, &bind_groups.sim_group_a, &[]);
                pass.set_pipeline(init_pipeline);
                pass.dispatch_workgroups(SIZE.0 / WORKGROUP_SIZE, SIZE.1 / WORKGROUP_SIZE, 1);
            }
            GameOfLifeState::Update(index) => {
                let (sim_bind_group, count_bind_group) = if index == 0 {
                    (&bind_groups.sim_group_a, &bind_groups.count_group_a)
                } else {
                    (&bind_groups.sim_group_b, &bind_groups.count_group_b)
                };

                let count_pipeline = pipeline_cache
                    .get_compute_pipeline(pipeline.count_pipeline)
                    .unwrap();
                let update_pipeline = pipeline_cache
                    .get_compute_pipeline(pipeline.update_pipeline)
                    .unwrap();

                command_encoder.clear_buffer(&pipeline.count_buffer, 0, None);
                {
                    let mut pass =
                        command_encoder.begin_compute_pass(&ComputePassDescriptor::default());
                    pass.set_pipeline(count_pipeline);
                    pass.set_bind_group(0, count_bind_group, &[]);
                    pass.dispatch_workgroups(SIZE.0 / WORKGROUP_SIZE, SIZE.1 / WORKGROUP_SIZE, 1);
                }

                {
                    let mut pass =
                        command_encoder.begin_compute_pass(&ComputePassDescriptor::default());
                    pass.set_bind_group(0, sim_bind_group, &[]);
                    pass.set_pipeline(update_pipeline);
                    pass.dispatch_workgroups(SIZE.0 / WORKGROUP_SIZE, SIZE.1 / WORKGROUP_SIZE, 1);
                }
            }
        }

        Ok(())
    }
}

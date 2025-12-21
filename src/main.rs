use cubecl::{
    CubeCount, CubeDim, Runtime, cube,
    frontend::CompilationArg,
    prelude::{
        ABSOLUTE_POS, Array, ArrayArg, Atomic, CUBE_DIM_X, CUBE_DIM_Y, CUBE_POS_X, CUBE_POS_Y,
        SharedMemory, Tensor, TensorArg, UNIT_POS, UNIT_POS_X, UNIT_POS_Y, sync_cube,
    },
    server::Allocation,
    wgpu::WgpuRuntime,
};
use rerun::external::arrow::datatypes::ToByteSlice;

const WIDTH: usize = 1500;
const HEIGHT: usize = 1000;

#[cube]
fn hash(value: u32) -> u32 {
    let mut state = value;
    state = state ^ 2747636419u32;
    state = state * 2654435769u32;
    state = state ^ state >> 16u32;
    state = state * 2654435769u32;
    state = state ^ state >> 16u32;
    state = state * 2654435769u32;
    state
}

#[cube]
fn random_float(value: u32) -> f32 {
    // Convert integer hash to 0.0..1.0 float
    hash(value) as f32 / 4294967295.0
}

#[cube(launch)]
fn count_reset_kernel(count_buffer: &mut Array<Atomic<u32>>) {
    if ABSOLUTE_POS == 0 {
        Atomic::store(&mut count_buffer[0], 0);
    }
}

#[cube(launch)]
fn init_kernel(output: &mut Tensor<u32>) {
    let width = output.shape(1);
    let idx = ABSOLUTE_POS;
    let x = idx % width;
    let y = idx / width;

    let seed = (y << 16) | x;
    let val = random_float(seed);

    // 10% chance to be alive
    output[idx] = if val > 0.9 { 1u32.into() } else { 0u32.into() };
}

#[cube(launch)]
fn update_kernel(
    input: &Tensor<u32>,
    output: &mut Tensor<u32>,
    prev_count_buffer: &Array<Atomic<u32>>,
    next_count_buffer: &Array<Atomic<u32>>,
) {
    // Place to put the count for the immediate cube.
    let mut cube_sum = SharedMemory::<Atomic<u32>>::new(1);
    // Only one needs to initialize it.
    if UNIT_POS == 0 {
        Atomic::store(&mut cube_sum[0], 0);
    }

    sync_cube();

    let x = (CUBE_POS_X * CUBE_DIM_X) + UNIT_POS_X;
    let y = (CUBE_POS_Y * CUBE_DIM_Y) + UNIT_POS_Y;

    let width = input.shape(1);
    let height = input.shape(0);
    let idx = (y * width) + x;

    if x < width && y < height {
        // Calculate neighbors with simple bounds checks instead of modulo (%)
        let coord_left = if x == 0 { width - 1 } else { x - 1 };
        let coord_right = if x == width - 1 { 0u32.into() } else { x + 1 };
        let coord_above = if y == 0 { height - 1 } else { y - 1 };
        let coord_below = if y == height - 1 { 0u32.into() } else { y + 1 };

        // Pre-calculate row offsets
        let row_above_offset = coord_above * width;
        let row_curr_offset = y * width;
        let row_below_offset = coord_below * width;

        let neighbors = 
            // Top row
            input[row_above_offset + coord_left] + 
            input[row_above_offset + x] + 
            input[row_above_offset + coord_right] +
            // Middle row
            input[row_curr_offset + coord_left] + 
            input[row_curr_offset + coord_right] +
            // Bottom row
            input[row_below_offset + coord_left] + 
            input[row_below_offset + x] + 
            input[row_below_offset + coord_right];

        let current = input[idx];
        let mut next: u32 = 0;

        // Conway's Rules
        if current == 1 {
            if neighbors == 2 || neighbors == 3 {
                next = 1;
            }
        } else {
            if neighbors == 3 {
                next = 1;
            }
        }

        // Homeostasis
        let total_pixels = WIDTH * HEIGHT;
        let alive_count: u32 = Atomic::load(&prev_count_buffer[0]);

        if alive_count < (total_pixels as u32 / 10) {
            // If current cell is dead...
            if next != 1 {
                let seed = ((y as u32) << 16) | (x as u32);
                // 0.1% chance to spawn
                if random_float(seed) < 0.001 {
                    next = 1;
                }
            }
        }

        // Set the pixel to alive or dead.
        output[idx] = next;

        // Since we're here, let's start the count for the next iteration.
        if next == 1 {
            Atomic::add(&cube_sum[0], 1);
        }
    }

    sync_cube();

    // Now that everyone's counted their piece, add the cube's count to the global one.
    // Only one unit needs to do the addition.
    if UNIT_POS == 0 {
        let local_count = Atomic::load(&cube_sum[0]);
        if local_count > 0 {
            Atomic::add(&next_count_buffer[0], local_count);
        }
    }
}

unsafe fn wrap_tensor<'a>(alloc: &'a Allocation, shape: &'a [usize]) -> TensorArg<'a, WgpuRuntime> {
    unsafe {
        TensorArg::<WgpuRuntime>::from_raw_parts::<u32>(&alloc.handle, &alloc.strides, shape, 1)
    }
}

unsafe fn wrap_array<'a>(alloc: &'a Allocation) -> ArrayArg<'a, WgpuRuntime> {
    unsafe { ArrayArg::<WgpuRuntime>::from_raw_parts::<u32>(&alloc.handle, alloc.strides[0], 1) }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let rec = rerun::RecordingStreamBuilder::new("simslime_cubecl").spawn()?;
    let client = cubecl::wgpu::WgpuRuntime::client(&Default::default());

    let shape = [HEIGHT, WIDTH];

    println!("Allocating GPU Memory...");

    let tensor_a_alloc = client.create_tensor(&[], &shape, 4);
    let tensor_b_alloc = client.create_tensor(&[], &shape, 4);
    let count_alloc_a = client.create_tensor(u32::MAX.to_byte_slice(), &[1], 4);
    let count_alloc_b = client.create_tensor(bytemuck::bytes_of(&0u32), &[1], 4);

    let cube_size_x = 16;
    let cube_size_y = 16;
    // Cube count is rounded up to ensure perfect coverage for any image size.
    let cube_count_x = (WIDTH + cube_size_x - 1) / cube_size_x;
    let cube_count_y = (HEIGHT + cube_size_y - 1) / cube_size_y;
    let cube_dim = CubeDim::new_2d(cube_size_x as u32, cube_size_y as u32);
    let cube_count = CubeCount::Static(cube_count_x as u32, cube_count_y as u32, 1);

    println!("Launching Init...");
    init_kernel::launch(&client, cube_count.clone(), cube_dim.clone(), unsafe {
        wrap_tensor(&tensor_a_alloc, &shape)
    });

    println!("Starting Loop...");

    let mut input_is_a = true;
    for step in 0..10000 {
        // Ping-Pong Logic
        let (input, output, prev_count, next_count) = if input_is_a {
            (
                &tensor_a_alloc,
                &tensor_b_alloc,
                &count_alloc_a,
                &count_alloc_b,
            )
        } else {
            (
                &tensor_b_alloc,
                &tensor_a_alloc,
                &count_alloc_b,
                &count_alloc_a,
            )
        };

        // Reset counter
        count_reset_kernel::launch(
            &client,
            CubeCount::Static(1, 1, 1),
            CubeDim::default(),
            unsafe { wrap_array(&next_count) },
        );

        // Update state
        update_kernel::launch(
            &client,
            CubeCount::Static(cube_count_x as u32, cube_count_y as u32, 1),
            CubeDim::new_2d(cube_size_x as u32, cube_size_y as u32),
            unsafe { wrap_tensor(input, &shape) },
            unsafe { wrap_tensor(output, &shape) },
            unsafe { wrap_array(&prev_count) },
            unsafe { wrap_array(&next_count) },
        );

        if step % 25 == 0 {
            let bytes = client.read_one(output.handle.clone());

            // Convert bytes to u32 slice
            let ints = bytemuck::cast_slice::<u8, u32>(&bytes);

            // int (0-1) -> Byte (0-255) for Image
            let image_buffer: Vec<u8> = ints.iter().map(|&v| (v * 255) as u8).collect();

            rec.set_time_sequence("step", step);
            rec.log(
                "world/grid",
                &rerun::Image::from_color_model_and_bytes(
                    image_buffer,           // Data
                    [WIDTH as u32, HEIGHT as u32], // Resolution
                    rerun::ColorModel::L,
                    rerun::ChannelDatatype::U8,
                ),
            )
            .unwrap();

            let count_bytes = client.read_one(next_count.handle.clone());
            let count_val = bytemuck::cast_slice::<u8, u32>(&count_bytes)[0];

            // Log the population graph
            rec.log("world/stats/population", &rerun::Scalars::new([count_val]))
                .unwrap();
        }

        // Swap and repeat
        input_is_a = !input_is_a;
    }

    Ok(())
}

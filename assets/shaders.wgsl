// The shader reads the previous frame's state from the `input` texture, and writes the new state of
// each pixel to the `output` texture. The textures are flipped each step to progress the
// simulation.
// Two textures are needed for the game of life as each pixel of step N depends on the state of its
// neighbors at step N-1.

struct CountBuffer {
    count: atomic<u32>,
}

@group(0) @binding(0) var input: texture_storage_2d<r32float, read>;
@group(0) @binding(1) var output: texture_storage_2d<r32float, write>;
@group(0) @binding(2) var<storage, read> count_buffer_in: CountBuffer;

@group(0) @binding(0) var texture_to_count: texture_storage_2d<r32float, read>;
@group(0) @binding(1) var<storage, read_write> count_buffer_out: CountBuffer;

fn hash(value: u32) -> u32 {
    var state = value;
    state = state ^ 2747636419u;
    state = state * 2654435769u;
    state = state ^ state >> 16u;
    state = state * 2654435769u;
    state = state ^ state >> 16u;
    state = state * 2654435769u;
    return state;
}

fn randomFloat(value: u32) -> f32 {
    return f32(hash(value)) / 4294967295.0;
}

@compute @workgroup_size(8, 8, 1)
fn init(@builtin(global_invocation_id) invocation_id: vec3<u32>, @builtin(num_workgroups) num_workgroups: vec3<u32>) {
    let location = vec2<i32>(i32(invocation_id.x), i32(invocation_id.y));

    let randomNumber = randomFloat(invocation_id.y << 16u | invocation_id.x);
    let alive = randomNumber > 0.9;
    let color = vec4<f32>(f32(alive));

    textureStore(output, location, color);
}

fn is_alive(location: vec2<i32>, offset_x: i32, offset_y: i32) -> i32 {
    let value: vec4<f32> = textureLoad(input, location + vec2<i32>(offset_x, offset_y));
    return i32(value.x);
}

fn count_alive(location: vec2<i32>) -> i32 {
    return is_alive(location, -1, -1) +
           is_alive(location, -1,  0) +
           is_alive(location, -1,  1) +
           is_alive(location,  0, -1) +
           is_alive(location,  0,  1) +
           is_alive(location,  1, -1) +
           is_alive(location,  1,  0) +
           is_alive(location,  1,  1);
}

@compute @workgroup_size(8, 8, 1)
fn update(@builtin(global_invocation_id) invocation_id: vec3<u32>) {
    let location = vec2<i32>(i32(invocation_id.x), i32(invocation_id.y));

    let n_alive = count_alive(location);

    var alive: bool;
    if (n_alive == 3) {
        alive = true;
    } else if (n_alive == 2) {
        let currently_alive = is_alive(location, 0, 0);
        alive = bool(currently_alive);
    } else {
        alive = false;
    }

    let dims = textureDimensions(input);
    let total_pixels = dims.x * dims.y;
    let alive_count = atomicLoad(&count_buffer_in.count);

    if (alive_count < (total_pixels / 10u)) {
        if (!alive) {
            let seed = (invocation_id.y << 16u | invocation_id.x) + u32(n_alive);
            let randomNumber = randomFloat(seed);
            if (randomNumber < 0.001) {
                alive = true;
            }
        }
    }

    let color = vec4<f32>(f32(alive));

    textureStore(output, location, color);
}

@compute @workgroup_size(8, 8, 1)
fn count_alive_pixels(@builtin(global_invocation_id) invocation_id: vec3<u32>) {
    let dims = textureDimensions(texture_to_count);
    if (invocation_id.x >= dims.x || invocation_id.y >= dims.y) {
        return;
    }
    let coords = vec2<i32>(invocation_id.xy);
    let texel_value = textureLoad(texture_to_count, coords);
    if (texel_value.r != 0.0) {
        atomicAdd(&count_buffer_out.count, 1u);
    }
}

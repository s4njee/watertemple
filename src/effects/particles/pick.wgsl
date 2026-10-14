// Finds the particle nearest to the pointer on screen, in two passes over the particles: the first takes the
// smallest screen distance with an atomic min (non-negative floats order like their bits), the second takes
// the lowest index among those at exactly that distance. A small bias towards the camera prefers the front
// particle of a deep cloud. The result is state[0] (0xffffffff when nothing is on screen), which is the
// layout the neighbour marker reads.

#include "camera.wgsl"

#include "pick_params.wgsl"

@group(0) @binding(0) var<uniform> cam: Camera;
@group(0) @binding(1) var<uniform> pick: PickParams;
@group(0) @binding(2) var<storage, read> positions: array<vec4f>;
// [0] best particle, [1] best distance as bits. Both start at 0xffffffff.
@group(0) @binding(3) var<storage, read_write> state: array<atomic<u32>>;

const OFF_SCREEN: f32 = 3.0e38;

fn score(i: u32) -> f32 {
  let h = cam.view_proj * vec4f(positions[i].xyz, 1.0);
  if (!(h.w > 0.0)) { return OFF_SCREEN; }
  let px = camera_ndc_to_uv(h.xy / h.w) * cam.viewport;
  let d = px - pick.pointer;
  return dot(d, d) + h.w * 0.02;
}

@compute @workgroup_size(64)
fn pick_distance(@builtin(global_invocation_id) id: vec3u) {
  let i = id.x;
  if (i >= arrayLength(&positions) || pick.active_flag == 0u) { return; }
  let s = score(i);
  if (s < 1.0e30) {
    atomicMin(&state[1], bitcast<u32>(s));
  }
}

@compute @workgroup_size(64)
fn pick_index(@builtin(global_invocation_id) id: vec3u) {
  let i = id.x;
  if (i >= arrayLength(&positions) || pick.active_flag == 0u) { return; }
  let s = score(i);
  if (s < 1.0e30 && bitcast<u32>(s) == atomicLoad(&state[1])) {
    atomicMin(&state[0], i);
  }
}

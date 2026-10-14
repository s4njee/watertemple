// Speed of every particle into a scalar the billboard renderer colours by (the 'speed' view). One f32 per
// particle, in the solver's current (cell) order, which is also the order the positions are drawn in.

@group(0) @binding(0) var<storage, read> velocities: array<vec4f>;
@group(0) @binding(1) var<storage, read_write> speed: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3u) {
  let i = id.x;
  if (i >= arrayLength(&speed)) { return; }
  speed[i] = length(velocities[i].xyz);
}

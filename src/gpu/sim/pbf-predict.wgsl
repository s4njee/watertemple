// pbf-predict.wgsl -- step 1 of a position-based fluid sub-step (story 7.4): apply the external forces and predict.
//
//   previous = x                    where the particle was when the step began (the velocity update needs it)
//   v        = v + g dt             gravity, then (first sub-step of an update only) the pointer impulses
//   v        = v exp(-damping dt)   optional drag; zero by default
//   x        = x + v dt, kept inside the container and outside the colliders
//
// `positions` holds the PREDICTED position from here on; the solver iterations correct it in place of x. The
// scalar in positions.w (the foam amount) passes through untouched. Bindings: see pbf.ts.

#include "pbf-core.wgsl"

@group(0) @binding(2) var<storage, read_write> positions: array<vec4f>;
@group(0) @binding(3) var<storage, read_write> velocities: array<vec4f>;
@group(0) @binding(4) var<storage, read_write> previous: array<vec4f>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3u) {
  let i = id.x;
  if (i >= params.count) { return; }
  let pw = positions[i];
  let p = pw.xyz;
  var v = velocities[i].xyz;
  previous[i] = vec4f(p, 0.0);

  v += params.gravity * tick.dt;
  if (tick.first != 0u) {
    v = pbf_impulses(p, v);
  }
  v *= exp(-params.damping * tick.dt);
  let speed = length(v);
  if (speed > params.max_speed) {
    v *= params.max_speed / speed;
  }
  v = pbf_finite3(v, vec3f(0.0));

  var x = pbf_contain(pbf_finite3(p + v * tick.dt, p), p, 0.0);
  positions[i] = vec4f(x, pw.w);
  velocities[i] = vec4f(v, 0.0);
}

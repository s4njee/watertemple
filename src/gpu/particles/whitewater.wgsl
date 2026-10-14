// Secondary particles are independent of the primary sort. Neighbour occupancy decides whether
// they fly ballistically (spray), follow the surface (foam), or rise through the fluid (bubbles).
#include "whitewater_params.wgsl"
#include "neighbor_bindings.wgsl"
#include "hash.wgsl"
@group(0) @binding(0) var<uniform> params: WhitewaterParams;
@group(0) @binding(1) var<storage, read> fluid_position: array<vec4f>;
@group(0) @binding(2) var<storage, read> fluid_velocity: array<vec4f>;
@group(0) @binding(3) var<storage, read_write> position: array<vec4f>;
@group(0) @binding(4) var<storage, read_write> velocity: array<vec4f>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3u) {
  let i = id.x;
  if (i >= arrayLength(&position)) { return; }
  var p = position[i];
  var v = velocity[i];
  v.w -= params.dt;
  if (v.w <= 0.0) {
    p = vec4f(1e6, 1e6, 1e6, 0.0);
    v = vec4f(0.0);
    if (params.count > 0u && params.emission > 0.0) {
      // Every dead slot samples one current source, never an index retained across a reorder.
      let seed = pcg_1d(i + params.frame * 747796405u);
      let source = seed % params.count;
      let fp = fluid_position[source];
      let fv = fluid_velocity[source];
      let probability = clamp(params.emission * params.dt * fp.w, 0.0, 1.0);
      if (length(fv.xyz) >= params.min_speed && f32(pcg_1d(seed) % 65536u) / 65536.0 < probability) {
        let jitter = vec3f(f32(seed % 31u), f32((seed >> 5u) % 31u), f32((seed >> 10u) % 31u)) / 30.0 - 0.5;
        p = vec4f(fp.xyz + jitter * params.radius, 1.0);
        v = vec4f(fv.xyz, params.lifetime * (0.6 + f32(seed % 101u) / 250.0));
      }
    }
  }
  if (v.w > 0.0) {
    let cell = neighbor_cell_of(neighbor_grid, p.xyz);
    let r2 = neighbor_grid.radius * neighbor_grid.radius;
    var occupancy = 0u;
    var flow = vec3f(0.0);
    for (var row = 0u; row < neighbor_grid.rows; row++) {
      let span = neighbor_row_span(cell, row);
      for (var k = span.x; k < span.y; k++) {
        let j = neighbor_particle(k);
        if (j < params.count && dot(fluid_position[j].xyz - p.xyz, fluid_position[j].xyz - p.xyz) < r2) {
          occupancy++;
          flow += fluid_velocity[j].xyz;
        }
      }
    }
    if (occupancy <= 2u) {
      p.w = 1.0;
      v = vec4f(v.xyz + params.gravity * params.dt, v.w);
    } else {
      flow /= f32(occupancy);
      p.w = select(2.0, 3.0, occupancy > params.bubble_neighbors);
      let buoyancy = select(0.0, params.buoyancy, p.w == 3.0);
      v = vec4f(mix(v.xyz, flow, 1.0 - exp(-params.drag * params.dt)) + vec3f(0.0, buoyancy * params.dt, 0.0), v.w);
    }
    let speed = length(v.xyz);
    v = vec4f(v.xyz * min(1.0, 8.0 / max(speed, 1e-6)), v.w);
    p = vec4f(p.xyz + v.xyz * params.dt, p.w);
    if (any(p.xyz < params.bounds_min) || any(p.xyz > params.bounds_max)) {
      p = vec4f(clamp(p.xyz, params.bounds_min, params.bounds_max), p.w);
      v = vec4f(v.xyz * 0.2, v.w);
    }
  }
  position[i] = p;
  velocity[i] = v;
}

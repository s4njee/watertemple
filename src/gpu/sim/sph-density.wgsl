// SPH pass 1: the density of every particle by kernel summation, and the pressure from the equation of state.
//     rho_i = m * (W(0) + sum over neighbours W_poly6(|x_i - x_j|)) + the wall term of sph-core.wgsl
// Output: state[i] = (rho_i, p_i). Reads only positions, so it can run right after the neighbour grid reorders them.

#include "sph_params.wgsl"
#include "sph-kernels.wgsl"
#include "neighbor_bindings.wgsl"

@group(0) @binding(0) var<uniform> params: SphParams;
@group(0) @binding(1) var<storage, read> positions: array<vec4f>;
@group(0) @binding(2) var<storage, read_write> state: array<vec2f>;
// The grid's bindings (neighbor_bindings.wgsl) start at binding 3.

#include "sph-core.wgsl"

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3u) {
  let i = id.x;
  if (i >= params.count) { return; }
  let p = positions[i].xy;
  let h = params.h;
  let h2 = h * h;
  let c = neighbor_cell_of(neighbor_grid, vec3f(p, 0.0));
  var sum = sph_poly6_2d(0.0, h);
  for (var row = 0u; row < neighbor_grid.rows; row++) {
    let span = neighbor_row_span(c, row);
    for (var k = span.x; k < span.y; k++) {
      let j = neighbor_particle(k);
      let d = p - positions[j].xy;
      let r2 = dot(d, d);
      if (j != i && r2 < h2) {
        sum += sph_poly6_2d(r2, h);
      }
    }
  }
  let rho = max(params.mass * sum + sph_wall_density(p), 1e-6 * params.rest_density);
  state[i] = vec2f(rho, sph_pressure(rho));
}

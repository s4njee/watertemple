// pbf-density.wgsl -- step 3a of a position-based fluid sub-step, once per solver iteration (story 7.4): the density
// constraint and its Lagrange multiplier, for every particle at once.
//
//   rho_i = (sum_j W_poly6(|x_i - x_j|) + wall)  / n0         dimensionless: 1 is the rest density
//   C_i   = rho_i - 1                                          the constraint
//   lambda_i = -C_i / (sum_k |grad_k C_i|^2 + eps)             the step along the constraint gradient that zeroes C
//
// The sum includes the particle itself and the missing fluid behind the walls (pbf_wall_terms). The gradient
// sum is |sum_j g_ij|^2 (moving particle i) plus sum_j |g_ij|^2 (moving each neighbour j), all over n0^2, with g
// the Spiky gradient; the walls add to the first term only, because a wall does not move. eps is the relaxation
// (params.eps, already in the same units): without it a particle with few neighbours divides by almost nothing.
// A negative C (fewer neighbours than at rest: a free surface, a splash) is scaled by params.neg_pressure, 0 by
// default, so that surface particles are not pulled inward (the tensile instability, see pbf.ts).
//
// Output: solve[i] = (lambda_i, rho_i). The neighbour loop is the one documented in src/wgsl/neighbors.wgsl.

#include "pbf-core.wgsl"
#include "neighbor_bindings.wgsl"

@group(0) @binding(2) var<storage, read> positions: array<vec4f>;
@group(0) @binding(5) var<storage, read_write> solve: array<vec2f>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3u) {
  let i = id.x;
  if (i >= params.count) { return; }
  let p = positions[i].xyz;
  let c = neighbor_cell_of(neighbor_grid, p);
  var density = pbf_poly6(0.0, params.h2, params.poly6);
  var grad_sum = vec3f(0.0);
  var grad_sq = 0.0;
  for (var row = 0u; row < neighbor_grid.rows; row++) {
    let span = neighbor_row_span(c, row);
    for (var k = span.x; k < span.y; k++) {
      let j = neighbor_particle(k);
      if (j == i) { continue; }
      var r = p - positions[j].xyz;
      var r2 = neighbor_dist2(neighbor_grid, r);
      if (r2 >= params.h2) { continue; }
      if (r2 < 1e-12) {
        r = pbf_nudge(i, j, params.h);
        r2 = dot(r, r);
      }
      density += pbf_poly6(r2, params.h2, params.poly6);
      let g = pbf_spiky_grad(r, params.h, params.spiky);
      grad_sum += g;
      grad_sq += dot(g, g);
    }
  }
  let wall = pbf_wall_terms(p);
  let rho = (density + wall.density) * params.inv_n0;
  var constraint = rho - 1.0;
  if (constraint < 0.0) {
    constraint *= params.neg_pressure;
  }
  grad_sum += wall.grad;
  let denominator = (dot(grad_sum, grad_sum) + grad_sq) * params.inv_n0 * params.inv_n0 + params.eps;
  solve[i] = vec2f(-constraint / denominator, rho);
}

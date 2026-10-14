// pbf-viscosity.wgsl -- step 4a of a position-based fluid sub-step (story 7.4): XSPH viscosity, and the vorticity
// of the velocity field for the confinement that follows.
//
//   v_i' = v_i + c * 1/n0 * sum_j (v_j - v_i) W_poly6(|x_i - x_j|)
//   omega_i = 1/n0 * sum_j (v_i - v_j) x grad W_spiky(x_i - x_j)       the curl of v (2 omega for a rigid rotation)
//
// XSPH moves each velocity towards the kernel-weighted mean of its neighbours: a viscosity that costs one pass and
// cannot make the solver unstable, because it is an average. It is applied to the velocity only, after the positions
// are final, so it does not fight the density constraint. It is a sum of pair terms (v_j - v_i) W that are
// antisymmetric, so it conserves linear momentum. `c` is a fraction per 1/60 s (params.viscosity), converted to the
// sub-step length so the damping does not depend on the step count.
//
// Writes the new velocity to a second buffer (Jacobi again) and omega (xyz) with its length (w) for pbf-vorticity.

#include "pbf-core.wgsl"
#include "neighbor_bindings.wgsl"

@group(0) @binding(2) var<storage, read> positions: array<vec4f>;
@group(0) @binding(3) var<storage, read> velocities: array<vec4f>;
@group(0) @binding(6) var<storage, read_write> omega: array<vec4f>;
@group(0) @binding(7) var<storage, read_write> velocities_out: array<vec4f>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3u) {
  let i = id.x;
  if (i >= params.count) { return; }
  let p = positions[i].xyz;
  let vi = velocities[i].xyz;
  let c = neighbor_cell_of(neighbor_grid, p);
  var smooth_sum = vec3f(0.0);
  var curl = vec3f(0.0);
  let want_curl = params.vorticity > 0.0;
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
      let vj = velocities[j].xyz;
      smooth_sum += (vj - vi) * pbf_poly6(r2, params.h2, params.poly6);
      if (want_curl) {
        curl += cross(vi - vj, pbf_spiky_grad(r, params.h, params.spiky));
      }
    }
  }
  let c_step = 1.0 - pow(1.0 - clamp(params.viscosity, 0.0, 0.999), tick.dt * 60.0);
  let v_new = pbf_finite3(vi + c_step * smooth_sum * params.inv_n0, vi);
  velocities_out[i] = vec4f(v_new, 0.0);
  let w = curl * params.inv_n0;
  omega[i] = vec4f(w, length(w));
}

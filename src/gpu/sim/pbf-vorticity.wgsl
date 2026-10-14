// pbf-vorticity.wgsl -- step 4b of a position-based fluid sub-step, optional (story 7.4): vorticity confinement
// (Fedkiw, Stam and Jensen 2001; the form of the position-based fluids paper).
//
// A particle method loses small eddies to numerical damping. Confinement finds where |omega| peaks and adds a force
// that spins the fluid around that peak:
//
//   eta_i = grad |omega|_i = 1/n0 * sum_j (|omega_j| - |omega_i|) grad W_spiky(x_i - x_j)
//   N = eta / |eta|,   dv = epsilon * h * 60 * dt * (N x omega_i)
//
// epsilon is params.vorticity; h * 60 gives it a velocity per second per second so that 1 is a strong effect
// at any particle size. It reads omega from pbf-viscosity and changes only the velocity of its own particle, so it is
// safe in place. It adds energy: keep it small (0.05 to 0.3) or the fluid will never come to rest.

#include "pbf-core.wgsl"
#include "neighbor_bindings.wgsl"

@group(0) @binding(2) var<storage, read> positions: array<vec4f>;
@group(0) @binding(3) var<storage, read_write> velocities: array<vec4f>;
@group(0) @binding(6) var<storage, read> omega: array<vec4f>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3u) {
  let i = id.x;
  if (i >= params.count) { return; }
  let p = positions[i].xyz;
  let wi = omega[i];
  let c = neighbor_cell_of(neighbor_grid, p);
  var eta = vec3f(0.0);
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
      eta += (omega[j].w - wi.w) * pbf_spiky_grad(r, params.h, params.spiky);
    }
  }
  eta *= params.inv_n0;
  let len = length(eta);
  if (len < 1e-6) { return; }
  let force = cross(eta / len, wi.xyz);
  let v = velocities[i];
  velocities[i] = vec4f(pbf_finite3(v.xyz + force * (params.vorticity * params.h * 60.0 * tick.dt), v.xyz), 0.0);
}

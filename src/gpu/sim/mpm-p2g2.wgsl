// mpm-p2g2.wgsl -- P2G, pass 2 of an MLS-MPM step (src/gpu/sim/mpm.ts): the fluid stress.
//
// DENSITY. The mass the first pass left on the grid is gathered back to each particle with the same weights,
//     rho_p / rho_0 = r = (sum_i w_ip m_i) / rest_ppc,
// where rest_ppc is the number of particles per cell at rest (a cell holds 8 particles of mass 1 at rest density,
// so 8 grid-mass units per cell volume ARE rho_0), corrected for the container walls (below). The ratio r is stored in affine0.w for the next pass (foam)
// and for the density view.
//
// WALLS. Near a wall the gather sees less mass (the fluid stops there) and r reads low, which would let the fluid
// pile up against it. The mass a fluid at exactly rest density would leave on node i is rest_ppc * F_i, where F_i is the
// share of the node's B-spline kernel inside the container (a product of three one-dimensional shares, each a cubic
// polynomial: mpm_kernel_above). Gathering that with the particle's weights gives the density a particle would read in
// a perfectly filled box; r is divided by it. Away from the walls it is exactly 1. The free surface is left alone: its
// deficit is real (there is no fluid beyond it) and gives the zero-pressure boundary condition.
//
// STRESS. Tait equation of state with negative pressure clamped,
//     p / rho_0 = (c^2 / gamma) (r^gamma - 1),     p >= -tension c^2 / gamma,
// whose small-compression sound speed is exactly c; plus the viscous stress of a Newtonian fluid, 2 mu D with the
// symmetric part of the velocity gradient D = (C + C^T) / 2 (the APIC matrix C is the gradient). The particle's
// volume is V = V_0 / r, so with unit mass V p = (p / rho_0) / r and V mu = nu / r (nu = kinematic viscosity). The
// MLS-MPM momentum change of node i from particle p (Hu et al. 2018, eq. 29) is
//     dq_i = w_ip (4 / h^2) dt (V p I - V 2 mu D) (x_i - x_p)
// and is added to the node's momentum, again with fixed-point atomics. Compression pushes the node away from the
// particle; tension pulls it in. Bindings: 1 uniform and 5 storage buffers.

#include "mpm-params.wgsl"
#include "mpm-core.wgsl"

@group(0) @binding(0) var<uniform> P: MpmParams;
@group(0) @binding(1) var<storage, read_write> grid: array<atomic<i32>>;
@group(0) @binding(2) var<storage, read> positions: array<vec4f>;
@group(0) @binding(3) var<storage, read_write> affine0: array<vec4f>;
@group(0) @binding(4) var<storage, read> affine1: array<vec4f>;
@group(0) @binding(5) var<storage, read> affine2: array<vec4f>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let p = gid.x;
  if (p >= P.count) {
    return;
  }
  let x = positions[p].xyz;
  let a0 = affine0[p];
  let C = mat3x3f(a0.xyz, affine1[p].xyz, affine2[p].xyz);

  let dims = vec3i(P.dims);
  let g = x * P.inv_h;
  let base = clamp(vec3i(floor(g - vec3f(0.5))), vec3i(0), dims - vec3i(3));
  let fx = g - vec3f(base);
  let W = mpm_weight_matrix(fx);

  // Density from the grid mass.
  var mass = 0.0;
  for (var k = 0; k < 3; k++) {
    for (var j = 0; j < 3; j++) {
      for (var i = 0; i < 3; i++) {
        let w = W[i].x * W[j].y * W[k].z;
        let node = u32(((base.z + k) * dims.y + (base.y + j)) * dims.x + (base.x + i)) * 4u;
        mass += w * mpm_from_fixed(atomicLoad(&grid[node]));
      }
    }
  }
  // Walls: a fluid at rest density filling the container gives each node rest_ppc times the share of its kernel
  // inside the walls, so dividing by the same share seen from this particle removes the deficit near them.
  var fill = 1.0;
  if (P.wall_comp != 0u) {
    let lo = P.bounds_min * P.inv_h;
    let hi = P.bounds_max * P.inv_h;
    for (var axis = 0; axis < 3; axis++) {
      var share = 0.0;
      for (var o = 0; o < 3; o++) {
        let node = f32(base[axis] + o);
        let inside = mpm_kernel_above(lo[axis] - node) - mpm_kernel_above(hi[axis] - node);
        share += W[o][axis] * inside;
      }
      fill *= share;
    }
  }
  let r = clamp(mass / (P.rest_ppc * max(fill, 0.05)), 0.0, 64.0);
  affine0[p] = vec4f(a0.xyz, r);

  // Pressure per rest density, clamped from above (r_max keeps the impulses inside the fixed-point range)
  // and from below (tension).
  let c2g = P.sound_speed * P.sound_speed / P.gamma;
  let rc = clamp(r, 0.001, P.r_max);
  let pressure = max(c2g * (pow(rc, P.gamma) - 1.0), -P.tension * c2g);
  let inv_r = 1.0 / max(rc, 0.25);
  let scale = 4.0 * P.inv_h * P.inv_h * P.dt;
  let strain = (C + transpose(C)) * P.viscosity;       // 2 nu D

  for (var k = 0; k < 3; k++) {
    for (var j = 0; j < 3; j++) {
      for (var i = 0; i < 3; i++) {
        let w = W[i].x * W[j].y * W[k].z;
        let d = (vec3f(f32(i), f32(j), f32(k)) - fx) * P.h;
        let dq = (w * scale * inv_r) * (pressure * d - strain * d);
        let node = u32(((base.z + k) * dims.y + (base.y + j)) * dims.x + (base.x + i)) * 4u;
        atomicAdd(&grid[node + 1u], mpm_to_fixed(dq.x));
        atomicAdd(&grid[node + 2u], mpm_to_fixed(dq.y));
        atomicAdd(&grid[node + 3u], mpm_to_fixed(dq.z));
      }
    }
  }
}

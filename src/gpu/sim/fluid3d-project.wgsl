// fluid3d-project.wgsl -- the pressure projection in 3D: divergence, the pressure solve (damped Jacobi
// sweeps, or a multigrid V-cycle built from the same sweeps plus a restriction and a prolongation), and the
// subtraction of the pressure gradient.
//
// THE MATHS. Same as the 2D solver (fluid-project.wgsl). After advection and forces the velocity u* has a
// divergence; the projection finds phi with  laplacian(phi) = div u*  and sets  u = u* - grad phi. On the MAC
// grid every operator is a compact difference of neighbouring values, so with the cell size scaled out
//     div(i, j, k)  = u(i+1,j,k) - u(i,j,k) + v(i,j+1,k) - v(i,j,k) + w(i,j,k+1) - w(i,j,k)
//     u'(i, j, k)   = u(i, j, k) - (phi(i, j, k) - phi(i-1, j, k))        v', w' likewise
//     lap phi       = sum of the six neighbours - 6 phi
// and  div u' = div u - lap phi  is zero to the accuracy of the linear solve. phi has the units of the velocity
// (pressure times dt / density, scaled by the cell size), independent of dt, so last step's solution is a good
// starting guess (the warm start). A wall or solid neighbour contributes nothing and does not count in the
// diagonal (zero normal gradient), so the diagonal is the number of fluid neighbours.
//
// MULTIGRID. Damped Jacobi removes error whose wavelength is a few cells in a few sweeps and the long
// wavelengths of an N-cell grid only after of the order of N^2 of them. A V-cycle fixes that: smooth, take the
// residual to a grid with half the cells (`cs_restrict`: the mean of the eight child residuals, times 4 because the
// unscaled laplacian of a cell twice as wide is a quarter as strong), solve for the correction there the same
// way, interpolate it back (`cs_prolong`, trilinear, zero-gradient at the walls) and smooth again. The
// cost is a few fine sweeps plus 1/8 of that for each coarser level. Obstacles are not represented on the
// coarse grids, so a solver with an obstacle uses plain sweeps.
//
// Every pressure kernel takes its grid size from the texture it reads (`textureDimensions`), so the same
// kernels run on every level.
//
// Bindings (besides the shared 0 `P`):
//   2 vel      velocity ((nx+1) x (ny+1) x (nz+1))                (cs_divergence, cs_project, cs_residual_fine unused)
//   3 rhs_tex  r32float right-hand side of the level               (cs_jacobi*, cs_restrict)
//   4 p_src    r32float pressure of the last iteration             (cs_jacobi*, cs_project, cs_restrict, cs_prolong)
//   5 p_dst    r32float storage: the new pressure / coarse rhs     (cs_jacobi*, cs_restrict, cs_prolong)
//   6 div_dst  r32float storage: the divergence                    (cs_divergence)
//   7 out_vel  rgba16float storage: projected velocity             (cs_project)
//   8 coarse   r32float pressure correction of the next coarser level (cs_prolong)

#include "fluid3d-core.wgsl"

@group(0) @binding(2) var vel: texture_3d<f32>;
@group(0) @binding(3) var rhs_tex: texture_3d<f32>;
@group(0) @binding(4) var p_src: texture_3d<f32>;
@group(0) @binding(5) var p_dst: texture_storage_3d<r32float, write>;
@group(0) @binding(6) var div_dst: texture_storage_3d<r32float, write>;
@group(0) @binding(7) var out_vel: texture_storage_3d<rgba16float, write>;
@group(0) @binding(8) var coarse: texture_3d<f32>;

// ---- Divergence -----------------------------------------------------------------------------------------------

@compute @workgroup_size(WG_X, WG_Y, WG_Z)
fn cs_divergence(@builtin(global_invocation_id) id: vec3u) {
  if (any(id >= P.grid)) {
    return;
  }
  let c = vec3i(id);
  var d = 0.0;
  if (!cell_solid(c)) {
    d = face_load(vel, 0u, c + vec3i(1, 0, 0)) - face_load(vel, 0u, c)
      + face_load(vel, 1u, c + vec3i(0, 1, 0)) - face_load(vel, 1u, c)
      + face_load(vel, 2u, c + vec3i(0, 0, 1)) - face_load(vel, 2u, c);
  }
  textureStore(div_dst, c, vec4f(d, 0.0, 0.0, 0.0));
}

// ---- Jacobi ---------------------------------------------------------------------------------------------------------

// (pressure of the neighbour, 1) for a fluid neighbour, (0, 0) for the wall or a solid; `use_prev` false
// means "all pressures are zero".
fn neighbour(c: vec3i, dims: vec3i, use_prev: bool) -> vec2f {
  if (solid_at(c, dims)) {
    return vec2f(0.0, 0.0);
  }
  return vec2f(select(0.0, textureLoad(p_src, c, 0).x, use_prev), 1.0);
}

fn jacobi(c: vec3i, dims: vec3i, use_prev: bool) -> f32 {
  if (solid_at(c, dims)) {
    return 0.0;
  }
  let sn = neighbour(c + vec3i(1, 0, 0), dims, use_prev) + neighbour(c + vec3i(-1, 0, 0), dims, use_prev)
         + neighbour(c + vec3i(0, 1, 0), dims, use_prev) + neighbour(c + vec3i(0, -1, 0), dims, use_prev)
         + neighbour(c + vec3i(0, 0, 1), dims, use_prev) + neighbour(c + vec3i(0, 0, -1), dims, use_prev);
  if (sn.y < 0.5) {
    return 0.0;
  }
  return (sn.x - textureLoad(rhs_tex, c, 0).x) / sn.y;
}

// Damped Jacobi: phi' = (1 - w) phi + w J(phi). The undamped iteration (w = 1) multiplies the highest-
// frequency (chequerboard) error mode by -1 every sweep: an odd number of sweeps then over-corrects it. With
// w = 0.8 that mode shrinks by 0.6 per sweep and the smooth ones are slowed by only a fifth. (For a 7-point
// laplacian the best smoothing weight is 6/7; 0.8 is the 2D solver's value and close enough.)
fn relaxed(c: vec3i, dims: vec3i, use_prev: bool) -> f32 {
  let own = select(0.0, textureLoad(p_src, c, 0).x, use_prev);
  return (1.0 - P.jacobi_w) * own + P.jacobi_w * jacobi(c, dims, use_prev);
}

@compute @workgroup_size(WG_X, WG_Y, WG_Z)
fn cs_jacobi(@builtin(global_invocation_id) id: vec3u) {
  let dims = vec3i(textureDimensions(p_src));
  if (any(vec3i(id) >= dims)) {
    return;
  }
  let c = vec3i(id);
  textureStore(p_dst, c, vec4f(relaxed(c, dims, true), 0.0, 0.0, 0.0));
}

// The first sweep of a solve that starts from zero (a cold start, or a correction on a coarse level).
@compute @workgroup_size(WG_X, WG_Y, WG_Z)
fn cs_jacobi_cold(@builtin(global_invocation_id) id: vec3u) {
  let dims = vec3i(textureDimensions(p_src));
  if (any(vec3i(id) >= dims)) {
    return;
  }
  let c = vec3i(id);
  textureStore(p_dst, c, vec4f(relaxed(c, dims, false), 0.0, 0.0, 0.0));
}

// ---- Multigrid ---------------------------------------------------------------------------------------------------------

fn residual(c: vec3i, dims: vec3i) -> f32 {
  if (solid_at(c, dims)) {
    return 0.0;
  }
  let here = textureLoad(p_src, c, 0).x;
  let sn = neighbour(c + vec3i(1, 0, 0), dims, true) + neighbour(c + vec3i(-1, 0, 0), dims, true)
         + neighbour(c + vec3i(0, 1, 0), dims, true) + neighbour(c + vec3i(0, -1, 0), dims, true)
         + neighbour(c + vec3i(0, 0, 1), dims, true) + neighbour(c + vec3i(0, 0, -1), dims, true);
  return textureLoad(rhs_tex, c, 0).x - (sn.x - sn.y * here);
}

// Residual of the fine level, restricted to the next coarser one: rhs_coarse = 4 * mean of the eight children.
@compute @workgroup_size(WG_X, WG_Y, WG_Z)
fn cs_restrict(@builtin(global_invocation_id) id: vec3u) {
  let coarse_dims = vec3i(textureDimensions(p_dst));
  if (any(vec3i(id) >= coarse_dims)) {
    return;
  }
  let fine_dims = vec3i(textureDimensions(p_src));
  let base = vec3i(id) * 2;
  var sum = 0.0;
  for (var n = 0; n < 8; n++) {
    sum += residual(base + vec3i(n & 1, (n >> 1) & 1, (n >> 2) & 1), fine_dims);
  }
  textureStore(p_dst, vec3i(id), vec4f(0.5 * sum, 0.0, 0.0, 0.0));
}

// p_dst = p_src + trilinear interpolation of the coarse correction.
@compute @workgroup_size(WG_X, WG_Y, WG_Z)
fn cs_prolong(@builtin(global_invocation_id) id: vec3u) {
  let dims = vec3i(textureDimensions(p_src));
  if (any(vec3i(id) >= dims)) {
    return;
  }
  let c = vec3i(id);
  let cd = vec3i(textureDimensions(coarse));
  let q = (vec3f(c) + 0.5) * 0.5 - 0.5;
  let b = vec3i(floor(q));
  let t = q - vec3f(b);
  var corr = 0.0;
  for (var n = 0; n < 8; n++) {
    let d = vec3i(n & 1, (n >> 1) & 1, (n >> 2) & 1);
    let w = select(1.0 - t.x, t.x, d.x == 1) * select(1.0 - t.y, t.y, d.y == 1) * select(1.0 - t.z, t.z, d.z == 1);
    corr += w * textureLoad(coarse, clamp(b + d, vec3i(0), cd - vec3i(1)), 0).x;
  }
  var value = 0.0;
  if (!solid_at(c, dims)) {
    value = textureLoad(p_src, c, 0).x + corr;
  }
  textureStore(p_dst, c, vec4f(value, 0.0, 0.0, 0.0));
}

// ---- Projection -----------------------------------------------------------------------------------------------------------

fn projected(axis: u32, t: vec3i) -> f32 {
  let f = face_of(axis, t);
  if (face_blocked(axis, f)) {
    return 0.0;
  }
  let gradient = textureLoad(p_src, f, 0).x - textureLoad(p_src, f - axis_step(axis), 0).x;
  return textureLoad(vel, f, 0)[axis] - gradient;
}

@compute @workgroup_size(WG_X, WG_Y, WG_Z)
fn cs_project(@builtin(global_invocation_id) id: vec3u) {
  if (any(id >= P.grid + vec3u(1u))) {
    return;
  }
  let t = vec3i(id);
  let v = vec3f(projected(0u, t), projected(1u, t), projected(2u, t));
  textureStore(out_vel, t, vec4f(finite3(v), 0.0));
}

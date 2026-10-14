// fluid-project.wgsl -- the pressure projection: divergence, Jacobi iterations of the pressure Poisson
// equation, subtraction of the pressure gradient, and the resolve pass that packs what consumers read.
//
// THE MATHS. Incompressibility is div u = 0. After advection and forces the velocity u* has some
// divergence; the projection finds phi with  laplacian(phi) = div u*  and sets  u = u* - grad phi,
// which then has div u = div u* - laplacian(phi) = 0. On the MAC grid every operator is a compact
// difference of neighbouring values, so with the cell size scaled out they are exactly
//     div(i, j)  = u(i+1, j) - u(i, j) + v(i, j+1) - v(i, j)
//     u'(i, j)   = u(i, j) - (phi(i, j) - phi(i-1, j))          v' likewise down the column
//     lap phi    = phi(i+1, j) + phi(i-1, j) + phi(i, j+1) + phi(i, j-1) - 4 phi(i, j)
// and  div u' = div u - lap phi  is zero to the accuracy of the linear solve, with no checkerboard
// null space. `phi` has the units of the velocity (image heights per second): it is the velocity
// potential of the correction, i.e. pressure times dt / density, scaled by the cell size. That makes
// it independent of dt, which is why last step's solution is a good starting guess.
//
// BOUNDARIES. A SOLID neighbour contributes nothing and does not count in the diagonal (zero normal
// gradient: the wall), an AIR neighbour (open boundary) counts as phi = 0 (the flow leaves freely),
// and a BLOCKED face keeps velocity 0 (set below). Jacobi: phi' = (sum of fluid neighbours - div) / n.
//
// SOLVERS. `cs_jacobi` is the Jacobi iteration (damped, see `relaxed`): every cell from its
// neighbours' previous values, one dispatch per iteration, converging like cos(pi / N)^k: hundreds of
// iterations for the longest wavelengths of an N-cell grid. `cs_sor_red` and `cs_sor_black` are red-black successive
// over-relaxation: the chequerboard is updated one colour at a time (a cell's four neighbours all
// have the other colour, so the update is exact Gauss-Seidel), each new value pushed past the
// Gauss-Seidel value by the factor omega (P.sor_omega, near 2 / (1 + sin(pi / N))). Without
// read-write storage textures a half-sweep reads one texture and writes another (copying the colour
// it does not update), so it costs the same as one Jacobi dispatch; and it converges like
// (1 - 2 pi / N)^k, about N / pi times faster in the long run. An "iteration" in the API is one such
// dispatch either way.
//
// Bindings (besides the shared 0 `P` and 1 `solid_mask`):
//   2 vel      velocity (W+1 x H+1)                 (cs_divergence, cs_project, cs_resolve)
//   3 div_tex  r32float divergence                  (cs_jacobi, cs_jacobi_cold)
//   4 p_src    r32float pressure of the last iteration (cs_jacobi; cs_project and cs_resolve read it at 4 too)
//   5 p_dst    r32float storage: the new pressure   (cs_jacobi, cs_jacobi_cold)
//   6 div_dst  r32float storage: the divergence     (cs_divergence)
//   7 out_vel  rgba16float storage: projected velocity (cs_project)
//   8 fields   rgba16float storage: the resolved output (cs_resolve)

#include "fluid-core.wgsl"

@group(0) @binding(2) var vel: texture_2d<f32>;
@group(0) @binding(3) var div_tex: texture_2d<f32>;
@group(0) @binding(4) var p_src: texture_2d<f32>;
@group(0) @binding(5) var p_dst: texture_storage_2d<r32float, write>;
@group(0) @binding(6) var div_dst: texture_storage_2d<r32float, write>;
@group(0) @binding(7) var out_vel: texture_storage_2d<rgba16float, write>;
@group(0) @binding(8) var fields: texture_storage_2d<rgba16float, write>;

// ---- Divergence ----------------------------------------------------------------------------------------

@compute @workgroup_size(16, 16)
fn cs_divergence(@builtin(global_invocation_id) id: vec3u) {
  if (any(id.xy >= P.grid)) {
    return;
  }
  let c = vec2i(id.xy);
  var d = 0.0;
  if (cell_class(c) == CELL_FLUID) {
    d = face_tap(vel, 0u, c.x + 1, c.y).x - face_tap(vel, 0u, c.x, c.y).x
      + face_tap(vel, 1u, c.x, c.y + 1).x - face_tap(vel, 1u, c.x, c.y).x;
  }
  textureStore(div_dst, c, vec4f(d, 0.0, 0.0, 0.0));
}

// ---- Jacobi ------------------------------------------------------------------------------------------------

// (sum of the neighbours' pressure, number of non-solid neighbours), `use_prev` false meaning "all zero".
fn neighbour(c: vec2i, use_prev: bool) -> vec2f {
  let cls = cell_class(c);
  if (cls == CELL_SOLID) {
    return vec2f(0.0, 0.0);
  }
  if (cls == CELL_FLUID && use_prev) {
    return vec2f(textureLoad(p_src, cell_wrapped(c), 0).x, 1.0);
  }
  return vec2f(0.0, 1.0);
}

fn jacobi(c: vec2i, use_prev: bool) -> f32 {
  if (cell_class(c) != CELL_FLUID) {
    return 0.0;
  }
  let sn = neighbour(c + vec2i(1, 0), use_prev) + neighbour(c + vec2i(-1, 0), use_prev)
         + neighbour(c + vec2i(0, 1), use_prev) + neighbour(c + vec2i(0, -1), use_prev);
  if (sn.y < 0.5) {
    return 0.0;
  }
  return (sn.x - textureLoad(div_tex, c, 0).x) / sn.y;
}

// Damped Jacobi: phi' = (1 - w) phi + w J(phi). The undamped iteration (w = 1) multiplies the
// highest-frequency (chequerboard) error mode by -1 every step: an odd number of iterations then
// OVER-corrects it (measured: 31 iterations left four times the divergence of 30) and an even number
// does not touch it. With w = 0.8 that mode shrinks by 0.6 per iteration, every other mode still
// converges, and the smooth ones are slowed by only a fifth.
fn relaxed(c: vec2i, use_prev: bool) -> f32 {
  let own = select(0.0, textureLoad(p_src, c, 0).x, use_prev);
  return (1.0 - P.jacobi_w) * own + P.jacobi_w * jacobi(c, use_prev);
}

@compute @workgroup_size(16, 16)
fn cs_jacobi(@builtin(global_invocation_id) id: vec3u) {
  if (any(id.xy >= P.grid)) {
    return;
  }
  let c = vec2i(id.xy);
  textureStore(p_dst, c, vec4f(relaxed(c, true), 0.0, 0.0, 0.0));
}

// The first iteration of a solve that does not warm-start: the previous pressure is taken as zero.
// Red-black SOR half-sweep: cells of colour `red` (x + y even) become (1 - omega) phi + omega GS(phi),
// the others are copied. `use_prev` false starts from zero (the first red half-sweep of a cold solve).
fn sor(c: vec2i, red: bool, use_prev: bool) -> f32 {
  let own = select(0.0, textureLoad(p_src, c, 0).x, use_prev);
  let is_red = ((c.x + c.y) & 1) == 0;
  if (is_red != red) {
    return own;
  }
  return (1.0 - P.sor_omega) * own + P.sor_omega * jacobi(c, use_prev);
}

@compute @workgroup_size(16, 16)
fn cs_sor_red(@builtin(global_invocation_id) id: vec3u) {
  if (any(id.xy >= P.grid)) {
    return;
  }
  let c = vec2i(id.xy);
  textureStore(p_dst, c, vec4f(sor(c, true, true), 0.0, 0.0, 0.0));
}

@compute @workgroup_size(16, 16)
fn cs_sor_red_cold(@builtin(global_invocation_id) id: vec3u) {
  if (any(id.xy >= P.grid)) {
    return;
  }
  let c = vec2i(id.xy);
  textureStore(p_dst, c, vec4f(sor(c, true, false), 0.0, 0.0, 0.0));
}

@compute @workgroup_size(16, 16)
fn cs_sor_black(@builtin(global_invocation_id) id: vec3u) {
  if (any(id.xy >= P.grid)) {
    return;
  }
  let c = vec2i(id.xy);
  textureStore(p_dst, c, vec4f(sor(c, false, true), 0.0, 0.0, 0.0));
}

@compute @workgroup_size(16, 16)
fn cs_jacobi_cold(@builtin(global_invocation_id) id: vec3u) {
  if (any(id.xy >= P.grid)) {
    return;
  }
  let c = vec2i(id.xy);
  textureStore(p_dst, c, vec4f(relaxed(c, false), 0.0, 0.0, 0.0));
}

// ---- Projection ----------------------------------------------------------------------------------------------

fn phi(c: vec2i) -> f32 {
  if (cell_class(c) != CELL_FLUID) {
    return 0.0;
  }
  return textureLoad(p_src, cell_wrapped(c), 0).x;
}

fn projected(axis: u32, c: vec2i) -> f32 {
  let f = face_of(axis, c);
  if (face_state(axis, f) != FACE_FREE) {
    return 0.0;
  }
  let cells = face_cells(axis, f);
  let v = face_tap(vel, axis, f.x, f.y).x - (phi(cells[1]) - phi(cells[0]));
  return clamp(finite_or_zero(v), -P.vmax, P.vmax);
}

@compute @workgroup_size(16, 16)
fn cs_project(@builtin(global_invocation_id) id: vec3u) {
  if (any(id.xy >= P.grid + vec2u(1u))) {
    return;
  }
  let c = vec2i(id.xy);
  textureStore(out_vel, c, vec4f(projected(0u, c), projected(1u, c), 0.0, 0.0));
}

// ---- Resolve -----------------------------------------------------------------------------------------------------

// The consumer texel of a cell (W x H): x, y the velocity at the cell centre (the mean of its two
// faces per axis, image heights per second), z the vorticity in 1 / s (same sign convention as
// cs_curl), w the pressure potential phi. Solid cells are all zero.
@compute @workgroup_size(16, 16)
fn cs_resolve(@builtin(global_invocation_id) id: vec3u) {
  if (any(id.xy >= P.grid)) {
    return;
  }
  let c = vec2i(id.xy);
  if (cell_class(c) != CELL_FLUID) {
    textureStore(fields, c, vec4f(0.0));
    return;
  }
  let u = 0.5 * (face_tap(vel, 0u, c.x, c.y).x + face_tap(vel, 0u, c.x + 1, c.y).x);
  let v = 0.5 * (face_tap(vel, 1u, c.x, c.y).x + face_tap(vel, 1u, c.x, c.y + 1).x);
  let dv = face_tap(vel, 1u, c.x + 1, c.y).x - face_tap(vel, 1u, c.x - 1, c.y).x
         + face_tap(vel, 1u, c.x + 1, c.y + 1).x - face_tap(vel, 1u, c.x - 1, c.y + 1).x;
  let du = face_tap(vel, 0u, c.x, c.y + 1).x - face_tap(vel, 0u, c.x, c.y - 1).x
         + face_tap(vel, 0u, c.x + 1, c.y + 1).x - face_tap(vel, 0u, c.x + 1, c.y - 1).x;
  let w = 0.25 * P.cells_per_h * (dv - du);
  textureStore(fields, c, vec4f(u, v, w, textureLoad(p_src, c, 0).x));
}

// fluid-multigrid.wgsl -- the extra kernels of the multigrid pressure solve (pressureSolver 'multigrid',
// story 6.8): restriction of the residual, prolongation of the coarse correction, the coarse obstacle
// mask, and a near-exact solve of the coarsest level (up to 32 x 32 cells) inside one workgroup. The smoothing sweeps and the
// divergence are the existing `cs_jacobi`, `cs_jacobi_cold` and `cs_divergence` of fluid-project.wgsl,
// bound to a level's own uniform block and mask.
//
// THE SCHEME. The pressure equation is A phi = d with  A phi = (sum of fluid neighbours' phi) - n phi
// (n = neighbours that are not solid), the same operator on every level, with the cell size scaled out.
// A V-cycle on level l:
//     smooth A phi = d (damped Jacobi, a couple of sweeps)           removes the short wavelengths
//     r = d - A phi; d' = restrict(r); solve A e = d' one level down   the long wavelengths are
//     phi += prolong(e); smooth again                                  short on the coarser grid
// Levels halve the grid (ceil(n / 2), so any size works). Scaling: A on a grid with cells twice as wide
// is 4x the operator of the finer grid, so the restricted residual is the SUM of the four children's
// residuals (4 x their mean; rescaled by 4 / count where an odd size leaves a coarse cell fewer children).
// Prolongation is bilinear between the four nearest coarse cell centres, renormalised over the ones
// that are not solid. A coarse cell is solid only when all its children are, so obstacles survive
// as long as they are wider than the grid is coarse, and the smoothers see the same stencil on every
// level: the boundary rules of fluid-core.wgsl (wall, open, wrap, solid cells) apply unchanged.
//
// Every kernel here is bound with the uniform block and mask of the level its `P.grid` describes: the
// FINE level for restrict, prolong and mask (they compute the coarse size as (grid + 1) / 2), the
// level itself for the coarse solve.
//
// Bindings (besides the shared 0 `P` and 1 `solid_mask`); an entry point uses only some:
//   2 rhs_tex     r32float right-hand side of this level (cs_mg_restrict, cs_mg_coarse)
//   3 phi_tex     r32float pressure of this level, fine side (cs_mg_restrict, cs_mg_prolong)
//   4 corr_tex    r32float correction of the coarser level (cs_mg_prolong)
//   5 out_r32     r32float storage output
//   6 coarse_mask the coarser level's solid mask (cs_mg_prolong)
//   7 out_mask    rgba8unorm storage: the coarser level's mask (cs_mg_mask)

#include "fluid-core.wgsl"

@group(0) @binding(2) var rhs_tex: texture_2d<f32>;
@group(0) @binding(3) var phi_tex: texture_2d<f32>;
@group(0) @binding(4) var corr_tex: texture_2d<f32>;
@group(0) @binding(5) var out_r32: texture_storage_2d<r32float, write>;
@group(0) @binding(6) var coarse_mask: texture_2d<f32>;
@group(0) @binding(7) var out_mask: texture_storage_2d<rgba8unorm, write>;

fn mg_coarse_dims() -> vec2i {
  return (vec2i(P.grid) + vec2i(1)) / 2;
}

// ---- Residual and restriction --------------------------------------------------------------------

// d - A phi at a fluid cell, 0 elsewhere.
fn mg_residual(c: vec2i) -> f32 {
  if (cell_class(c) != CELL_FLUID) {
    return 0.0;
  }
  var sum = 0.0;
  var n = 0.0;
  for (var k = 0; k < 4; k++) {
    let q = c + select(select(vec2i(1, 0), vec2i(-1, 0), k == 1), select(vec2i(0, 1), vec2i(0, -1), k == 3), k >= 2);
    let cls = cell_class(q);
    if (cls == CELL_SOLID) {
      continue;
    }
    n += 1.0;
    if (cls == CELL_FLUID) {
      sum += textureLoad(phi_tex, cell_wrapped(q), 0).x;
    }
  }
  return textureLoad(rhs_tex, c, 0).x - (sum - n * textureLoad(phi_tex, c, 0).x);
}

@compute @workgroup_size(16, 16)
fn cs_mg_restrict(@builtin(global_invocation_id) id: vec3u) {
  let coarse = mg_coarse_dims();
  let k = vec2i(id.xy);
  if (any(k >= coarse)) {
    return;
  }
  let g = vec2i(P.grid);
  var sum = 0.0;
  var count = 0.0;
  for (var dy = 0; dy < 2; dy++) {
    for (var dx = 0; dx < 2; dx++) {
      let c = k * 2 + vec2i(dx, dy);
      if (c.x < g.x && c.y < g.y) {
        sum += mg_residual(c);
        count += 1.0;
      }
    }
  }
  textureStore(out_r32, k, vec4f(sum * 4.0 / max(count, 1.0), 0.0, 0.0, 0.0));
}

// The coarser level's mask: solid where every child is.
@compute @workgroup_size(16, 16)
fn cs_mg_mask(@builtin(global_invocation_id) id: vec3u) {
  let coarse = mg_coarse_dims();
  let k = vec2i(id.xy);
  if (any(k >= coarse)) {
    return;
  }
  let g = vec2i(P.grid);
  var all_solid = true;
  for (var dy = 0; dy < 2; dy++) {
    for (var dx = 0; dx < 2; dx++) {
      let c = k * 2 + vec2i(dx, dy);
      if (c.x < g.x && c.y < g.y && textureLoad(solid_mask, c, 0).r < 0.5) {
        all_solid = false;
      }
    }
  }
  textureStore(out_mask, k, vec4f(select(0.0, 1.0, all_solid), 0.0, 0.0, 1.0));
}

// ---- Prolongation --------------------------------------------------------------------------------

// (correction, weight) of a coarse cell; a cell outside the domain follows the boundary rule.
fn mg_coarse_tap(k: vec2i, coarse: vec2i) -> vec2f {
  var q = k;
  if (k.x < 0 || k.y < 0 || k.x >= coarse.x || k.y >= coarse.y) {
    if (P.boundary == BOUNDARY_OPEN) {
      return vec2f(0.0, 1.0);
    }
    if (P.boundary == BOUNDARY_WRAP) {
      q = vec2i(wrap_index(k.x, coarse.x), wrap_index(k.y, coarse.y));
    } else {
      q = clamp(k, vec2i(0), coarse - vec2i(1));
    }
  }
  if (P.solid != 0u && textureLoad(coarse_mask, q, 0).r >= 0.5) {
    return vec2f(0.0, 0.0);
  }
  return vec2f(textureLoad(corr_tex, q, 0).x, 1.0);
}

// phi + the bilinear interpolation of the coarse correction.
@compute @workgroup_size(16, 16)
fn cs_mg_prolong(@builtin(global_invocation_id) id: vec3u) {
  if (any(id.xy >= P.grid)) {
    return;
  }
  let c = vec2i(id.xy);
  if (cell_class(c) != CELL_FLUID) {
    textureStore(out_r32, c, vec4f(0.0));
    return;
  }
  let coarse = mg_coarse_dims();
  // Fine cell centres sit at c + 1/2; coarse ones at 2 (k + 1/2).
  let q = (vec2f(c) + 0.5) * 0.5 - 0.5;
  let b = floor(q);
  let t = q - b;
  let k = vec2i(b);
  let a00 = mg_coarse_tap(k, coarse);
  let a10 = mg_coarse_tap(k + vec2i(1, 0), coarse);
  let a01 = mg_coarse_tap(k + vec2i(0, 1), coarse);
  let a11 = mg_coarse_tap(k + vec2i(1, 1), coarse);
  let w00 = (1.0 - t.x) * (1.0 - t.y) * a00.y;
  let w10 = t.x * (1.0 - t.y) * a10.y;
  let w01 = (1.0 - t.x) * t.y * a01.y;
  let w11 = t.x * t.y * a11.y;
  let w = w00 + w10 + w01 + w11;
  var correction = 0.0;
  if (w > 1e-6) {
    correction = (w00 * a00.x + w10 * a10.x + w01 * a01.x + w11 * a11.x) / w;
  }
  textureStore(out_r32, c, vec4f(textureLoad(phi_tex, c, 0).x + correction, 0.0, 0.0, 0.0));
}

// ---- Coarsest level ------------------------------------------------------------------------------

// The whole level (at most 32 x 32 cells, four to a thread) in one workgroup: `P.mg_sweeps` sweeps of
// red-black SOR (omega in P.sor_omega) in shared memory, from zero, with a barrier between the colours.
// One dispatch instead of dozens of dependent ones, and the coarsest level ends up solved far better than
// a handful of Jacobi sweeps could.
var<workgroup> mg_phi: array<f32, 1024>;

const MG_SIDE: i32 = 32;

@compute @workgroup_size(16, 16)
fn cs_mg_coarse(@builtin(local_invocation_id) lid: vec3u) {
  let base = vec2i(lid.xy) * 2;
  let g = vec2i(P.grid);
  // Per cell of this thread's 2 x 2 block: the shared-memory index of each fluid neighbour (-1: none),
  // the number of neighbours that count in the diagonal, and the right-hand side.
  var nbr = array<i32, 16>(-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1);
  var diag = array<f32, 4>(0.0, 0.0, 0.0, 0.0);
  var rhs = array<f32, 4>(0.0, 0.0, 0.0, 0.0);
  var live = 0u;
  for (var q = 0; q < 4; q++) {
    let c = base + vec2i(q & 1, q >> 1);
    mg_phi[c.y * MG_SIDE + c.x] = 0.0;
    if (any(c >= g) || cell_class(c) != CELL_FLUID) {
      continue;
    }
    rhs[q] = textureLoad(rhs_tex, c, 0).x;
    for (var k = 0; k < 4; k++) {
      let n = c + select(select(vec2i(1, 0), vec2i(-1, 0), k == 1), select(vec2i(0, 1), vec2i(0, -1), k == 3), k >= 2);
      let cls = cell_class(n);
      if (cls == CELL_SOLID) {
        continue;
      }
      diag[q] += 1.0;
      if (cls == CELL_FLUID) {
        let w = cell_wrapped(n);
        nbr[q * 4 + k] = w.y * MG_SIDE + w.x;
      }
    }
    if (diag[q] > 0.5) {
      live |= 1u << u32(q);
    }
  }
  workgroupBarrier();
  for (var sweep = 0u; sweep < P.mg_sweeps; sweep++) {
    for (var colour = 0; colour < 2; colour++) {
      for (var q = 0; q < 4; q++) {
        let c = base + vec2i(q & 1, q >> 1);
        let red = ((c.x + c.y) & 1) == 0;
        if ((live & (1u << u32(q))) != 0u && red == (colour == 0)) {
          var sum = 0.0;
          for (var k = 0; k < 4; k++) {
            let at = nbr[q * 4 + k];
            if (at >= 0) {
              sum += mg_phi[u32(at)];
            }
          }
          let at = c.y * MG_SIDE + c.x;
          mg_phi[at] = (1.0 - P.sor_omega) * mg_phi[at] + P.sor_omega * (sum - rhs[q]) / diag[q];
        }
      }
      workgroupBarrier();
    }
  }
  for (var q = 0; q < 4; q++) {
    let c = base + vec2i(q & 1, q >> 1);
    if (all(c < g)) {
      let on = (live & (1u << u32(q))) != 0u;
      textureStore(out_r32, c, vec4f(select(0.0, mg_phi[c.y * MG_SIDE + c.x], on), 0.0, 0.0, 0.0));
    }
  }
}

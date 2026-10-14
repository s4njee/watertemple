// heightfield-core.wgsl -- the maths of the heightfield solver, shared by the render-pass and the
// compute-pass variant so that both run the SAME code and differ only in how they are launched.
//
// Included (as an effect-local chunk) by heightfield-render.wgsl and heightfield-compute.wgsl, after
// the generated `sim_params.wgsl` (struct SimParams). Bindings, group 0:
//   0  P           uniform SimParams
//   1  state_in    rgba16float   the previous state:  x height, y velocity (z, w unused)
//   2  solid_mask  r8unorm       obstacle mask, grid sized: >= 0.5 is solid
//
// UNITS. Heights are in CELLS (1 = one grid cell, so a stored gradient is a dimensionless slope) and
// velocity in cells per STEP. Time is only in the uniform: k = (c dt / cell)^2 is the squared Courant
// number, v_decay and h_decay are per-step multipliers the CPU computed from rates per second.
//
// SCHEME. Symplectic (semi-implicit) Euler for u_tt = c^2 laplacian(u) - damping u_t:
//     v' = (v + k L(h)) v_decay          h' = (h + v') h_decay
// which is the leapfrog scheme with the velocity kept on half steps: second-order accurate, no
// numerical damping of its own, stable while k (1 - cos) combined along the worst mode stays below 4,
// i.e. C = sqrt(k) <= 0.866 for the 9-point laplacian below (0.707 for the 5-point one).
//
// L is the isotropic 9-point laplacian [1 4 1; 4 -20 4; 1 4 1] / 6. The plain 5-point stencil is
// four times as cheap in taps but propagates diagonally slower than along the axes, so rings turn
// into rounded diamonds; this one is round to the fourth order.
//
// WALLS. A neighbour that lies outside the grid or inside a solid cell is replaced by the centre
// cell's own height: zero normal gradient, i.e. a reflecting wall without a sign flip. Solid cells
// themselves hold height 0 and velocity 0.

#include "sim_params.wgsl"

@group(0) @binding(0) var<uniform> P: SimParams;
@group(0) @binding(1) var state_in: texture_2d<f32>;
@group(0) @binding(2) var solid_mask: texture_2d<f32>;

const SOLID_THRESHOLD: f32 = 0.5;
// Hard limits. The Courant clamp makes them unreachable; they keep a bad impulse from ever turning
// into Inf in a half-float texture (65504 max).
const HEIGHT_LIMIT: f32 = 512.0;
const VELOCITY_LIMIT: f32 = 64.0;

fn sim_is_solid(c: vec2i) -> bool {
  if (P.solid == 0u) {
    return false;
  }
  return textureLoad(solid_mask, c, 0).r >= SOLID_THRESHOLD;
}

// The height cell `c` presents to a neighbour with height `centre`.
fn sim_height(c: vec2i, centre: f32) -> f32 {
  let e = clamp(c, vec2i(0), vec2i(P.size) - vec2i(1));
  if (sim_is_solid(e)) {
    return centre;
  }
  return textureLoad(state_in, e, 0).x;
}

fn sim_bump(q: f32) -> f32 {
  let t = max(1.0 - q * q, 0.0);
  return t * t;
}

// Adds the queued impulses. Impulse i is two vec4s:
//   a = (x, y, radius, amplitude) in cells; b = (x1, y1, kind, 0)
//   kind 0: a bump of height `amplitude` at (x, y)
//   kind 1: a footprint moving from (x, y) to (x1, y1): amplitude * (bump(from) - bump(to))
fn sim_splat(c: vec2i) -> vec4f {
  var s = textureLoad(state_in, c, 0);
  if (sim_is_solid(c)) {
    return vec4f(0.0);
  }
  let p = vec2f(c) + 0.5;
  var dh = 0.0;
  for (var i = 0u; i < P.impulse_count; i++) {
    let a = P.impulses[2u * i];
    let b = P.impulses[2u * i + 1u];
    var w = sim_bump(length(p - a.xy) / a.z);
    if (b.z > 0.5) {
      w -= sim_bump(length(p - b.xy) / a.z);
    }
    dh += a.w * w;
  }
  s.x = clamp(s.x + dh, -HEIGHT_LIMIT, HEIGHT_LIMIT);
  return s;
}

fn sim_step(c: vec2i) -> vec4f {
  if (sim_is_solid(c)) {
    return vec4f(0.0);
  }
  let s = textureLoad(state_in, c, 0);
  let h = s.x;
  let n = sim_height(c + vec2i(0, -1), h);
  let so = sim_height(c + vec2i(0, 1), h);
  let e = sim_height(c + vec2i(1, 0), h);
  let w = sim_height(c + vec2i(-1, 0), h);
  let ne = sim_height(c + vec2i(1, -1), h);
  let nw = sim_height(c + vec2i(-1, -1), h);
  let se = sim_height(c + vec2i(1, 1), h);
  let sw = sim_height(c + vec2i(-1, 1), h);
  let lap = (4.0 * (n + so + e + w) + (ne + nw + se + sw) - 20.0 * h) * (1.0 / 6.0);

  var v_decay = P.v_decay;
  var h_decay = P.h_decay;
  if (P.sponge_cells > 0.0) {
    // The sponge: extra damping that ramps up quadratically over the last `sponge_cells` cells, so
    // a wave is eaten before it can reflect and the ramp itself reflects almost nothing.
    let edge = min(min(c.x, c.y), min(i32(P.size.x) - 1 - c.x, i32(P.size.y) - 1 - c.y));
    let t = max(1.0 - f32(edge) / P.sponge_cells, 0.0);
    let sigma = P.sponge_rate * t * t;
    v_decay *= 1.0 - sigma;
    h_decay *= 1.0 - 0.5 * sigma;
  }
  let v = clamp((s.y + P.k * lap) * v_decay, -VELOCITY_LIMIT, VELOCITY_LIMIT);
  let next = clamp((h + v) * h_decay, -HEIGHT_LIMIT, HEIGHT_LIMIT);
  return vec4f(next, v, 0.0, 0.0);
}

// The consumer-facing texel: height and velocity unchanged, plus the Sobel gradient of the height
// (central difference smoothed across the neighbouring rows), in cells per cell = a pure slope.
// x grows right, y grows DOWN, matching uv.
fn sim_resolve(c: vec2i) -> vec4f {
  if (sim_is_solid(c)) {
    return vec4f(0.0);
  }
  let s = textureLoad(state_in, c, 0);
  let h = s.x;
  let n = sim_height(c + vec2i(0, -1), h);
  let so = sim_height(c + vec2i(0, 1), h);
  let e = sim_height(c + vec2i(1, 0), h);
  let w = sim_height(c + vec2i(-1, 0), h);
  let ne = sim_height(c + vec2i(1, -1), h);
  let nw = sim_height(c + vec2i(-1, -1), h);
  let se = sim_height(c + vec2i(1, 1), h);
  let sw = sim_height(c + vec2i(-1, 1), h);
  let gx = ((ne + 2.0 * e + se) - (nw + 2.0 * w + sw)) * 0.125;
  let gy = ((sw + 2.0 * so + se) - (nw + 2.0 * n + ne)) * 0.125;
  return vec4f(h, s.y, gx, gy);
}

// fluid-core.wgsl -- the shared maths of the stable-fluids solver (src/gpu/sim/fluid.ts): the grid
// arrangement, boundary handling and the samplers every pass uses. Included by fluid-advect.wgsl,
// fluid-forces.wgsl and fluid-project.wgsl after the generated `fluid_uniforms.wgsl`.
//
// GRID. A MAC ("marker and cell") grid, W x H square cells. Lengths inside the solver are in CELLS with
// the origin at the top-left corner of the domain and y pointing DOWN (the uv direction): cell (i, j)
// covers [i, i+1] x [j, j+1]. Velocity is stored on the FACES between cells, not at cell centres:
//     u(i, j)  x-velocity of the face on the LEFT of cell (i, j)   at (i,       j + 1/2)   i = 0..W, j = 0..H-1
//     v(i, j)  y-velocity of the face on the TOP  of cell (i, j)   at (i + 1/2, j      )   i = 0..W-1, j = 0..H
// both in image heights per second, in one rgba16float texture of (W+1) x (H+1) texels with u in .x and
// v in .y (the row H of u and the column W of v are padding that repeats the last row / column, so a
// hardware-filtered tap at the edge behaves like clamp-to-edge). Pressure, divergence and vorticity
// live at cell centres (W x H). Why not a plain cell-centred ("collocated") grid? The pressure solve
// uses the compact 5-point laplacian, and only face velocities make divergence and the pressure
// gradient the matching compact differences: the projected field is divergence-free to the solver's
// tolerance (collocated grids leave the checkerboard mode in the null space and a sizeable residual
// at short wavelengths), and a solid face is exactly zero, so nothing leaks through a one-cell wall.
//
// CELLS AND FACES have a class. A cell is FLUID, SOLID (obstacle mask >= 0.5, or outside the domain with
// the `wall` boundary) or AIR (outside the domain with the `open` boundary: pressure fixed at 0, flow
// may leave). `wrap` maps outside cells back inside. A face is FREE (both sides fluid or air), BLOCKED
// (exactly one side solid: its velocity is held at 0, it lies on a surface) or INSIDE (both sides solid:
// meaningless, ignored by every interpolation).
//
// Bindings shared by every pass: 0 `P` the uniform block, 1 `solid_mask` (r8unorm, grid sized).

#include "fluid_uniforms.wgsl"

@group(0) @binding(0) var<uniform> P: FluidUniforms;
@group(0) @binding(1) var solid_mask: texture_2d<f32>;

const BOUNDARY_WALL: u32 = 0u;
const BOUNDARY_OPEN: u32 = 1u;
const BOUNDARY_WRAP: u32 = 2u;

const CELL_FLUID: u32 = 0u;
const CELL_SOLID: u32 = 1u;
const CELL_AIR: u32 = 2u;

const FACE_FREE: u32 = 0u;
const FACE_BLOCKED: u32 = 1u;
const FACE_INSIDE: u32 = 2u;

const WORKGROUP_SIDE: u32 = 16u;

fn wrap_index(i: i32, n: i32) -> i32 {
  return ((i % n) + n) % n;
}

// 0 for NaN and +-Inf, x otherwise. Tests the exponent bits because `x != x` may be folded away.
fn finite_or_zero(x: f32) -> f32 {
  return select(x, 0.0, (bitcast<u32>(x) & 0x7f800000u) == 0x7f800000u);
}

fn finite2(v: vec2f) -> vec2f {
  return vec2f(finite_or_zero(v.x), finite_or_zero(v.y));
}

// ---- Cells ---------------------------------------------------------------------------------------

fn cell_class(c: vec2i) -> u32 {
  let g = vec2i(P.grid);
  var q = c;
  if (c.x < 0 || c.y < 0 || c.x >= g.x || c.y >= g.y) {
    if (P.boundary == BOUNDARY_WALL) {
      return CELL_SOLID;
    }
    if (P.boundary == BOUNDARY_OPEN) {
      return CELL_AIR;
    }
    q = vec2i(wrap_index(c.x, g.x), wrap_index(c.y, g.y));
  }
  if (P.solid != 0u && textureLoad(solid_mask, q, 0).r >= 0.5) {
    return CELL_SOLID;
  }
  return CELL_FLUID;
}

// The in-range cell a FLUID neighbour lives in (only `wrap` ever leaves the range).
fn cell_wrapped(c: vec2i) -> vec2i {
  let g = vec2i(P.grid);
  if (P.boundary == BOUNDARY_WRAP) {
    return vec2i(wrap_index(c.x, g.x), wrap_index(c.y, g.y));
  }
  return c;
}

// ---- Faces ---------------------------------------------------------------------------------------

// The two cells a face separates: axis 0 is a u face (left of cell f), axis 1 a v face (top of cell f).
fn face_cells(axis: u32, f: vec2i) -> array<vec2i, 2> {
  if (axis == 0u) {
    return array<vec2i, 2>(vec2i(f.x - 1, f.y), f);
  }
  return array<vec2i, 2>(vec2i(f.x, f.y - 1), f);
}

fn face_state(axis: u32, f: vec2i) -> u32 {
  if (P.solid == 0u) {
    // No obstacles: only a closed boundary blocks faces, and only those on the edge. This is the hot
    // path (every tap of every interpolation asks), so it skips the cell lookups below.
    if (P.boundary == BOUNDARY_WALL) {
      let g = vec2i(P.grid);
      let on_edge = select(f.y == 0 || f.y == g.y, f.x == 0 || f.x == g.x, axis == 0u);
      return select(FACE_FREE, FACE_BLOCKED, on_edge);
    }
    return FACE_FREE;
  }
  let cells = face_cells(axis, f);
  let a = cell_class(cells[0]);
  let b = cell_class(cells[1]);
  if (a == CELL_SOLID && b == CELL_SOLID) {
    return FACE_INSIDE;
  }
  if (a == CELL_SOLID || b == CELL_SOLID) {
    return FACE_BLOCKED;
  }
  return FACE_FREE;
}

// The face a texel of the velocity texture is responsible for, for one component: the padding row of
// u and column of v repeat the last real one.
fn face_of(axis: u32, texel: vec2i) -> vec2i {
  let g = vec2i(P.grid);
  if (axis == 0u) {
    return vec2i(texel.x, min(texel.y, g.y - 1));
  }
  return vec2i(min(texel.x, g.x - 1), texel.y);
}

// Where the face sits, in cells.
fn face_position(axis: u32, f: vec2i) -> vec2f {
  if (axis == 0u) {
    return vec2f(f32(f.x), f32(f.y) + 0.5);
  }
  return vec2f(f32(f.x) + 0.5, f32(f.y));
}

// Face coordinates mapped onto stored texels: clamped to the real faces, or wrapped.
fn face_index(axis: u32, i: i32, j: i32) -> vec2i {
  let g = vec2i(P.grid);
  if (P.boundary == BOUNDARY_WRAP) {
    return vec2i(wrap_index(i, g.x), wrap_index(j, g.y));
  }
  if (axis == 0u) {
    return vec2i(clamp(i, 0, g.x), clamp(j, 0, g.y - 1));
  }
  return vec2i(clamp(i, 0, g.x - 1), clamp(j, 0, g.y));
}

// One face of a velocity texture as (value, weight): weight 0 for a face inside a solid, value 0 for
// a blocked one. Interpolation renormalises by the weights, which extends the fluid's own velocity
// into solid cells (free-slip along obstacles) instead of dragging it towards zero.
fn face_tap(tex: texture_2d<f32>, axis: u32, i: i32, j: i32) -> vec2f {
  let f = face_index(axis, i, j);
  let s = textureLoad(tex, f, 0);
  let value = select(s.y, s.x, axis == 0u);
  let state = face_state(axis, f);
  if (state == FACE_INSIDE) {
    return vec2f(0.0, 0.0);
  }
  return vec2f(select(value, 0.0, state == FACE_BLOCKED), 1.0);
}

// Bilinear interpolation of one velocity component at position p (cells).
fn face_sample(tex: texture_2d<f32>, axis: u32, p: vec2f) -> f32 {
  let q = select(vec2f(p.x - 0.5, p.y), vec2f(p.x, p.y - 0.5), axis == 0u);
  let b = floor(q);
  let t = q - b;
  let i = i32(b.x);
  let j = i32(b.y);
  let a00 = face_tap(tex, axis, i, j);
  let a10 = face_tap(tex, axis, i + 1, j);
  let a01 = face_tap(tex, axis, i, j + 1);
  let a11 = face_tap(tex, axis, i + 1, j + 1);
  let w00 = (1.0 - t.x) * (1.0 - t.y) * a00.y;
  let w10 = t.x * (1.0 - t.y) * a10.y;
  let w01 = (1.0 - t.x) * t.y * a01.y;
  let w11 = t.x * t.y * a11.y;
  let w = w00 + w10 + w01 + w11;
  if (w < 1e-6) {
    return 0.0;
  }
  return (w00 * a00.x + w10 * a10.x + w01 * a01.x + w11 * a11.x) / w;
}

// The same interpolation, plus the smallest and largest value among the taps that contributed:
// (value, min, max). The bounds are what the MacCormack and BFECC limiters clamp to.
fn face_sample_bounds(tex: texture_2d<f32>, axis: u32, p: vec2f) -> vec3f {
  let q = select(vec2f(p.x - 0.5, p.y), vec2f(p.x, p.y - 0.5), axis == 0u);
  let b = floor(q);
  let t = q - b;
  let i = i32(b.x);
  let j = i32(b.y);
  let a00 = face_tap(tex, axis, i, j);
  let a10 = face_tap(tex, axis, i + 1, j);
  let a01 = face_tap(tex, axis, i, j + 1);
  let a11 = face_tap(tex, axis, i + 1, j + 1);
  let w00 = (1.0 - t.x) * (1.0 - t.y) * a00.y;
  let w10 = t.x * (1.0 - t.y) * a10.y;
  let w01 = (1.0 - t.x) * t.y * a01.y;
  let w11 = t.x * t.y * a11.y;
  let w = w00 + w10 + w01 + w11;
  if (w < 1e-6) {
    return vec3f(0.0);
  }
  var lo = 1e30;
  var hi = -1e30;
  // A tap that carries weight is a tap that exists; the zero-weight ones (inside a solid) are skipped.
  if (a00.y > 0.0) { lo = min(lo, a00.x); hi = max(hi, a00.x); }
  if (a10.y > 0.0) { lo = min(lo, a10.x); hi = max(hi, a10.x); }
  if (a01.y > 0.0) { lo = min(lo, a01.x); hi = max(hi, a01.x); }
  if (a11.y > 0.0) { lo = min(lo, a11.x); hi = max(hi, a11.x); }
  return vec3f((w00 * a00.x + w10 * a10.x + w01 * a01.x + w11 * a11.x) / w, lo, hi);
}

// Velocity (image heights per second) at p (cells).
fn vel_at(tex: texture_2d<f32>, p: vec2f) -> vec2f {
  return vec2f(face_sample(tex, 0u, p), face_sample(tex, 1u, p));
}

// ---- Tracing ---------------------------------------------------------------------------------------

fn clamp_length(d: vec2f, limit: f32) -> vec2f {
  let l = length(d);
  return select(d, d * (limit / l), l > limit);
}

// Where a parcel that is at p now was (way -1) or will be (way +1) one step away, in cells: an
// explicit Euler step, or the midpoint method when `rk2` is set (second order: rotation stays a
// rotation instead of spiralling outwards). The displacement is capped at `max_disp` cells.
fn trace(tex: texture_2d<f32>, p: vec2f, way: f32) -> vec2f {
  let k = way * P.dt * P.cells_per_h;
  var d = clamp_length(vel_at(tex, p) * k, P.max_disp);
  if (P.rk2 != 0u) {
    d = clamp_length(vel_at(tex, p + 0.5 * d) * k, P.max_disp);
  }
  return p + d;
}

// ---- Higher-order velocity (story 6.8) -----------------------------------------------------------------

// Cubic reconstruction of the staggered velocity for the DYE trace. `vel_at` is bilinear: continuous but
// with a kink in its slope at every cell border, which a fine dye grid shows as corners in streaklines
// (the dye is 4 times finer than the velocity, so a kink every 4 texels). These read the velocity
// texture through the hardware filter, nine or four taps instead of the sixteen of a gather, so the
// price is about that of the eight loads of `vel_at`. They assume the plain domain: no obstacles (a
// solid face would have to be left out of the stencil) and no periodic wrap (the texture is one face
// wider than the grid); the caller falls back to `vel_at` otherwise. `samp` is linear, clamp-to-edge.
//   kind 1  cubic B-spline          C2, reproduces linear fields exactly, smooths slightly (not interpolating)
//   kind 2  Catmull-Rom             C1, reproduces linear fields exactly, interpolates the stored faces,
//                                   with small negative lobes (the field may overshoot its faces by a few percent)
//   kind 3  hardware bilinear       the same function as `vel_at`, two filtered taps instead of eight loads and
//                                   the face bookkeeping (to the precision of the filter's weights)

// A texture reconstructed by a cubic B-spline at t (texels, centres at +0.5), four bilinear taps.
fn bspline_at(tex: texture_2d<f32>, samp: sampler, t: vec2f, size: vec2f) -> vec4f {
  let coord = t - 0.5;
  let base = floor(coord);
  let f = coord - base;
  let f2 = f * f;
  let f3 = f2 * f;
  let w0 = (1.0 - 3.0 * f + 3.0 * f2 - f3) / 6.0;
  let w1 = (4.0 - 6.0 * f2 + 3.0 * f3) / 6.0;
  let w2 = (1.0 + 3.0 * f + 3.0 * f2 - 3.0 * f3) / 6.0;
  let w3 = f3 / 6.0;
  let g0 = w0 + w1;
  let g1 = w2 + w3;
  let p0 = (base - 1.0 + w1 / g0 + 0.5) / size;
  let p1 = (base + 1.0 + w3 / g1 + 0.5) / size;
  let t00 = textureSampleLevel(tex, samp, vec2f(p0.x, p0.y), 0.0);
  let t10 = textureSampleLevel(tex, samp, vec2f(p1.x, p0.y), 0.0);
  let t01 = textureSampleLevel(tex, samp, vec2f(p0.x, p1.y), 0.0);
  let t11 = textureSampleLevel(tex, samp, vec2f(p1.x, p1.y), 0.0);
  return g0.y * (g0.x * t00 + g1.x * t10) + g1.y * (g0.x * t01 + g1.x * t11);
}

// The same with a Catmull-Rom spline: nine bilinear taps (the two middle texels of each axis share one).
fn catmull_rom_at(tex: texture_2d<f32>, samp: sampler, t: vec2f, size: vec2f) -> vec4f {
  let pos1 = floor(t - 0.5) + 0.5;
  let f = t - pos1;
  let w0 = f * (-0.5 + f * (1.0 - 0.5 * f));
  let w1 = 1.0 + f * f * (-2.5 + 1.5 * f);
  let w2 = f * (0.5 + f * (2.0 - 1.5 * f));
  let w3 = f * f * (-0.5 + 0.5 * f);
  let w12 = w1 + w2;
  let p0 = (pos1 - 1.0) / size;
  let p3 = (pos1 + 2.0) / size;
  let p12 = (pos1 + w2 / w12) / size;
  var sum = textureSampleLevel(tex, samp, vec2f(p0.x, p0.y), 0.0) * (w0.x * w0.y);
  sum += textureSampleLevel(tex, samp, vec2f(p12.x, p0.y), 0.0) * (w12.x * w0.y);
  sum += textureSampleLevel(tex, samp, vec2f(p3.x, p0.y), 0.0) * (w3.x * w0.y);
  sum += textureSampleLevel(tex, samp, vec2f(p0.x, p12.y), 0.0) * (w0.x * w12.y);
  sum += textureSampleLevel(tex, samp, vec2f(p12.x, p12.y), 0.0) * (w12.x * w12.y);
  sum += textureSampleLevel(tex, samp, vec2f(p3.x, p12.y), 0.0) * (w3.x * w12.y);
  sum += textureSampleLevel(tex, samp, vec2f(p0.x, p3.y), 0.0) * (w0.x * w3.y);
  sum += textureSampleLevel(tex, samp, vec2f(p12.x, p3.y), 0.0) * (w12.x * w3.y);
  sum += textureSampleLevel(tex, samp, vec2f(p3.x, p3.y), 0.0) * (w3.x * w3.y);
  return sum;
}

// Velocity (image heights per second) at p (cells) for a grid of `cells` cells: u from a reconstruction
// centred on the u faces (texel p + (1/2, 0)), v on the v faces (texel p + (0, 1/2)).
fn vel_at_cubic(tex: texture_2d<f32>, samp: sampler, p: vec2f, cells: vec2f, kind: u32) -> vec2f {
  let size = cells + vec2f(1.0);
  let tu = vec2f(p.x + 0.5, p.y);
  let tv = vec2f(p.x, p.y + 0.5);
  if (kind == 3u) {
    return vec2f(
      textureSampleLevel(tex, samp, tu / size, 0.0).x,
      textureSampleLevel(tex, samp, tv / size, 0.0).y,
    );
  }
  if (kind == 2u) {
    return vec2f(catmull_rom_at(tex, samp, tu, size).x, catmull_rom_at(tex, samp, tv, size).y);
  }
  return vec2f(bspline_at(tex, samp, tu, size).x, bspline_at(tex, samp, tv, size).y);
}

// `trace` with the cubic velocity.
fn trace_cubic(tex: texture_2d<f32>, samp: sampler, p: vec2f, way: f32) -> vec2f {
  let cells = vec2f(P.grid);
  let k = way * P.dt * P.cells_per_h;
  var d = clamp_length(vel_at_cubic(tex, samp, p, cells, P.dye_filter) * k, P.max_disp);
  if (P.rk2 != 0u) {
    d = clamp_length(vel_at_cubic(tex, samp, p + 0.5 * d, cells, P.dye_filter) * k, P.max_disp);
  }
  return p + d;
}

// ---- Dye -----------------------------------------------------------------------------------------

struct DyeTap {
  v: vec4f,
  w: f32,
}

// The sim cell a dye texel lies in.
fn dye_cell(t: vec2i) -> vec2i {
  let g = vec2f(P.grid);
  let d = vec2f(P.dye_size);
  return min(vec2i(floor((vec2f(t) + 0.5) * g / d)), vec2i(P.grid) - vec2i(1));
}

fn sim_to_dye(p: vec2f) -> vec2f {
  return p * vec2f(P.dye_size) / vec2f(P.grid);
}

fn dye_to_sim(q: vec2f) -> vec2f {
  return q * vec2f(P.grid) / vec2f(P.dye_size);
}

// A dye texel as (value, weight). Outside the domain: clamped for `wall`, wrapped for `wrap`, clear
// water (zero) for `open`. Texels in solid cells have weight 0.
fn dye_tap(tex: texture_2d<f32>, t: vec2i) -> DyeTap {
  let d = vec2i(P.dye_size);
  var q = t;
  if (t.x < 0 || t.y < 0 || t.x >= d.x || t.y >= d.y) {
    if (P.boundary == BOUNDARY_OPEN) {
      return DyeTap(vec4f(0.0), 1.0);
    }
    if (P.boundary == BOUNDARY_WRAP) {
      q = vec2i(wrap_index(t.x, d.x), wrap_index(t.y, d.y));
    } else {
      q = clamp(t, vec2i(0), d - vec2i(1));
    }
  }
  if (P.solid != 0u && textureLoad(solid_mask, dye_cell(q), 0).r >= 0.5) {
    return DyeTap(vec4f(0.0), 0.0);
  }
  return DyeTap(textureLoad(tex, q, 0), 1.0);
}

struct DyeSample {
  v: vec4f,
  lo: vec4f,
  hi: vec4f,
}

// Bilinear interpolation at q (dye texels, texel centres at +0.5) with the per-channel bounds of the
// contributing taps.
fn dye_sample(tex: texture_2d<f32>, q: vec2f) -> DyeSample {
  let b = floor(q - 0.5);
  let t = q - 0.5 - b;
  let i = vec2i(b);
  let a00 = dye_tap(tex, i);
  let a10 = dye_tap(tex, i + vec2i(1, 0));
  let a01 = dye_tap(tex, i + vec2i(0, 1));
  let a11 = dye_tap(tex, i + vec2i(1, 1));
  let w00 = (1.0 - t.x) * (1.0 - t.y) * a00.w;
  let w10 = t.x * (1.0 - t.y) * a10.w;
  let w01 = (1.0 - t.x) * t.y * a01.w;
  let w11 = t.x * t.y * a11.w;
  let w = w00 + w10 + w01 + w11;
  if (w < 1e-6) {
    return DyeSample(vec4f(0.0), vec4f(0.0), vec4f(0.0));
  }
  var lo = vec4f(1e30);
  var hi = vec4f(-1e30);
  if (a00.w > 0.0) { lo = min(lo, a00.v); hi = max(hi, a00.v); }
  if (a10.w > 0.0) { lo = min(lo, a10.v); hi = max(hi, a10.v); }
  if (a01.w > 0.0) { lo = min(lo, a01.v); hi = max(hi, a01.v); }
  if (a11.w > 0.0) { lo = min(lo, a11.v); hi = max(hi, a11.v); }
  return DyeSample((w00 * a00.v + w10 * a10.v + w01 * a01.v + w11 * a11.v) / w, lo, hi);
}

// ---- Splats --------------------------------------------------------------------------------------

// A capsule of influence: the segment a -> b in uv, inflated by a Gaussian of radius `shape.x` (image
// heights, the distance at which the weight is 1/e). A point is a segment with a == b. `shape.y` is
// the half-width of a smooth start and end ramp along the segment (0 = a plain flat capsule): with a
// ramp, segments laid end to end (a pointer's path, frame after frame) add up to a uniform stroke,
// because the falling edge of one is exactly the rising edge of the next. `value` is what the pass
// adds at full weight: a change of velocity (xy) or dye (rgba).
struct Splat {
  seg: vec4f,
  shape: vec4f,
  value: vec4f,
}

fn splat_weight(uv: vec2f, s: Splat) -> f32 {
  let k = vec2f(P.aspect, 1.0);
  let a = s.seg.xy * k;
  let ab = s.seg.zw * k - a;
  let p = uv * k;
  let len = length(ab);
  let ramp = s.shape.y;
  var along = 1.0;
  var d = length(p - a);
  if (len > 1e-6) {
    let dir = ab / len;
    let rel = p - a;
    let t = dot(rel, dir);
    if (ramp > 0.0) {
      along = smoothstep(-ramp, ramp, t) * (1.0 - smoothstep(len - ramp, len + ramp, t));
      d = length(rel - dir * t);
    } else {
      d = length(rel - dir * clamp(t, 0.0, len));
    }
  }
  let x = d / max(s.shape.x, 1e-5);
  if (x > 3.0) {
    return 0.0;
  }
  return along * exp(-x * x);
}

// fluid3d-core.wgsl -- the shared maths of the 3D stable-fluids solver (src/gpu/sim/fluid3d.ts): the grid
// arrangement, the boundary rule and the samplers every pass uses. Included by fluid3d-advect.wgsl,
// fluid3d-forces.wgsl and fluid3d-project.wgsl after the generated `fluid3d_uniforms.wgsl` (the
// uniform struct plus the workgroup size constants WG_X, WG_Y, WG_Z).
//
// GRID. A MAC ("marker and cell") grid like the 2D solver's, in three dimensions: nx x ny x nz cubic cells,
// y UP, cell (i, j, k) covers [i, i+1] x [j, j+1] x [k, k+1] in CELLS. Velocity lives on the faces
// between cells, one component per face orientation:
//     u(i, j, k)   x-velocity of the face on the -x side of cell (i, j, k)   at (i,       j + 1/2, k + 1/2)
//     v(i, j, k)   y-velocity of the face on the -y side                      at (i + 1/2, j,       k + 1/2)
//     w(i, j, k)   z-velocity of the face on the -z side                      at (i + 1/2, j + 1/2, k)
// all in one rgba16float 3D texture of (nx+1) x (ny+1) x (nz+1) texels with u in .x, v in .y, w in .z (the
// extra row of each component repeats the last real face, so a hardware-filtered tap at the edge behaves
// like clamp-to-edge). Pressure, divergence, vorticity live at cell centres (nx x ny x nz). The compact
// 7-point laplacian then matches the compact divergence and gradient exactly: no chequerboard null space and a
// wall face is exactly zero. The price against a collocated grid is three filtered taps to read one
// velocity (u, v and w are at different places): `vel_at`.
//
// UNITS. The solver works in cells inside the shaders (P.inv_h converts length to cells); the uniform
// block and the public API are in LENGTH units (the domain is size = (nx, ny, nz) * h long, h = P.h).
// Velocity: length per second. The pressure potential has the units of velocity (see fluid3d-project.wgsl).
//
// BOUNDARY. Closed box, free-slip: the normal velocity on the six walls is held at 0 by every pass that
// writes velocity, the tangential velocity is free (a filtered tap past the wall clamps to the edge, which
// repeats it). An optional sphere (P.obstacle, centre + radius in length units, radius 0 = none) makes the
// cells whose centre is inside it solid; faces touching a solid cell are held at 0 the same way.

#include "fluid3d_uniforms.wgsl"

@group(0) @binding(0) var<uniform> P: Fluid3dUniforms;

// 0 for NaN and +-Inf, x otherwise. Tests the exponent bits because `x != x` may be folded away.
fn finite_or_zero(x: f32) -> f32 {
  return select(x, 0.0, (bitcast<u32>(x) & 0x7f800000u) == 0x7f800000u);
}

fn finite3(v: vec3f) -> vec3f {
  return vec3f(finite_or_zero(v.x), finite_or_zero(v.y), finite_or_zero(v.z));
}

fn finite4(v: vec4f) -> vec4f {
  return select(v, vec4f(0.0), (bitcast<vec4u>(v) & vec4u(0x7f800000u)) == vec4u(0x7f800000u));
}

// ---- Cells and faces -------------------------------------------------------------------------------------

// True outside the box and inside the obstacle sphere. `dims` is the size of the grid the cell index
// belongs to; the sphere only exists on the full-resolution grid.
fn solid_at(c: vec3i, dims: vec3i) -> bool {
  if (any(c < vec3i(0)) || any(c >= dims)) {
    return true;
  }
  if (P.obstacle.w > 0.0 && all(dims == vec3i(P.grid))) {
    return distance((vec3f(c) + 0.5) * P.h, P.obstacle.xyz) < P.obstacle.w;
  }
  return false;
}

fn cell_solid(c: vec3i) -> bool {
  return solid_at(c, vec3i(P.grid));
}

fn axis_step(axis: u32) -> vec3i {
  return vec3i(select(0, 1, axis == 0u), select(0, 1, axis == 1u), select(0, 1, axis == 2u));
}

// A face is blocked (its velocity is held at 0) when it lies on the wall or touches a solid cell.
fn face_blocked(axis: u32, f: vec3i) -> bool {
  let g = vec3i(P.grid);
  if (P.obstacle.w <= 0.0) {
    return f[axis] <= 0 || f[axis] >= g[axis];
  }
  return cell_solid(f - axis_step(axis)) || cell_solid(f);
}

// The face a texel of the velocity texture is responsible for, for one component: the padding rows
// repeat the last real face.
fn face_of(axis: u32, texel: vec3i) -> vec3i {
  var f = min(texel, vec3i(P.grid) - vec3i(1));
  f[axis] = texel[axis];
  return f;
}

// Where a face sits, in cells.
fn face_position(axis: u32, f: vec3i) -> vec3f {
  return vec3f(f) + vec3f(0.5) - 0.5 * vec3f(axis_step(axis));
}

// One component of one face (0 for a blocked one).
fn face_load(tex: texture_3d<f32>, axis: u32, f: vec3i) -> f32 {
  let s = textureLoad(tex, f, 0);
  return select(s[axis], 0.0, face_blocked(axis, f));
}

// ---- Sampling ---------------------------------------------------------------------------------------------

// Filtered velocity component at position p (cells): one hardware trilinear tap. The texture coordinate
// of face (i, j, k) is (i + 1/2, j + 1/2, k + 1/2) texels, its position is (i, j + 1/2, k + 1/2) for u, hence the
// half-cell shift along the component's own axis.
fn comp_sample(tex: texture_3d<f32>, samp: sampler, axis: u32, p: vec3f) -> f32 {
  let q = clamp(p, vec3f(0.0), vec3f(P.grid));
  let tc = (q + 0.5 * vec3f(axis_step(axis))) / vec3f(P.grid + vec3u(1u));
  return textureSampleLevel(tex, samp, tc, 0.0)[axis];
}

// Smallest and largest value among the eight faces comp_sample interpolates: the MacCormack limiter's range.
fn comp_bounds(tex: texture_3d<f32>, axis: u32, p: vec3f) -> vec2f {
  let q = clamp(p, vec3f(0.0), vec3f(P.grid)) - vec3f(0.5) + 0.5 * vec3f(axis_step(axis));
  let b = vec3i(floor(q));
  let hi = vec3i(P.grid);
  var lo_v = 1e30;
  var hi_v = -1e30;
  for (var n = 0; n < 8; n++) {
    let d = vec3i(n & 1, (n >> 1) & 1, (n >> 2) & 1);
    var f = clamp(b + d, vec3i(0), hi - vec3i(1));
    f[axis] = clamp(b[axis] + d[axis], 0, hi[axis]);
    let s = textureLoad(tex, f, 0)[axis];
    lo_v = min(lo_v, s);
    hi_v = max(hi_v, s);
  }
  return vec2f(lo_v, hi_v);
}

// Velocity at position p (cells), length per second.
fn vel_at(tex: texture_3d<f32>, samp: sampler, p: vec3f) -> vec3f {
  return vec3f(comp_sample(tex, samp, 0u, p), comp_sample(tex, samp, 1u, p), comp_sample(tex, samp, 2u, p));
}

fn clamp_length(d: vec3f, limit: f32) -> vec3f {
  let l = length(d);
  return select(d, d * (limit / l), l > limit);
}

// Where a parcel that is at p now was (way -1) or will be (way +1) one step away, in cells: an explicit
// Euler step, or the midpoint method when `rk2` is set (second order: rotation stays a rotation instead
// of spiralling outwards). The displacement is capped at `max_disp` cells and the result stays in the box.
fn trace(tex: texture_3d<f32>, samp: sampler, p: vec3f, way: f32) -> vec3f {
  let k = way * P.dt * P.inv_h;
  var d = clamp_length(vel_at(tex, samp, p) * k, P.max_disp);
  if (P.rk2 != 0u) {
    d = clamp_length(vel_at(tex, samp, p + 0.5 * d) * k, P.max_disp);
  }
  return clamp(p + d, vec3f(0.0), vec3f(P.grid));
}

// ---- The scalar grid (ink) ----------------------------------------------------------------------------------

// The scalar grid covers the same box with its own number of texels. Positions in CELLS of the velocity
// grid are turned into texture coordinates by dividing by the grid size (the same for both grids).

// Position (cells) of the centre of scalar texel t.
fn scalar_pos(t: vec3i) -> vec3f {
  return (vec3f(t) + 0.5) * vec3f(P.grid) / vec3f(P.scalar_size);
}

fn scalar_solid(t: vec3i) -> bool {
  if (P.obstacle.w <= 0.0) {
    return false;
  }
  return distance((vec3f(t) + 0.5) / vec3f(P.scalar_size) * P.size, P.obstacle.xyz) < P.obstacle.w;
}

fn scalar_sample(tex: texture_3d<f32>, samp: sampler, p: vec3f) -> vec4f {
  return textureSampleLevel(tex, samp, p / vec3f(P.grid), 0.0);
}

// Per-channel smallest and largest value among the eight texels scalar_sample interpolates: (lo, hi).
struct Bounds {
  lo: vec4f,
  hi: vec4f,
}

fn scalar_bounds(tex: texture_3d<f32>, p: vec3f) -> Bounds {
  let s = vec3i(P.scalar_size);
  let q = p / vec3f(P.grid) * vec3f(P.scalar_size) - 0.5;
  let b = vec3i(floor(q));
  var lo = vec4f(1e30);
  var hi = vec4f(-1e30);
  for (var n = 0; n < 8; n++) {
    let d = vec3i(n & 1, (n >> 1) & 1, (n >> 2) & 1);
    let v = textureLoad(tex, clamp(b + d, vec3i(0), s - vec3i(1)), 0);
    lo = min(lo, v);
    hi = max(hi, v);
  }
  return Bounds(lo, hi);
}

// ---- Splats ---------------------------------------------------------------------------------------------------

// A capsule of influence: the segment a -> b (length units) inflated by a Gaussian of radius r (the
// distance at which the weight is 1/e). A point is a segment with a == b. `ramp` is the half-width of a
// smooth start and end along the segment (0 = a plain flat capsule): with a ramp, segments laid end to
// end (a pointer's path, frame after frame) add up to a uniform stroke. `value` is what the pass adds at
// full weight: a change of velocity (xyz) or of the scalar (rgba).
struct Splat {
  a: vec4f,     // xyz start, w radius
  b: vec4f,     // xyz end, w ramp
  value: vec4f,
}

fn splat_weight(p: vec3f, s: Splat) -> f32 {
  let a = s.a.xyz;
  let ab = s.b.xyz - a;
  let len = length(ab);
  let ramp = s.b.w;
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
  let x = d / max(s.a.w, 1e-5);
  if (x > 3.0) {
    return 0.0;
  }
  return along * exp(-x * x);
}

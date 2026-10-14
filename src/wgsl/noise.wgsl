// noise.wgsl -- lattice noise: value noise and gradient (Perlin-style) noise, 2D and 3D.
//
// Conventions
//  * Every function returns a SIGNED value in [-1, 1] (value noise: (-1, 1)). Map to [0, 1] with
//    0.5 + 0.5 * n. In practice gradient noise reaches about +-0.93 (2D) / +-0.89 (3D) and
//    value noise nearly the whole range.
//  * `p` is in lattice units: cells are 1.0 wide, so scale it for frequency (`p = uv * 8.0`).
//  * 3D noise doubles as animated 2D noise: pass vec3f(p, time).
//  * Quintic (C2) interpolation, so derivatives are continuous. Keep |p| below ~1e4; past that f32
//    can no longer resolve the fractional part. Integer lattice hashing => identical on all GPUs.
//  * `_d` functions return the value AND its analytic gradient w.r.t. p: .x = value, .yz (or .yzw)
//    = d/dx, d/dy (, d/dz). Use them for water normals, curl noise, slope-based shading.
//  * `_tile` functions repeat exactly every `period` cells per axis (period >= 1): f(p) ==
//    f(p + period). For a seamless texture of size N cells, pass uv * vec2f(period) as p. A z period
//    makes a seamless time loop; pass a large z period (e.g. 1 << 20) to leave z free.
//  * Inside one period a tile function equals its plain twin, except for the cells touching the seam.
//
// Value noise (smooth random lattice values; cheap, blobby, visible grid at low octaves)
//   vnoise_2d(p: vec2f) -> f32                         4 hashes
//   vnoise_3d(p: vec3f) -> f32                         8 hashes
//   vnoise_2d_d(p: vec2f) -> vec3f                     (value, d/dx, d/dy)
//   vnoise_2d_tile(p: vec2f, period: vec2i) -> f32
//
// Gradient noise (random lattice gradients; zero at lattice points, better looking than value noise)
//   gnoise_2d(p: vec2f) -> f32                         4 hashes, range [-1, 1], zero on integer points
//   gnoise_3d(p: vec3f) -> f32                         8 hashes
//   gnoise_2d_d(p: vec2f) -> vec3f                     (value, d/dx, d/dy)
//   gnoise_3d_d(p: vec3f) -> vec4f                     (value, d/dx, d/dy, d/dz)
//   gnoise_2d_tile(p: vec2f, period: vec2i) -> f32
//   gnoise_3d_tile(p: vec3f, period: vec3i) -> f32
//
// Technique: Perlin's improved noise (quintic fade, gradients from a hash); derivative formulas by
// differentiating the (tri)linear blend of corner contributions, as popularised by Inigo Quilez.

#include "hash.wgsl"
#include "math.wgsl"

// Scales that bring the raw gradient-noise sums to a worst-case magnitude of just under 1.
// 2D: unit gradients peak at sqrt(1/2). 3D: edge gradients (length sqrt 2) peak at ~1.036.
const _GNOISE_2D_SCALE: f32 = 1.4142;
const _GNOISE_3D_SCALE: f32 = 0.964;

fn _fade_2(t: vec2f) -> vec2f {
  return t * t * t * (t * (t * 6.0 - 15.0) + 10.0);
}

fn _fade_3(t: vec3f) -> vec3f {
  return t * t * t * (t * (t * 6.0 - 15.0) + 10.0);
}

// Derivative of the quintic fade: 30 t^2 (1 - t)^2.
fn _fade_d_2(t: vec2f) -> vec2f {
  let a = t * (1.0 - t);
  return 30.0 * a * a;
}

fn _fade_d_3(t: vec3f) -> vec3f {
  let a = t * (1.0 - t);
  return 30.0 * a * a;
}

fn _cell_2d(x: i32, y: i32) -> u32 {
  return pcg_2d(bitcast<vec2u>(vec2i(x, y))).x;
}

fn _cell_3d(x: i32, y: i32, z: i32) -> u32 {
  return pcg_3d(bitcast<vec3u>(vec3i(x, y, z))).x;
}

// Cell hashes of the corners (x0,y0) (x1,y0) (x0,y1) (x1,y1) of one square, in that order.
fn _quad_2d(x0: i32, x1: i32, y0: i32, y1: i32) -> vec4u {
  return vec4u(_cell_2d(x0, y0), _cell_2d(x1, y0), _cell_2d(x0, y1), _cell_2d(x1, y1));
}

fn _quad_3d(x0: i32, x1: i32, y0: i32, y1: i32, z: i32) -> vec4u {
  return vec4u(_cell_3d(x0, y0, z), _cell_3d(x1, y0, z), _cell_3d(x0, y1, z), _cell_3d(x1, y1, z));
}

fn _quad_2d_tile(c: vec2i, period: vec2i) -> vec4u {
  let a = wrap_2d(c, period);
  let b = wrap_2d(c + vec2i(1), period);
  return _quad_2d(a.x, b.x, a.y, b.y);
}

fn _vnoise_2d_core(u: vec2f, h: vec4u) -> f32 {
  let v = to_unorm_4d(h);
  return 2.0 * mix(mix(v.x, v.y, u.x), mix(v.z, v.w, u.x), u.y) - 1.0;
}

fn _vnoise_3d_core(u: vec3f, lo: vec4u, hi: vec4u) -> f32 {
  let a = to_unorm_4d(lo);
  let b = to_unorm_4d(hi);
  let z0 = mix(mix(a.x, a.y, u.x), mix(a.z, a.w, u.x), u.y);
  let z1 = mix(mix(b.x, b.y, u.x), mix(b.z, b.w, u.x), u.y);
  return 2.0 * mix(z0, z1, u.z) - 1.0;
}

// f: position inside the cell, u: faded f, h: corner hashes.
fn _gnoise_2d_core(f: vec2f, u: vec2f, h: vec4u) -> f32 {
  let va = dot(grad_2d(h.x), f);
  let vb = dot(grad_2d(h.y), f - vec2f(1.0, 0.0));
  let vc = dot(grad_2d(h.z), f - vec2f(0.0, 1.0));
  let vd = dot(grad_2d(h.w), f - vec2f(1.0, 1.0));
  return _GNOISE_2D_SCALE * mix(mix(va, vb, u.x), mix(vc, vd, u.x), u.y);
}

fn _gnoise_2d_core_d(f: vec2f, u: vec2f, du: vec2f, h: vec4u) -> vec3f {
  let ga = grad_2d(h.x);
  let gb = grad_2d(h.y);
  let gc = grad_2d(h.z);
  let gd = grad_2d(h.w);
  let va = dot(ga, f);
  let vb = dot(gb, f - vec2f(1.0, 0.0));
  let vc = dot(gc, f - vec2f(0.0, 1.0));
  let vd = dot(gd, f - vec2f(1.0, 1.0));
  let k = va - vb - vc + vd;
  let value = va + (vb - va) * u.x + (vc - va) * u.y + k * u.x * u.y;
  // Gradient = blend of the corner gradients + the change of the blend weights.
  let g = mix(mix(ga, gb, u.x), mix(gc, gd, u.x), u.y);
  let d = g + du * vec2f((vb - va) + k * u.y, (vc - va) + k * u.x);
  return _GNOISE_2D_SCALE * vec3f(value, d);
}

fn _gnoise_3d_core(f: vec3f, u: vec3f, lo: vec4u, hi: vec4u) -> f32 {
  let va = dot(grad_3d(lo.x), f);
  let vb = dot(grad_3d(lo.y), f - vec3f(1.0, 0.0, 0.0));
  let vc = dot(grad_3d(lo.z), f - vec3f(0.0, 1.0, 0.0));
  let vd = dot(grad_3d(lo.w), f - vec3f(1.0, 1.0, 0.0));
  let ve = dot(grad_3d(hi.x), f - vec3f(0.0, 0.0, 1.0));
  let vf = dot(grad_3d(hi.y), f - vec3f(1.0, 0.0, 1.0));
  let vg = dot(grad_3d(hi.z), f - vec3f(0.0, 1.0, 1.0));
  let vh = dot(grad_3d(hi.w), f - vec3f(1.0, 1.0, 1.0));
  let z0 = mix(mix(va, vb, u.x), mix(vc, vd, u.x), u.y);
  let z1 = mix(mix(ve, vf, u.x), mix(vg, vh, u.x), u.y);
  return _GNOISE_3D_SCALE * mix(z0, z1, u.z);
}

fn _gnoise_3d_core_d(f: vec3f, u: vec3f, du: vec3f, lo: vec4u, hi: vec4u) -> vec4f {
  let ga = grad_3d(lo.x);
  let gb = grad_3d(lo.y);
  let gc = grad_3d(lo.z);
  let gd = grad_3d(lo.w);
  let ge = grad_3d(hi.x);
  let gf = grad_3d(hi.y);
  let gg = grad_3d(hi.z);
  let gh = grad_3d(hi.w);
  let va = dot(ga, f);
  let vb = dot(gb, f - vec3f(1.0, 0.0, 0.0));
  let vc = dot(gc, f - vec3f(0.0, 1.0, 0.0));
  let vd = dot(gd, f - vec3f(1.0, 1.0, 0.0));
  let ve = dot(ge, f - vec3f(0.0, 0.0, 1.0));
  let vf = dot(gf, f - vec3f(1.0, 0.0, 1.0));
  let vg = dot(gg, f - vec3f(0.0, 1.0, 1.0));
  let vh = dot(gh, f - vec3f(1.0, 1.0, 1.0));
  // Trilinear blend written as a polynomial in u.
  let k1 = vb - va;
  let k2 = vc - va;
  let k3 = ve - va;
  let k4 = va - vb - vc + vd;
  let k5 = va - vc - ve + vg;
  let k6 = va - vb - ve + vf;
  let k7 = -va + vb + vc - vd + ve - vf - vg + vh;
  let value =
    va + k1 * u.x + k2 * u.y + k3 * u.z + k4 * u.x * u.y + k5 * u.y * u.z + k6 * u.z * u.x +
    k7 * u.x * u.y * u.z;
  let g = mix(
    mix(mix(ga, gb, u.x), mix(gc, gd, u.x), u.y),
    mix(mix(ge, gf, u.x), mix(gg, gh, u.x), u.y),
    u.z,
  );
  let d =
    g +
    du *
      vec3f(
        k1 + k4 * u.y + k6 * u.z + k7 * u.y * u.z,
        k2 + k4 * u.x + k5 * u.z + k7 * u.x * u.z,
        k3 + k5 * u.y + k6 * u.x + k7 * u.x * u.y,
      );
  return _GNOISE_3D_SCALE * vec4f(value, d);
}

fn vnoise_2d(p: vec2f) -> f32 {
  let ip = floor(p);
  let c = vec2i(ip);
  return _vnoise_2d_core(_fade_2(p - ip), _quad_2d(c.x, c.x + 1, c.y, c.y + 1));
}

fn vnoise_3d(p: vec3f) -> f32 {
  let ip = floor(p);
  let c = vec3i(ip);
  return _vnoise_3d_core(
    _fade_3(p - ip),
    _quad_3d(c.x, c.x + 1, c.y, c.y + 1, c.z),
    _quad_3d(c.x, c.x + 1, c.y, c.y + 1, c.z + 1),
  );
}

fn vnoise_2d_d(p: vec2f) -> vec3f {
  let ip = floor(p);
  let f = p - ip;
  let c = vec2i(ip);
  let u = _fade_2(f);
  let du = _fade_d_2(f);
  let v = to_unorm_4d(_quad_2d(c.x, c.x + 1, c.y, c.y + 1));
  let k = v.x - v.y - v.z + v.w;
  let value = v.x + (v.y - v.x) * u.x + (v.z - v.x) * u.y + k * u.x * u.y;
  let dx = du.x * ((v.y - v.x) + k * u.y);
  let dy = du.y * ((v.z - v.x) + k * u.x);
  return vec3f(2.0 * value - 1.0, 2.0 * dx, 2.0 * dy);
}

fn vnoise_2d_tile(p: vec2f, period: vec2i) -> f32 {
  let ip = floor(p);
  return _vnoise_2d_core(_fade_2(p - ip), _quad_2d_tile(vec2i(ip), period));
}

fn gnoise_2d(p: vec2f) -> f32 {
  let ip = floor(p);
  let f = p - ip;
  let c = vec2i(ip);
  return _gnoise_2d_core(f, _fade_2(f), _quad_2d(c.x, c.x + 1, c.y, c.y + 1));
}

fn gnoise_3d(p: vec3f) -> f32 {
  let ip = floor(p);
  let f = p - ip;
  let c = vec3i(ip);
  return _gnoise_3d_core(
    f,
    _fade_3(f),
    _quad_3d(c.x, c.x + 1, c.y, c.y + 1, c.z),
    _quad_3d(c.x, c.x + 1, c.y, c.y + 1, c.z + 1),
  );
}

fn gnoise_2d_d(p: vec2f) -> vec3f {
  let ip = floor(p);
  let f = p - ip;
  let c = vec2i(ip);
  return _gnoise_2d_core_d(f, _fade_2(f), _fade_d_2(f), _quad_2d(c.x, c.x + 1, c.y, c.y + 1));
}

fn gnoise_3d_d(p: vec3f) -> vec4f {
  let ip = floor(p);
  let f = p - ip;
  let c = vec3i(ip);
  return _gnoise_3d_core_d(
    f,
    _fade_3(f),
    _fade_d_3(f),
    _quad_3d(c.x, c.x + 1, c.y, c.y + 1, c.z),
    _quad_3d(c.x, c.x + 1, c.y, c.y + 1, c.z + 1),
  );
}

fn gnoise_2d_tile(p: vec2f, period: vec2i) -> f32 {
  let ip = floor(p);
  let f = p - ip;
  return _gnoise_2d_core(f, _fade_2(f), _quad_2d_tile(vec2i(ip), period));
}

fn gnoise_3d_tile(p: vec3f, period: vec3i) -> f32 {
  let ip = floor(p);
  let f = p - ip;
  let a = wrap_3d(vec3i(ip), period);
  let b = wrap_3d(vec3i(ip) + vec3i(1), period);
  return _gnoise_3d_core(
    f,
    _fade_3(f),
    _quad_3d(a.x, b.x, a.y, b.y, a.z),
    _quad_3d(a.x, b.x, a.y, b.y, b.z),
  );
}

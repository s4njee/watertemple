// math.wgsl -- shared constants and small helpers.
//
// Constants
//   PI, TAU (= 2 PI), HALF_PI                       f32
// Functions
//   rotation_2d(angle) -> mat2x2f     counter-clockwise rotation; `rotation_2d(a) * v` rotates v
//   rotate_2d(p, angle) -> vec2f      p rotated counter-clockwise about the origin (radians)
//   remap(x, in_lo, in_hi, out_lo, out_hi) -> f32     linear map between ranges, NOT clamped
//   smootherstep(lo, hi, x) -> f32    quintic smoothstep: 0 below lo, 1 above hi, C2 at both ends
//   wrap_2d(c, period) -> vec2i       per-component floor-modulo, result in [0, period)
//   wrap_3d(c, period) -> vec3i       same for 3 components
//
// WGSL's `%` keeps the sign of the dividend (-1 % 4 == -1); wrap_* is the periodic-lattice
// version (-1 -> 3). `period` is clamped to >= 1 per axis. All of it is a few ALU ops.

const PI: f32 = 3.14159265358979;
const TAU: f32 = 6.28318530717959;
const HALF_PI: f32 = 1.5707963267949;

fn rotation_2d(angle: f32) -> mat2x2f {
  let c = cos(angle);
  let s = sin(angle);
  return mat2x2f(c, s, -s, c);
}

fn rotate_2d(p: vec2f, angle: f32) -> vec2f {
  let c = cos(angle);
  let s = sin(angle);
  return vec2f(c * p.x - s * p.y, s * p.x + c * p.y);
}

fn remap(x: f32, in_lo: f32, in_hi: f32, out_lo: f32, out_hi: f32) -> f32 {
  return out_lo + (x - in_lo) * (out_hi - out_lo) / (in_hi - in_lo);
}

fn smootherstep(lo: f32, hi: f32, x: f32) -> f32 {
  let t = clamp((x - lo) / (hi - lo), 0.0, 1.0);
  return t * t * t * (t * (t * 6.0 - 15.0) + 10.0);
}

fn wrap_2d(c: vec2i, period: vec2i) -> vec2i {
  let n = max(period, vec2i(1));
  let m = c % n;
  return m + select(vec2i(0), n, m < vec2i(0));
}

fn wrap_3d(c: vec3i, period: vec3i) -> vec3i {
  let n = max(period, vec3i(1));
  let m = c % n;
  return m + select(vec3i(0), n, m < vec3i(0));
}

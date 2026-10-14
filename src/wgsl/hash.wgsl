// hash.wgsl -- integer hashes (PCG family) with float wrappers. No sin(), no float-precision
// dependence: the same input bits give the same output bits on every device.
//
// Credits: pcg_1d is the PCG output permutation (M. O'Neill, pcg-random.org); pcg_2d/3d/4d are the
// multi-component PCG hashes from Jarzynski & Olano, "Hash Functions for GPU Rendering" (JCGT 2020).
//
// Raw integer hashes (u32 in, u32 out; every output bit is balanced and depends on the whole input)
//   pcg_1d(v: u32) -> u32            pcg_2d(v: vec2u) -> vec2u
//   pcg_3d(v: vec3u) -> vec3u        pcg_4d(v: vec4u) -> vec4u
//
// u32 -> float in [0, 1), exactly 24 bits of precision (never returns 1.0)
//   to_unorm_1d(h: u32) -> f32       to_unorm_2d(h: vec2u) -> vec2f
//   to_unorm_3d(h: vec3u) -> vec3f   to_unorm_4d(h: vec4u) -> vec4f
//
// Float-input hashes, named hash_<in>to<out>: hash the BIT PATTERN of the inputs, output in [0, 1).
// Any change in the input bits changes the output completely (white noise), so for per-cell
// randomness hash floor(p). -0.0 and 0.0 hash differently; NaN/inf are not meaningful inputs.
//   hash_1to1(x: f32) -> f32         hash_1to2(x: f32) -> vec2f
//   hash_2to1(p: vec2f) -> f32       hash_2to2(p: vec2f) -> vec2f       hash_2to3(p: vec2f) -> vec3f
//   hash_3to1(p: vec3f) -> f32       hash_3to3(p: vec3f) -> vec3f
//
// Integer-lattice hashes, named ihash_<in>to<out>: hash an integer cell coordinate, output in [0, 1).
// Use these when the input is already a cell index (negative cells are fine).
//   ihash_2to1(c: vec2i) -> f32      ihash_2to2(c: vec2i) -> vec2f
//   ihash_3to1(c: vec3i) -> f32      ihash_3to3(c: vec3i) -> vec3f
//
// Gradient picking, for gradient/simplex noise: the low bits of any u32 hash select a direction.
// Computed with selects, not constant arrays, so nothing is indexed dynamically.
//   grad_2d(h: u32) -> vec2f         one of 8 unit vectors, 45 degrees apart
//   grad_3d(h: u32) -> vec3f         one of the 12 cube-edge directions (+-1, +-1, 0 and
//                                    permutations, length sqrt 2); 4 of them occur twice to fill 16 slots
//
// Cost: pcg_1d ~ 6 integer ops; pcg_2d/3d/4d ~ 15-25; float-input hashes add one pcg_1d per component.
// Everything is branch-free.

const _UNORM_SCALE: f32 = 1.0 / 16777216.0; // 2^-24

fn pcg_1d(v: u32) -> u32 {
  let state = v * 747796405u + 2891336453u;
  let word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
  return (word >> 22u) ^ word;
}

fn pcg_2d(v_in: vec2u) -> vec2u {
  var v = v_in * 1664525u + 1013904223u;
  v.x += v.y * 1664525u;
  v.y += v.x * 1664525u;
  v ^= v >> vec2u(16u);
  v.x += v.y * 1664525u;
  v.y += v.x * 1664525u;
  v ^= v >> vec2u(16u);
  return v;
}

fn pcg_3d(v_in: vec3u) -> vec3u {
  var v = v_in * 1664525u + 1013904223u;
  v.x += v.y * v.z;
  v.y += v.z * v.x;
  v.z += v.x * v.y;
  v ^= v >> vec3u(16u);
  v.x += v.y * v.z;
  v.y += v.z * v.x;
  v.z += v.x * v.y;
  return v;
}

fn pcg_4d(v_in: vec4u) -> vec4u {
  var v = v_in * 1664525u + 1013904223u;
  v.x += v.y * v.w;
  v.y += v.z * v.x;
  v.z += v.x * v.y;
  v.w += v.y * v.z;
  v ^= v >> vec4u(16u);
  v.x += v.y * v.w;
  v.y += v.z * v.x;
  v.z += v.x * v.y;
  v.w += v.y * v.z;
  return v;
}

fn to_unorm_1d(h: u32) -> f32 {
  return f32(h >> 8u) * _UNORM_SCALE;
}

fn to_unorm_2d(h: vec2u) -> vec2f {
  return vec2f(h >> vec2u(8u)) * _UNORM_SCALE;
}

fn to_unorm_3d(h: vec3u) -> vec3f {
  return vec3f(h >> vec3u(8u)) * _UNORM_SCALE;
}

fn to_unorm_4d(h: vec4u) -> vec4f {
  return vec4f(h >> vec4u(8u)) * _UNORM_SCALE;
}

// Float bit patterns are low-entropy (consecutive integers differ only in a few top mantissa bits), which
// the cheap multi-component PCG mixers do not fully diffuse. Each component is run through pcg_1d first.
fn _float_bits_2d(p: vec2f) -> vec2u {
  let b = bitcast<vec2u>(p);
  return vec2u(pcg_1d(b.x), pcg_1d(b.y));
}

fn _float_bits_3d(p: vec3f) -> vec3u {
  let b = bitcast<vec3u>(p);
  return vec3u(pcg_1d(b.x), pcg_1d(b.y), pcg_1d(b.z));
}

fn hash_1to1(x: f32) -> f32 {
  return to_unorm_1d(pcg_1d(bitcast<u32>(x)));
}

fn hash_1to2(x: f32) -> vec2f {
  return to_unorm_2d(pcg_2d(vec2u(pcg_1d(bitcast<u32>(x)), 0u)));
}

fn hash_2to1(p: vec2f) -> f32 {
  return to_unorm_1d(pcg_2d(_float_bits_2d(p)).x);
}

fn hash_2to2(p: vec2f) -> vec2f {
  return to_unorm_2d(pcg_2d(_float_bits_2d(p)));
}

fn hash_2to3(p: vec2f) -> vec3f {
  return to_unorm_3d(pcg_3d(vec3u(_float_bits_2d(p), 0u)));
}

fn hash_3to1(p: vec3f) -> f32 {
  return to_unorm_1d(pcg_3d(_float_bits_3d(p)).x);
}

fn hash_3to3(p: vec3f) -> vec3f {
  return to_unorm_3d(pcg_3d(_float_bits_3d(p)));
}

fn ihash_2to1(c: vec2i) -> f32 {
  return to_unorm_1d(pcg_2d(bitcast<vec2u>(c)).x);
}

fn ihash_2to2(c: vec2i) -> vec2f {
  return to_unorm_2d(pcg_2d(bitcast<vec2u>(c)));
}

fn ihash_3to1(c: vec3i) -> f32 {
  return to_unorm_1d(pcg_3d(bitcast<vec3u>(c)).x);
}

fn ihash_3to3(c: vec3i) -> vec3f {
  return to_unorm_3d(pcg_3d(bitcast<vec3u>(c)));
}

const _SQRT_HALF: f32 = 0.70710678118654752; // sqrt(1/2)

// The directions are built with selects instead of a constant array, so nothing is indexed dynamically
// and the result does not depend on how a particular WGSL compiler treats that.
fn grad_2d(h: u32) -> vec2f {
  let m = h & 3u;
  let axis = select(vec2f(1.0, 0.0), vec2f(0.0, 1.0), m == 2u); // 0 and 90 degrees
  let diagonal = select(vec2f(_SQRT_HALF, _SQRT_HALF), vec2f(-_SQRT_HALF, _SQRT_HALF), m == 3u); // 45 and 135
  let g = select(axis, diagonal, (m & 1u) == 1u);
  return select(g, -g, (h & 4u) != 0u); // the other half of the circle
}

fn grad_3d(h: u32) -> vec3f {
  let k = h & 15u;
  let a = select(vec3f(1.0, 0.0, 0.0), vec3f(0.0, 1.0, 0.0), k >= 8u);
  let b = select(
    select(vec3f(0.0, 0.0, 1.0), vec3f(1.0, 0.0, 0.0), k == 12u || k == 14u),
    vec3f(0.0, 1.0, 0.0),
    k < 4u,
  );
  return select(a, -a, (k & 1u) != 0u) + select(b, -b, (k & 2u) != 0u);
}

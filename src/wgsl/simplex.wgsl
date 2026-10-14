// simplex.wgsl -- simplex noise: gradient noise on a triangular (2D) / tetrahedral (3D) lattice.
// Fewer corners than lattice noise in 3D (4 vs 8), no axis-aligned grid look, smoother and more
// isotropic. Use it when grid artefacts from noise.wgsl are visible (low octave counts, big features).
//
//   snoise_2d(p: vec2f) -> f32       range [-1, 1]; 3 corners, 3 hashes
//   snoise_2d_d(p: vec2f) -> vec3f   (value, d/dx, d/dy), analytic gradient; same range for the value
//   snoise_3d(p: vec3f) -> f32       range [-1, 1]; 4 corners, 4 hashes. Animated 2D: vec3f(p, time)
//
// `p` is in lattice units (feature size ~1.0 per unit). Zero mean, signed like noise.wgsl, deterministic
// integer hashing. Keep |p| below ~1e4. No tileable form: use gnoise_*_tile from noise.wgsl for that.
//
// Technique: Perlin's simplex noise (2001); radial kernel (0.5 - r^2)^4 in 2D and (0.6 - r^2)^4 in 3D
// follow Stefan Gustavson's "Simplex noise demystified". Code written from the published algorithm.

#include "hash.wgsl"

const _SKEW_2D: f32 = 0.36602540378443865; // (sqrt(3) - 1) / 2
const _UNSKEW_2D: f32 = 0.21132486540518713; // (3 - sqrt(3)) / 6
const _SKEW_3D: f32 = 0.3333333333333333; // 1 / 3
const _UNSKEW_3D: f32 = 0.16666666666666666; // 1 / 6

// Scales that bring the worst case (every corner's gradient perfectly aligned) to just under 1.
const _SNOISE_2D_SCALE: f32 = 99.0;
const _SNOISE_3D_SCALE: f32 = 32.6;

fn _simplex_hash_2d(c: vec2i) -> u32 {
  return pcg_2d(bitcast<vec2u>(c)).x;
}

fn _simplex_hash_3d(c: vec3i) -> u32 {
  return pcg_3d(bitcast<vec3u>(c)).x;
}

// One corner's contribution t^4 (g . r), r = vector from the corner to the sample point.
fn _simplex_corner_2d(h: u32, r: vec2f) -> f32 {
  let t = max(0.5 - dot(r, r), 0.0);
  let t2 = t * t;
  return t2 * t2 * dot(grad_2d(h), r);
}

// (contribution, d/dx, d/dy) of one corner.
fn _simplex_corner_2d_d(h: u32, r: vec2f) -> vec3f {
  let t = max(0.5 - dot(r, r), 0.0);
  let t2 = t * t;
  let g = grad_2d(h);
  let gr = dot(g, r);
  return vec3f(t2 * t2 * gr, t2 * t2 * g - 8.0 * t2 * t * gr * r);
}

fn _simplex_corner_3d(h: u32, r: vec3f) -> f32 {
  let t = max(0.6 - dot(r, r), 0.0);
  let t2 = t * t;
  return t2 * t2 * dot(grad_3d(h), r);
}

fn snoise_2d(p: vec2f) -> f32 {
  let i = floor(p + (p.x + p.y) * _SKEW_2D);
  let p0 = p - (i - (i.x + i.y) * _UNSKEW_2D);
  // Which of the two triangles of the skewed cell the point is in decides the middle corner.
  let step_x = p0.x > p0.y;
  let o = select(vec2f(0.0, 1.0), vec2f(1.0, 0.0), step_x);
  let p1 = p0 - o + _UNSKEW_2D;
  let p2 = p0 - 1.0 + 2.0 * _UNSKEW_2D;
  let c = vec2i(i);
  let n =
    _simplex_corner_2d(_simplex_hash_2d(c), p0) +
    _simplex_corner_2d(_simplex_hash_2d(c + vec2i(o)), p1) +
    _simplex_corner_2d(_simplex_hash_2d(c + vec2i(1)), p2);
  return _SNOISE_2D_SCALE * n;
}

fn snoise_2d_d(p: vec2f) -> vec3f {
  let i = floor(p + (p.x + p.y) * _SKEW_2D);
  let p0 = p - (i - (i.x + i.y) * _UNSKEW_2D);
  let step_x = p0.x > p0.y;
  let o = select(vec2f(0.0, 1.0), vec2f(1.0, 0.0), step_x);
  let p1 = p0 - o + _UNSKEW_2D;
  let p2 = p0 - 1.0 + 2.0 * _UNSKEW_2D;
  let c = vec2i(i);
  let n =
    _simplex_corner_2d_d(_simplex_hash_2d(c), p0) +
    _simplex_corner_2d_d(_simplex_hash_2d(c + vec2i(o)), p1) +
    _simplex_corner_2d_d(_simplex_hash_2d(c + vec2i(1)), p2);
  return _SNOISE_2D_SCALE * n;
}

fn snoise_3d(p: vec3f) -> f32 {
  let i = floor(p + (p.x + p.y + p.z) * _SKEW_3D);
  let p0 = p - (i - (i.x + i.y + i.z) * _UNSKEW_3D);
  // Rank the three coordinates (ties broken consistently so ranks are always 0, 1, 2).
  // The tetrahedron's path from the origin corner steps along the largest, then the second largest.
  let rank = vec3f(
    f32(p0.x >= p0.y) + f32(p0.x >= p0.z),
    f32(p0.y > p0.x) + f32(p0.y >= p0.z),
    f32(p0.z > p0.x) + f32(p0.z > p0.y),
  );
  let o1 = step(vec3f(2.0), rank);
  let o2 = step(vec3f(1.0), rank);
  let p1 = p0 - o1 + _UNSKEW_3D;
  let p2 = p0 - o2 + 2.0 * _UNSKEW_3D;
  let p3 = p0 - 1.0 + 3.0 * _UNSKEW_3D;
  let c = vec3i(i);
  let n =
    _simplex_corner_3d(_simplex_hash_3d(c), p0) +
    _simplex_corner_3d(_simplex_hash_3d(c + vec3i(o1)), p1) +
    _simplex_corner_3d(_simplex_hash_3d(c + vec3i(o2)), p2) +
    _simplex_corner_3d(_simplex_hash_3d(c + vec3i(1)), p3);
  return _SNOISE_3D_SCALE * n;
}

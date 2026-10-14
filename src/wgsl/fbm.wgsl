// fbm.wgsl -- fractal sums of gradient noise (gnoise_* from noise.wgsl) and domain warping.
//
// Shared parameters
//   octaves     i32, clamped to [0, FBM_MAX_OCTAVES = 16]; cost is one noise evaluation per octave
//   lacunarity  frequency multiplier per octave, typically 2.0
//   gain        amplitude multiplier per octave, typically 0.5 (keep >= 0)
// Each result is divided by the sum of the octave amplitudes, so the stated range holds for any octave
// count, lacunarity and gain. In practice a 4-8 octave sum stays within about +-0.7 (std dev ~0.18):
// multiply by ~1.4 if you need the full span. Octave 0 uses `p` unchanged, so fbm_*(p, 1, ..) ==
// gnoise_*(p). Later octaves are rotated (36.87 degrees; about z for 3D) and shifted so the lattice
// axes and zero points of different octaves do not line up.
//
//   fbm_2d(p: vec2f, octaves, lacunarity, gain) -> f32          [-1, 1]
//   fbm_3d(p: vec3f, octaves, lacunarity, gain) -> f32          [-1, 1]; z is time or a third axis
//   fbm_2d_d(p: vec2f, octaves, lacunarity, gain) -> vec3f      (value, d/dx, d/dy), analytic
//   fbm_3d_d(p: vec3f, octaves, lacunarity, gain) -> vec4f      (value, d/dx, d/dy, d/dz), analytic
//   ridged_2d(p: vec2f, octaves, lacunarity, gain) -> f32       [0, 1]; (1 - |noise|)^2 per octave, so bright
//                                                               creases where the noise crosses zero: wave
//                                                               crests, veins, caustic-like lines
//   fbm_2d_tile(p: vec2f, period: vec2i, octaves, gain) -> f32  [-1, 1]; repeats every `period` cells of
//                                                               octave 0 (lacunarity fixed at 2, no rotation)
//   domain_warp_2d(p, strength, octaves, lacunarity, gain) -> vec2f
//                                  p displaced by up to `strength` (in p units) per axis along two fbm fields.
//                                  Feed the result to any pattern: fbm_2d(domain_warp_2d(p, ..), ..)
//   domain_warp_2d_anim(p, time, strength, octaves, lacunarity, gain) -> vec2f
//                                  same, evolving with `time` (3D fbm, about 2x the cost)
//
// The _d gradients are w.r.t. p (octave frequency and rotation are accounted for), so they are not
// unit-scale: expect magnitudes of 1-3. Cost is about 1.5x the plain form. Domain warping: I. Quilez, "warp".

#include "noise.wgsl"

const FBM_MAX_OCTAVES: i32 = 16;

// Per-octave transform q' = lacunarity * R q + shift. 0.8/0.6 is a 3-4-5 rotation (36.87 degrees).
const _FBM_ROT_2D = mat2x2f(0.8, 0.6, -0.6, 0.8);
const _FBM_ROT_3D = mat3x3f(0.8, 0.6, 0.0, -0.6, 0.8, 0.0, 0.0, 0.0, 1.0);
const _FBM_SHIFT_2D = vec2f(1.37, 2.71);
const _FBM_SHIFT_3D = vec3f(1.37, 2.71, 0.93);
const _WARP_OFFSET = vec2f(5.2, 1.3);

fn fbm_2d(p: vec2f, octaves: i32, lacunarity: f32, gain: f32) -> f32 {
  let n = clamp(octaves, 0, FBM_MAX_OCTAVES);
  let step_q = lacunarity * _FBM_ROT_2D;
  var q = p;
  var amp = 1.0;
  var sum = 0.0;
  var norm = 0.0;
  for (var i = 0; i < n; i++) {
    sum += amp * gnoise_2d(q);
    norm += abs(amp);
    amp *= gain;
    q = step_q * q + _FBM_SHIFT_2D;
  }
  return sum / max(norm, 1e-6);
}

fn fbm_3d(p: vec3f, octaves: i32, lacunarity: f32, gain: f32) -> f32 {
  let n = clamp(octaves, 0, FBM_MAX_OCTAVES);
  let step_q = lacunarity * _FBM_ROT_3D;
  var q = p;
  var amp = 1.0;
  var sum = 0.0;
  var norm = 0.0;
  for (var i = 0; i < n; i++) {
    sum += amp * gnoise_3d(q);
    norm += abs(amp);
    amp *= gain;
    q = step_q * q + _FBM_SHIFT_3D;
  }
  return sum / max(norm, 1e-6);
}

fn fbm_2d_d(p: vec2f, octaves: i32, lacunarity: f32, gain: f32) -> vec3f {
  let n = clamp(octaves, 0, FBM_MAX_OCTAVES);
  let step_q = lacunarity * _FBM_ROT_2D;
  var q = p;
  var dq = mat2x2f(1.0, 0.0, 0.0, 1.0); // dq/dp, the chain-rule factor of the current octave
  var amp = 1.0;
  var sum = vec3f(0.0);
  var norm = 0.0;
  for (var i = 0; i < n; i++) {
    let d = gnoise_2d_d(q);
    sum += amp * vec3f(d.x, d.yz * dq);
    norm += abs(amp);
    amp *= gain;
    q = step_q * q + _FBM_SHIFT_2D;
    dq = step_q * dq;
  }
  return sum / max(norm, 1e-6);
}

fn fbm_3d_d(p: vec3f, octaves: i32, lacunarity: f32, gain: f32) -> vec4f {
  let n = clamp(octaves, 0, FBM_MAX_OCTAVES);
  let step_q = lacunarity * _FBM_ROT_3D;
  var q = p;
  var dq = mat3x3f(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0);
  var amp = 1.0;
  var sum = vec4f(0.0);
  var norm = 0.0;
  for (var i = 0; i < n; i++) {
    let d = gnoise_3d_d(q);
    sum += amp * vec4f(d.x, d.yzw * dq);
    norm += abs(amp);
    amp *= gain;
    q = step_q * q + _FBM_SHIFT_3D;
    dq = step_q * dq;
  }
  return sum / max(norm, 1e-6);
}

fn ridged_2d(p: vec2f, octaves: i32, lacunarity: f32, gain: f32) -> f32 {
  let n = clamp(octaves, 0, FBM_MAX_OCTAVES);
  let step_q = lacunarity * _FBM_ROT_2D;
  var q = p;
  var amp = 1.0;
  var sum = 0.0;
  var norm = 0.0;
  for (var i = 0; i < n; i++) {
    let r = 1.0 - abs(gnoise_2d(q));
    sum += amp * r * r;
    norm += abs(amp);
    amp *= gain;
    q = step_q * q + _FBM_SHIFT_2D;
  }
  return sum / max(norm, 1e-6);
}

fn fbm_2d_tile(p: vec2f, period: vec2i, octaves: i32, gain: f32) -> f32 {
  let n = clamp(octaves, 0, FBM_MAX_OCTAVES);
  var amp = 1.0;
  var sum = 0.0;
  var norm = 0.0;
  var scale = 1.0;
  var per = max(period, vec2i(1));
  for (var i = 0; i < n; i++) {
    // Any real shift keeps the pattern periodic; it only keeps octaves from all being zero at the
    // same lattice points. Octave 0 is unshifted.
    let shift = f32(min(i, 1)) * hash_1to2(f32(i));
    sum += amp * gnoise_2d_tile(p * scale + shift, per);
    norm += abs(amp);
    amp *= gain;
    scale *= 2.0;
    per *= 2;
  }
  return sum / max(norm, 1e-6);
}

fn domain_warp_2d(p: vec2f, strength: f32, octaves: i32, lacunarity: f32, gain: f32) -> vec2f {
  let q = vec2f(
    fbm_2d(p, octaves, lacunarity, gain),
    fbm_2d(p + _WARP_OFFSET, octaves, lacunarity, gain),
  );
  return p + strength * q;
}

fn domain_warp_2d_anim(
  p: vec2f,
  time: f32,
  strength: f32,
  octaves: i32,
  lacunarity: f32,
  gain: f32,
) -> vec2f {
  let q = vec2f(
    fbm_3d(vec3f(p, time), octaves, lacunarity, gain),
    fbm_3d(vec3f(p + _WARP_OFFSET, time), octaves, lacunarity, gain),
  );
  return p + strength * q;
}

// curl.wgsl -- divergence-free flow fields from the curl of a noise potential.
//
// A velocity field v = curl(psi) neither creates nor destroys "stuff" (div v = 0), so particles
// advected through it swirl and mix without bunching up: ink, foam drift, floating debris, spray.
// The fields are computed from the ANALYTIC gradient of the potential, so they are exactly
// divergence-free up to float rounding (no finite differences, no extra samples).
//
//   curl_2d(p: vec2f, octaves, lacunarity, gain) -> vec2f
//       v = (dpsi/dy, -dpsi/dx) with psi = fbm_2d(p). Static field.
//   curl_2d_anim(p: vec2f, time: f32, octaves, lacunarity, gain) -> vec2f
//       same with psi = fbm_3d(vec3f(p, time)); the field evolves smoothly. Still divergence-free in
//       x/y at every instant.
//   curl_3d(p: vec3f) -> vec3f
//       v = curl of a vector potential made from three decorrelated gnoise_3d. Single octave.
//
// Scale: the field is the gradient of a unit-amplitude potential: |v| averages about 1.2 (curl_3d: 1.5) per
// lattice unit and rarely exceeds 4. It is NOT normalised; multiply by your speed. `p` is in lattice units, so
// raising the frequency of p also raises |v| per world unit: advect with
// pos += curl_2d(pos * freq, ...) * dt / freq to keep the speed independent of frequency.
// Cost: one fbm _d evaluation (about 1.5x the matching fbm; curl_3d: three gnoise_3d_d).
// Technique: R. Bridson, J. Hourihan, M. Nordenstam, "Curl-Noise for Procedural Fluid Flow" (2007).

#include "fbm.wgsl"

const _CURL_3D_OFFSET_1 = vec3f(0.0, 0.0, 0.0);
const _CURL_3D_OFFSET_2 = vec3f(31.416, 47.853, 12.793);
const _CURL_3D_OFFSET_3 = vec3f(-23.157, 19.642, 61.288);

fn curl_2d(p: vec2f, octaves: i32, lacunarity: f32, gain: f32) -> vec2f {
  let d = fbm_2d_d(p, octaves, lacunarity, gain);
  return vec2f(d.z, -d.y);
}

fn curl_2d_anim(p: vec2f, time: f32, octaves: i32, lacunarity: f32, gain: f32) -> vec2f {
  let d = fbm_3d_d(vec3f(p, time), octaves, lacunarity, gain);
  return vec2f(d.z, -d.y);
}

fn curl_3d(p: vec3f) -> vec3f {
  let a = gnoise_3d_d(p + _CURL_3D_OFFSET_1);
  let b = gnoise_3d_d(p + _CURL_3D_OFFSET_2);
  let c = gnoise_3d_d(p + _CURL_3D_OFFSET_3);
  // Components of d are (value, d/dx, d/dy, d/dz); curl = (c_y - b_z, a_z - c_x, b_x - a_y).
  return vec3f(c.z - b.w, a.w - c.y, b.y - a.z);
}

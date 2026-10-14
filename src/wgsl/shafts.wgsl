// shafts.wgsl -- light shafts ("god rays") seen from below the surface: bands of sunlight that fan
// down through the water, shifting slowly as the surface above them moves. An analytic, noise-driven
// construction in screen space: no ray march, no depth buffer, one fan coordinate and two noise
// evaluations per pixel.
//
// CONVENTIONS
//  * `p` is a screen position in aspect-corrected units: x in [0, aspect], y in [0, 1] growing DOWN
//    (so p = vec2f(uv.x * aspect, uv.y)). Depth below the surface is roughly p.y.
//  * `source` is where the rays seem to come from, in the same units. Put it ABOVE the frame
//    (negative y): the farther away, the more parallel the shafts; a source just above the top edge
//    fans them out like a spotlight. The source's x sets the slant: a source to the left of the frame
//    centre leans the shafts to the right as they go down.
//  * `time` in seconds-like units, any rate. Shafts drift sideways slowly and breathe in strength.
//  * Results are dimensionless light factors in [0, 1] (multiply by a light colour and an intensity).
//
// Functions
//   shafts_fan(p, source) -> f32
//       the ray coordinate: tan of the angle between the ray through `p` and straight down, so it is
//       constant along each ray. Clamped near the source, where it would blow up.
//   shafts_beams(coord, time, frequency, contrast) -> f32
//       [0, 1] pattern across the fan: two octaves of animated gradient noise in (coord, time).
//       `frequency` is shafts per unit of the fan coordinate (about 3 to 12); `contrast` in [0, 1]
//       takes it from a soft wash (0) to separate crisp bands with dark water between them (1).
//   shafts_falloff(p, source, reach) -> f32
//       [0, 1]: 1 at the surface, decaying with distance along the ray (exp(-d / reach); `reach` in
//       frame heights, 0.5 to 2), with a slow ripple along each ray so shafts are not uniform rods.
//   shafts_light(p, source, time, frequency, contrast, reach) -> f32
//       beams * falloff: what the demo uses.
//
// Cost: shafts_beams = 2 gnoise_2d (8 hashes) + a smoothstep; falloff = 1 exp, 1 sin, 1 sqrt
// and a divide (the fan coordinate). About 90 ALU in all. Per pixel that is cheaper than one octave of 3D FBM.
// Why not a ray march: a march of N samples against a shadow map costs N texture reads per pixel and
// needs the surface's height; the shafts of a calm background only need to look like a fan of bands.
// Pitfalls: with the source inside the frame the fan coordinate is singular at it; keep the source
// at least 0.1 above the top edge (the fan clamps its denominator at 0.05 as a guard).
// Credits: screen-space volumetric light as in Y. Dobashi et al. (2002) and the many "fake god ray"
// shaders; the noise-in-ray-angle form is the usual cheap approximation of it.

#include "noise.wgsl"
#include "math.wgsl"

fn shafts_fan(p: vec2f, source: vec2f) -> f32 {
  return (p.x - source.x) / max(p.y - source.y, 0.05);
}

fn shafts_beams(coord: f32, time: f32, frequency: f32, contrast: f32) -> f32 {
  let x = coord * frequency;
  // Two octaves: broad swaths and finer streaks, drifting in different directions and speeds.
  let n1 = gnoise_2d(vec2f(x + time * 0.11, time * 0.17));
  let n2 = gnoise_2d(vec2f(x * 2.3 + 7.1 - time * 0.19, time * 0.29 + 3.0));
  let v = saturate(0.5 + 0.55 * (0.7 * n1 + 0.3 * n2));
  // Contrast: the band edges tighten around the middle of the range as it rises.
  let lo = mix(0.0, 0.42, saturate(contrast));
  let hi = mix(1.0, 0.72, saturate(contrast));
  return smoothstep(lo, hi, v);
}

fn shafts_falloff(p: vec2f, source: vec2f, reach: f32) -> f32 {
  let coord = shafts_fan(p, source);
  // Distance travelled below the surface (y = 0) along the ray through p.
  let travelled = max(p.y, 0.0) * sqrt(1.0 + coord * coord);
  let fade = exp(-travelled / max(reach, 0.05));
  // Each ray brightens and dims slowly along its length.
  let ripple = 0.8 + 0.2 * sin(travelled * 5.0 + coord * 3.0);
  return fade * ripple;
}

fn shafts_light(p: vec2f, source: vec2f, time: f32, frequency: f32, contrast: f32, reach: f32) -> f32 {
  let beams = shafts_beams(shafts_fan(p, source), time, frequency, contrast);
  return beams * shafts_falloff(p, source, reach);
}

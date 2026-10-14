// wetness.wgsl -- the usual wet-surface model: where the water is, how wet the ground around it is,
// and what wetness does to a material (darker and more saturated, smoother, mirror-like).
//
// A dry porous surface scatters light inside a rough, bright skin. Water fills the pores and coats the
// grains: the skin becomes translucent (darker, more saturated colour), the micro-roughness is smoothed
// by a film (the surface gets glossier and reflects the sky), and a real puddle on top is a flat mirror
// with Fresnel reflection. These functions split the job so a shader can combine them: a flood mask
// from a height field and a water level, a wetness field with a damp band around the water, and the
// material changes driven by wetness.
//
// Conventions
//  * `height` and `level` (the water level) are in the same unit, metres in practice; so are `reach`,
//    `depth` and `edge`. The ground is flooded where height < level.
//  * Wetness is [0, 1]: 0 bone dry, 1 soaked. Albedo and colours are LINEAR rgb.
//
// Where the water is
//   wet_depth(height, level) -> f32           water depth over the ground, max(level - height, 0)
//   wet_water_mask(depth, edge) -> f32        coverage [0, 1]: 0 at depth <= 0, 1 at depth >= edge, smooth and
//                                             monotonic between. edge is the width of the shoreline in depth
//                                             units; set it to about one pixel of the shore's slope (slope times
//                                             pixel footprint) for a crisp anti-aliased edge, larger for a soft
//                                             one. edge <= 0 gives a hard step
//   wet_damp(height, level, reach, base) -> f32
//                                             wetness of the ground: 1 at and below the waterline, falling to
//                                             `base` (the general dampness, 0 dry .. 1 soaked) over `reach`
//                                             above it. Never below base; monotonic in base and in level
// What wetness does to the material
//   wet_albedo(albedo, wetness) -> vec3f      darker and more saturated: per-channel pow(albedo, 1 + 0.9 w), then
//                                             saturation x (1 + 0.3 w) about the luminance
//                                             (limited so that no channel goes below 0). Output in [0, 1];
//                                             luminance never increases with wetness; w = 0 returns albedo
//   wet_roughness(roughness, wetness) -> f32  roughness * (1 - 0.75 w): a film of water smooths the surface
//   wet_reflectance(cos_theta, wetness, water, roughness) -> f32
//                                             how much of the environment the surface mirrors, [0, 1]. Over a
//                                             real puddle (`water` 1) it is the Fresnel reflectance of water;
//                                             on damp ground a dull sheen: the same Fresnel curve times
//                                             wetness squared times smoothness, capped at 0.8 of the puddle.
//                                             cos_theta = dot(normal, view), view pointing to the eye
//   wet_through_water(bed, depth, sigma) -> vec3f
//                                             the bed seen through `depth` of water with absorption `sigma`
//                                             per metre (absorb_transmittance): bed * exp(-sigma depth)
//
// Cost: a few ALU ops each; wet_albedo has 3 pow, wet_through_water 3 exp.
//
// References: the darker-albedo, lower-roughness wet-surface model is the one S. Lagarde describes in
// "Water drop 3b - Physically based wet surfaces" (2013); the puddle as a flat Fresnel mirror follows
// the same article.

#include "color.wgsl"
#include "fresnel.wgsl"
#include "absorption.wgsl"

fn wet_depth(height: f32, level: f32) -> f32 {
  return max(level - height, 0.0);
}

fn wet_water_mask(depth: f32, edge: f32) -> f32 {
  if (edge <= 0.0) {
    return select(0.0, 1.0, depth > 0.0);
  }
  return smoothstep(0.0, edge, depth);
}

fn wet_damp(height: f32, level: f32, reach: f32, base: f32) -> f32 {
  let above = max(height - level, 0.0);
  let band = select(0.0, 1.0 - smoothstep(0.0, reach, above), reach > 0.0);
  let at_water = select(0.0, 1.0, height <= level);
  return max(max(saturate(base), band), at_water);
}

fn wet_albedo(albedo: vec3f, wetness: f32) -> vec3f {
  let w = saturate(wetness);
  let dark = pow(clamp(albedo, vec3f(0.0), vec3f(1.0)), vec3f(1.0 + 0.9 * w));
  let l = luminance(dark);
  // More saturation about the luminance, but never so much that a channel goes negative: clamping
  // would add light and make a wetter surface brighter.
  let lowest = min(dark.x, min(dark.y, dark.z));
  let boost = min(1.0 + 0.3 * w, l / max(l - lowest, 1e-4));
  let rich = vec3f(l) + (dark - vec3f(l)) * max(boost, 1.0);
  return clamp(rich, vec3f(0.0), vec3f(1.0));
}

fn wet_roughness(roughness: f32, wetness: f32) -> f32 {
  return clamp(roughness, 0.0, 1.0) * (1.0 - 0.75 * saturate(wetness));
}

fn wet_reflectance(cos_theta: f32, wetness: f32, water: f32, roughness: f32) -> f32 {
  let f = fresnel_water(cos_theta);
  let w = saturate(wetness);
  let sheen = f * w * w * (1.0 - clamp(roughness, 0.0, 1.0)) * 0.8;
  return mix(sheen, f, saturate(water));
}

fn wet_through_water(bed: vec3f, depth: f32, sigma: vec3f) -> vec3f {
  return bed * absorb_transmittance(sigma, depth);
}

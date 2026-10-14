// absorption.wgsl -- Beer-Lambert light absorption and in-scattering for light that travels through water.
//
// Water is not a flat tint: red is absorbed within a few metres, green within tens, blue survives
// longest, so colour shifts from the bed's own colour toward teal, then deep blue, with depth. The model
// here is the standard cheap one: per-channel exponential transmittance exp(-sigma * path) plus a single
// "fog" term that scatters light toward a scatter colour as the path gets longer. PATH LENGTH (or depth)
// is always a parameter: feed it an analytic depth today, a depth-buffer difference in a 3D scene later.
//
// UNITS. Path lengths and depths are in metres, coefficients in 1/metre, whatever scale your scene uses
// as long as you stay consistent. Colours are LINEAR rgb.
//
// Designer-friendly coefficients
//   absorb_coefficients_from_color(color: vec3f, depth: f32) -> vec3f
//       sigma such that white light has turned into `color` after `depth`: sigma = -ln(color) / depth.
//       color is clamped to [1e-4, 1]. "Clear tropical water turns (0.06, 0.5, 0.72) at 6 m".
//   absorb_turbidity_from_visibility(visibility: f32) -> f32
//       scattering coefficient from how far you can see (Secchi depth, metres): 1.7 / visibility.
//
// Transmittance and slant paths
//   absorb_transmittance(sigma: vec3f, path: f32) -> vec3f     exp(-sigma * path); 1 at path 0, non-increasing,
//                                                              always in [0, 1] (negative path counts as 0)
//   absorb_cos_refracted(cos_view: f32, ior: f32) -> f32       cosine of the ray's angle inside the water,
//                                                              given the cosine in the air (Snell's law)
//   absorb_path_length(depth, cos_view, ior) -> f32            vertical `depth` -> length along the refracted
//                                                              view ray: depth / cos_refracted (cos clamped >= 0.05).
//                                                              cos_view = 1 looking straight down; ior = 1.333
//   absorb_path_length_two_way(depth, cos_view, cos_light, ior) -> f32
//                                                              sun going down plus eye coming up: the light that
//                                                              lit the bed took a slant path too
//
// Compositing
//   struct AbsorbWater { sigma: vec3f, scatter: vec3f, turbidity: f32 }
//       sigma: absorption per metre; scatter: colour of the light scattered back (sky / sun tinted, roughly
//       the colour of very deep water divided by its brightness); turbidity: scattering coefficient per metre
//   absorb_composite(bed: vec3f, sigma: vec3f, turbidity: f32, scatter: vec3f, path: f32) -> vec3f
//       what the eye sees: bed * T + scatter * (turbidity / extinction) * (1 - T) per channel, with
//       extinction = sigma + turbidity and T = exp(-extinction * path). Path 0 returns the bed exactly;
//       turbidity 0 returns bed * absorb_transmittance; path -> infinity returns absorb_deep_color. Never
//       brighter than max(bed, scatter) per channel.
//   absorb_deep_color(sigma, turbidity, scatter) -> vec3f      the colour of water too deep to see the bed
//   absorb_water(kind: i32) -> AbsorbWater                     named water types, derived from "colour at depth":
//       0 clear tropical (6 m)   1 coastal (3 m)   2 murky green (1.5 m)   3 tea lake (1.5 m)
//       4 glacial milk (4 m)     5 swimming pool (3 m)   6 open ocean (20 m).   kind is clamped to 0..6
//   ABSORB_TYPE_COUNT = 7
//
// Cost: transmittance is one exp per channel (3 exp); composite is 3 exp + ~15 ALU; absorb_water is 3 log
// and a select chain, so call it once outside any loop (and hoist it if `kind` is uniform). This is the
// single-scatter, uniformly lit approximation: no sun-shafts, no depth-dependent light colour for the
// in-scattered part. Good for tints and fog, not for underwater volumetrics.

const ABSORB_TYPE_COUNT: i32 = 7;
const ABSORB_MIN_COS: f32 = 0.05;

struct AbsorbWater {
  sigma: vec3f,
  scatter: vec3f,
  turbidity: f32,
}

fn absorb_coefficients_from_color(color: vec3f, depth: f32) -> vec3f {
  let c = clamp(color, vec3f(1e-4), vec3f(1.0));
  return -log(c) / max(depth, 1e-4);
}

fn absorb_turbidity_from_visibility(visibility: f32) -> f32 {
  return 1.7 / max(visibility, 1e-3);
}

fn absorb_transmittance(sigma: vec3f, path: f32) -> vec3f {
  return exp(-max(sigma, vec3f(0.0)) * max(path, 0.0));
}

fn absorb_cos_refracted(cos_view: f32, ior: f32) -> f32 {
  let c = clamp(cos_view, 0.0, 1.0);
  let n = max(ior, 1.0);
  return sqrt(max(1.0 - (1.0 - c * c) / (n * n), 0.0));
}

fn absorb_path_length(depth: f32, cos_view: f32, ior: f32) -> f32 {
  return max(depth, 0.0) / max(absorb_cos_refracted(cos_view, ior), ABSORB_MIN_COS);
}

fn absorb_path_length_two_way(depth: f32, cos_view: f32, cos_light: f32, ior: f32) -> f32 {
  let down = 1.0 / max(absorb_cos_refracted(cos_light, ior), ABSORB_MIN_COS);
  let up = 1.0 / max(absorb_cos_refracted(cos_view, ior), ABSORB_MIN_COS);
  return max(depth, 0.0) * (down + up);
}

fn absorb_composite(bed: vec3f, sigma: vec3f, turbidity: f32, scatter: vec3f, path: f32) -> vec3f {
  let extinction = max(sigma, vec3f(0.0)) + vec3f(max(turbidity, 0.0));
  let t = exp(-extinction * max(path, 0.0));
  let albedo = vec3f(max(turbidity, 0.0)) / max(extinction, vec3f(1e-6));
  return bed * t + scatter * albedo * (vec3f(1.0) - t);
}

fn absorb_deep_color(sigma: vec3f, turbidity: f32, scatter: vec3f) -> vec3f {
  let extinction = max(sigma, vec3f(0.0)) + vec3f(max(turbidity, 0.0));
  return scatter * vec3f(max(turbidity, 0.0)) / max(extinction, vec3f(1e-6));
}

fn absorb_water(kind: i32) -> AbsorbWater {
  var color = vec3f(0.06, 0.5, 0.72); // colour white light turns after `depth` metres
  var depth = 6.0;
  var scatter = vec3f(0.3, 0.5, 0.95);
  var turbidity = 0.03;
  let k = clamp(kind, 0, ABSORB_TYPE_COUNT - 1);
  if (k == 1) { // coastal: green-turquoise, a little silt
    color = vec3f(0.12, 0.55, 0.5);
    depth = 3.0;
    scatter = vec3f(0.35, 0.65, 0.55);
    turbidity = 0.15;
  } else if (k == 2) { // murky green: algae and silt
    color = vec3f(0.1, 0.5, 0.12);
    depth = 1.5;
    scatter = vec3f(0.3, 0.55, 0.2);
    turbidity = 0.6;
  } else if (k == 3) { // tea lake: tannins eat blue first
    color = vec3f(0.35, 0.15, 0.04);
    depth = 1.5;
    scatter = vec3f(0.35, 0.18, 0.06);
    turbidity = 0.25;
  } else if (k == 4) { // glacial milk: rock flour, strong scattering
    color = vec3f(0.25, 0.65, 0.8);
    depth = 4.0;
    scatter = vec3f(0.55, 0.85, 0.9);
    turbidity = 0.8;
  } else if (k == 5) { // swimming pool
    color = vec3f(0.35, 0.85, 0.95);
    depth = 3.0;
    scatter = vec3f(0.5, 0.9, 1.0);
    turbidity = 0.02;
  } else if (k == 6) { // open ocean
    color = vec3f(0.02, 0.45, 0.75);
    depth = 20.0;
    scatter = vec3f(0.25, 0.42, 0.8);
    turbidity = 0.01;
  }
  return AbsorbWater(absorb_coefficients_from_color(color, depth), scatter, turbidity);
}

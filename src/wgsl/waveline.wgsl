// waveline.wgsl -- a wavy line as a signed-distance field: section dividers, footers, wave layers.
//
// A "wave line" is the graph y = line_y(x) of a few summed sines plus a little noise, so it never
// repeats exactly or reads as a pure sine. Everything is measured in ONE length unit chosen by the
// caller (CSS pixels is the natural one), and y grows DOWNWARD like uv.y. The filled side is below the
// line (larger y). The distance is signed like sdf.wgsl: negative inside the fill, positive above it.
//
//   waveline_curve(x, phase, wavelength, amplitude, roughness, seed) -> vec2f
//       (y, dy/dx) of the curve relative to its own mean line (add the baseline yourself). Three sines
//       at 1x, 1.93x and 0.57x the base frequency (weights 0.58, 0.27, 0.15) plus gradient noise with
//       weight `roughness` in [0, 1]. `phase` in radians moves the whole pattern toward -x as it grows
//       (the pattern is f(x + phase * wavelength / TAU), each harmonic at its own speed, so the line
//       evolves instead of sliding rigidly). `seed` decorrelates layers: use a different value per
//       layer (any float; a few units apart). |y| <= 1.6 * amplitude in the worst case, usually < amplitude.
//   waveline_distance(y, line_y, slope) -> f32
//       signed distance from the point (x, y) to the curve, from the vertical gap and the slope:
//       (line_y - y) / sqrt(1 + slope^2). Negative below the line. It is a first-order distance (the
//       true distance is a little larger away from the line), exact to first order at the line, with
//       unit gradient there: moving one pixel across the line changes it by one pixel. Pass
//       line_y = baseline + curve.x and slope = curve.y.
//   waveline_coverage(d, px) -> f32
//       anti-aliased fill from a distance: 1 below the line, 0 above, a ramp `px` wide centred on the
//       line. Because the distance has unit gradient, `px` is simply the size of one pixel in the
//       distance's units (1 / dpr for CSS pixels): no derivative instructions are needed, so the caller
//       may branch freely, and edges are exactly one pixel wide at any DPR without MSAA.
//   waveline_band(d, centre, half_width, px) -> f32
//       anti-aliased coverage of the strip |d - centre| <= half_width: a thin highlight or outline
//       that follows the curve. centre = -half_width puts the strip just inside the fill.
//   waveline_foam(depth, p, time, width, amount, edge) -> f32
//       foam along a line, in [0, 1]: round bubbles that merge into a solid band right at depth 0 and
//       thin out to nothing by `depth` = `width`. `depth` = distance from the foam's start edge (same
//       units as `width`; use -d plus a margin to let it reach past the line), `p` the texture
//       coordinate in feature units (one bubble cell is 1.0: pixels / 9 reads as fine foam), `time` in
//       radians (the bubbles boil), `amount` in [0, 1] the coverage, `edge` the softness of a bubble's
//       rim in feature units (one pixel = 1 / cell size in pixels). Costs one Voronoi and one gradient
//       noise; call it only for pixels with depth < width.
//
// Cost: curve = 3 sin + 3 cos + 1 gnoise_2d_d (about 40 ALU); distance and coverage a few ops.
// Technique: a sum of sines (the simplest Gerstner-less wave), a height-field distance estimate
// d = f / |grad f| (I. Quilez, "distance to implicit functions"), analytic-gradient anti-aliasing.

#include "math.wgsl"
#include "noise.wgsl"
#include "sdf.wgsl"
#include "foam.wgsl"
#include "voronoi.wgsl"

fn waveline_curve(
  x: f32, phase: f32, wavelength: f32, amplitude: f32, roughness: f32, seed: f32,
) -> vec2f {
  let k = TAU / max(wavelength, 1e-3);
  let a1 = k * x + phase;
  let a2 = k * 1.93 * x + phase * 1.31 + 1.7 + seed * 2.3;
  let a3 = k * 0.57 * x - phase * 0.73 + 4.1 + seed * 5.1;
  let sines = 0.58 * sin(a1) + 0.27 * sin(a2) + 0.15 * sin(a3);
  let dsines = k * (0.58 * cos(a1) + 0.27 * 1.93 * cos(a2) + 0.15 * 0.57 * cos(a3));

  // Noise along x, drifting slowly in its second axis so the shape keeps changing.
  let nk = k * 0.9;
  let n = gnoise_2d_d(vec2f(x * nk + seed * 13.0, phase * 0.12 + seed * 3.0));
  let rough = clamp(roughness, 0.0, 1.0) * 0.6;
  return amplitude * vec2f(sines + rough * n.x, dsines + rough * n.y * nk);
}

fn waveline_distance(y: f32, line_y: f32, slope: f32) -> f32 {
  return (line_y - y) * inverseSqrt(1.0 + slope * slope);
}

fn waveline_coverage(d: f32, px: f32) -> f32 {
  return sd_coverage(d, px);
}

fn waveline_band(d: f32, centre: f32, half_width: f32, px: f32) -> f32 {
  return sd_coverage(abs(d - centre) - half_width, px);
}

fn waveline_foam(depth: f32, p: vec2f, time: f32, width: f32, amount: f32, edge: f32) -> f32 {
  // Patchiness along the crest, so the foam comes in clumps and fingers.
  let patch_noise = 0.5 + 0.5 * gnoise_2d(vec2f(p.x * 0.3, time * 0.15));
  let reach = foam_mask(depth, width, patch_noise, 0.45) * clamp(amount, 0.0, 1.0);
  let v = voronoi_2d_anim(p, 0.9, time);
  // Bubbles shrink with distance from the crest and the small ones go first.
  let radius = 0.8 * reach * (0.35 + 0.65 * v.id);
  let soft = max(edge, 1e-3);
  // A bubble of zero radius is nothing, not the half-covered dot a bare ramp would leave.
  return (1.0 - smoothstep(radius - soft, radius + soft, v.f1)) * smoothstep(0.0, 2.0 * soft, radius);
}

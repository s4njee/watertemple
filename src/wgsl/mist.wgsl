// mist.wgsl -- drifting mist and steam: billowing density layers, a density profile that thins toward
// one edge, and an opacity curve with a hard cap, for overlays that must never hide what is under them.
//
// A mist layer is a low-frequency density field. Its look comes from two moving things: the base
// pattern slides along the drift direction, and a warp field bends it while sliding at a DIFFERENT
// speed, so the wisps billow and fold instead of translating rigidly. Render the layers at a fraction
// of the output size (they are soft by nature) and apply `mist_opacity` after upsampling.
//
//   mist_layer(p, base_shift, warp_shift, time, warp, coverage) -> f32
//       Density of one layer in [0, 1]. `p` in feature units (about one wisp per unit; stretch an axis
//       before calling to elongate the wisps). `base_shift` is added to p for the base pattern and
//       `warp_shift` for the warp field: pass drift * phase and drift * phase * 0.6, plus any scroll
//       offset in both, so a layer moves as a whole when the page scrolls. `time` evolves the shape
//       in place. `warp` in [0, 1.5] bends the pattern. `coverage` in [0, 1] is how much of the area is
//       mist: 0 is nearly clear, 1 is mostly filled. Five 3D gradient-noise evaluations.
//   mist_profile(s, height, falloff) -> f32
//       Vertical density profile in [0, 1]. `s` is the distance from the source edge as a fraction of
//       the canvas (0 at the edge, 1 at the far side). 1 up to height - falloff, 0 past
//       height + falloff, smooth between: `height` is where the mist ends, `falloff` how gradually.
//   mist_opacity(density, max_opacity) -> f32
//       Opacity of the overlay for a density sum: monotonic, smooth, 0 at density 0, and at most
//       `max_opacity` however large the density (density is clamped to [0, 1]; density 1 gives exactly the cap,
//       and the curve flattens toward it so the mist thins out smoothly). Use it as the
//       straight alpha of a premultiplied overlay: rgb = tint * alpha.
//
// MIST_ALPHA_LIMIT (0.8) is the highest cap the demo allows: a veil that opaque hides what it is over.
// Cost: mist_layer about 5 gnoise_3d; profile and opacity a few ALU and one exp.
// Technique: animated fbm with a counter-moving domain warp ("flow billowing", after I. Quilez's
// domain warping), the exponential opacity curve of Beer-Lambert extinction with a ceiling.

#include "noise.wgsl"
#include "fbm.wgsl"

const MIST_ALPHA_LIMIT: f32 = 0.8;

fn mist_layer(
  p: vec2f, base_shift: vec2f, warp_shift: vec2f, time: f32, warp: f32, coverage: f32,
) -> f32 {
  let w = vec2f(
    fbm_3d(vec3f(p * 0.6 + warp_shift, time * 0.5), 2, 2.0, 0.5),
    fbm_3d(vec3f(p * 0.6 + warp_shift + vec2f(31.4, 7.9), time * 0.5), 2, 2.0, 0.5),
  );
  let q = p + base_shift + warp * 0.9 * w;
  let n = fbm_3d(vec3f(q, time * 0.35), 3, 2.0, 0.5);
  // Wisps: a soft threshold of the noise, moved by `coverage`.
  let level = mix(0.55, -0.2, clamp(coverage, 0.0, 1.0));
  return smoothstep(level, level + 0.55, n * 1.5);
}

fn mist_profile(s: f32, height: f32, falloff: f32) -> f32 {
  let f = max(falloff, 1e-3);
  return 1.0 - smoothstep(height - f, height + f, s);
}

fn mist_opacity(density: f32, max_opacity: f32) -> f32 {
  let d = clamp(density, 0.0, 1.0);
  let cap = clamp(max_opacity, 0.0, MIST_ALPHA_LIMIT);
  return cap * (1.0 - exp(-3.0 * d)) / (1.0 - exp(-3.0));
}

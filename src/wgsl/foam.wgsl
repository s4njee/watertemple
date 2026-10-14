// foam.wgsl -- depth-driven foam, soft intersection fade and wet-sand helpers for shorelines.
//
// Foam appears where water is shallow or touches something: along a beach, around rocks, where a surface
// cuts through geometry. All of it keys off ONE number, `depth`: the vertical (or distance-to-geometry)
// thickness of water at this pixel, in whatever unit you like, as long as `width` / `fade` use the same.
// depth <= 0 means "at or inside the geometry". Feed analytic depth today, depth-buffer differences later.
//
// The foam look is built in three steps so each can be tuned or replaced on its own:
//   1. coverage  = foam_mask(depth, width, noise, breakup)   where foam is allowed, [0, 1]
//   2. thickness = foam_texture(p, time)                      what foam looks like, lace and bubbles, [0, 1]
//   3. foam      = foam_dissolve(thickness, coverage, soft)   thickness thresholded by coverage, [0, 1]
// As coverage falls the thin lace goes first and the thick clumps last, so foam dissolves into holes
// instead of fading out as a grey wash.
//
// Mask and dissolve
//   foam_mask(depth, width, noise01, breakup) -> f32
//       1 at depth <= 0, 0 at depth >= width, monotonic between, always in [0, 1] for any noise. `noise01` in
//       [0, 1] (e.g. 0.5 + 0.5 * gnoise) shortens the reach of the foam where it is low; `breakup` in [0, 1]
//       says how much (0 = a clean smooth band, 1 = ragged, fingers of foam reaching up to `width`).
//   foam_dissolve(thickness, coverage, softness) -> f32
//       smoothstep threshold: coverage 1 gives 1 everywhere, coverage 0 gives 0 everywhere, in between the
//       foam survives where thickness is largest. softness in (0, 1]: edge width of the holes.
//   foam_texture(p: vec2f, time: f32) -> f32
//       procedural foam thickness in [0, 1]: two scales of animated Voronoi (lace on the cell walls, bubbles
//       in the cells) modulated by a slow patchiness. p is in feature units (cells about 1.0 across; use
//       world * 12 for a fine bubbly look, world * 4 for big clumps). `time` is in radians: the cell seeds
//       orbit, so the lace boils instead of sliding. Cost: 2 voronoi + 1 gnoise.
//   foam_texture_lite(p, time) -> f32      one scale, no patchiness: 1 voronoi, about 40% of the cost
// Animation
//   foam_swash(phase: f32, rise: f32) -> f32
//       wave run-up cycle in [0, 1], periodic in `phase` (cycles). Zero at integer phase, peak 1 at
//       fract(phase) = rise (rush up fast, ease out), then drains back slowly. rise in (0, 1), typically 0.25.
//       Add a noise offset to the phase per position so wave lines lap in and out at different times.
// Soft intersection
//   foam_edge_fade(depth, fade) -> f32
//       0 at depth <= 0 (the intersection), 1 at depth >= fade, smooth (C1) between: multiply a water
//       surface's alpha by this to hide the hard line where it cuts through geometry
//   foam_rim(depth, width) -> f32          bright line hugging the contact: 1 at depth 0, 0 at width, concentrated
//                                          near 0 (squared), for the thin crisp edge on top of the foam body
// Wet ground
//   foam_wetness(height_above, reach) -> f32     1 at the waterline (height_above <= 0), 0 at `reach` above it
//   foam_darken_wet(color: vec3f, wetness: f32, strength: f32) -> vec3f
//       wet surfaces are darker and more saturated: pow(color, 1 + strength * wetness), a little extra
//       darkening. wetness 0 returns the colour unchanged. strength ~0.6..1.2 for sand
//
// Cost: mask / dissolve / swash / fade / rim are a handful of ALU each; the texture dominates (see above).
// Credits: Voronoi lace after the usual shoreline-foam recipes (S. Worley 1996 for the cells).

#include "noise.wgsl"
#include "voronoi.wgsl"

fn foam_mask(depth: f32, width: f32, noise01: f32, breakup: f32) -> f32 {
  let reach = max(width, 1e-5) * mix(1.0, clamp(noise01, 0.0, 1.0), clamp(breakup, 0.0, 1.0));
  return 1.0 - smoothstep(0.0, max(reach, 1e-5), depth);
}

fn foam_dissolve(thickness: f32, coverage: f32, softness: f32) -> f32 {
  let s = clamp(softness, 0.01, 1.0);
  let lo = 1.0 - clamp(coverage, 0.0, 1.0) * (1.0 + s);
  return smoothstep(lo, lo + s, clamp(thickness, 0.0, 1.0));
}

fn foam_texture_lite(p: vec2f, time: f32) -> f32 {
  let v = voronoi_2d_anim(p, 1.0, time);
  let wall = 1.0 - smoothstep(0.0, 0.3, v.f2 - v.f1);
  let bubble = 1.0 - smoothstep(0.1, 0.65, v.f1);
  return clamp(0.65 * wall + 0.45 * bubble * (0.4 + 0.6 * v.id), 0.0, 1.0);
}

fn foam_texture(p: vec2f, time: f32) -> f32 {
  let a = foam_texture_lite(p, time);
  let b = foam_texture_lite(p * 2.3 + vec2f(13.7, 5.1), time * 1.3 + 2.0);
  let clump = 0.5 + 0.5 * gnoise_2d(p * 0.35 + vec2f(0.0, time * 0.05));
  return clamp((0.6 * a + 0.5 * b) * (0.55 + 0.7 * clump), 0.0, 1.0);
}

fn foam_swash(phase: f32, rise: f32) -> f32 {
  let r = clamp(rise, 0.02, 0.98);
  let x = fract(phase);
  if (x < r) {
    return sin(x / r * 1.5707963);
  }
  let u = (x - r) / (1.0 - r);
  return 0.5 + 0.5 * cos(u * 3.1415927);
}

fn foam_edge_fade(depth: f32, fade: f32) -> f32 {
  let t = clamp(depth / max(fade, 1e-5), 0.0, 1.0);
  return t * t * (3.0 - 2.0 * t);
}

fn foam_rim(depth: f32, width: f32) -> f32 {
  let t = 1.0 - clamp(depth / max(width, 1e-5), 0.0, 1.0);
  return t * t;
}

fn foam_wetness(height_above: f32, reach: f32) -> f32 {
  return 1.0 - smoothstep(0.0, max(reach, 1e-5), height_above);
}

fn foam_darken_wet(color: vec3f, wetness: f32, strength: f32) -> vec3f {
  let w = clamp(wetness, 0.0, 1.0);
  let c = max(color, vec3f(0.0));
  return pow(c, vec3f(1.0 + max(strength, 0.0) * w)) * (1.0 - 0.12 * w);
}

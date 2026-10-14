// Stylised toon water, one pass: a top-down island with rocks in flat banded water, crisp animated
// cell-noise foam, a foam ring and ink outline along every shore, and hard-edged glints.
//
// Everything is flat colour with an anti-aliased edge, and every edge width is known (a pixel in world
// units is constant for this flat scene), so lines keep a steady width and nothing shimmers. The
// reusable stylisation lives in src/wgsl/toon.wgsl; this file is the scene.
#include "fullscreen.wgsl"
#include "toon.wgsl"
#include "noise.wgsl"
#include "voronoi.wgsl"
#include "color.wgsl"
#include "toon_params.wgsl"

@group(0) @binding(0) var<uniform> P: ToonParams;
@group(0) @binding(1) var sdf_tex: texture_2d<f32>;
@group(0) @binding(2) var sdf_samp: sampler;

// Palette slots in P.palette (linear rgb in xyz).
const C_DEEP: i32 = 0;
const C_MID_DEEP: i32 = 1;
const C_MID_SHALLOW: i32 = 2;
const C_SHALLOW: i32 = 3;
const C_FOAM: i32 = 4;
const C_INK: i32 = 5;
const C_SAND: i32 = 6;
const C_GRASS: i32 = 7;
const C_BUSH: i32 = 8;
const C_ROCK: i32 = 9;
const C_ROCK_DARK: i32 = 10;
const C_SPARKLE: i32 = 11;

fn colour_at(i: i32) -> vec3f {
  return P.palette[i].xyz;
}

// Island and rock distances come from the baked field (sdf-bake.wgsl): r = island, g = rocks.
fn shapes_at(uv: vec2f) -> vec2f {
  return textureSampleLevel(sdf_tex, sdf_samp, uv, 0.0).rg;
}

@fragment fn fs(in: FullscreenVarying) -> @location(0) vec4f {
  // World position: one unit is the canvas height, zoomed out on narrow canvases so the island fits.
  let p = (in.uv - vec2f(0.5)) * vec2f(P.aspect, 1.0) * P.zoom;
  let px = P.zoom / P.resolution.y;           // one pixel, in world units
  let height_scale = P.resolution.y / 600.0;  // line widths are given for a 600 px tall canvas

  // ---- Land and depth ---------------------------------------------------------------------------
  let shapes = shapes_at(in.uv);
  let d_island = shapes.x;
  let d_rock = shapes.y;
  let d = min(d_island, d_rock);

  // Depth bands. A slow wobble in the distance keeps the contours from being perfect offset curves.
  let wobble = 0.011 * (sin(p.x * 17.0 + P.time * 0.5) * sin(p.y * 13.0 - P.time * 0.35) + 0.5 * sin(p.x * 31.0 - p.y * 23.0 + P.time * 0.7));
  let t = 1.0 - clamp((d + wobble) / P.depth, 0.0, 1.0);   // 1 at the shore, 0 in deep water
  let level = toon_band(t, P.bands, px / P.depth);
  var water = toon_ramp(level, colour_at(C_DEEP), colour_at(C_MID_DEEP), colour_at(C_MID_SHALLOW), colour_at(C_SHALLOW));

  // ---- Foam network (uniform control flow: these use derivatives) -------------------------------
  // The water sways a little so the whole network breathes instead of only the cells morphing.
  let sway = 0.010 * vec2f(sin(p.y * 9.0 + P.time * 0.9), cos(p.x * 8.0 + P.time * 0.7));
  let q = (p + sway) * P.foam_scale;
  let f_a = toon_foam_cells(q, P.foam_time, 0.85);
  let g_a = length(vec2f(dpdx(f_a), dpdy(f_a)));
  let line_a = toon_cell_edge(f_a, g_a, P.foam_width * height_scale);

  // The faint second layer (bigger cells, slower, thicker) is a second Voronoi per pixel, about a third
  // of this pass. `deep_foam` is a uniform, so skipping it keeps the derivatives in uniform control flow.
  let q2 = (p + sway) * P.foam_scale * 0.45 + vec2f(5.3, 1.7);
  var line_b = 0.0;
  if (P.deep_foam > 0.5) {
    let f_b = toon_foam_cells(q2, P.foam_time * 0.6, 0.85);
    let g_b = length(vec2f(dpdx(f_b), dpdy(f_b)));
    line_b = toon_cell_edge(f_b, g_b, P.foam_width * height_scale * 1.7);
  }

  // Where the foam is allowed: patches that drift, more of them near the shore. The patch edges are
  // hard too, so a line is cut off cleanly instead of fading.
  // One noise field serves both layers: the shallow network lives where it is high, the faint deep one
  // where it is low, so the two never fight over the same patch.
  let patch_n = gnoise_2d(q * 0.16 + vec2f(P.time * 0.06, 0.0));
  let patch_a = patch_n + (t - 0.4) * 1.3;
  let patch_b = -patch_n + (0.35 - t) * 1.1;
  let show_a = toon_step(0.05, patch_a, fwidth(patch_a));
  let show_b = toon_step(0.1, patch_b, fwidth(patch_b));

  // ---- Water surface -----------------------------------------------------------------------------
  let foam_c = colour_at(C_FOAM);
  let in_water = toon_step(0.0, d, px);

  // Faint deep network first, then the stronger shallow one on top of it.
  water = mix(water, mix(water, foam_c, 0.5), line_b * show_b);
  water = mix(water, foam_c, line_a * show_a);

  // Swooshes and sparkles: hard-edged highlights, mostly over deeper water.
  let swoosh_n = gnoise_2d(vec2f(p.x * 4.5 + P.time * 0.22, p.y * 20.0 + sin(p.x * 3.0 + P.time * 0.4) * 1.5));
  let swoosh = toon_step(0.62 - 0.1 * P.highlight, swoosh_n, fwidth(swoosh_n)) * (1.0 - toon_step(0.4, t, px / P.depth));
  water = mix(water, foam_c, swoosh * min(1.0, P.highlight * 1.6) * 0.8);
  let sparkle = toon_sparkle(p * 13.0, P.time, 0.5 * P.highlight, 0.33, px * 13.0);
  water = mix(water, colour_at(C_SPARKLE), sparkle * (1.0 - toon_step(0.7, t, px / P.depth)));

  // Shore: a breathing foam ring, a detached swash line behind it, and the ink outline on the coast.
  let phase = P.time * 1.1 + 2.2 * sin(p.x * 6.0 + p.y * 4.0 + P.time * 0.2) + 1.6 * sin(p.y * 9.0 - p.x * 5.0);
  let breath = 0.5 + 0.5 * sin(phase);
  let ring_w = P.shore * (0.012 + 0.03 * breath);
  let ring = toon_ring(d, 0.0, ring_w, px);
  let swash_at = ring_w + 0.014 + 0.035 * P.shore * (0.5 + 0.5 * sin(phase * 0.8 + 2.0));
  let swash = toon_ring(d, swash_at, swash_at + 0.0065 * (0.4 + P.shore), px) * min(1.0, P.shore * 8.0);
  water = mix(water, foam_c, max(ring, swash));
  let ink_w = 2.4 * px * height_scale;
  let outline = toon_ring(d, 0.0, ink_w, px) * P.outline;
  water = mix(water, colour_at(C_INK), outline);

  // ---- Land --------------------------------------------------------------------------------------
  var land = vec3f(0.0);
  if (in_water < 1.0) {
    let on_rock = d_rock < d_island;
    // Island: a sand rim, then grass dotted with round bushes (a static Voronoi, one disc per cell).
    let beach = toon_step(-0.055 - 0.008 * sin(p.x * 31.0 + p.y * 17.0), d_island, px);
    var grass = colour_at(C_GRASS);
    let v = voronoi_2d(p * 15.0, 0.9);
    let radius = 0.2 + 0.2 * v.id;
    let bush = (1.0 - toon_step(radius, v.f1, px * 15.0)) * (1.0 - step(0.62, v.id));
    let bush_hi = (1.0 - toon_step(radius * 0.6, length(p * 15.0 - v.site - vec2f(-0.08, -0.08)), px * 15.0)) * bush;
    grass = mix(grass, colour_at(C_BUSH), bush);
    grass = mix(grass, colour_at(C_GRASS) * 1.18, bush_hi);
    let island_c = mix(grass, colour_at(C_SAND), beach);

    // Rocks: lit cap on the upper left, dark body, flat.
    // A point is lit if the rock still covers it after shifting toward the shadow side.
    let shifted_uv = in.uv + vec2f(0.013, 0.013) / (vec2f(P.aspect, 1.0) * P.zoom);
    let cap = 1.0 - toon_step(0.0, shapes_at(shifted_uv).y, px);
    let rock_c = mix(colour_at(C_ROCK_DARK), colour_at(C_ROCK), cap);
    land = select(island_c, rock_c, on_rock);
    // Ink outline inside the coast, so the shape reads against any water colour.
    land = mix(land, colour_at(C_INK), toon_ring(d, -ink_w, 0.0, px) * P.outline);
  }

  let colour = mix(land, water, in_water);
  return vec4f(linear_to_srgb(clamp(colour, vec3f(0.0), vec3f(1.0))), 1.0);
}

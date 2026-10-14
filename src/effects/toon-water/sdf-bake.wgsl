// The static geometry of the toon scene, baked once per canvas size: signed distance (world units,
// negative inside) to the island in r and to the rocks in g. The main pass reads it with one bilinear
// fetch instead of evaluating two noise octaves and eight circles per pixel. A distance field is smooth
// enough that half the canvas resolution still gives sub-pixel-accurate, perfectly straight-edged coasts.
#include "fullscreen.wgsl"
#include "noise.wgsl"
#include "sdf.wgsl"
#include "toon_params.wgsl"

@group(0) @binding(0) var<uniform> P: ToonParams;

// Rocks: x, y, radius. World units, the screen is 1.0 tall at zoom 1, y down.
const ROCK_COUNT: i32 = 8;
const ROCKS = array<vec3f, 8>(
  vec3f(0.44, 0.10, 0.050),
  vec3f(0.53, 0.02, 0.030),
  vec3f(0.40, -0.02, 0.022),
  vec3f(-0.43, -0.12, 0.055),
  vec3f(-0.52, 0.04, 0.030),
  vec3f(0.10, -0.40, 0.045),
  vec3f(-0.22, 0.38, 0.050),
  vec3f(0.70, -0.26, 0.032),
);

fn island_sd(p: vec2f) -> f32 {
  let c = p - vec2f(0.0, 0.02);
  let shape = 0.07 * gnoise_2d(c * 3.0 + vec2f(1.7, 0.4)) + 0.02 * gnoise_2d(c * 9.0 + vec2f(4.0, 2.0));
  return length(c * vec2f(0.8, 1.05)) - 0.30 + shape;
}

fn rocks_sd(p: vec2f) -> f32 {
  var rocks = ROCKS;
  var d = 1e3;
  for (var i = 0; i < ROCK_COUNT; i++) {
    let r = rocks[i];
    d = min(d, sd_circle(p - r.xy, r.z));
  }
  return d;
}

@fragment fn fs(in: FullscreenVarying) -> @location(0) vec4f {
  let p = (in.uv - vec2f(0.5)) * vec2f(P.aspect, 1.0) * P.zoom;
  return vec4f(island_sd(p), rocks_sd(p), 0.0, 1.0);
}

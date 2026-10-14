// Bakes the static beach once per resize, into two targets:
//   ground: rgb = dry ground colour (sand and rocks), a = terrain height in metres above mean sea level
//   aux:    r = signed distance to the nearest rock in image-height units (negative inside),
//           g = smooth noise used to offset each stretch of shoreline's wave timing
#include "fullscreen.wgsl"
#include "noise.wgsl"
#include "backdrop.wgsl"
#include "ground_params.wgsl"

@group(0) @binding(0) var<uniform> G: GroundParams;

struct GroundOut {
  @location(0) ground: vec4f,
  @location(1) aux: vec2f,
}

const SLOPE: f32 = 6.0;     // metres of height per unit of uv.x: the beach profile
const SHORE_U: f32 = 0.4;   // where terrain height crosses 0 (before meander)
const ROCK_TOP: f32 = 1.8;  // rocks stand this far above mean sea level, clear of the highest run-up

fn terrain_height(uv: vec2f, q: vec2f) -> f32 {
  let u = uv.x + 0.04 * gnoise_2d(vec2f(q.y * 2.2, 3.0)) + 0.015 * gnoise_2d(q * 7.0);
  var h = SLOPE * (SHORE_U - u) - 9.0 * smoothstep(0.45, 1.0, u); // beach, then a shelf falling away
  // A sandbar offshore: waves feel it first, so it carries its own foam line when the water is high.
  let bar = (u - 0.66) / 0.05;
  h += 0.95 * exp(-bar * bar) * (0.75 + 0.25 * gnoise_2d(vec2f(q.y * 3.0, 9.0)));
  h += 0.1 * gnoise_2d(q * 5.0);
  return h;
}

@fragment fn fs(in: FullscreenVarying) -> GroundOut {
  let uv = in.uv;
  let q = vec2f(uv.x * G.aspect, uv.y);
  let h = terrain_height(uv, q);

  var sand = backdrop_sand(q * 9.6, 0.1);
  // Dry, bleached upper beach.
  sand = mix(sand, sand * vec3f(1.1, 1.06, 1.0), smoothstep(0.2, 1.8, h));

  // Rocks: centre.x in uv, centre.y and radius in image heights.
  var rocks = array<vec4f, 5>(
    vec4f(0.47, 0.2, 0.05, 1.0),
    vec4f(0.55, 0.52, 0.065, 2.0),
    vec4f(0.62, 0.82, 0.042, 3.0),
    vec4f(0.34, 0.64, 0.055, 4.0),
    vec4f(0.76, 0.36, 0.05, 5.0),
  );
  var rock_d = 1.0e3;
  var nearest = vec4f(0.0);
  for (var i = 0; i < 5; i++) {
    let r = rocks[i];
    let c = vec2f(r.x * G.aspect, r.y);
    let d = length(q - c) - r.z + 0.012 * gnoise_2d(q * 32.0 + r.w * 11.0);
    if (d < rock_d) {
      rock_d = d;
      nearest = vec4f(c, r.z, r.w);
    }
  }
  let cover = clamp(0.5 - rock_d / max(G.px, 1e-5), 0.0, 1.0);
  let local = (q - nearest.xy) / max(nearest.z, 1e-3);
  let dome = normalize(vec3f(local, max(sqrt(max(1.0 - dot(local, local), 0.0)), 0.3)));
  let shade = 0.4 + 0.8 * max(dot(dome, normalize(vec3f(-0.5, -0.6, 0.65))), 0.0);
  let speckle = 0.8 + 0.4 * (0.5 + 0.5 * gnoise_2d(q * 70.0)) + 0.15 * gnoise_2d(q * 18.0);
  let rock = srgb_to_linear(vec3f(0.5, 0.47, 0.44)) * shade * speckle;

  let phase_noise = gnoise_2d(vec2f(q.y * 2.6, 7.0)) + 0.5 * gnoise_2d(q * vec2f(1.2, 6.0));
  var result: GroundOut;
  result.ground = vec4f(mix(sand, rock, cover), mix(h, max(h, ROCK_TOP), cover));
  result.aux = vec2f(rock_d, phase_noise);
  return result;
}

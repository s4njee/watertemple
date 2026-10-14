#include "fullscreen.wgsl"
#include "noise.wgsl"
#include "color.wgsl"
#include "absorption.wgsl"
#include "foam.wgsl"
#include "surf_params.wgsl"

@group(0) @binding(0) var<uniform> P: SurfParams;
@group(0) @binding(1) var ground_tex: texture_2d<f32>;
@group(0) @binding(2) var aux_tex: texture_2d<f32>;
@group(0) @binding(3) var samp: sampler;

const RISE: f32 = 0.28;           // share of each wave cycle spent rushing up the beach
const METRES_PER_HEIGHT: f32 = 6.0; // converts rock distance (image heights) into the depth unit
const FOAM_COLOUR: vec3f = vec3f(0.92, 0.95, 0.97);

@fragment fn fs(in: FullscreenVarying) -> @location(0) vec4f {
  let uv = in.uv;
  let q = vec2f(uv.x * P.aspect, uv.y);
  let ground = textureSampleLevel(ground_tex, samp, uv, 0.0);
  let aux = textureSampleLevel(aux_tex, samp, uv, 0.0).xy;
  let height = ground.a;

  // Where the water is. Each stretch of shoreline runs on its own clock (aux.y), and the swell
  // reaches the beach later the further out it starts, so wave lines are never parallel or in step.
  let cycles = P.wave_cycles - (uv.x - 0.4) * P.travel + 0.3 * aux.y;
  let swash = foam_swash(cycles, RISE);
  let level = P.runup * swash;
  let depth = level - height;

  let w = absorb_water(P.water_type);
  let ripple = pow(1.0 - abs(gnoise_3d(vec3f(q * 3.0, P.ripple_time))), 3.0);
  let light = 1.0 + P.shimmer * (ripple * 2.4 - 0.5) * exp(-max(depth, 0.0) * 0.5);

  // Wet sand: always damp below the highest reach of the waves, soaked right at the waterline.
  let damp_zone = 1.0 - smoothstep(0.5 * P.runup, 1.1 * P.runup + 0.05, height);
  let wetness = max(0.8 * damp_zone, foam_wetness(height - level, 0.15));
  let sand = foam_darken_wet(ground.rgb, wetness, P.wet_strength);

  // Under water the bed is always saturated; the water over it tints by absorption of its depth.
  let wet_bed = foam_darken_wet(ground.rgb, 1.0, P.wet_strength) * light;
  let water = absorb_composite(wet_bed, w.sigma, w.turbidity, w.scatter, max(depth, 0.0));
  // Surface: a little sky reflection that brightens with the swell passing under it.
  let sheen = vec3f(0.55, 0.72, 0.9) * (0.04 + 0.12 * swash) * (0.6 + 0.8 * ripple);
  let contact = foam_edge_fade(depth, P.edge_soft);
  var colour = mix(sand, water + sheen, contact);

  // Foam. Coverage comes from distance to the waterline (depth) and to each rock, broken up by noise
  // so it reaches in fingers; the foam texture is carried with the water so it moves with the wave.
  // Most pixels are nowhere near either, so the noise and the Voronoi lace are skipped for them.
  let in_water = smoothstep(-0.04, 0.02, depth);
  let rock_dist = max(aux.x, 0.0) * METRES_PER_HEIGHT;
  let near_foam = in_water > 0.0 && (depth < P.foam_width || rock_dist < P.foam_width * 1.1);
  if (near_foam) {
    let breakup_noise = 0.5 + 0.5 * gnoise_3d(vec3f(q * 9.0, P.foam_time * 0.2));
    let x = fract(cycles);
    let energy = 1.0 - 0.65 * smoothstep(RISE, 1.0, x);   // fresh at the leading edge, thinning as it drains
    let shore = foam_mask(depth, P.foam_width, breakup_noise, P.breakup) * energy;
    let rock_ring = foam_mask(rock_dist, P.foam_width * 1.1, breakup_noise, P.breakup) * (0.5 + 0.5 * energy);
    // Full coverage is a solid white sheet; 0.85 keeps lace holes in the thickest foam.
    let coverage = clamp(max(shore, rock_ring) * in_water * P.foam_amount * 0.85, 0.0, 1.0);

    let carried = q * P.foam_scale + vec2f(level * P.foam_scale * 0.5, 0.0);
    let thickness = foam_texture(carried, P.foam_time);
    let foam = foam_dissolve(thickness, coverage, 0.35);
    // A thin crisp rim of white right at the contact, under the lace.
    let rim = foam_rim(depth, P.foam_width * 0.25) * in_water * energy * 0.6 * step(0.001, P.foam_amount);
    colour = mix(colour, FOAM_COLOUR * (0.85 + 0.15 * thickness), clamp(max(foam * 0.95, rim), 0.0, 1.0));
  }

  return vec4f(linear_to_srgb(clamp(colour, vec3f(0.0), vec3f(1.0))), 1.0);
}

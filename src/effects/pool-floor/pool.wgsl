#include "fullscreen.wgsl"
#include "color.wgsl"
#include "math.wgsl"
#include "normals.wgsl"
#include "refraction.wgsl"
#include "absorption.wgsl"
#include "specular.wgsl"
#include "caustics.wgsl"
#include "pool_params.wgsl"

@group(0) @binding(0) var<uniform> P: PoolParams;
@group(0) @binding(1) var floor_tex: texture_2d<f32>;
@group(0) @binding(2) var floor_samp: sampler;
@group(0) @binding(3) var normal_tex: texture_2d<f32>;
@group(0) @binding(4) var normal_samp: sampler;
@group(0) @binding(5) var caustic_tex: texture_2d<f32>;
@group(0) @binding(6) var caustic_samp: sampler;

// The sun, almost overhead, as a unit vector pointing at it (z up out of the water).
const SUN_DIR: vec3f = vec3f(-0.16, -0.2, 0.965);
// Light on the floor: sky fill that is always there, plus the sun, which arrives as caustics.
const AMBIENT: f32 = 0.42;
const SUN: f32 = 0.95;

// Surface normal from two scrolling taps of the baked ripple map, plus the pointer's lens and rings.
fn surface_normal(pw: vec2f) -> vec3f {
  let a = normal_decode_signed(
    textureSampleLevel(
      normal_tex, normal_samp, pw * P.ripple_scale + vec2f(0.37, 0.11) * P.time, P.normal_lod,
    ),
  );
  let b = normal_decode_signed(
    textureSampleLevel(
      normal_tex, normal_samp,
      rotate_2d(pw, 0.9) * (P.ripple_scale * 1.7) + vec2f(-0.21, 0.29) * P.time, P.normal_lod,
    ),
  );
  var n = normal_blend_whiteout(normal_scale(a, P.ripple), normal_scale(b, P.ripple * 0.6));

  let d = pw - P.pointer;
  let r = length(d);
  let dir = d / max(r, 1e-4);
  let s = 0.12;
  var slope = -P.lens * (d / s) * exp(-0.5 * r * r / (s * s));
  slope += P.ring * exp(-r * 3.2) * smoothstep(0.0, 0.04, r) * cos(r * 55.0 - P.ring_phase) * dir;
  n = normal_blend_whiteout(n, refraction_normal_from_slope(slope));
  return n;
}

@fragment fn fs(in: FullscreenVarying) -> @location(0) vec4f {
  let uv = in.uv;
  let pw = vec2f(uv.x * P.aspect, uv.y);
  let n = surface_normal(pw);

  // Refraction: where the ray that reaches this pixel hits the floor.
  let offset_world = refraction_offset_cheap(n, 1.333, P.thickness);
  let offset_uv = refraction_to_uv(offset_world, P.aspect);
  let lod = refraction_blur_lod(P.blur_px, P.max_lod);
  let floor_colour = refraction_sample(floor_tex, floor_samp, uv, offset_uv, lod, 0.05);

  // Caustics, evaluated at the floor position the ray reached, so they bend with the surface.
  let water = absorb_water(P.water_kind);
  let sun_through = absorb_transmittance(water.sigma, P.depth);
  let floor_pos = pw + offset_world;
  let c = caustics_baked(
    caustic_tex, caustic_samp, floor_pos * P.caustic_scale, P.caustic_time, P.caustic_lod,
    CAUSTICS_BAKED_AVG,
  );
  let irradiance = vec3f(AMBIENT) + vec3f(SUN * P.caustic * c) * sun_through;
  let lit = floor_colour * irradiance;

  // Depth tint and fog for the path the eye's ray takes through the water.
  let path = absorb_path_length(P.depth, 1.0, 1.333);
  var colour = absorb_composite(lit, water.sigma, water.turbidity, water.scatter * 0.8, path);

  // The sun glinting off the surface itself.
  let glint = spec_sun(n, vec3f(0.0, 0.0, 1.0), normalize(SUN_DIR), 0.04, 0.0093);
  colour += vec3f(1.0, 0.96, 0.88) * glint * P.shine * 3.0;

  colour = tonemap_reinhard_white(colour, 3.0);
  return vec4f(linear_to_srgb(colour), 1.0);
}

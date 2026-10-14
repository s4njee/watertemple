// The water pass: refract the tiled floor through the live surface, light it with the live caustics,
// absorb. The caustics texture is indexed by FLOOR position, so the light is looked up where the
// refracted ray hits the floor and stays registered to the tiles however the surface moves.
//
// Views: lit floor, the caustics alone, and two side-by-side comparisons of the REAL caustics (this
// project's heightfield solver) against the PROCEDURAL fake of caustics.wgsl (iterated sine warps,
// no simulation), left half real, right half fake.
#include "fullscreen.wgsl"
#include "heightfield.wgsl"
#include "refraction.wgsl"
#include "absorption.wgsl"
#include "specular.wgsl"
#include "caustics.wgsl"
#include "color.wgsl"
#include "shade_params.wgsl"

@group(0) @binding(0) var<uniform> P: ShadeParams;
@group(0) @binding(1) var floor_tex: texture_2d<f32>;
@group(0) @binding(2) var floor_samp: sampler;
@group(0) @binding(3) var field_tex: texture_2d<f32>;
@group(0) @binding(4) var caustic_tex: texture_2d<f32>;
@group(0) @binding(5) var linear_samp: sampler;

const VIEW_LIT: u32 = 0u;
const VIEW_CAUSTICS: u32 = 1u;
const VIEW_COMPARE_LIT: u32 = 2u;
const VIEW_COMPARE_CAUSTICS: u32 = 3u;

fn sun_light(c: vec3f) -> vec3f {
  // Contrast about the physical value: 0 is uniform light, 1 the simulation as computed.
  return max(vec3f(1.0) + (c - vec3f(1.0)) * P.intensity, vec3f(0.0));
}

fn lit_floor(floor_colour: vec3f, caustic: vec3f) -> vec3f {
  let sun_through = vec3f(exp(-P.sigma.x * P.depth), exp(-P.sigma.y * P.depth), exp(-P.sigma.z * P.depth));
  return floor_colour * (vec3f(P.ambient) + vec3f(P.sun) * sun_light(caustic) * sun_through);
}

@fragment fn fs(in: FullscreenVarying) -> @location(0) vec4f {
  let uv = in.uv;
  let pw = vec2f(uv.x * P.aspect, uv.y);
  let s = heightfield_sample(field_tex, linear_samp, uv);
  let n = heightfield_normal(s, P.strength);

  // Where the ray that reaches this pixel hits the floor, and what is there.
  let offset_world = refraction_offset_physical(n, vec3f(0.0, 0.0, -1.0), P.ior, P.depth);
  let offset_uv = refraction_to_uv(offset_world, P.aspect);
  let floor_uv = refraction_safe_uv(uv, offset_uv, P.fade);
  let floor_colour = textureSampleLevel(floor_tex, floor_samp, floor_uv, 0.0).rgb;

  let real = textureSampleLevel(caustic_tex, linear_samp, floor_uv, 0.0).rgb;
  let floor_world = vec2f(floor_uv.x * P.aspect, floor_uv.y);
  let fake = vec3f(caustics_stack(CAUSTICS_SINE, floor_world * P.fake_scale, P.fake_cells, P.time, P.fake_sharp));

  let on_right = uv.x >= P.split;
  var caustic = real;
  if (P.view >= VIEW_COMPARE_LIT && on_right) {
    caustic = fake;
  }

  var colour = vec3f(0.0);
  if (P.view == VIEW_CAUSTICS || P.view == VIEW_COMPARE_CAUSTICS) {
    // The caustics alone, exposed for looking at: 1 (flat light) sits at mid-grey.
    let c = caustic * P.intensity + vec3f(1.0 - P.intensity);
    colour = vec3f(1.0) - exp(-c * 0.7);
    colour = srgb_to_linear(colour * vec3f(0.9, 1.0, 1.05));
  } else {
    colour = lit_floor(floor_colour, caustic);
    let path = absorb_path_length(P.depth, 1.0, P.ior);
    colour = absorb_composite(colour, P.sigma, P.turbidity, P.scatter, path);
    // The sun glinting off the surface itself.
    let glint = spec_sun(n, vec3f(0.0, 0.0, 1.0), P.light, 0.04, 0.0093);
    colour += vec3f(1.0, 0.96, 0.88) * glint * P.shine * 3.0;
    colour = tonemap_reinhard_white(colour, P.white);
  }

  if (P.view >= VIEW_COMPARE_LIT) {
    let divider = 1.0 - smoothstep(0.0012, 0.0026, abs(uv.x - P.split));
    colour = mix(colour, vec3f(1.0), divider);
  }
  return vec4f(linear_to_srgb(clamp(colour, vec3f(0.0), vec3f(1.0))), 1.0);
}

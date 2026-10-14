// The water surface: reads the heightfield solver's output texture (one tap), turns its slope into
// a normal, refracts the backdrop through it and adds a specular glint. All the physics already
// happened in the solver; this pass is pure shading.
#include "fullscreen.wgsl"
#include "heightfield.wgsl"
#include "refraction.wgsl"
#include "specular.wgsl"
#include "color.wgsl"
#include "water_params.wgsl"

@group(0) @binding(0) var<uniform> P: WaterParams;
@group(0) @binding(1) var backdrop_tex: texture_2d<f32>;
@group(0) @binding(2) var backdrop_samp: sampler;
@group(0) @binding(3) var field_tex: texture_2d<f32>;
@group(0) @binding(4) var field_samp: sampler;

const VIEW_WATER: u32 = 0u;
const VIEW_SPLIT: u32 = 1u;
const VIEW_HEIGHT: u32 = 2u;
const VIEW_NORMALS: u32 = 3u;
const VIEW_VELOCITY: u32 = 4u;

const LIGHT: vec3f = vec3f(-0.4, -0.5, 0.77);

// Blue below the rest level, red above, white-ish at rest: the raw height, for judging the solver.
fn diverging(x: f32) -> vec3f {
  let t = clamp(x, -1.0, 1.0);
  let cold = vec3f(0.1, 0.3, 0.9);
  let hot = vec3f(0.95, 0.25, 0.1);
  return mix(vec3f(0.5), select(cold, hot, t > 0.0), abs(t));
}

@fragment fn fs(in: FullscreenVarying) -> @location(0) vec4f {
  let uv = in.uv;
  let s = heightfield_sample(field_tex, field_samp, uv);
  let n = heightfield_normal(s, P.strength);

  let offset_world = refraction_offset_physical(n, vec3f(0.0, 0.0, -1.0), P.ior, P.thickness);
  let offset_uv = refraction_to_uv(offset_world, P.aspect);

  var colour = vec3f(0.0);
  if (P.dispersion < 0.0005) {
    colour = refraction_sample(backdrop_tex, backdrop_samp, uv, offset_uv, 0.0, P.fade);
  } else {
    let scales = refraction_dispersion_scales(P.ior, P.dispersion);
    colour = refraction_sample_rgb(
      backdrop_tex, backdrop_samp, uv, refraction_rgb_offsets(offset_uv, scales), 0.0, P.fade,
    );
  }

  // Thicker water is bluer; the tint is a gentle multiply.
  colour *= mix(vec3f(1.0), P.tint, saturate(P.thickness * 1.6));

  // Relief: the side of each wave that faces the light is a little brighter, the far side darker.
  // Top-down water has almost no Fresnel to show the waves, and this is what makes a ripple read as
  // a ridge on a calm surface rather than a distortion alone.
  colour *= 1.0 + P.relief * 4.0 * dot(n.xy, normalize(LIGHT.xy)) * length(LIGHT.xy);

  // Sun glint on the crests.
  let light = normalize(LIGHT);
  let glint = spec_sun(n, vec3f(0.0, 0.0, 1.0), light, 0.04, 0.0093);
  colour += vec3f(1.0, 0.97, 0.9) * glint * P.shine * 30.0;

  if (P.view == VIEW_SPLIT) {
    let original = textureSampleLevel(backdrop_tex, backdrop_samp, uv, 0.0).rgb;
    let divider = 1.0 - smoothstep(0.0015, 0.003, abs(uv.x - 0.5));
    colour = mix(select(colour, original, uv.x < 0.5), vec3f(1.0), divider);
  } else if (P.view == VIEW_HEIGHT) {
    colour = srgb_to_linear(diverging(heightfield_height(s) * P.height_gain));
  } else if (P.view == VIEW_NORMALS) {
    colour = srgb_to_linear(n * 0.5 + 0.5);
  } else if (P.view == VIEW_VELOCITY) {
    colour = srgb_to_linear(diverging(heightfield_velocity(s) * P.height_gain * 8.0));
  }
  return vec4f(linear_to_srgb(clamp(colour, vec3f(0.0), vec3f(1.0))), 1.0);
}

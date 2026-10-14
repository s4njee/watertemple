// The reactive surface: the ripples shading (a backdrop refracted through the live heightfield, a sun
// glint and some relief) plus one optional detail, a small status dot in the bottom-right corner that
// tells a visitor what the audio input is doing. The solver already did the physics; this is shading.
//
// The dot (`dot_mode`): 0 = hidden. 1 = the audio context is waiting for a first click or tap (a hollow
// ring that breathes slowly, an invitation, not an alert). 2 = listening (a filled dot that swells with
// the overall level and flashes on an onset).
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
const VIEW_HEIGHT: u32 = 1u;

const LIGHT: vec3f = vec3f(-0.4, -0.5, 0.77);

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
  let scales = refraction_dispersion_scales(P.ior, P.dispersion);
  var colour = refraction_sample_rgb(
    backdrop_tex, backdrop_samp, uv, refraction_rgb_offsets(offset_uv, scales), 0.0, P.fade,
  );

  colour *= mix(vec3f(1.0), P.tint, saturate(P.thickness * 1.6));
  colour *= 1.0 + P.relief * 4.0 * dot(n.xy, normalize(LIGHT.xy)) * length(LIGHT.xy);
  let glint = spec_sun(n, vec3f(0.0, 0.0, 1.0), normalize(LIGHT), 0.04, 0.0093);
  colour += vec3f(1.0, 0.97, 0.9) * glint * P.shine * 30.0;

  if (P.view == VIEW_HEIGHT) {
    let h = heightfield_height(s) * P.height_gain;
    colour = srgb_to_linear(diverging(sign(h) * sqrt(abs(h))));
  }

  if (P.dot_mode > 0u) {
    let p = vec2f(uv.x * P.aspect, uv.y);
    let centre = vec2f(P.aspect - 0.045, 0.94);
    let d = length(p - centre);
    let pixel = P.pixel;
    var shape = 0.0;
    if (P.dot_mode == 1u) {
      let breath = 0.5 + 0.5 * sin(P.clock * 1.6);
      let radius = 0.012 + 0.002 * breath;
      shape = (1.0 - smoothstep(pixel, 2.5 * pixel, abs(d - radius))) * (0.35 + 0.35 * breath);
    } else {
      let radius = 0.005 + 0.011 * P.level + 0.008 * P.flash;
      shape = (1.0 - smoothstep(radius - pixel, radius + pixel, d)) * (0.55 + 0.45 * P.flash);
    }
    colour = mix(colour, vec3f(0.95, 0.98, 1.0), shape * 0.9);
  }
  return vec4f(linear_to_srgb(clamp(colour, vec3f(0.0), vec3f(1.0))), 1.0);
}

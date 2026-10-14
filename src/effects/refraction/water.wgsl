#include "fullscreen.wgsl"
#include "fbm.wgsl"
#include "color.wgsl"
#include "refraction.wgsl"
#include "water_params.wgsl"

@group(0) @binding(0) var<uniform> P: WaterParams;
@group(0) @binding(1) var backdrop_tex: texture_2d<f32>;
@group(0) @binding(2) var backdrop_samp: sampler;

const MODEL_PHYSICAL: u32 = 0u;
const VIEW_FINAL: u32 = 0u;
const VIEW_SPLIT: u32 = 1u;
const VIEW_NORMALS: u32 = 2u;
const VIEW_OFFSET: u32 = 3u;

// Slope (gradient of the water height) at a position in image-height units. Three layers:
//  1. two octaves of animated gradient noise, slope straight from the analytic derivative;
//  2. a smooth lens bump under the pointer (a bulge that magnifies what is below it);
//  3. rings that run outward from the pointer, fed by how fast it moves.
fn surface_slope(pw: vec2f) -> vec2f {
  let field = fbm_3d_d(vec3f(pw * P.ripple_scale, P.time), 2, 2.0, 0.5);
  var slope = field.yz * P.ripple_amount * 0.3;

  let d = pw - P.pointer;
  let r = length(d);
  let dir = d / max(r, 1e-4);
  let s = 0.13;
  slope -= P.lens * (d / s) * exp(-0.5 * r * r / (s * s));
  let envelope = exp(-r * 3.0) * smoothstep(0.0, 0.04, r);
  slope += P.ring * envelope * cos(r * 52.0 - P.ring_phase) * dir;
  return slope;
}

@fragment fn fs(in: FullscreenVarying) -> @location(0) vec4f {
  let uv = in.uv;
  let pw = vec2f(uv.x * P.aspect, uv.y);
  let n = refraction_normal_from_slope(surface_slope(pw));

  var offset_world = refraction_offset_cheap(n, P.ior, P.thickness);
  if (P.model == MODEL_PHYSICAL) {
    offset_world = refraction_offset_physical(n, vec3f(0.0, 0.0, -1.0), P.ior, P.thickness);
  }
  let offset_uv = refraction_to_uv(offset_world, P.aspect);
  let lod = refraction_blur_lod(P.blur_px, P.max_lod);

  var colour = vec3f(0.0);
  if (P.dispersion < 0.0005) {
    colour = refraction_sample(backdrop_tex, backdrop_samp, uv, offset_uv, lod, P.fade);
  } else if (P.quality == 0u) {
    let scales = refraction_dispersion_scales(P.ior, P.dispersion);
    colour = refraction_sample_rgb(
      backdrop_tex, backdrop_samp, uv, refraction_rgb_offsets(offset_uv, scales), lod, P.fade,
    );
  } else {
    colour = refraction_sample_spectral(
      backdrop_tex, backdrop_samp, uv, offset_uv, P.ior, P.dispersion, lod, P.fade, P.taps,
    );
  }

  // A touch of water: tint grows with thickness, plus a tight highlight where the normal faces the light.
  colour *= mix(vec3f(1.0), P.tint, saturate(P.thickness * 2.0));
  let half_vec = normalize(normalize(vec3f(-0.35, -0.45, 0.8)) + vec3f(0.0, 0.0, 1.0));
  colour += vec3f(1.0, 0.98, 0.92) * pow(max(dot(n, half_vec), 0.0), 700.0) * P.shine * 1.6;

  if (P.view == VIEW_SPLIT) {
    let original = textureSampleLevel(backdrop_tex, backdrop_samp, uv, 0.0).rgb;
    let divider = 1.0 - smoothstep(0.0015, 0.003, abs(uv.x - 0.5));
    colour = mix(select(colour, original, uv.x < 0.5), vec3f(1.0), divider);
  } else if (P.view == VIEW_NORMALS) {
    colour = srgb_to_linear(n * 0.5 + 0.5);
  } else if (P.view == VIEW_OFFSET) {
    colour = srgb_to_linear(vec3f(0.5 + offset_uv * 14.0, 0.5));
  }
  return vec4f(linear_to_srgb(clamp(colour, vec3f(0.0), vec3f(1.0))), 1.0);
}

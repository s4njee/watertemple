// Final pass of the HDR chain: scene (+ bloom) -> exposure -> tonemap -> sRGB -> dither -> canvas.
#include "fullscreen.wgsl"
#include "color.wgsl"
#include "output-params.wgsl"

@group(0) @binding(0) var scene_tex: texture_2d<f32>;
@group(0) @binding(1) var bloom_tex: texture_2d<f32>;
@group(0) @binding(2) var samp: sampler;
@group(0) @binding(3) var<uniform> post: OutputParams;

const TONEMAP_NONE = 0u;
const TONEMAP_ACES = 1u;
const TONEMAP_REINHARD = 2u;
const VIEW_FINAL = 0u;
const VIEW_SCENE = 1u;
const VIEW_BLOOM = 2u;

// The tonemap curve itself, not clamped for `none` so the HDR shoulder can continue past 1.
fn post_curve(c: vec3f, mode: u32) -> vec3f {
  if (mode == TONEMAP_ACES) { return tonemap_aces(c); }
  if (mode == TONEMAP_REINHARD) { return tonemap_reinhard(c); }
  return max(c, vec3f(0.0));
}

// Display-referred linear colour in [0, 1] for an SDR canvas.
fn post_map_sdr(c: vec3f, mode: u32) -> vec3f {
  return min(post_curve(c, mode), vec3f(1.0));
}

// Extended-range mapping for an HDR canvas. Up to scene value 1.0 (after exposure) it is exactly the
// SDR curve, so SDR content looks the same as it does on an SDR canvas. Above that, instead of
// clipping to 1 the curve keeps going with the slope the SDR curve had at 1.0 and rolls off
// smoothly (C1-continuous) towards `headroom`, in multiples of SDR white:
//   y(x) = f0 + d * slope / (1 + d * slope / (headroom - f0)),   d = x - 1
fn post_map_hdr(c: vec3f, mode: u32, headroom: f32) -> vec3f {
  let f0 = post_curve(vec3f(1.0), mode).x;
  let eps = 0.001;
  let slope = max((f0 - post_curve(vec3f(1.0 - eps), mode).x) / eps, 1e-3);
  // A headroom of 1 (or less) means "no HDR": the shoulder then tops out at SDR white.
  let room = max(max(headroom, 1.0) - f0, 1e-3);
  let d = max(c - vec3f(1.0), vec3f(0.0));
  let above = vec3f(f0) + d * slope / (vec3f(1.0) + d * slope / room);
  return select(above, post_map_sdr(c, mode), c <= vec3f(1.0));
}

// Interleaved gradient noise (Jimenez 2014): cheap, well distributed, no texture. Two evaluations
// at unrelated offsets summed give triangular-PDF noise in [-1, 1], which decorrelates the
// quantisation error from the signal (plain uniform noise leaves a little noise-modulated banding).
fn post_ign(p: vec2f) -> f32 {
  return fract(52.9829189 * fract(dot(p, vec2f(0.06711056, 0.00583715))));
}

fn post_dither(pixel: vec2f, frame: f32) -> f32 {
  let shift = vec2f(5.588238 * frame);
  return post_ign(pixel + shift) + post_ign(pixel.yx + vec2f(37.0, 91.0) + shift) - 1.0;
}

@fragment fn fs(in: FullscreenVarying) -> @location(0) vec4f {
  let scene = textureSampleLevel(scene_tex, samp, in.uv, 0.0).rgb;
  let glow = textureSampleLevel(bloom_tex, samp, in.uv, 0.0).rgb;

  var radiance = scene + glow;
  if (post.view == VIEW_SCENE) { radiance = scene; }
  if (post.view == VIEW_BLOOM) { radiance = glow; }

  let exposed = max(radiance, vec3f(0.0)) * post.exposure;
  var mapped = post_map_sdr(exposed, post.tonemap);
  if (post.hdr == 1u) { mapped = post_map_hdr(exposed, post.tonemap, post.headroom); }

  let encoded = linear_to_srgb(mapped);
  let noise = post_dither(in.position.xy, post.frame) * post.dither_lsb;
  return vec4f(max(encoded + vec3f(noise), vec3f(0.0)), 1.0);
}

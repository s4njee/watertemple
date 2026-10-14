// color.wgsl -- colour-space transfer functions, luminance, tonemapping, cosine palette.
//
// Pipeline assumed by these functions: do all lighting maths in LINEAR RGB (Rec.709 / sRGB primaries),
// tonemap if the values can exceed 1, then encode to sRGB as the very last step. Decode sRGB textures
// and colour picked by eye (hex codes, CSS colours) to linear first. If the canvas / render target
// format ends in `-srgb` the hardware encodes for you: do NOT call linear_to_srgb yourself then. The
// preferred canvas format (bgra8unorm / rgba8unorm) is not sRGB-encoding: call it.
//
// Transfer functions: the exact IEC 61966-2-1 piecewise curve (not the pow 2.2 approximation).
// Input is clamped at 0 (negative values give 0); values above 1 pass through (HDR).
//   srgb_to_linear(c: vec3f) -> vec3f      linear_to_srgb(c: vec3f) -> vec3f
//   srgb_to_linear_f(x: f32) -> f32        linear_to_srgb_f(x: f32) -> f32     single-channel forms
//
// Luminance
//   luminance(c: vec3f) -> f32     relative luminance (Rec.709 weights 0.2126, 0.7152, 0.0722) of LINEAR rgb
//
// Tonemappers: input is linear scene-referred RGB >= 0 (1.0 = diffuse white, higher is HDR);
// output is LINEAR display-referred in [0, 1]: apply linear_to_srgb after. All map 0 -> 0 and are
// monotonic per channel. Apply exposure beforehand: tonemap_x(color * exposure).
//   tonemap_reinhard(c) -> vec3f            c / (1 + c), per channel. Never clips, washes out highlights
//   tonemap_reinhard_white(c, white) -> vec3f   extended Reinhard: input `white` (> 0) maps to exactly 1.0,
//                                           anything brighter clips at 1.0
//   tonemap_aces(c) -> vec3f                Narkowicz's ACES filmic fit: contrasty, 1.0 -> ~0.80,
//                                           0.18 -> ~0.27, clips to 1.0 above ~7.2 (hard shoulder);
//                                           shifts hue in bright highlights
//
// Palette
//   palette_cosine(t, a, b, c, d) -> vec3f  a + b * cos(TAU * (c * t + d)), t typically 0..1. A cheap smooth
//                                           gradient: a offset, b amplitude, c frequency, d phase (all vec3f)
//
// Cost: the transfer functions use one pow() per channel; the rest is a few ALU ops. Palette: 3 cos.
// Credits: ACES fit by K. Narkowicz (2016); cosine palettes from I. Quilez's "palettes" article.

#include "math.wgsl"

fn srgb_to_linear_f(x: f32) -> f32 {
  let c = max(x, 0.0);
  let lo = c / 12.92;
  let hi = pow((c + 0.055) / 1.055, 2.4);
  return select(hi, lo, c <= 0.04045);
}

fn linear_to_srgb_f(x: f32) -> f32 {
  let c = max(x, 0.0);
  let lo = c * 12.92;
  let hi = 1.055 * pow(c, 1.0 / 2.4) - 0.055;
  return select(hi, lo, c <= 0.0031308);
}

fn srgb_to_linear(c: vec3f) -> vec3f {
  let v = max(c, vec3f(0.0));
  let lo = v / 12.92;
  let hi = pow((v + 0.055) / 1.055, vec3f(2.4));
  return select(hi, lo, v <= vec3f(0.04045));
}

fn linear_to_srgb(c: vec3f) -> vec3f {
  let v = max(c, vec3f(0.0));
  let lo = v * 12.92;
  let hi = 1.055 * pow(v, vec3f(1.0 / 2.4)) - 0.055;
  return select(hi, lo, v <= vec3f(0.0031308));
}

fn luminance(c: vec3f) -> f32 {
  return dot(c, vec3f(0.2126, 0.7152, 0.0722));
}

fn tonemap_reinhard(c: vec3f) -> vec3f {
  let v = max(c, vec3f(0.0));
  return v / (1.0 + v);
}

fn tonemap_reinhard_white(c: vec3f, white: f32) -> vec3f {
  let v = max(c, vec3f(0.0));
  let w2 = max(white * white, 1e-12);
  return min(v * (1.0 + v / w2) / (1.0 + v), vec3f(1.0));
}

fn tonemap_aces(c: vec3f) -> vec3f {
  let v = max(c, vec3f(0.0));
  let mapped = (v * (2.51 * v + 0.03)) / (v * (2.43 * v + 0.59) + 0.14);
  return clamp(mapped, vec3f(0.0), vec3f(1.0));
}

fn palette_cosine(t: f32, a: vec3f, b: vec3f, c: vec3f, d: vec3f) -> vec3f {
  return a + b * cos(TAU * (c * t + d));
}

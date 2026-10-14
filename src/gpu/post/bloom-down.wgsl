// Bloom downsample chain. The 13-tap filter is the shared one in passes/downsample.wgsl, included as
// the chunk "downsample13.wgsl" (it brings `src`, `samp`, `tap` and the plain `fs` entry point).
// This file adds the first-level entry point: soft-knee threshold + Karis average, then intensity.
#include "downsample13.wgsl"
#include "color.wgsl"
#include "bloom-params.wgsl"

@group(0) @binding(2) var<uniform> bloom: BloomParams;

// Unity-style soft knee on the brightest channel. Everything at or below (threshold - knee) is
// removed; between there and (threshold + knee) a quadratic fades the contribution in so the glow
// does not pop on and off as a pixel crosses the threshold. Above that the excess over the
// threshold is kept, with hue preserved by scaling the whole colour.
fn _bloom_prefilter(colour: vec3f) -> vec3f {
  let c = clamp(colour, vec3f(0.0), vec3f(65000.0));
  let brightness = max(c.r, max(c.g, c.b));
  let knee = max(bloom.knee, 1e-5);
  let soft = clamp(brightness - bloom.threshold + knee, 0.0, 2.0 * knee);
  let curve = soft * soft / (4.0 * knee);
  let kept = max(curve, brightness - bloom.threshold) / max(brightness, 1e-5);
  return c * kept;
}

// Karis average: weight each of the five 4-tap boxes by 1 / (1 + luma) so one very bright pixel
// cannot dominate a box. Without it a single hot pixel becomes a flickering firefly at lower mips as
// the image moves. Weights are renormalised, so the result is a weighted mean (this does remove
// energy from isolated hot pixels on purpose; turn it off with `karis = 0` to conserve it).
fn _bloom_weight(box_colour: vec3f, base: f32) -> f32 {
  if (bloom.karis < 0.5) { return base; }
  return base / (1.0 + luminance(box_colour));
}

@fragment fn fs_prefilter(in: FullscreenVarying) -> @location(0) vec4f {
  let uv = in.uv;
  let texel = 1.0 / vec2f(textureDimensions(src));

  let a = tap(uv, texel, -2.0, -2.0).rgb;
  let b = tap(uv, texel, 0.0, -2.0).rgb;
  let c = tap(uv, texel, 2.0, -2.0).rgb;
  let d = tap(uv, texel, -2.0, 0.0).rgb;
  let e = tap(uv, texel, 0.0, 0.0).rgb;
  let f = tap(uv, texel, 2.0, 0.0).rgb;
  let g = tap(uv, texel, -2.0, 2.0).rgb;
  let h = tap(uv, texel, 0.0, 2.0).rgb;
  let i = tap(uv, texel, 2.0, 2.0).rgb;
  let j = tap(uv, texel, -1.0, -1.0).rgb;
  let k = tap(uv, texel, 1.0, -1.0).rgb;
  let l = tap(uv, texel, -1.0, 1.0).rgb;
  let m = tap(uv, texel, 1.0, 1.0).rgb;

  // The same five overlapping boxes as the plain filter: centre 0.5, four corners 0.125 each.
  let p0 = _bloom_prefilter((a + b + d + e) * 0.25);
  let p1 = _bloom_prefilter((b + c + e + f) * 0.25);
  let p2 = _bloom_prefilter((d + e + g + h) * 0.25);
  let p3 = _bloom_prefilter((e + f + h + i) * 0.25);
  let p4 = _bloom_prefilter((j + k + l + m) * 0.25);
  let w0 = _bloom_weight(p0, 0.125);
  let w1 = _bloom_weight(p1, 0.125);
  let w2 = _bloom_weight(p2, 0.125);
  let w3 = _bloom_weight(p3, 0.125);
  let w4 = _bloom_weight(p4, 0.5);
  let mean = (p0 * w0 + p1 * w1 + p2 * w2 + p3 * w3 + p4 * w4) / (w0 + w1 + w2 + w3 + w4);
  return vec4f(mean * bloom.intensity, 1.0);
}

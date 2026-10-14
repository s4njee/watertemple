#include "fullscreen.wgsl"

@group(0) @binding(0) var src: texture_2d<f32>;
@group(0) @binding(1) var samp: sampler;

// 13 bilinear taps, offsets in source texels from the target pixel's centre (a is -2,-2; j is -1,-1):
//   a . b . c
//   . j . k .
//   d . e . f
//   . l . m .
//   g . h . i
fn tap(uv: vec2f, texel: vec2f, x: f32, y: f32) -> vec4f {
  return textureSampleLevel(src, samp, uv + vec2f(x, y) * texel, 0.0);
}

@fragment fn fs(in: FullscreenVarying) -> @location(0) vec4f {
  let uv = in.uv;
  let texel = 1.0 / vec2f(textureDimensions(src));

  let a = tap(uv, texel, -2.0, -2.0);
  let b = tap(uv, texel, 0.0, -2.0);
  let c = tap(uv, texel, 2.0, -2.0);
  let d = tap(uv, texel, -2.0, 0.0);
  let e = tap(uv, texel, 0.0, 0.0);
  let f = tap(uv, texel, 2.0, 0.0);
  let g = tap(uv, texel, -2.0, 2.0);
  let h = tap(uv, texel, 0.0, 2.0);
  let i = tap(uv, texel, 2.0, 2.0);
  let j = tap(uv, texel, -1.0, -1.0);
  let k = tap(uv, texel, 1.0, -1.0);
  let l = tap(uv, texel, -1.0, 1.0);
  let m = tap(uv, texel, 1.0, 1.0);

  // Five boxes of four taps each: the centre box (j k l m) weighs 0.5 and the four corner boxes
  // (a b d e, b c e f, d e g h, e f h i) 0.125 each, so e is shared by four of them. Weights sum to 1.
  return e * 0.125 + (a + c + g + i) * 0.03125 + (b + d + f + h) * 0.0625 + (j + k + l + m) * 0.125;
}

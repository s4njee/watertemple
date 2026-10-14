// liquid-2d wide pass: the field reduced to roughly a quarter of its size with a box filter, ready to be
// blurred into the thickness map. Four bilinear taps at the corners of each output texel's footprint cover
// it exactly (each tap averages a 2x2 block), so sparse particles cannot be skipped the way a single
// bilinear tap of a 4x reduction would skip them.

#include "fullscreen.wgsl"
#include "liquid2d_params.wgsl"

@group(0) @binding(0) var<uniform> P: Liquid2dParams;
@group(0) @binding(1) var field_tex: texture_2d<f32>;
@group(0) @binding(3) var samp: sampler;

@fragment
fn fs(in: FullscreenVarying) -> @location(0) vec4f {
  let e = 0.25 / P.wide_size; // a quarter of an output texel, in uv
  var sum = textureSampleLevel(field_tex, samp, in.uv + vec2f(-e.x, -e.y), 0.0);
  sum += textureSampleLevel(field_tex, samp, in.uv + vec2f(e.x, -e.y), 0.0);
  sum += textureSampleLevel(field_tex, samp, in.uv + vec2f(-e.x, e.y), 0.0);
  sum += textureSampleLevel(field_tex, samp, in.uv + vec2f(e.x, e.y), 0.0);
  return sum * 0.25;
}

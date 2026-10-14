#include "fullscreen.wgsl"

@group(0) @binding(0) var src: texture_2d<f32>;
@group(0) @binding(1) var samp: sampler;

// 3x3 tent filter, weights 1 2 1 / 2 4 2 / 1 2 1 over 16, taps one texel of `src` apart and
// bilinearly filtered, so the coarse level is reconstructed smoothly rather than blockily.
// The pass blends this over the finer level with a constant factor (see bloom.ts), so the shader
// itself does not know about radius.
fn up_tap(uv: vec2f, texel: vec2f, x: f32, y: f32) -> vec3f {
  return textureSampleLevel(src, samp, uv + vec2f(x, y) * texel, 0.0).rgb;
}

@fragment fn fs(in: FullscreenVarying) -> @location(0) vec4f {
  let uv = in.uv;
  let texel = 1.0 / vec2f(textureDimensions(src));
  var sum = up_tap(uv, texel, 0.0, 0.0) * 4.0;
  sum += (up_tap(uv, texel, -1.0, 0.0) + up_tap(uv, texel, 1.0, 0.0) +
          up_tap(uv, texel, 0.0, -1.0) + up_tap(uv, texel, 0.0, 1.0)) * 2.0;
  sum += up_tap(uv, texel, -1.0, -1.0) + up_tap(uv, texel, 1.0, -1.0) +
         up_tap(uv, texel, -1.0, 1.0) + up_tap(uv, texel, 1.0, 1.0);
  return vec4f(sum / 16.0, 1.0);
}

#include "fullscreen.wgsl"

@group(0) @binding(0) var src: texture_2d<f32>;
@group(0) @binding(1) var samp: sampler;

// textureLoad keeps nearest working on unfilterable formats (r32float, rgba32float).
@fragment fn fs_nearest(in: FullscreenVarying) -> @location(0) vec4f {
  let size = vec2f(textureDimensions(src));
  let texel = vec2u(clamp(in.uv * size, vec2f(0.0), size - 1.0));
  return textureLoad(src, texel, 0);
}

@fragment fn fs_linear(in: FullscreenVarying) -> @location(0) vec4f {
  return textureSampleLevel(src, samp, in.uv, 0.0);
}

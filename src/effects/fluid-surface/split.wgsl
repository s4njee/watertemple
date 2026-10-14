// Puts two renderings of the liquid side by side: `left` on the left half, `right` on the right, with
// a thin divider. Used by the filter comparison view.

#include "fullscreen.wgsl"

@group(0) @binding(0) var left: texture_2d<f32>;
@group(0) @binding(1) var right: texture_2d<f32>;

@fragment fn fs(in: FullscreenVarying) -> @location(0) vec4f {
  let pixel = vec2i(in.position.xy);
  let size = vec2f(textureDimensions(left));
  let divider = 1.0 - smoothstep(0.5, 1.5, abs(in.position.x - 0.5 * size.x));
  var c = textureLoad(right, pixel, 0);
  if (in.uv.x < 0.5) {
    c = textureLoad(left, pixel, 0);
  }
  return vec4f(mix(c.rgb, vec3f(1.0), divider), 1.0);
}

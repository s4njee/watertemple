// One separable Gaussian pass over the thickness image (rg16float: thickness and thickness times
// the particle scalar). Radius and sigma follow `blur_radius`; zero copies.

#include "fluid-frame.wgsl"
#include "fullscreen.wgsl"

@group(0) @binding(0) var<uniform> frame: FluidFrame;
@group(0) @binding(1) var<uniform> pass_info: FluidPass;
@group(0) @binding(2) var source: texture_2d<f32>;

@fragment fn fs(in: FullscreenVarying) -> @location(0) vec4f {
  let pixel = vec2i(in.position.xy);
  let size = vec2i(frame.thick_size);
  let radius = frame.blur_radius;
  if (radius <= 0) {
    return textureLoad(source, pixel, 0);
  }
  let sigma = (f32(radius) + 1.0) / 3.0;
  var total = vec4f(0.0);
  var weight_sum = 0.0;
  for (var i = -radius; i <= radius; i++) {
    // Clamp to the edge: thickness continues past the border of the image.
    let q = clamp(pixel + pass_info.direction * i, vec2i(0), size - 1);
    let w = exp(-0.5 * f32(i * i) / (sigma * sigma));
    total += w * textureLoad(source, q, 0);
    weight_sum += w;
  }
  return total / weight_sum;
}

#include "fullscreen.wgsl"
#include "blur_params.wgsl"

@group(0) @binding(0) var<uniform> params: BlurParams;
@group(0) @binding(1) var src: texture_2d<f32>;
@group(0) @binding(2) var samp: sampler;

// One axis of a Gaussian, exp(-i^2 / 2 sigma^2) for taps i = -radius..radius, normalised by the
// sum of the weights actually used so a truncated kernel still conserves energy.
//
// Taps i and i + 1 are fetched together: a bilinear sample placed between them, at their
// weight-weighted mean offset, returns exactly w_i * texel_i + w_(i+1) * texel_(i+1), so the loop
// makes half the fetches. That needs a filterable source and the clamp-to-edge linear sampler.
@fragment fn fs(in: FullscreenVarying) -> @location(0) vec4f {
  let step = params.direction / vec2f(textureDimensions(src));
  let k = -0.5 / (params.sigma * params.sigma);

  var sum = textureSampleLevel(src, samp, in.uv, 0.0);
  var total = 1.0;
  for (var i = 1; i <= params.radius; i += 2) {
    let w0 = exp(k * f32(i * i));
    let has_next = i < params.radius;
    let w1 = select(0.0, exp(k * f32((i + 1) * (i + 1))), has_next);
    let w = w0 + w1;
    if (w < 1e-6) { break; }
    let offset = (f32(i) * w0 + f32(i + 1) * w1) / w;
    let a = textureSampleLevel(src, samp, in.uv + step * offset, 0.0);
    let b = textureSampleLevel(src, samp, in.uv - step * offset, 0.0);
    sum += (a + b) * w;
    total += 2.0 * w;
  }
  return sum / total;
}

// One separable pass of the depth filter. `direction` is (1, 0) or (0, 1); a full round is two draws.
// Both filters take a Gaussian average along the line, with a kernel whose radius follows the
// projected size of a particle at the pixel's depth, so the smoothing is the same in world terms
// near and far. They differ in what they do with a neighbour whose depth is far from the centre's:
//   bilateral     weights it by exp(-dz^2 / 2 range^2): far depths are dropped, so a slanted surface
//                 only averages a thin sliver of its neighbours and the blobs show through
//   narrow-range  keeps it but clamps its depth to centre +- range: a foreign surface can move the
//                 result by at most `range`, and a slanted surface still averages everything
// Background pixels (depth 0) are skipped by both, so an isolated droplet keeps its own shape and a
// silhouette is never pulled outwards.

#include "fluid-frame.wgsl"
#include "fullscreen.wgsl"
#include "camera.wgsl"
#include "fluid-surface.wgsl"

@group(0) @binding(0) var<uniform> cam: Camera;
@group(0) @binding(1) var<uniform> frame: FluidFrame;
@group(0) @binding(2) var<uniform> pass_info: FluidPass;
@group(0) @binding(3) var source: texture_2d<f32>;

// At most 33 taps per direction. A wider kernel takes every n-th texel, which is only safe on a
// surface that is already smooth: on raw depth it makes each parity class of pixels average a
// different set of texels and leaves a checkered texture, so the default cap stays below it.
const TAPS_PER_SIDE = 16.0;

@fragment fn fs(in: FullscreenVarying) -> @location(0) f32 {
  let pixel = vec2i(in.position.xy);
  let size = vec2i(frame.chain_size);
  let centre = textureLoad(source, pixel, 0).r;
  if (centre <= 0.0) {
    return 0.0;
  }
  let pixel_radius = fluid_surface_pixel_radius(cam, centre, frame.radius, frame.chain_size.y);
  let radius = fluid_surface_filter_radius(pixel_radius, frame.filter_scale, frame.max_radius);
  let stride = max(1, i32(ceil(radius / TAPS_PER_SIDE)));
  let taps = i32(floor(radius / f32(stride)));
  var total = 0.0;
  var weight_sum = 0.0;
  for (var i = -taps; i <= taps; i++) {
    let offset = i * stride;
    let q = pixel + pass_info.direction * offset;
    if (any(q < vec2i(0)) || any(q >= size)) {
      continue;
    }
    let z = textureLoad(source, q, 0).r;
    if (z <= 0.0) {
      continue;
    }
    var weight = fluid_surface_kernel_weight(f32(offset), radius);
    var value = z;
    if (frame.filter_kind == 0u) {
      weight = fluid_surface_bilateral_weight(weight, z - centre, frame.range);
    } else {
      value = fluid_surface_narrow_clamp(z, centre, frame.range);
    }
    total += weight * value;
    weight_sum += weight;
  }
  return total / max(weight_sum, 1e-6);
}

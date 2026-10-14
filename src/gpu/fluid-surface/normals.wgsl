// Normals from a depth image. The view-space position of each pixel is rebuilt from its linear
// depth; the normal is the cross product of the position differences to the neighbours, and in each
// direction the one-sided difference with the smaller depth change is used (or the only valid one),
// so a pixel on a silhouette takes its normal from the surface side and never from across the gap.
// Output: xyz is the unit view-space normal facing the camera, w is 1 on the fluid and 0 off it.

#include "fluid-frame.wgsl"
#include "fullscreen.wgsl"
#include "camera.wgsl"
#include "fluid-surface.wgsl"

@group(0) @binding(0) var<uniform> cam: Camera;
@group(0) @binding(1) var<uniform> frame: FluidFrame;
@group(0) @binding(2) var depth: texture_2d<f32>;

fn position_at(pixel: vec2i) -> vec4f {
  let size = vec2i(frame.chain_size);
  if (any(pixel < vec2i(0)) || any(pixel >= size)) {
    return vec4f(0.0);
  }
  let z = textureLoad(depth, pixel, 0).r;
  if (z <= 0.0) {
    return vec4f(0.0);
  }
  let uv = (vec2f(pixel) + 0.5) / frame.chain_size;
  return vec4f(fluid_surface_view_pos(cam, uv, z), 1.0);
}

@fragment fn fs(in: FullscreenVarying) -> @location(0) vec4f {
  let pixel = vec2i(in.position.xy);
  let here = position_at(pixel);
  if (here.w == 0.0) {
    return vec4f(0.0);
  }
  let right = position_at(pixel + vec2i(1, 0));
  let left = position_at(pixel + vec2i(-1, 0));
  let up = position_at(pixel + vec2i(0, -1));
  let down = position_at(pixel + vec2i(0, 1));
  let dx = fluid_surface_pick_difference(right.xyz - here.xyz, here.xyz - left.xyz, right.w > 0.0, left.w > 0.0);
  // Screen y grows downwards, so the pixel above is towards +y in view space.
  let dy = fluid_surface_pick_difference(up.xyz - here.xyz, here.xyz - down.xyz, up.w > 0.0, down.w > 0.0);
  return vec4f(fluid_surface_normal(dx, dy), 1.0);
}

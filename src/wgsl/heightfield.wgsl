// heightfield.wgsl -- read the output texture of the heightfield solver (src/gpu/sim/heightfield.ts)
// without knowing how it is laid out. Everything a shading pass needs: height, slope, normal.
//
// THE TEXTURE. `solver.view` is an rgba16float texture, filterable, one texel per grid cell, covering
// the whole canvas: uv (0, 0) is its top-left corner and (1, 1) the bottom-right, the same as the
// fullscreen pass's `uv`. Channels, per texel:
//     x  height     in CELLS (1.0 = one grid cell tall; multiply by solver.grid.cellSize for image heights)
//     y  velocity   in cells per simulation step (internal to the scheme; handy as a foam / activity hint)
//     z  slope dx   d height / d x, per cell, x to the right
//     w  slope dy   d height / d y, per cell, y DOWN (the uv direction)
// Height and the two slopes use the SAME unit (cells), so the slope is a dimensionless tangent: 0.1 is
// a 5.7 degree tilt. The slopes are a Sobel gradient of the height of the same instant, so they are
// consistent with the height channel, and they are smooth enough to shade with after bilinear
// filtering. Solid obstacle cells read as all zeros (the mask itself is solver.obstacles).
//
// FRAME. Normals live in the screen frame of refraction.wgsl: x right, y down, z toward the viewer, and
// a bump of positive height faces the viewer. Slopes are taken per CELL; because the height is also in
// cells, the slope is the same number whatever the grid resolution (a 128 grid and a 1024 grid that
// hold the same wave give the same slope).
//
// Decoding a texel `s` (a vec4f from the texture)
//   heightfield_height(s) -> f32            height in cells
//   heightfield_height_world(s, cell_size) -> f32   height in image heights; cell_size = solver.grid.cellSize
//   heightfield_velocity(s) -> f32          cells per step
//   heightfield_gradient(s) -> vec2f        (d height/dx, d height/dy), dimensionless
//   heightfield_slope(s, strength) -> vec2f gradient times a strength; the form refraction_normal_from_slope
//                                           takes. strength 1 is the physical surface, larger exaggerates
//   heightfield_normal(s, strength) -> vec3f unit normal of the surface z = height scaled by `strength`:
//                                           normalize(-strength * gradient, 1). Same convention as
//                                           refraction_normal_from_slope, so it feeds straight into
//                                           refraction_offset_cheap / _physical and spec_sun
//   heightfield_is_solid(mask_value) -> bool  true for an obstacle-mask texel (r8unorm .r) that is solid
//
// Fetching (the texture and sampler are parameters; this chunk declares no bindings)
//   heightfield_sample(tex, samp, uv) -> vec4f          one bilinear tap at uv; needs a filtering sampler
//   heightfield_sample_normal(tex, samp, uv, strength) -> vec3f   the normal at uv from that one tap
//   heightfield_load(tex, cell: vec2i) -> vec4f         the exact texel of a cell (clamped to the grid),
//                                                       for taps that must not be filtered
//
// Cost: one filtered tap plus a normalize (about 10 ALU). The normal needs no extra taps because the
// solver stores the slopes: that is the point of the layout.
//
// Why bilinear taps of the slope are fine: the slope field is as smooth as the height field, and a
// bilinear blend of two Sobel gradients is a gradient of a slightly smoother height. Bilinear taps of
// the HEIGHT channel are not a good normal source on their own (they flatten every crest).

const HEIGHTFIELD_SOLID_THRESHOLD: f32 = 0.5;

fn heightfield_height(s: vec4f) -> f32 {
  return s.x;
}

fn heightfield_height_world(s: vec4f, cell_size: f32) -> f32 {
  return s.x * cell_size;
}

fn heightfield_velocity(s: vec4f) -> f32 {
  return s.y;
}

fn heightfield_gradient(s: vec4f) -> vec2f {
  return s.zw;
}

fn heightfield_slope(s: vec4f, strength: f32) -> vec2f {
  return s.zw * strength;
}

fn heightfield_normal(s: vec4f, strength: f32) -> vec3f {
  return normalize(vec3f(-s.zw * strength, 1.0));
}

fn heightfield_is_solid(mask_value: f32) -> bool {
  return mask_value >= HEIGHTFIELD_SOLID_THRESHOLD;
}

fn heightfield_sample(tex: texture_2d<f32>, samp: sampler, uv: vec2f) -> vec4f {
  return textureSampleLevel(tex, samp, uv, 0.0);
}

fn heightfield_sample_normal(tex: texture_2d<f32>, samp: sampler, uv: vec2f, strength: f32) -> vec3f {
  return heightfield_normal(textureSampleLevel(tex, samp, uv, 0.0), strength);
}

fn heightfield_load(tex: texture_2d<f32>, cell: vec2i) -> vec4f {
  let size = vec2i(textureDimensions(tex));
  return textureLoad(tex, clamp(cell, vec2i(0), size - vec2i(1)), 0);
}

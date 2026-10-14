// obstacle-raster.wgsl -- draws the shape table into the fluid's obstacle mask: one r8unorm texel per
// solver cell, 1 where the CELL CENTRE is inside any shape (distance <= T.grow), else 0. The solver
// reads >= 0.5 as solid (src/gpu/sim/obstacles.ts). A plain fragment pass over the mask; uv origin is
// top-left, which is row 0 of the mask.
//
// The mask is binary and blocky (a staircase of cells); the shapes are drawn crisply by the shading
// pass from the same table, which is what hides the staircase.

#include "fullscreen.wgsl"
#include "obstacle-shapes.wgsl"
#include "obstacle_table.wgsl"

@group(0) @binding(0) var<uniform> T: ObstacleTable;

@fragment fn fs(in: FullscreenVarying) -> @location(0) vec4f {
  let p = in.uv * vec2f(T.aspect, 1.0);
  var d = 1e9;
  for (var i = 0u; i < T.count; i++) {
    d = min(d, obstacle_sd(T.data[3u * i], T.data[3u * i + 1u], p));
  }
  return vec4f(select(0.0, 1.0, d <= T.grow), 0.0, 0.0, 1.0);
}

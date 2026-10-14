// Low-resolution pass: the pigment field (bloom shape, tone, edge detail) for every texel of a
// target 1/4 the size of the canvas per axis. rgba16float: signed values, alpha unused.
#include "fullscreen.wgsl"
#include "wash.wgsl"
#include "field_params.wgsl"

@group(0) @binding(0) var<uniform> field: FieldParams;

@fragment fn fs(in: FullscreenVarying) -> @location(0) vec4f {
  let p = in.uv * field.extent * field.scale + field.origin;
  return vec4f(wash_field(p, field.time, field.warp), 0.0);
}

// The pool floor: tiles drawn once into a texture (with mips, for depth blur), exactly like the image
// a real site would refract.
#include "fullscreen.wgsl"
#include "backdrop.wgsl"
#include "floor_params.wgsl"

@group(0) @binding(0) var<uniform> floor_params: FloorParams;

@fragment fn fs(in: FullscreenVarying) -> @location(0) vec4f {
  let colour = backdrop_sample(0, in.uv, floor_params.aspect, floor_params.aa, floor_params.density);
  return vec4f(colour, 1.0);
}

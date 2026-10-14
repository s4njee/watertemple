// Bakes the paper (grain and fibres) once into a small tileable texture: r = grain, g = fibre.
#include "fullscreen.wgsl"
#include "wash.wgsl"

@fragment fn fs(in: FullscreenVarying) -> @location(0) vec4f {
  let paper = wash_paper_tile(in.uv);
  return vec4f(paper, 0.0, 1.0);
}

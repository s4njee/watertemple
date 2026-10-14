// Bakes the paper (grain and fibres) once into a small tileable texture: r = grain height, g = fibre.
// The pigment layer (src/gpu/sim/pigment-step.wgsl) reads it one texel per dye texel to modulate
// diffusion and deposition, and display.wgsl reads it at the same scale for the paper's look, so the
// valleys that took more pigment are the valleys that are drawn dark.
#include "fullscreen.wgsl"
#include "wash.wgsl"

@fragment fn fs(in: FullscreenVarying) -> @location(0) vec4f {
  let paper = wash_paper_tile(in.uv);
  return vec4f(paper, 0.0, 1.0);
}

// The floor under the water, drawn once into a texture. In a real site this is a photograph of tiles
// or sand; the water pass does not care.
#include "fullscreen.wgsl"
#include "backdrop.wgsl"
#include "scene_params.wgsl"

@group(0) @binding(0) var<uniform> scene: SceneParams;

@fragment fn fs(in: FullscreenVarying) -> @location(0) vec4f {
  // Linear light into an -srgb target: the hardware encodes it, so bilinear taps through the
  // refraction offset blend in linear space.
  return vec4f(backdrop_sample(scene.kind, in.uv, scene.aspect, scene.aa, scene.density), 1.0);
}

// The scene behind the water, drawn once into a texture. In a real site this is an image, a video
// frame or a captured page; the water pass does not care.
#include "fullscreen.wgsl"
#include "backdrop.wgsl"
#include "scene_params.wgsl"

@group(0) @binding(0) var<uniform> scene: SceneParams;

@fragment fn fs(in: FullscreenVarying) -> @location(0) vec4f {
  // Linear light into an -srgb target: the hardware encodes it, so later filtering (bilinear taps
  // through the refraction offset) happens in linear space.
  return vec4f(backdrop_sample(scene.kind, in.uv, scene.aspect, scene.aa, scene.density), 1.0);
}

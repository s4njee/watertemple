// The "scene behind the water": one of the shared backdrops, drawn once into a texture. In a real
// site this texture would be an image, a video frame or the page; the water pass does not care.
#include "fullscreen.wgsl"
#include "backdrop.wgsl"
#include "scene_params.wgsl"

@group(0) @binding(0) var<uniform> scene: SceneParams;

@fragment fn fs(in: FullscreenVarying) -> @location(0) vec4f {
  // Written as linear light into an -srgb target: the hardware encodes it, and later filtering
  // (bilinear, mips, refraction taps) happens in linear space.
  let colour = backdrop_sample(scene.kind, in.uv, scene.aspect, scene.aa, scene.density);
  return vec4f(colour, 1.0);
}

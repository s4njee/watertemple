// Headless stand-in for the demo picture: the shared test card drawn once into a texture, for when
// there is no DOM to paint the harbour on (the tests, and any page without a 2D canvas).
#include "fullscreen.wgsl"
#include "backdrop.wgsl"

struct SceneParams {
  aspect: f32,
  aa: f32,
}
@group(0) @binding(0) var<uniform> scene: SceneParams;

@fragment fn fs(in: FullscreenVarying) -> @location(0) vec4f {
  return vec4f(backdrop_sample(BACKDROP_KIND_CARD, in.uv, scene.aspect, scene.aa, 1.0), 1.0);
}

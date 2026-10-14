// Draws the sky behind the scene objects, once for the real camera and once for the mirrored one.
#include "fullscreen.wgsl"
#include "camera.wgsl"
#include "scene-light.wgsl"
#include "rf-sky.wgsl"

@group(0) @binding(0) var<uniform> cam: Camera;
@group(0) @binding(1) var<uniform> light: SceneLight;

@fragment fn fs(in: FullscreenVarying) -> @location(0) vec4f {
  return vec4f(rf_sky(light, camera_world_ray(cam, in.uv)), 1.0);
}

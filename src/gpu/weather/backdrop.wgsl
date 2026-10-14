#include "fullscreen.wgsl"
#include "weather_params.wgsl"
#include "weather_scene.wgsl"
@group(0) @binding(0) var<uniform> P: WeatherParams;
@fragment fn fs(in: FullscreenVarying) -> @location(0) vec4f {
  return vec4f(weather_scene(in.uv,P.aspect,P.background),1);
}

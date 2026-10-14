// The wall behind the tank, drawn once per resize into a linear rgba16float texture that the liquid refracts: a
// warm tiled wall inside the tank (straight lines, so the lens of every ripple and splash shows), a darker room
// around it, a soft light from the upper left. Pool tiles come from the shared backdrop.wgsl.

#include "fullscreen.wgsl"
#include "backdrop.wgsl"
#include "color.wgsl"
#include "sdf.wgsl"
#include "backdrop_params.wgsl"

@group(0) @binding(0) var<uniform> params: BackdropParams;

@fragment
fn fs(in: FullscreenVarying) -> @location(0) vec4f {
  let ndc = vec2f(in.uv.x * 2.0 - 1.0, 1.0 - in.uv.y * 2.0);
  let world = params.center + vec2f(ndc.x * params.aspect, ndc.y) * params.half_height;
  let pixel = 2.0 * params.half_height / params.height_px;
  let tile = 0.2;
  let tiles = backdrop_pool_tiles(world / tile, pixel / tile);
  // Keep only the tile pattern's lightness: a warm plaster wall that does not fight the water's colour.
  let l = clamp(luminance(tiles) * 1.15, 0.0, 1.0);
  let wall = mix(vec3f(0.30, 0.24, 0.20), vec3f(0.86, 0.80, 0.70), l);
  let room = vec3f(0.035, 0.04, 0.05) * (0.7 + 0.3 * smoothstep(-0.6, 0.6, world.y - params.center.y));

  // Inside the tank (the box), a hair of glass edge shadow at the border.
  let lo = params.box_min;
  let hi = params.box_max;
  let d = sd_box_2d(world - 0.5 * (lo + hi), 0.5 * (hi - lo));
  let inside = 1.0 - smoothstep(-pixel, pixel, d);
  var color = mix(room, wall, inside);
  // Light falloff across the wall, from the upper left, and a dim vignette.
  let light = 0.78 + 0.32 * dot(normalize(vec2f(-0.6, 0.8)), (world - params.center) / params.half_height) * -0.5;
  color *= mix(1.0, light, inside);
  color *= 1.0 - 0.35 * smoothstep(-0.3, 0.0, d) * inside;
  return vec4f(color, 1.0);
}

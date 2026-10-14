// The scene the liquid is rendered into: a sky and a tiled floor, drawn inside the forward renderer's pass
// (group 0 is its camera and light). The sky is the same function the liquid reflects; the floor is the legible
// pattern the refraction bends, in warm colours so that the (blue) water reads against it. Tiles are 12 cm, so a
// 1.6 m tank holds about 13 across and the water visibly shears them.

#include "fullscreen.wgsl"
#include "camera.wgsl"
#include "scene-light.wgsl"
#include "sky.wgsl"
#include "backdrop.wgsl"

@group(0) @binding(0) var<uniform> cam: Camera;
@group(0) @binding(1) var<uniform> light: SceneLight;

@fragment fn fs_sky(in: FullscreenVarying) -> @location(0) vec4f {
  return vec4f(sky_radiance(camera_world_ray(cam, in.uv), light.sun_dir), 1.0);
}

struct FloorVarying {
  @builtin(position) position: vec4f,
  @location(0) world: vec3f,
}

const FLOOR_HALF = 90.0;
const TILE_SIZE = 0.12;

@vertex fn vs_floor(@builtin(vertex_index) index: u32) -> FloorVarying {
  var corners = array<vec2f, 6>(
    vec2f(-1.0, -1.0), vec2f(-1.0, 1.0), vec2f(1.0, -1.0),
    vec2f(1.0, -1.0), vec2f(-1.0, 1.0), vec2f(1.0, 1.0),
  );
  let p = corners[index] * FLOOR_HALF;
  var out: FloorVarying;
  out.world = vec3f(p.x, 0.0, p.y);
  out.position = cam.view_proj * vec4f(out.world, 1.0);
  return out;
}

@fragment fn fs_floor(in: FloorVarying) -> @location(0) vec4f {
  let p = in.world.xz / TILE_SIZE;
  let aa = max(fwidth(p.x), fwidth(p.y));
  // The aqua pool tiles, recoloured warm (terracotta and cream): the water is blue, so it reads against them.
  let tiles = backdrop_pool_tiles(p, aa);
  let l = clamp((dot(tiles, vec3f(0.3, 0.55, 0.15)) - 0.2) / 0.6, 0.0, 1.0);
  let albedo = mix(vec3f(0.07, 0.035, 0.03), vec3f(0.92, 0.66, 0.46), l);
  let sun = normalize(light.sun_dir);
  let lit = albedo * (light.sky_color * light.ambient + light.sun_color * light.sun_intensity * sun.y);
  // Fade into the horizon of the sky so the floor has no edge.
  let to_point = in.world - cam.position;
  let distance = length(to_point);
  let horizon = sky_radiance(normalize(vec3f(to_point.x, 0.002, to_point.z)), sun);
  let fog = 1.0 - exp(-light.fog_density * distance);
  return vec4f(mix(lit, horizon, fog), 1.0);
}

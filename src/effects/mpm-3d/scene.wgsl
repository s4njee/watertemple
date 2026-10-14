// The room the tank stands in, drawn inside the forward renderer's pass (group 0 is its camera and light): the
// sky, and one floor. Inside the tank's footprint the floor is the pool tiles of backdrop.wgsl at a size where
// every line is straight, so the water's refraction is easy to judge; outside it is a dark plinth with a faint
// grid, fading into the sky at the horizon. Group 1 holds the tank rectangle and the floor height.

#include "fullscreen.wgsl"
#include "camera.wgsl"
#include "scene-light.wgsl"
#include "sky.wgsl"
#include "backdrop.wgsl"
#include "mpm_scene.wgsl"

@group(0) @binding(0) var<uniform> cam: Camera;
@group(0) @binding(1) var<uniform> light: SceneLight;
@group(1) @binding(0) var<uniform> room: MpmScene;

@fragment fn fs_sky(in: FullscreenVarying) -> @location(0) vec4f {
  return vec4f(sky_radiance(camera_world_ray(cam, in.uv), light.sun_dir), 1.0);
}

struct FloorVarying {
  @builtin(position) position: vec4f,
  @location(0) world: vec3f,
}

const FLOOR_HALF = 90.0;

@vertex fn vs_floor(@builtin(vertex_index) index: u32) -> FloorVarying {
  var corners = array<vec2f, 6>(
    vec2f(-1.0, -1.0), vec2f(-1.0, 1.0), vec2f(1.0, -1.0),
    vec2f(1.0, -1.0), vec2f(-1.0, 1.0), vec2f(1.0, 1.0),
  );
  let p = corners[index] * FLOOR_HALF;
  var out: FloorVarying;
  out.world = vec3f(p.x, room.floor_y, p.y);
  out.position = cam.view_proj * vec4f(out.world, 1.0);
  return out;
}

@fragment fn fs_floor(in: FloorVarying) -> @location(0) vec4f {
  let xz = in.world.xz;
  // Derivatives first: they must be taken in uniform control flow.
  let tiles = (xz - room.tank_min) / room.tile;
  let tile_aa = max(fwidth(tiles.x), fwidth(tiles.y));
  let grid = xz;
  let grid_aa = max(fwidth(grid.x), fwidth(grid.y));

  let inside = all(xz >= room.tank_min) && all(xz <= room.tank_max);
  var albedo: vec3f;
  if (inside) {
    albedo = backdrop_pool_tiles(tiles, tile_aa);
  } else {
    // Plinth: dark stone, a faint square grid, brighter right at the tank so its footprint reads.
    let f = abs(fract(grid) - 0.5);
    let line = 1.0 - smoothstep(0.47 - grid_aa, 0.5 + grid_aa, max(f.x, f.y));
    let edge = max(room.tank_min - xz, xz - room.tank_max);
    let rim = 1.0 - smoothstep(0.0, 0.35, max(edge.x, edge.y));
    albedo = vec3f(0.07, 0.075, 0.085) + vec3f(0.05) * line + vec3f(0.07, 0.07, 0.08) * rim;
  }
  let sun = normalize(light.sun_dir);
  let lit = albedo * (light.sky_color * light.ambient + light.sun_color * light.sun_intensity * sun.y);
  let to_point = in.world - cam.position;
  let distance = length(to_point);
  let horizon = sky_radiance(normalize(vec3f(to_point.x, 0.002, to_point.z)), sun);
  let fog = 1.0 - exp(-light.fog_density * distance);
  return vec4f(mix(lit, horizon, fog), 1.0);
}

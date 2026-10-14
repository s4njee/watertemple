// The picture around the water in pbf-2d, in two fragment entry points that share one uniform block:
//
//   fs_backdrop  drawn first, opaque: the room, the tank (a dark gradient with a faint ruled grid, every 0.1 m, so a
//                column height or a wave can be read off), the piston of the wave-maker (everything in the tank to the
//                left of the moving wall), the static obstacles and the glass outline of the CURRENT container.
//   fs_overlay   drawn last, premultiplied over the water: a ring at every pointer that is pressed or hovering.
//
// World units are metres, +y up. The tank is the fixed rectangle [tank_min, tank_max]; the container [wall_min,
// wall_max] is what the solver currently uses (the wave-maker's wall moves, nothing else does).

#include "fullscreen.wgsl"
#include "color.wgsl"
#include "sdf.wgsl"
#include "backdrop_params.wgsl"

@group(0) @binding(0) var<uniform> params: BackdropParams;

fn world_of(uv: vec2f) -> vec2f {
  let ndc = vec2f(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0);
  return params.center + vec2f(ndc.x * params.aspect, ndc.y) * params.half_height;
}

fn encode(rgb: vec3f) -> vec3f {
  return select(rgb, linear_to_srgb(rgb), params.encode_srgb != 0u);
}

// Signed distance to the nearest obstacle (negative inside one).
fn obstacle_distance(world: vec2f) -> f32 {
  var d = 1e5;
  for (var k = 0u; k < params.collider_count; k++) {
    let a = params.colliders[k * 2u];
    let b = params.colliders[k * 2u + 1u];
    if (u32(a.w + 0.5) == 1u) {
      d = min(d, length(world - a.xy) - a.z);
    } else {
      d = min(d, sd_box_2d(world - 0.5 * (a.xy + b.xy), 0.5 * (b.xy - a.xy)));
    }
  }
  return d;
}

@fragment
fn fs_backdrop(in: FullscreenVarying) -> @location(0) vec4f {
  let world = world_of(in.uv);
  let pixel = 2.0 * params.half_height / params.height_px;

  // The room: a dark vertical gradient.
  let vertical = clamp((world.y - params.tank_min.y) / (params.tank_max.y - params.tank_min.y), 0.0, 1.0);
  var color = mix(vec3f(0.004, 0.008, 0.016), vec3f(0.012, 0.022, 0.04), vertical);

  // The tank interior.
  let tank_mid = 0.5 * (params.tank_min + params.tank_max);
  let tank_half = 0.5 * (params.tank_max - params.tank_min);
  let in_tank = 1.0 - smoothstep(-pixel, pixel, sd_box_2d(world - tank_mid, tank_half));
  var inside = mix(vec3f(0.050, 0.078, 0.115), vec3f(0.018, 0.034, 0.062), 1.0 - vertical);
  // Ruled grid: lines every 0.1 m, stronger every 0.5 m, one pixel wide.
  let cell = world / 0.1;
  let line = abs(fract(cell - 0.5) - 0.5) * 0.1 / pixel;
  let minor = 1.0 - smoothstep(0.0, 1.0, min(line.x, line.y));
  let major_cell = world / 0.5;
  let major_line = abs(fract(major_cell - 0.5) - 0.5) * 0.5 / pixel;
  let major = 1.0 - smoothstep(0.0, 1.2, min(major_line.x, major_line.y));
  inside += vec3f(0.55, 0.75, 0.95) * (0.020 * minor + 0.035 * major);
  color = mix(color, inside, in_tank);

  // The piston: the part of the tank left of the moving wall.
  let behind = (1.0 - smoothstep(-pixel, pixel, world.x - params.wall_min.x)) * in_tank;
  let slab = mix(vec3f(0.115, 0.135, 0.16), vec3f(0.075, 0.09, 0.11), clamp((params.wall_min.x - world.x) / 0.12, 0.0, 1.0));
  color = mix(color, slab, behind);
  let face = (1.0 - smoothstep(0.0, 2.0 * pixel, abs(world.x - params.wall_min.x))) * in_tank;
  color += vec3f(0.35, 0.45, 0.55) * 0.35 * face;

  // Obstacles.
  let od = obstacle_distance(world);
  let solid = 1.0 - smoothstep(-pixel, pixel, od);
  let shade = mix(vec3f(0.19, 0.22, 0.26), vec3f(0.11, 0.13, 0.16), clamp(-od / 0.05, 0.0, 1.0));
  color = mix(color, shade, solid);
  let rim = (1.0 - smoothstep(0.0, 2.0 * pixel, abs(od)));
  color += vec3f(0.35, 0.5, 0.62) * 0.35 * rim;

  // The glass: an outline of the container the solver is using.
  let box_mid = 0.5 * (params.wall_min + params.wall_max);
  let box_half = 0.5 * (params.wall_max - params.wall_min);
  let edge = 1.0 - smoothstep(0.6 * pixel, 1.8 * pixel, abs(sd_box_2d(world - box_mid, box_half)));
  color += vec3f(0.30, 0.55, 0.72) * 0.55 * edge;
  return vec4f(encode(color), 1.0);
}

@fragment
fn fs_overlay(in: FullscreenVarying) -> @location(0) vec4f {
  let world = world_of(in.uv);
  let pixel = 2.0 * params.half_height / params.height_px;
  var alpha = 0.0;
  for (var k = 0u; k < params.pointer_count; k++) {
    let p = params.pointers[k];
    let d = length(world - p.xy);
    let ring = 1.0 - smoothstep(0.7 * pixel, 1.9 * pixel, abs(d - p.z));
    let disc = 1.0 - smoothstep(p.z - pixel, p.z + pixel, d);
    alpha = max(alpha, p.w * (0.85 * ring + 0.07 * disc));
  }
  let rgb = encode(vec3f(0.85, 0.95, 1.0));
  return vec4f(rgb * alpha, alpha);
}

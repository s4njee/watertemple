// Drawn over the finished liquid, premultiplied: the glass edge of the tank, the solid obstacles standing in the
// water (circles and rounded boxes, shaded as rounded pebbles), and a ring around each pointer. All 2D signed
// distance fields in world units, anti-aliased with the size of a pixel.

#include "fullscreen.wgsl"
#include "sdf.wgsl"
#include "color.wgsl"
#include "overlay_params.wgsl"

@group(0) @binding(0) var<uniform> params: OverlayParams;

fn obstacle_sd(k: u32, p: vec2f) -> f32 {
  let head = params.obstacles[2u * k];
  let tail = params.obstacles[2u * k + 1u];
  let q = p - head.yz;
  if (head.x < 1.5) {
    return sd_circle(q, tail.x);
  }
  return sd_rounded_box_2d(q, tail.xy, tail.z);
}

@fragment
fn fs(in: FullscreenVarying) -> @location(0) vec4f {
  let ndc = vec2f(in.uv.x * 2.0 - 1.0, 1.0 - in.uv.y * 2.0);
  let world = params.center + vec2f(ndc.x * params.aspect, ndc.y) * params.half_height;
  let pixel = 2.0 * params.half_height / params.height_px;
  var rgb = vec3f(0.0);
  var alpha = 0.0;

  // The glass: a thin pale line just outside the box and a dark rim inside it.
  let lo = params.box_min;
  let hi = params.box_max;
  let d_box = sd_box_2d(world - 0.5 * (lo + hi), 0.5 * (hi - lo));
  let glass = 1.0 - smoothstep(0.0, 1.5 * pixel, abs(d_box - 0.6 * pixel));
  rgb = mix(rgb, vec3f(0.75, 0.82, 0.88), glass * 0.55);
  alpha = max(alpha, glass * 0.55);

  // Obstacles.
  for (var k = 0u; k < params.obstacle_count; k++) {
    let d = obstacle_sd(k, world);
    let coverage = 1.0 - smoothstep(-0.5 * pixel, 0.5 * pixel, d);
    if (coverage > 0.0) {
      let e = 2.0 * pixel;
      let n = normalize(vec2f(
        obstacle_sd(k, world + vec2f(e, 0.0)) - obstacle_sd(k, world - vec2f(e, 0.0)),
        obstacle_sd(k, world + vec2f(0.0, e)) - obstacle_sd(k, world - vec2f(0.0, e)),
      ) + vec2f(1e-9));
      let light = normalize(vec2f(-0.55, 0.83));
      let edge = clamp(-d / (0.12 * params.half_height), 0.0, 1.0);
      let shade = 0.45 + 0.35 * dot(n, light) * (1.0 - edge) + 0.12 * edge;
      let base = mix(vec3f(0.34, 0.28, 0.24), vec3f(0.62, 0.52, 0.44), shade);
      let rim = smoothstep(2.5 * pixel, 0.0, -d) * 0.25;
      let color = base + vec3f(rim);
      rgb = mix(rgb, color, coverage);
      alpha = max(alpha, coverage);
    }
  }

  // Pointer rings.
  for (var k = 0u; k < params.pointer_count; k++) {
    let head = params.pointers[k];
    let r = length(world - head.xy);
    let ring = 1.0 - smoothstep(0.0, 1.5 * pixel, abs(r - head.z));
    let a = ring * head.w;
    rgb = mix(rgb, vec3f(1.0), a);
    alpha = max(alpha, a);
  }
  let encoded = select(rgb, linear_to_srgb(rgb), params.encode_srgb != 0u);
  return vec4f(encoded * alpha, alpha);
}

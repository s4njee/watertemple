// composite.wgsl -- puts the ink (march.wgsl's radiance and transmittance, at a fraction of the canvas
// resolution, upsampled bilinearly) between a tank and its backdrop, at full resolution, and writes the canvas.
//
// Back to front along each camera ray:
//   1. the backdrop: a dim studio gradient and a table. The tank has no floor of its own; the table is its floor,
//      lit a little brighter inside the footprint. Water tints what is seen through it (Beer-Lambert).
//   2. the tank's far edges (the three edges of the back faces), attenuated by the ink in front of them.
//   3. the ink: result = behind * T + L.
//   4. the tank's near edges and a faint sheen on the near face, on top of everything: the glass is in front.
// Edges are found analytically (the ray's entry and exit points on the box, and how close each is to a box edge,
// antialiased to one pixel), so the tank costs no geometry and its depth order against the ink is exact.
// Then exposure, ACES tonemapping and sRGB encoding. The debug views replace all of it with a slice through the
// velocity or the ink.
//
// Bindings: 0 `P`, 1 camera, 2 radiance, 3 transmittance, 4 linear sampler, 5 velocity at the cell centres (3D),
// 6 the ink (3D).

#include "fullscreen.wgsl"
#include "camera.wgsl"
#include "volume.wgsl"
#include "ink.wgsl"
#include "color.wgsl"

@group(0) @binding(1) var<uniform> cam: Camera;
@group(0) @binding(2) var radiance_tex: texture_2d<f32>;
@group(0) @binding(3) var transmittance_tex: texture_2d<f32>;
@group(0) @binding(4) var samp: sampler;
@group(0) @binding(5) var velocity_tex: texture_3d<f32>;
@group(0) @binding(6) var dens: texture_3d<f32>;

const EDGE_COLOR: vec3f = vec3f(0.62, 0.74, 0.8);

// Distance on the box surface from p to the nearest box edge: a point on a face is near an edge when two of its
// three distances to the bounding planes are small, so it is the median of the three.
fn edge_distance(p: vec3f) -> f32 {
  let d = min(p - P.box_min, P.box_max - p);
  let a = min(d.x, min(d.y, d.z));
  let c = max(d.x, max(d.y, d.z));
  return d.x + d.y + d.z - a - c;
}

// Antialiased edge coverage at distance t along the ray: a line about 2 pixels wide with soft sides (a line a
// pixel wide steps in and out of coverage as it slides across pixel centres, which reads as dashes).
fn edge_cover(p: vec3f, t: f32) -> f32 {
  let pixel = 2.0 * t / (cam.proj[1][1] * cam.viewport.y);
  return 1.0 - smoothstep(0.4 * pixel, 1.6 * pixel, edge_distance(p));
}

fn studio_sky(d: vec3f) -> vec3f {
  let up = smoothstep(-0.25, 0.9, d.y);
  return mix(vec3f(0.02, 0.028, 0.036), vec3f(0.09, 0.115, 0.14), up);
}

// The table at y = 0 and the studio behind the tank, for a ray from `o` along `d`: a pool of light on the table
// round the tank, a soft contact shadow just outside its footprint and a slightly lighter table inside it.
fn backdrop(o: vec3f, d: vec3f) -> vec3f {
  var col = studio_sky(d);
  if (d.y < -1e-4) {
    let tp = -o.y / d.y;
    let hit = o + d * tp;
    let r = length(hit.xz);
    let mid = 0.5 * (P.box_min + P.box_max);
    let q = abs(hit.xz - mid.xz) - 0.5 * (P.box_max.xz - P.box_min.xz);
    let outside = length(max(q, vec2f(0.0))) + min(max(q.x, q.y), 0.0);
    let lit = 0.55 * exp(-r * r * 0.35) + 0.12;
    var table = vec3f(0.2, 0.22, 0.25) * lit;
    table *= select(1.0 - 0.6 * exp(-outside * 9.0), 1.08, outside < 0.0);
    col = mix(table, studio_sky(vec3f(d.x, 0.02, d.z)), smoothstep(2.5, 9.0, r));
  }
  return col;
}

// Unit normal of the box face that p lies on (the face whose plane p is closest to).
fn box_normal(p: vec3f) -> vec3f {
  let lo = p - P.box_min;
  let hi = P.box_max - p;
  let m = min(min(lo.x, hi.x), min(min(lo.y, hi.y), min(lo.z, hi.z)));
  if (lo.x == m) { return vec3f(-1.0, 0.0, 0.0); }
  if (hi.x == m) { return vec3f(1.0, 0.0, 0.0); }
  if (lo.y == m) { return vec3f(0.0, -1.0, 0.0); }
  if (hi.y == m) { return vec3f(0.0, 1.0, 0.0); }
  if (lo.z == m) { return vec3f(0.0, 0.0, -1.0); }
  return vec3f(0.0, 0.0, 1.0);
}

// The obstacle: a matte grey ball, lit by the sun and the sky, with a cool rim.
fn ball_color(p: vec3f, dir: vec3f) -> vec3f {
  let n = normalize(p - P.sphere.xyz);
  let diffuse = max(dot(n, P.light_dir), 0.0);
  let rim = pow(1.0 - max(dot(n, -dir), 0.0), 3.0);
  let albedo = vec3f(0.42, 0.45, 0.5);
  return albedo * (P.light_color * 0.55 * diffuse + P.ambient * (0.5 + 0.5 * n.y)) + EDGE_COLOR * rim * 0.12;
}

fn heat(t: f32) -> vec3f {
  return palette_cosine(t, vec3f(0.5), vec3f(0.5), vec3f(1.0), vec3f(0.0, 0.33, 0.67));
}

// A slice through the volume at z = P.slice, drawn over the whole canvas with the box's aspect.
fn debug_slice(uv: vec2f) -> vec3f {
  let aspect_box = (P.box_max.x - P.box_min.x) / (P.box_max.y - P.box_min.y);
  let aspect_view = cam.viewport.x / cam.viewport.y;
  let scale = min(aspect_view / aspect_box, 1.0);
  var q = (uv - 0.5) / vec2f(0.9 * scale * aspect_box / aspect_view, 0.9) + 0.5;
  q = vec2f(q.x, 1.0 - q.y);
  if (any(q < vec2f(0.0)) || any(q > vec2f(1.0))) {
    return vec3f(0.02, 0.025, 0.03);
  }
  let uvw = vec3f(q, P.slice);
  if (P.view == 1u) {
    let v = textureSampleLevel(velocity_tex, samp, uvw, 0.0).xyz;
    let speed = length(v) * P.debug_gain;
    // atan2(0, 0) is NaN on some GPUs, and NaN survives mix(a, b, 0): keep the arguments off the origin.
    let hue = atan2(v.y, v.x + 1e-9) / 6.2831853 + 0.5;
    return mix(vec3f(0.03), heat(hue), 1.0 - exp(-speed));
  
  }
  let d = textureSampleLevel(dens, samp, uvw, 0.0);
  let c = ink_tint(d);
  return mix(vec3f(0.03), c, 1.0 - exp(-d.a * P.debug_gain));
}

@fragment
fn fs(in: FullscreenVarying) -> @location(0) vec4f {
  if (P.view != 0u) {
    return vec4f(linear_to_srgb(debug_slice(in.uv)), 1.0);
  }
  let origin = cam.position;
  let dir = camera_world_ray(cam, in.uv);
  let span = volume_ray_box(origin, dir, P.box_min, P.box_max);
  var col = backdrop(origin, dir);
  if (volume_hit(span)) {
    let near_p = origin + dir * span.x;
    let far_p = origin + dir * span.y;
    // Behind the ink: what the ray meets after leaving the tank, seen through the water.
    var behind = backdrop(far_p, dir);
    var t_behind = span.y;
    behind = mix(behind, EDGE_COLOR * 0.4, 0.55 * edge_cover(far_p, span.y) * P.edge_strength);
    let t_sphere = sphere_hit(origin, dir);
    if (t_sphere > 0.0 && t_sphere < span.y) {
      behind = ball_color(origin + dir * t_sphere, dir);
      t_behind = t_sphere;
    }
    behind *= volume_slab_transmittance(P.water, t_behind - span.x);
    let T = textureSampleLevel(transmittance_tex, samp, in.uv, 0.0).rgb;
    let L = textureSampleLevel(radiance_tex, samp, in.uv, 0.0).rgb;
    col = behind * T + L;
    // In front of the ink: the near edges and a grazing sheen on the glass.
    let n = box_normal(near_p);
    let sheen = pow(1.0 - clamp(abs(dot(dir, n)), 0.0, 1.0), 5.0) * 0.12;
    col += studio_sky(reflect(dir, n)) * sheen * P.edge_strength;
    // Only the edges of the near side: a near point's edge is in front of everything behind it.
    col = mix(col, EDGE_COLOR, 0.5 * edge_cover(near_p, span.x) * P.edge_strength);
  }
  col *= P.exposure;
  return vec4f(linear_to_srgb(tonemap_aces(col)), 1.0);
}

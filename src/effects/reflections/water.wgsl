// The water, drawn as a post-process on the finished scene: for every pixel, intersect the camera ray
// with the plane y = water_level, and if that happens in front of the scene's depth it is a water
// pixel. There is no water mesh. The pass then blends what is seen through the surface (the scene
// colour, tinted by the water depth we know from the depth buffer) with the reflection, by Fresnel.
//
// Three reflection sources, chosen by `params.strategy` (0 planar, 1 ssr, 2 sky, 3 split):
//   planar  sample the mirrored-camera render at the pixel's mirrored position, nudged by the normal
//   ssr     ray-march the scene depth buffer along reflect(view, normal); misses fall back to the sky
//   sky     reflect only the sky model, the zero-cost baseline

#include "fullscreen.wgsl"
#include "camera.wgsl"
#include "ssr.wgsl"
#include "color.wgsl"
#include "scene-light.wgsl"
#include "params.wgsl"
#include "rf-sky.wgsl"

@group(0) @binding(0) var<uniform> cam: Camera;
@group(0) @binding(1) var<uniform> mcam: Camera;
@group(0) @binding(2) var<uniform> light: SceneLight;
@group(0) @binding(3) var<uniform> params: Params;
@group(0) @binding(4) var scene_color: texture_2d<f32>;
@group(0) @binding(5) var scene_depth: texture_depth_2d;
@group(0) @binding(6) var refl_color: texture_2d<f32>;
@group(0) @binding(7) var samp: sampler;

const STRATEGY_PLANAR: u32 = 0u;
const STRATEGY_SSR: u32 = 1u;
const STRATEGY_SKY: u32 = 2u;
const STRATEGY_SPLIT: u32 = 3u;
const DEBUG_BUFFER: u32 = 1u;
const DEBUG_MASK: u32 = 2u;

// A calm surface: a few sines, their analytic slope summed. Amplitude times wavenumber is the
// slope each contributes (about 0.06), so ripple = 1 tilts the normal by a few degrees.
fn wave_gradient(p: vec2f, t: f32, angle: f32, k: f32, amp: f32) -> vec2f {
  let d = vec2f(cos(angle), sin(angle));
  let phase = k * dot(d, p) + sqrt(9.81 * k) * t;
  return d * (amp * k * cos(phase));
}

fn ripple_gradient(p: vec2f, t: f32) -> vec2f {
  var g = wave_gradient(p, t, 0.3, 2.1, 0.030);
  g += wave_gradient(p, t, 1.4, 3.3, 0.020);
  g += wave_gradient(p, t, 2.5, 5.7, 0.012);
  g += wave_gradient(p, t, 3.9, 8.9, 0.008);
  g += wave_gradient(p, t, 5.1, 13.0, 0.005);
  return g;
}

// Interleaved gradient noise (Jimenez): cheap, well distributed, no sin().
fn ign(pixel: vec2f) -> f32 {
  return fract(52.9829189 * fract(dot(pixel, vec2f(0.06711056, 0.00583715))));
}

struct Refl {
  // Linear radiance of what the surface mirrors.
  color: vec3f,
  // 1 where the source had real data, 0 where it fell back (SSR miss) or was clamped (planar edge).
  confidence: f32,
  // What the "buffer" debug view shows for this source.
  debug: vec3f,
}

fn planar_reflection(p: vec3f, n: vec3f) -> Refl {
  // The mirrored camera sees the plane exactly where the real one does, so the point projects to the
  // right place in the reflection image. The normal then pushes the lookup around: a hack, since it
  // moves the image rather than bending the ray, but the reflection stays smooth.
  let base = camera_world_to_uv(mcam, p).xy;
  let uv = base + n.xz * params.distortion * 0.25;
  var r: Refl;
  r.color = textureSampleLevel(refl_color, samp, uv, 0.0).rgb;
  r.confidence = select(0.0, 1.0, all(uv >= vec2f(0.0)) && all(uv <= vec2f(1.0)));
  r.debug = r.color;
  return r;
}

fn ssr_reflection(p: vec3f, view: vec3f, n: vec3f, pixel: vec2f) -> Refl {
  // The same slider, applied the physical way: it scales how far the normal leans, and the ray
  // follows.
  let lean = params.distortion * 2.0;
  let nd = normalize(vec3f(n.x * lean, 1.0, n.z * lean));
  var dir = reflect(-view, nd);
  dir = normalize(vec3f(dir.x, max(dir.y, 0.02), dir.z)); // never march into the water
  let jitter = select(0.0, ign(pixel), params.ssr_jitter == 1u);
  let hit = ssr_trace(
    scene_depth, cam, p + vec3f(0.0, 0.002, 0.0), dir,
    params.ssr_steps, params.ssr_distance, params.ssr_thickness, jitter,
  );
  let fallback = rf_sky(light, dir);
  let found = textureSampleLevel(scene_color, samp, hit.uv, 0.0).rgb;
  var r: Refl;
  r.color = mix(fallback, found, hit.confidence);
  r.confidence = hit.confidence;
  r.debug = vec3f(hit.uv, hit.confidence);
  return r;
}

fn sky_reflection(view: vec3f, n: vec3f) -> Refl {
  var dir = reflect(-view, n);
  dir = normalize(vec3f(dir.x, max(dir.y, 0.02), dir.z));
  var r: Refl;
  r.color = rf_sky(light, dir);
  r.confidence = 0.0;
  r.debug = r.color;
  return r;
}

fn finish(hdr: vec3f, pixel: vec2f) -> vec4f {
  var rgb = linear_to_srgb(tonemap_aces(hdr * params.exposure));
  if (params.strategy == STRATEGY_SPLIT && abs(pixel.x - params.split * cam.viewport.x) < 1.0) {
    rgb = vec3f(1.0);
  }
  return vec4f(rgb, 1.0);
}

@fragment fn fs(in: FullscreenVarying) -> @location(0) vec4f {
  let pixel = in.position.xy;
  let px = vec2i(pixel);
  let uv = in.uv;
  let planar_side = params.strategy == STRATEGY_PLANAR
    || (params.strategy == STRATEGY_SPLIT && uv.x < params.split);
  let ssr_side = params.strategy == STRATEGY_SSR
    || (params.strategy == STRATEGY_SPLIT && uv.x >= params.split);

  // Debug: the planar reflection image itself, as rendered, across the whole canvas.
  if (params.debug == DEBUG_BUFFER && planar_side) {
    return finish(textureSampleLevel(refl_color, samp, uv, 0.0).rgb, pixel);
  }

  let scene = textureLoad(scene_color, px, 0).rgb;
  let depth = textureLoad(scene_depth, px, 0);
  let ray = camera_world_ray(cam, uv);

  var is_water = false;
  var t_plane = 0.0;
  var scene_dist = 1e30;
  if (ray.y < -1e-4) {
    t_plane = (params.water_level - cam.position.y) / ray.y;
    if (depth < 1.0) {
      scene_dist = length(camera_world_pos(cam, uv, depth) - cam.position);
    }
    let p = cam.position + ray * t_plane;
    is_water = t_plane > 0.0 && t_plane < scene_dist
      && abs(p.x) < params.basin && abs(p.z) < params.basin;
  }

  if (!is_water) {
    var rgb = scene;
    if (params.debug != 0u) {
      rgb = vec3f(luminance(scene)) * 0.25;
    }
    return finish(rgb, pixel);
  }

  let p = cam.position + ray * t_plane;
  let view = -ray;
  // Small ripples fade with distance: they alias into noise long before they are resolved.
  let g = ripple_gradient(p.xz, params.time) * params.ripple / (1.0 + 0.04 * t_plane);
  let n = normalize(vec3f(-g.x, 1.0, -g.y));
  let fresnel = 0.02 + 0.98 * pow(1.0 - clamp(dot(n, view), 0.0, 1.0), 5.0);

  var refl: Refl;
  if (planar_side) {
    refl = planar_reflection(p, n);
  } else if (ssr_side) {
    refl = ssr_reflection(p, view, n, pixel);
  } else {
    refl = sky_reflection(view, n);
  }

  // Seen through the surface: Beer-Lambert absorption over the water depth along the ray.
  let under = clamp(scene_dist - t_plane, 0.0, 20.0);
  let transmit = exp(-under * (vec3f(1.0) - params.tint) * params.murk);
  let through = scene * transmit + params.tint * 0.5 * (vec3f(1.0) - transmit);

  var hdr = mix(through, refl.color, fresnel);
  if (params.debug == DEBUG_BUFFER) {
    hdr = refl.debug;
  } else if (params.debug == DEBUG_MASK) {
    hdr = mix(vec3f(0.85, 0.1, 0.1), vec3f(0.1, 0.8, 0.25), refl.confidence);
  }
  return finish(hdr, pixel);
}

// ssr.wgsl -- screen-space reflection ray march against a depth buffer.
//
// struct SsrHit { uv: vec2f, confidence: f32, distance: f32 }
//   uv          where on screen the ray hit (top-left origin); meaningless when confidence is 0
//   confidence  0 = miss (ray left the screen, ran out of distance, or was never in front of the
//               scene); up to 1 = solid hit. Already faded towards the screen edges (10% border)
//               and towards `max_distance` (last 40%). Use it as the blend weight against a fallback.
//   distance    world-space distance from the origin to the hit surface (max_distance on a miss)
//
//   ssr_trace(depth_tex, cam, origin, dir, steps, max_distance, thickness, jitter) -> SsrHit
//     depth_tex     the depth buffer the scene was drawn with, bound as texture_depth_2d and read
//                   with textureLoad (no sampler, no filtering). Same camera as `cam`, standard
//                   (not reversed) 0..1 depth. Pixels at depth 1.0 are treated as empty sky.
//     cam           the camera that drew depth_tex (camera.wgsl); NOT a mirrored/oblique camera
//     origin, dir   world-space ray start and unit direction (e.g. a water point and reflect(V, N))
//     steps         linear samples along the ray's on-screen path (>= 1); resolution is path length
//                   in pixels / steps, so thin objects can slip between samples at low counts
//     max_distance  world-space length of the ray; the segment is also cut at the near plane
//     thickness     view-space depth (world units) a surface is assumed to have behind its visible
//                   face: a ray that ends up farther behind the surface than this (plus the depth
//                   it travels in one step) passed behind it and keeps going. Larger = fewer
//                   misses and more ghost hits behind thin occluders
//     jitter        0..1 per-pixel offset (e.g. interleaved gradient noise) that shifts the sample
//                   positions by up to 3/4 of a step: trades banding for fine noise. 0 disables it
//   ssr_edge_fade(uv, width) -> f32   1 in the interior, smoothly 0 at the screen edge over `width`
//                                     (fraction of the screen); the fade ssr_trace applies
//
// Method: project the ray's start and end into screen space once, then step linearly in screen
// space. 1 / view-z is affine along a projected line, so the ray's depth at every step is one
// reciprocal of a lerp, with no matrix multiply per step. A step whose ray depth has gone behind
// the depth buffer (within thickness) is a hit; it is refined by 5 bisections over the last step.
//
// Cost: setup is two matrix-vector products and four divides. Each step is one textureLoad of a
// 32-bit depth, one lerp, two reciprocals and a handful of compares: roughly 15 ALU + 1 fetch, so
// `steps` of 48 is about 750 ALU and 48 fetches per traced pixel. A hit adds 5 more steps.
// The cost is per pixel traced and independent of scene complexity (the opposite of a planar pass).
//
// Known limits: it can only reflect what is on screen and visible from the camera. There is no
// information about the back or underside of a surface, off-screen content, or what an occluder
// hides; rays that leave the screen miss. Reflections of surfaces the ray meets at grazing angles
// are stretched. Thin geometry under the step size is missed. Callers fall back to something else
// (a sky colour, a probe) with weight 1 - confidence.

#include "camera.wgsl"

struct SsrHit {
  uv: vec2f,
  confidence: f32,
  distance: f32,
}

const SSR_REFINE_STEPS: i32 = 5;
const SSR_EDGE_WIDTH: f32 = 0.1;
const SSR_FAR_FADE_START: f32 = 0.6;

fn ssr_edge_fade(uv: vec2f, width: f32) -> f32 {
  let to_edge = min(min(uv.x, 1.0 - uv.x), min(uv.y, 1.0 - uv.y));
  return smoothstep(0.0, max(width, 1e-5), to_edge);
}

// View-space depth of the surface at uv; effectively infinite for empty sky so no ray is behind it.
fn _ssr_scene_depth(depth_tex: texture_depth_2d, cam: Camera, uv: vec2f, size: vec2f) -> f32 {
  let texel = vec2i(clamp(uv * size, vec2f(0.0), size - vec2f(1.0)));
  let d = textureLoad(depth_tex, texel, 0);
  return select(camera_linear_depth(cam, d), 1e30, d >= 1.0);
}

fn ssr_trace(
  depth_tex: texture_depth_2d,
  cam: Camera,
  origin: vec3f,
  dir: vec3f,
  steps: u32,
  max_distance: f32,
  thickness: f32,
  jitter: f32,
) -> SsrHit {
  var result: SsrHit;
  result.uv = vec2f(0.0);
  result.confidence = 0.0;
  result.distance = max_distance;

  let size = vec2f(textureDimensions(depth_tex));
  let o = camera_world_to_view(cam, origin);
  let d = (cam.view * vec4f(dir, 0.0)).xyz;

  // Keep the whole segment in front of the near plane (view-space z is negative in front).
  let near_z = -cam.near * 1.001;
  if (o.z > near_z) {
    return result;
  }
  var ray_length = max_distance;
  if (d.z > 1e-5) {
    ray_length = min(ray_length, (near_z - o.z) / d.z);
  }

  let h0 = cam.proj * vec4f(o, 1.0);
  let h1 = cam.proj * vec4f(o + d * ray_length, 1.0);
  let k0 = 1.0 / h0.w;
  let k1 = 1.0 / h1.w;
  let s0 = camera_ndc_to_uv(h0.xy * k0);
  let s1 = camera_ndc_to_uv(h1.xy * k1);

  let count = max(steps, 1u);
  let inv_count = 1.0 / f32(count);
  var previous = 0.0;
  for (var i = 1u; i <= count; i++) {
    let t = (f32(i) - 0.75 * jitter) * inv_count;
    let uv = mix(s0, s1, t);
    if (any(uv < vec2f(0.0)) || any(uv >= vec2f(1.0))) {
      break;
    }
    let ray_z = 1.0 / mix(k0, k1, t);
    let scene_z = _ssr_scene_depth(depth_tex, cam, uv, size);
    let step_z = abs(ray_z - 1.0 / mix(k0, k1, previous));
    // In front of the surface, or so far behind it that the ray passed behind: keep marching.
    if (ray_z < scene_z || ray_z - scene_z > thickness + step_z) {
      previous = t;
      continue;
    }

    // Bisect the last step for the crossing.
    var near_t = previous;
    var far_t = t;
    for (var r = 0; r < SSR_REFINE_STEPS; r++) {
      let mid = 0.5 * (near_t + far_t);
      let mid_z = 1.0 / mix(k0, k1, mid);
      if (mid_z < _ssr_scene_depth(depth_tex, cam, mix(s0, s1, mid), size)) {
        near_t = mid;
      } else {
        far_t = mid;
      }
    }
    let hit_uv = mix(s0, s1, far_t);
    let texel = vec2i(clamp(hit_uv * size, vec2f(0.0), size - vec2f(1.0)));
    let hit_world = camera_world_pos(cam, hit_uv, textureLoad(depth_tex, texel, 0));
    let hit_distance = length(hit_world - origin);

    result.uv = hit_uv;
    result.distance = hit_distance;
    result.confidence = ssr_edge_fade(hit_uv, SSR_EDGE_WIDTH)
      * (1.0 - smoothstep(SSR_FAR_FADE_START, 1.0, hit_distance / max_distance));
    return result;
  }
  return result;
}

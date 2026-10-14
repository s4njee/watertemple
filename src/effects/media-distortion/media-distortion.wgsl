// Water over an image or a video: one pass.
//
//   1. surface normal: two scrolling layers of the baked ripple normal map (no noise per pixel),
//      blended with the whiteout blend, plus the pointer's lens and rings and any click drops;
//   2. the normal becomes a uv offset (refraction_offset_cheap), in canvas space;
//   3. the fit (cover or contain, computed on the CPU by fitScale in media.ts) maps canvas uv to media
//      uv, and the offset is carried into media space after the cover/contain fit, and the media is sampled
//      there with edge-safe uvs (refraction_safe_uv), once or once per colour channel;
//   4. a highlight and a faint relief shading from the same normal make it read as a surface.
#include "fullscreen.wgsl"
#include "normals.wgsl"
#include "refraction.wgsl"
#include "media.wgsl"
#include "color.wgsl"
#include "math.wgsl"
#include "distortion_params.wgsl"

@group(0) @binding(0) var<uniform> P: DistortionParams;
@group(0) @binding(1) var media_tex: texture_2d<f32>;
@group(0) @binding(2) var media_samp: sampler;
@group(0) @binding(3) var normal_tex: texture_2d<f32>;
@group(0) @binding(4) var normal_samp: sampler;

const IOR: f32 = 1.333;
// Two layers of the same baked map, turned and scaled apart so the tile never shows.
const ROT_B: f32 = 0.9;
const LIGHT: vec3f = vec3f(-0.4, -0.5, 0.75);   // toward the light; screen frame, y down

fn layer(p: vec2f, scale: f32, drift: vec2f, rotation: f32) -> vec3f {
  let uv = rotate_2d(p * scale + drift, rotation);
  let n = normal_decode_signed(textureSample(normal_tex, normal_samp, uv));
  return normal_yaw(n, -rotation);
}

// Slope contributed by the pointer: a lens bump that follows it, rings that run out from its
// movement, and the click drops. All in canvas-height units; zero when the controls are off.
fn pointer_slope(p: vec2f) -> vec2f {
  var slope = vec2f(0.0);
  let d = p - P.pointer;
  let r = length(d);
  let dir = d / max(r, 1e-4);
  let s = 0.12;
  slope -= P.lens * (d / s) * exp(-0.5 * r * r / (s * s));
  slope += P.ring * exp(-r * 3.2) * smoothstep(0.0, 0.04, r) * cos(r * 55.0 - P.ring_phase) * dir;
  for (var i = 0; i < 4; i++) {
    let drop = P.drops[i];
    let age = P.clock - drop.z;
    if (drop.w > 0.0 && age >= 0.0 && age < 4.0) {
      let dd = p - drop.xy;
      let rr = length(dd);
      let front = age * 0.4;
      let env = exp(-age * 1.5) * exp(-pow((rr - front) / 0.075, 2.0));
      slope += drop.w * env * cos((rr - front) * 48.0) * (dd / max(rr, 1e-4));
    }
  }
  return slope;
}

fn surface_normal(uv: vec2f) -> vec3f {
  let p = vec2f(uv.x * P.aspect, uv.y);
  // The hover mode keeps the water still away from the pointer: amplitude is a blob around it.
  let blob = 0.2 + 0.8 * exp(-0.5 * dot(p - P.pointer, p - P.pointer) / (0.4 * 0.4));
  let local = P.amp * mix(1.0, blob, P.focus);
  let a = layer(p, P.scale, vec2f(P.phase * 0.05, P.phase * 0.03), 0.0);
  let b = layer(p, P.scale * 1.7, vec2f(-P.phase * 0.04, P.phase * 0.07), ROT_B);
  var n = normal_blend_whiteout(normal_scale(a, 0.8 * local), normal_scale(b, 0.4 * local));
  n = normal_blend_whiteout(n, refraction_normal_from_slope(pointer_slope(p)));
  return n;
}

@fragment fn fs(in: FullscreenVarying) -> @location(0) vec4f {
  let uv = in.uv;
  let contain = P.contain != 0u;
  let fit = P.fit;
  let muv = media_fit_uv(uv, fit);
  let lod = media_lod(P.media_px, fit, P.resolution);

  let n = surface_normal(uv);
  let offset_canvas = refraction_to_uv(refraction_offset_cheap(n, IOR, P.strength), P.aspect);
  let offset = media_offset_to_media(offset_canvas, fit);

  var colour: vec3f;
  if (P.dispersion < 0.0005) {
    colour = refraction_sample(media_tex, media_samp, muv, offset, lod, 0.04);
  } else {
    let scales = refraction_dispersion_scales(IOR, P.dispersion);
    colour = refraction_sample_rgb(
      media_tex, media_samp, muv, refraction_rgb_offsets(offset, scales), lod, 0.04,
    );
  }

  // Water tint grows with how hard the water is working; white leaves the picture alone.
  colour *= mix(vec3f(1.0), P.tint, saturate(P.strength * 0.6));
  // Relief shading: slopes that face the light brighten, the others darken, around the flat value.
  let lit = dot(n, normalize(LIGHT)) - normalize(LIGHT).z;
  colour *= 1.0 + lit * P.highlight * 1.4;
  let half_vec = normalize(normalize(LIGHT) + vec3f(0.0, 0.0, 1.0));
  colour += vec3f(1.0, 0.98, 0.92) * pow(max(dot(n, half_vec), 0.0), 600.0) * P.highlight * 1.0;

  // Original for the comparison view (same fit, no water).
  let original = textureSampleLevel(media_tex, media_samp, muv, lod).rgb;
  if (P.compare != 0u) {
    let divider = 1.0 - smoothstep(0.0015, 0.003, abs(uv.x - 0.5));
    colour = mix(select(colour, original, uv.x < 0.5), vec3f(1.0), divider);
  }

  if (contain) {
    // Bars: a dim, heavily blurred copy of the media filling the canvas, so the page never shows black.
    let px = max(fit.x / P.resolution.x, fit.y / P.resolution.y);
    let inside = media_inside(muv, px);
    let backdrop = textureSampleLevel(
      media_tex, media_samp, media_fit_uv(uv, P.cover_fit), max(lod, P.bar_lod),
    ).rgb * 0.4;
    colour = mix(backdrop, colour, inside);
  }

  // A file is being dragged over the canvas: dim the picture and ring it.
  let edge = min(min(uv.x, 1.0 - uv.x) * P.aspect, min(uv.y, 1.0 - uv.y));
  let ring = 1.0 - smoothstep(0.012, 0.016, edge);
  colour = mix(colour, mix(colour * 0.5, vec3f(0.25, 0.7, 1.0), ring), P.drag);

  return vec4f(linear_to_srgb(clamp(colour, vec3f(0.0), vec3f(1.0))), 1.0);
}

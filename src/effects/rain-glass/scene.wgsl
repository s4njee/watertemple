// The world behind the window, drawn once into a texture: a street at night (or dusk, or a grey day)
// as a pile of procedural lights, lit windows and neon signs. HDR on purpose: the lights are far
// brighter than 1, so after the blur chain they stay glowing blobs instead of fading to grey.
//   uv      0..1 over the whole image, top-left origin
//   p       the same in image-height units (x scaled by the aspect ratio)
#include "fullscreen.wgsl"
#include "hash.wgsl"
#include "math.wgsl"
#include "scene_params.wgsl"

@group(0) @binding(0) var<uniform> S: SceneParams;

const KIND_NIGHT: i32 = 0;
const KIND_DUSK: i32 = 1;
const KIND_DAY: i32 = 2;

// Light colours: sodium and tungsten dominate, with some red tail lights, cold LEDs and neon.
fn light_colour(h: f32, kind: i32) -> vec3f {
  if (kind == KIND_DAY) {
    return select(vec3f(1.0, 0.8, 0.55), vec3f(1.0, 0.25, 0.12), h > 0.7);
  }
  if (h < 0.28) { return vec3f(1.0, 0.72, 0.38); }   // tungsten
  if (h < 0.5) { return vec3f(1.0, 0.45, 0.12); }    // sodium
  if (h < 0.64) { return vec3f(1.0, 0.1, 0.06); }    // tail lights
  if (h < 0.78) { return vec3f(0.35, 0.8, 1.0); }    // LED
  if (h < 0.88) { return vec3f(0.3, 1.0, 0.5); }     // traffic green
  return vec3f(1.0, 0.2, 0.75);                      // neon
}

// One layer of point lights, one candidate per cell. `y` bounds in image heights keep the street
// lights below the roof line. Positions stay inside the middle of their cell, so one cell is enough.
fn light_layer(p: vec2f, cell: vec2f, prob: f32, power: f32, sigma: f32, y_lo: f32, y_hi: f32, seed: i32) -> vec3f {
  let u = p / cell;
  let c = vec2i(floor(u));
  let h = ihash_3to3(vec3i(c, seed));
  if (h.x >= prob || p.y < y_lo || p.y > y_hi) {
    return vec3f(0.0);
  }
  let g = ihash_3to3(vec3i(c.x + 977, c.y - 311, seed));
  let q = (vec2f(c) + 0.25 + 0.5 * g.xy) * cell;
  let d = p - q;
  let s = sigma * (0.6 + 0.8 * g.z);
  let r2 = dot(d, d) / (s * s);
  let falloff = exp(-r2) + 0.05 * exp(-sqrt(r2) * 0.35);
  return light_colour(h.y, S.kind) * power * (0.4 + 0.6 * h.z) * falloff;
}

fn windows(p: vec2f, cell: vec2f, seed: i32, lit: f32, power: f32, aa: f32) -> vec3f {
  let u = p / cell;
  let c = vec2i(floor(u));
  let l = fract(u);
  let h = ihash_3to3(vec3i(c, seed));
  if (h.x > lit) { return vec3f(0.0); }
  let inset = 0.2;
  let box = smoothstep(inset, inset + aa, l.x) * smoothstep(inset, inset + aa, 1.0 - l.x)
    * smoothstep(inset, inset + aa, l.y) * smoothstep(inset, inset + aa, 1.0 - l.y);
  let warm = mix(vec3f(1.0, 0.62, 0.3), vec3f(1.0, 0.85, 0.6), h.y);
  let cold = vec3f(0.55, 0.75, 1.0);
  return select(warm, cold, h.z > 0.8) * power * (0.35 + 0.65 * h.y) * box;
}

fn sky_colour(uv: vec2f, horizon: f32) -> vec3f {
  let t = saturate(uv.y / horizon);
  if (S.kind == KIND_DAY) {
    return mix(vec3f(0.30, 0.34, 0.40), vec3f(0.55, 0.58, 0.60), t);
  }
  if (S.kind == KIND_DUSK) {
    return mix(vec3f(0.05, 0.06, 0.16), vec3f(0.95, 0.38, 0.16), t * t) * 0.9;
  }
  return mix(vec3f(0.012, 0.018, 0.045), vec3f(0.17, 0.08, 0.06), t * t);
}

@fragment fn fs(in: FullscreenVarying) -> @location(0) vec4f {
  let uv = in.uv;
  let p = vec2f(uv.x * S.aspect, uv.y);
  let horizon = 0.64;
  var col = sky_colour(uv, horizon);

  let day = S.kind == KIND_DAY;
  let light_power = select(select(7.0, 5.0, S.kind == KIND_DUSK), 1.4, day);

  // Two skylines. The far one is lower, hazier and has fewer lit windows.
  let far_w = 0.11;
  let far_c = floor(p.x / far_w);
  let far_h = ihash_2to1(vec2i(i32(far_c), S.seed + 5));
  let far_top = horizon - (0.1 + 0.2 * far_h);
  if (uv.y > far_top && uv.y < horizon + 0.02) {
    let haze = select(vec3f(0.05, 0.05, 0.085), vec3f(0.3, 0.33, 0.36), day);
    col = select(haze, vec3f(0.22, 0.12, 0.12), S.kind == KIND_DUSK);
    col += windows(p, vec2f(far_w / 7.0, 0.02), S.seed + 11, 0.45, 1.3 * light_power / 7.0, 0.2);
  }
  let near_w = 0.2;
  let near_c = floor(p.x / near_w + 0.37);
  let near_h = ihash_2to1(vec2i(i32(near_c), S.seed + 9));
  let near_top = horizon - (0.06 + 0.3 * near_h * near_h);
  if (uv.y > near_top && uv.y < horizon + 0.02) {
    col = select(vec3f(0.006, 0.008, 0.014), vec3f(0.12, 0.13, 0.14), day);
    col = select(col, vec3f(0.06, 0.03, 0.04), S.kind == KIND_DUSK);
    let q = vec2f(p.x + 0.37 * near_w, p.y);
    col += windows(q, vec2f(near_w / 6.0, 0.024), S.seed + 21, 0.5, light_power * 0.5, 0.18);
    // A neon sign on some of them.
    let sign_h = ihash_3to3(vec3i(i32(near_c), 4, S.seed));
    if (sign_h.x < 0.35 && !day) {
      let sx = (fract(q.x / near_w) - 0.5) * near_w;
      let sy = p.y - (near_top + 0.06 + 0.1 * sign_h.y);
      if (abs(sx) < 0.045 && abs(sy) < 0.012) {
        col += light_colour(sign_h.z, S.kind) * 1.6 * light_power * 0.4;
      }
    }
  }

  // Shopfronts along the street: bright panels under the skyline.
  if (uv.y > horizon - 0.045 && uv.y < horizon + 0.02) {
    let shop_w = 0.1;
    let shop_c = floor(p.x / shop_w);
    let shop = ihash_3to3(vec3i(i32(shop_c), 8, S.seed));
    let sx = fract(p.x / shop_w);
    if (shop.x < 0.6 && sx > 0.08 && sx < 0.92 && !day) {
      col = light_colour(shop.y, S.kind) * light_power * 0.12 * (0.5 + 0.5 * shop.z);
    }
  }

  // Street.
  if (uv.y > horizon + 0.02) {
    let t = (uv.y - horizon) / (1.0 - horizon);
    col = select(vec3f(0.02, 0.018, 0.025), vec3f(0.1, 0.1, 0.11), day) * (1.0 - 0.5 * t);
    col += select(vec3f(0.2, 0.1, 0.06), vec3f(0.0), day) * exp(-t * 3.0) * 0.25;
    // Reflections of the lights on the wet road: the same light field squashed vertically.
    let sq = vec2f(p.x, horizon * 0.5 + (p.y - horizon) * 0.07);
    col += light_layer(sq, vec2f(0.05, 0.02), 0.5, light_power * 0.35, 0.006, 0.0, 9.0, S.seed + 40) * 0.55;
  }

  // Lights at several distances: street lamps and signals up high, traffic lower, many small ones far away.
  col += light_layer(p, vec2f(0.17, 0.12), 0.5, light_power * 1.8, 0.0055, 0.1, 0.62, S.seed + 1);
  col += light_layer(p, vec2f(0.1, 0.075), 0.5, light_power * 1.4, 0.0042, 0.28, 0.9, S.seed + 2);
  col += light_layer(p, vec2f(0.06, 0.05), 0.5, light_power * 1.0, 0.0032, 0.4, 0.86, S.seed + 3);
  col += light_layer(p, vec2f(0.035, 0.03), 0.4, light_power * 0.7, 0.0023, 0.45, 0.8, S.seed + 4);
  return vec4f(col, 1.0);
}

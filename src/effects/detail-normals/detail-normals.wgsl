#include "fullscreen.wgsl"
#include "normals.wgsl"
#include "fbm.wgsl"
#include "color.wgsl"
#include "math.wgsl"
#include "params.wgsl"

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var normal_map: texture_2d<f32>;
@group(0) @binding(2) var normal_samp: sampler;

// A camera on a horizontal plane of water, looking toward the horizon.
const EYE_HEIGHT: f32 = 1.6;
const PITCH: f32 = 0.13;
const TAN_HALF_FOV: f32 = 0.62;

// Per layer: rotation of the tile grid (so the layers' seams never line up), scroll direction
// offset (in units of `spread`), scroll speed (tiles per second), and weight.
const LAYER_ROTATION = array<f32, 3>(0.0, 0.62, 1.31);
const LAYER_HEADING = array<f32, 3>(0.0, 1.0, -1.0);
const LAYER_SPEED = array<f32, 3>(0.035, 0.052, 0.081);
const LAYER_WEIGHT = array<f32, 3>(1.0, 0.8, 0.6);

// Blend ids, in the order of the `blend` parameter.
const BLEND_WHITEOUT: i32 = 0;
const BLEND_RNM: i32 = 1;
const BLEND_UDN: i32 = 2;
const BLEND_PD: i32 = 3;
const BLEND_LINEAR: i32 = 4;

// One layer's tangent-space normal, expressed in WORLD xz axes so layers with different tile
// rotations can be blended: the stored xy are in the texture's own axes and must be turned back.
fn layer_normal(q: vec2f, i: i32) -> vec3f {
  let scale = params.scale * pow(params.ratio, f32(i));
  let heading = 0.45 + LAYER_HEADING[i] * radians(params.spread);
  let drift = vec2f(cos(heading), sin(heading)) * (params.phase * LAYER_SPEED[i]);
  let rotation = LAYER_ROTATION[i];
  // Water appears to move along `heading`, so the lookup position moves against it.
  let uv = rotate_2d(q * scale - drift, rotation);
  var n: vec3f;
  if (params.source == 0u) {
    n = normal_decode_signed(textureSample(normal_map, normal_samp, uv));
  } else {
    // The same look computed per pixel: five octaves of analytic-gradient noise, at the baked
    // map's base frequency (6 cells per tile). This is the cost the texture saves.
    let g = fbm_2d_d(uv * 6.0, 5, 2.0, 0.5);
    n = normal_from_gradient(g.yz * 0.2 * 1.4, 1.0);
  }
  return normal_scale(normal_yaw(n, -rotation), params.strength * LAYER_WEIGHT[i]);
}

fn blend_two(a: vec3f, b: vec3f, mode: i32) -> vec3f {
  switch (mode) {
    case BLEND_WHITEOUT: { return normal_blend_whiteout(a, b); }
    case BLEND_RNM: { return normal_blend_rnm(a, b); }
    case BLEND_UDN: { return normal_blend_udn(a, b); }
    case BLEND_PD: { return normal_blend_pd(a, b); }
    default: { return normal_blend_linear(a, b); }
  }
}

// The first layer is the base; each further one is detail laid on top.
fn combine(l0: vec3f, l1: vec3f, l2: vec3f, mode: i32) -> vec3f {
  var n = l0;
  if (params.layers >= 2) { n = blend_two(n, l1, mode); }
  if (params.layers >= 3) { n = blend_two(n, l2, mode); }
  return n;
}

// ---- a small local lighting model: just enough to make the normals readable --------------------

fn sky(r: vec3f, sun: vec3f) -> vec3f {
  let horizon = vec3f(0.42, 0.58, 0.76);
  let zenith = vec3f(0.06, 0.20, 0.52);
  var c = mix(horizon, zenith, pow(clamp(r.y, 0.0, 1.0), 0.45));
  let s = max(dot(r, sun), 0.0);
  c += vec3f(1.0, 0.82, 0.55) * (pow(s, 1200.0) * 30.0 + pow(s, 40.0) * 0.12);
  return c;
}

fn shade(n_world: vec3f, dir: vec3f, sun: vec3f, dist: f32) -> vec3f {
  var r = reflect(dir, n_world);
  r.y = abs(r.y);
  let ndv = clamp(dot(n_world, -dir), 0.0, 1.0);
  let fresnel = 0.02 + 0.98 * pow(1.0 - ndv, 5.0);
  let body = params.water * (0.5 + 0.5 * clamp(n_world.y, 0.0, 1.0));
  let lit = mix(body, sky(r, sun), fresnel);
  // Distance fog toward the horizon colour.
  let fog = 1.0 - exp(-dist * 0.016);
  return mix(lit, sky(vec3f(0.0, 0.0, 1.0), sun), fog);
}

@fragment fn fs(in: FullscreenVarying) -> @location(0) vec4f {
  let ndc = vec2f((in.uv.x * 2.0 - 1.0) * params.aspect, 1.0 - in.uv.y * 2.0);
  var dir = normalize(vec3f(ndc * TAN_HALF_FOV, 1.0));
  dir = vec3f(dir.x, dir.y * cos(PITCH) - dir.z * sin(PITCH), dir.y * sin(PITCH) + dir.z * cos(PITCH));
  let sun_elevation = radians(params.sun_height);
  let sun = normalize(vec3f(0.0, sin(sun_elevation), cos(sun_elevation)));

  // Intersect the plane y = 0. Sky pixels are computed too (with a clamped distance) so every
  // texture sample below is in uniform control flow and gets proper derivatives.
  let dist = EYE_HEIGHT / max(-dir.y, 2e-3);
  // The top-down debug view looks straight down at four world units of water instead.
  let top_down = vec2f((in.uv.x - 0.5) * params.aspect, in.uv.y - 0.5) * 4.0;
  let q = select((dir * dist).xz, top_down, params.debug == 2u);

  // `layers` is a uniform, so skipping a layer here keeps the sampling in uniform control flow.
  let flat = vec3f(0.0, 0.0, 1.0);
  var l1 = flat;
  var l2 = flat;
  let l0 = layer_normal(q, 0);
  if (params.layers >= 2) { l1 = layer_normal(q, 1); }
  if (params.layers >= 3) { l2 = layer_normal(q, 2); }

  // The left half of the compare view always uses the naive linear blend.
  let naive = params.compare != 0u && in.uv.x < 0.5;
  let mode = select(params.blend, BLEND_LINEAR, naive);
  let n = combine(l0, l1, l2, mode);

  var color: vec3f;
  if (params.debug == 2u) {
    return vec4f(normal_encode_unorm(n), 1.0);
  } else if (dir.y >= -2e-3) {
    color = sky(dir, sun);
  } else if (params.debug == 1u) {
    return vec4f(normal_encode_unorm(n), 1.0);
  } else {
    color = shade(normal_to_world_up(n), dir, sun, dist);
  }
  if (params.compare != 0u && abs(in.uv.x - 0.5) < 0.0012) { color = vec3f(1.0); }
  return vec4f(linear_to_srgb(tonemap_aces(color)), 1.0);
}

// liquid-2d composite: threshold the particle field into a water body and shade it. Three entry points:
//
//   fs           the finished liquid (view 'final')
//   fs_field     the raw field as a grey ramp with the threshold iso-line marked (view 'field')
//   fs_backdrop  only the backdrop, or transparent without one (the first half of view 'discs')
//
// Everything is in target pixels. uv has its origin top-left, like the field texture (row 0 = top), so the
// field, the backdrop and the target all share one uv. Normals and slopes are in the SCREEN frame of
// refraction.wgsl (x right, y DOWN, z toward the viewer); lighting converts to y up.
//
// Coverage. F = field.r is a kernel-smoothed density; the half-density iso-line F = threshold is where a
// free surface sits. The edge is antialiased in SCREEN space: with g = |grad F| per pixel (central
// differences of the bilinear field) the first-order signed distance to the iso-line is (F - T) / g pixels,
// and coverage is a smoothstep of that distance over `edge_softness` pixels. So the edge is as sharp as a
// pixel allows wherever the field is steep and equally soft where it is shallow. No dpdx, so it is legal in
// non-uniform control flow.
//
// Thickness. The distance to the edge saturates quickly in a kernel-smoothed field, so the thickness proxy
// comes from a WIDE map: the field box-reduced to a quarter size and blurred with sigma ~24 px at the 560 px
// reference height (wide.wgsl plus a Blur). Its value W is the fraction of liquid within a few sigma: at
// depth d inside a straight edge it is Phi(d / sigma), 1/2 on the edge, so 2 W - 1 clamped to [0, 1] is 0 at
// the edge and saturates at about 2 sigma deep, and it is 0 for droplets and thin necks. A short ramp of the
// fine field itself (smoothstep(T, 1, F)) adds the first few pixels. The gradient of W, times sigma, is the
// slope of a broad dome that the fine field gradient lacks. Both are smooth: no tap pattern, no banding.

#include "fullscreen.wgsl"
#include "math.wgsl"
#include "color.wgsl"
#include "absorption.wgsl"
#include "refraction.wgsl"
#include "fresnel.wgsl"
#include "specular.wgsl"
#include "foam.wgsl"
#include "liquid2d_params.wgsl"

@group(0) @binding(0) var<uniform> P: Liquid2dParams;
@group(0) @binding(1) var field_tex: texture_2d<f32>;
@group(0) @binding(2) var backdrop_tex: texture_2d<f32>;
@group(0) @binding(3) var samp: sampler;
@group(0) @binding(4) var wide_tex: texture_2d<f32>;

// Look constants (the params expose the ones worth tuning).
const K_FINE: f32 = 1.6;          // field gradient (per reference pixel) to normal slope
const K_BROAD: f32 = 2.5;         // wide-map gradient (times sigma) to normal slope
const LIGHT_ELEVATION: f32 = 0.8; // radians above the screen plane
const SPEC_ROUGHNESS: f32 = 0.3;
const SPEC_SUN_RADIUS: f32 = 0.1;
const NO_BACKDROP_BED: vec3f = vec3f(0.72, 0.86, 0.95);
const SKY: vec3f = vec3f(0.62, 0.78, 0.98);
const FOAM_COLOR: vec3f = vec3f(0.92, 0.96, 1.0);

fn encode(rgb: vec3f) -> vec3f {
  return select(rgb, linear_to_srgb(rgb), P.encode_srgb != 0u);
}

fn field(uv: vec2f) -> vec4f {
  return textureSampleLevel(field_tex, samp, uv, 0.0);
}

fn backdrop_at(uv: vec2f) -> vec3f {
  return select(vec3f(0.0), textureSampleLevel(backdrop_tex, samp, uv, 0.0).rgb, P.has_backdrop != 0u);
}

struct Shaded {
  // Antialiased coverage of the liquid body, 0 outside.
  cov: f32,
  // Liquid colour (linear) to blend over a backdrop with `cov`.
  over: vec3f,
  // Straight (unpremultiplied) colour and opacity before coverage, for the see-through case.
  straight: vec3f,
  opacity: f32,
}

fn shade(uv: vec2f) -> Shaded {
  var s: Shaded;
  s.cov = 0.0;
  s.over = vec3f(0.0);
  s.straight = vec3f(0.0);
  s.opacity = 0.0;

  let f0 = field(uv);
  let big_f = f0.r;
  if (big_f < P.threshold * 0.1) { return s; } // far outside: nothing here can be covered

  let px = 1.0 / P.out_size;
  let gx = field(uv + vec2f(px.x, 0.0)).r - field(uv - vec2f(px.x, 0.0)).r;
  let gy = field(uv + vec2f(0.0, px.y)).r - field(uv - vec2f(0.0, px.y)).r;
  let grad = 0.5 * vec2f(gx, gy); // per target pixel, y down, points into the liquid
  let g = max(length(grad), 1e-4);
  let dist = (big_f - P.threshold) / g; // signed distance to the iso-line in pixels, positive inside
  let t = clamp(dist / max(P.edge_softness, 0.2) + 0.5, 0.0, 1.0);
  let cov = t * t * (3.0 - 2.0 * t);
  if (cov <= 0.0) { return s; }

  // Thickness and the broad dome, from the wide map.
  let wide_px = 1.0 / P.wide_size;
  let w0 = textureSampleLevel(wide_tex, samp, uv, 0.0).r;
  let wgx = textureSampleLevel(wide_tex, samp, uv + vec2f(wide_px.x, 0.0), 0.0).r -
    textureSampleLevel(wide_tex, samp, uv - vec2f(wide_px.x, 0.0), 0.0).r;
  let wgy = textureSampleLevel(wide_tex, samp, uv + vec2f(0.0, wide_px.y), 0.0).r -
    textureSampleLevel(wide_tex, samp, uv - vec2f(0.0, wide_px.y), 0.0).r;
  // Per target pixel, times sigma in pixels: dimensionless, 0.4 at a straight edge, y down.
  let broad = vec2f(wgx, wgy) / (2.0 * wide_px * P.out_size) * P.wide_sigma;
  let thick = clamp(0.3 * smoothstep(P.threshold, 1.0, big_f) + 0.7 * clamp(2.0 * w0 - 1.0, 0.0, 1.0), 0.0, 1.0);

  // Normal: the surface height rises into the liquid, so the normal leans out of it.
  let slope = -(grad * (P.res * K_FINE) + broad * K_BROAD);
  let n = normalize(vec3f(slope, 1.0));
  let n_up = vec3f(n.x, -n.y, n.z);

  // Light.
  let la = radians(P.light_angle);
  let l = vec3f(cos(la) * cos(LIGHT_ELEVATION), sin(la) * cos(LIGHT_ELEVATION), sin(LIGHT_ELEVATION));
  let nl = max(dot(n_up, l), 0.0);

  // Refracted bed.
  var bed = NO_BACKDROP_BED;
  if (P.has_backdrop != 0u) {
    let aspect = P.out_size.x / P.out_size.y;
    let offset = refraction_offset_cheap(n, P.ior, P.refraction * (0.2 + 0.8 * thick));
    bed = refraction_sample(backdrop_tex, samp, uv, refraction_to_uv(offset, aspect), 0.0, 0.05);
  }

  // Body colour by thickness.
  let water = absorb_water(P.water_type);
  let path = P.depth_scale * thick;
  var body = absorb_composite(bed, water.sigma, water.turbidity, water.scatter, path);
  body = body * (0.82 + 0.3 * nl);

  // Foam: the mean foam weight around here, broken up by a cheap cellular pattern.
  var foam_amt = 0.0;
  if (P.foam > 0.0) {
    let ratio = clamp(f0.g / max(f0.r, 0.05), 0.0, 1.0);
    if (ratio > 0.02) {
      // Soft bubbles and a low-frequency patchiness, not the cell walls of foam_texture_lite (which read as cracks).
      let fp = uv * P.out_size / (5.0 * P.res);
      let bubbles = 1.0 - smoothstep(0.15, 0.7, voronoi_2d_anim(fp, 1.0, 0.0).f1);
      let patches = 0.5 + 0.5 * gnoise_2d(fp * 0.45 + vec2f(3.1, 7.7));
      let cells = clamp(0.55 * bubbles + 0.6 * patches, 0.0, 1.0);
      foam_amt = P.foam * smoothstep(0.25, 0.8, ratio * (0.45 + 0.9 * cells));
    }
  }

  // Light added on top: specular glint, fresnel reflection of the sky, a thin meniscus line.
  let spec = spec_sun(n_up, vec3f(0.0, 0.0, 1.0), l, SPEC_ROUGHNESS, SPEC_SUN_RADIUS) * P.highlight;
  let reflection = SKY * (fresnel_water(n.z) * P.rim * 2.5);
  let facing = 0.6 + 0.4 * clamp(dot(n_up.xy, l.xy) / max(length(n_up.xy), 1e-3), -1.0, 1.0);
  let line_d = (dist - 1.2) / 0.7;
  let line = exp(-0.5 * line_d * line_d) * facing * P.rim * 0.45;
  var extra = vec3f(spec) + reflection + vec3f(line);

  let lit_foam = FOAM_COLOR * (0.75 + 0.25 * nl);
  body = mix(body, lit_foam, foam_amt * 0.9);
  extra = extra * (1.0 - foam_amt * 0.6);

  s.cov = cov;
  s.over = body + extra;

  // See-through case: opacity higher in the deep body, lower at thin edges and droplets.
  let body_opacity = clamp(
    P.opacity * mix(P.thin_opacity, 1.0, smoothstep(0.0, 0.35, thick)) * (1.0 - foam_amt) + foam_amt,
    0.0, 1.0);
  let premul = body * body_opacity + extra;
  s.opacity = clamp(body_opacity + max(extra.x, max(extra.y, extra.z)) * (1.0 - body_opacity), 0.0, 1.0);
  s.straight = clamp(premul / max(s.opacity, 1e-4), vec3f(0.0), vec3f(1.0));
  return s;
}

@fragment
fn fs(in: FullscreenVarying) -> @location(0) vec4f {
  let s = shade(in.uv);
  if (P.has_backdrop != 0u) {
    let bd = backdrop_at(in.uv);
    return vec4f(encode(clamp(mix(bd, s.over, s.cov), vec3f(0.0), vec3f(1.0))), 1.0);
  }
  let a = clamp(s.cov * s.opacity, 0.0, 1.0);
  return vec4f(encode(s.straight) * a, a);
}

@fragment
fn fs_backdrop(in: FullscreenVarying) -> @location(0) vec4f {
  if (P.has_backdrop != 0u) {
    return vec4f(encode(clamp(backdrop_at(in.uv), vec3f(0.0), vec3f(1.0))), 1.0);
  }
  return vec4f(0.0);
}

@fragment
fn fs_field(in: FullscreenVarying) -> @location(0) vec4f {
  let uv = in.uv;
  let f = field(uv);
  let px = 1.0 / P.out_size;
  let gx = field(uv + vec2f(px.x, 0.0)).r - field(uv - vec2f(px.x, 0.0)).r;
  let gy = field(uv + vec2f(0.0, px.y)).r - field(uv - vec2f(0.0, px.y)).r;
  let g = max(0.5 * length(vec2f(gx, gy)), 1e-4);
  let dist = (f.r - P.threshold) / g;
  let foam_ratio = clamp(f.g / max(f.r, 0.05), 0.0, 1.0) * step(0.05, f.r);
  var c = vec3f(clamp(f.r, 0.0, 1.0));
  c = mix(c, vec3f(0.2, 0.3, 1.0), 0.5 * foam_ratio);
  let iso = 1.0 - smoothstep(0.3, 1.0, abs(dist));
  c = mix(c, vec3f(1.0, 0.2, 0.08), iso * 0.9);
  // Over-dense regions (more than 30% above rest density) get a green tint.
  c = mix(c, vec3f(0.2, 0.9, 0.3), 0.4 * clamp((f.r - 1.3) * 2.0, 0.0, 1.0));
  return vec4f(c, 1.0);
}

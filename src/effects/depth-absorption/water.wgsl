#include "fullscreen.wgsl"
#include "noise.wgsl"
#include "color.wgsl"
#include "absorption.wgsl"
#include "view_params.wgsl"

@group(0) @binding(0) var<uniform> P: ViewParams;
@group(0) @binding(1) var bed_tex: texture_2d<f32>;
@group(0) @binding(2) var bed_samp: sampler;

const WATER_IOR: f32 = 1.333;
const VIEW_FINAL: u32 = 0u;
const VIEW_DEPTH: u32 = 1u;
const VIEW_TRANSMITTANCE: u32 = 2u;
const RAMP_TOP: f32 = 0.925;

struct Shaded {
  colour: vec3f,
  transmittance: vec3f,
}

// The whole water model for one pixel: sunlight goes down through the water to the bed, the bed
// reflects it, and the view ray comes back up through the same water picking up scattered light.
fn shade(bed: vec3f, depth_m: f32, caustic: f32, w: AbsorbWater, turbidity: f32) -> Shaded {
  let extinction = w.sigma + vec3f(turbidity);
  let sun_path = absorb_path_length(depth_m, P.cos_sun, WATER_IOR);
  let view_path = absorb_path_length(depth_m, P.cos_view, WATER_IOR);
  // Caustic light focuses and washes out within a few metres of the surface.
  let focus = mix(1.0, caustic, exp(-depth_m * 0.3));
  let lit_bed = bed * P.sun * focus * absorb_transmittance(extinction, sun_path);
  let colour = absorb_composite(lit_bed, w.sigma, turbidity, w.scatter * P.sun, view_path);
  return Shaded(colour, absorb_transmittance(extinction, view_path));
}

@fragment fn fs(in: FullscreenVarying) -> @location(0) vec4f {
  let uv = in.uv;
  let aspect = P.aspect;
  let q = vec2f(uv.x * aspect, uv.y);
  let w = absorb_water(P.water_type);
  let turbidity = w.turbidity * P.turbidity_scale;

  // Tilted view: the bed point we see lies further "up the screen" the deeper it is. Two fixed-point
  // steps are enough because the bed depth varies slowly.
  var bed_uv = uv;
  for (var i = 0; i < 2; i++) {
    let depth_n = textureSampleLevel(bed_tex, bed_samp, bed_uv, 0.0).a;
    bed_uv = uv + vec2f(0.0, P.parallax * depth_n);
  }
  let baked = textureSampleLevel(bed_tex, bed_samp, bed_uv, 0.0);
  let depth_m = baked.a * P.depth_scale;

  let ripple = pow(1.0 - abs(gnoise_3d(vec3f(q * 2.4, P.time * 0.5))), 3.0);
  let caustic = 1.0 + P.ripple * (ripple * 3.0 - 0.6);

  var s = shade(baked.rgb, depth_m, caustic, w, turbidity);
  var colour = s.colour;

  // Sky reflection on the surface (Schlick, with the view cosine): invisible from above, strong at a slant.
  let fresnel = 0.02 + 0.98 * pow(1.0 - P.cos_view, 5.0);
  colour += vec3f(0.35, 0.5, 0.75) * fresnel * (0.7 + 0.6 * ripple);

  // Derivatives need uniform control flow, so take them before any per-pixel branch.
  let step_m = max(P.depth_scale / 10.0, 1e-4);
  let contour_x = depth_m / step_m;
  let contour_w = fwidth(contour_x);
  let tick_x = uv.x * 10.0;
  let tick_w = fwidth(tick_x);
  let edge_w = fwidth(uv.y);

  if (P.view == VIEW_DEPTH) {
    let t = baked.a;
    // Shallow yellow, mid teal, deep violet: monotonic in lightness, so it reads in greyscale too.
    let shallow_c = vec3f(0.96, 0.82, 0.25);
    let mid_c = vec3f(0.1, 0.55, 0.5);
    let deep_c = vec3f(0.15, 0.05, 0.35);
    var heat = mix(mix(shallow_c, mid_c, smoothstep(0.0, 0.5, t)), deep_c, smoothstep(0.4, 1.0, t));
    let f = fract(contour_x);
    let line = smoothstep(0.0, 1.5 * contour_w, min(f, 1.0 - f));
    heat = select(heat, vec3f(0.12, 0.12, 0.14), baked.a < 0.002);
    colour = heat * mix(0.25, 1.0, line);
  } else if (P.view == VIEW_TRANSMITTANCE) {
    colour = s.transmittance;
  }

  // Depth ramp: what a white tile looks like at every depth from 0 to the deepest point, with ticks
  // at each tenth. Same shading code as the bed, so the strip and the basin always agree.
  if (P.ramp > 0u && uv.y > RAMP_TOP) {
    let ramp_depth = uv.x * P.depth_scale;
    let strip = shade(vec3f(0.85), ramp_depth, 1.0, w, turbidity);
    let tick = 1.0 - smoothstep(0.0, 2.0 * tick_w, min(fract(tick_x), 1.0 - fract(tick_x)));
    let top_edge = 1.0 - smoothstep(0.0, 2.0 * edge_w, uv.y - RAMP_TOP);
    colour = mix(strip.colour, vec3f(0.02), max(tick * step(uv.y, RAMP_TOP + 0.025), top_edge));
  }

  return vec4f(linear_to_srgb(clamp(colour, vec3f(0.0), vec3f(1.0))), 1.0);
}

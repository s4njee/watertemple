// display.wgsl -- draws the fluid-obstacles effect: the dye over a dark or light ground, the solid shapes
// crisp from their signed distance (not from the blocky mask the solver uses), or a debug view of the
// solver's fields.
//
// THE SHAPES. The solver sees each shape as the cells whose centre is inside it: a staircase. Here every
// shape is evaluated per pixel from the same table (obstacle-shapes.wgsl) and drawn anti-aliased, grown by
// P.skin (about half a cell) so that it covers the staircase and no halo of dye-free water shows
// between the true outline and the cells the solver blocked. Shapes from page elements (index >=
// P.dom_start) are drawn as plain ground: the real element is on top of the canvas.
//
// DYE: as fluid-dye. Emissive (rgb = colour * amount) on a dark ground, shaded through a soft shoulder;
// subtractive (optical density) on a light ground.

#include "fullscreen.wgsl"
#include "color.wgsl"
#include "sdf.wgsl"
#include "fluid.wgsl"
#include "obstacle-shapes.wgsl"
#include "obstacle_table.wgsl"
#include "display_params.wgsl"

@group(0) @binding(0) var<uniform> P: DisplayParams;
@group(0) @binding(1) var dye_tex: texture_2d<f32>;
@group(0) @binding(2) var samp: sampler;
@group(0) @binding(3) var fields_tex: texture_2d<f32>;
@group(0) @binding(4) var mask_tex: texture_2d<f32>;
@group(0) @binding(5) var<uniform> T: ObstacleTable;

const VIEW_DYE: u32 = 0u;
const VIEW_VELOCITY: u32 = 1u;
const VIEW_VORTICITY: u32 = 2u;
const VIEW_PRESSURE: u32 = 3u;
const VIEW_MASK: u32 = 4u;

const LIGHT: vec2f = vec2f(-0.6, -0.8);

fn emissive(d: vec4f) -> vec3f {
  let e = max(d.rgb, vec3f(0.0)) * P.gain;
  let peak = max(e.r, max(e.g, e.b));
  let mapped = e * ((1.0 - exp(-peak)) / max(peak, 1e-4));
  let core = smoothstep(1.2, 4.0, peak);
  return (mapped + vec3f(core * 0.18)) * smoothstep(0.01, 0.1, peak);
}

fn subtractive(d: vec4f) -> vec3f {
  let a = max(d.rgb, vec3f(0.0)) * P.gain;
  let toe = smoothstep(0.004, 0.06, max(a.r, max(a.g, a.b)));
  return P.ground * fluid_transmittance(a * toe);
}

// Smallest distance over the analytic shapes (x) and over the page elements (y), image heights.
fn scene_sd(p: vec2f) -> vec2f {
  var da = 1e9;
  var dd = 1e9;
  for (var i = 0u; i < T.count; i++) {
    let d = obstacle_sd(T.data[3u * i], T.data[3u * i + 1u], p);
    if (i < P.dom_start) {
      da = min(da, d);
    } else {
      dd = min(dd, d);
    }
  }
  return vec2f(da, dd);
}

@fragment fn fs(in: FullscreenVarying) -> @location(0) vec4f {
  let uv = in.uv;
  let p = uv * vec2f(P.aspect, 1.0);
  let d = scene_sd(p);
  let aa = P.pixel;
  var colour = P.ground;

  if (P.view == VIEW_DYE) {
    let s = textureSampleLevel(dye_tex, samp, uv, 0.0);
    colour = select(P.ground + emissive(s), subtractive(s), P.mode == 1u);
  } else if (P.view == VIEW_MASK) {
    let cell = textureSampleLevel(mask_tex, samp, uv, 0.0).r;
    let edge = 1.0 - smoothstep(0.0, 1.5 * aa, abs(min(d.x, d.y)));
    colour = vec3f(0.02) + vec3f(0.9, 0.15, 0.1) * cell + vec3f(0.1, 1.0, 0.3) * edge;
    return vec4f(linear_to_srgb(clamp(colour, vec3f(0.0), vec3f(1.0))), 1.0);
  } else {
    let s = fluid_sample_fields(fields_tex, samp, uv);
    var debug = vec3f(0.0);
    if (P.view == VIEW_VELOCITY) {
      debug = fluid_flow_color(fluid_velocity(s), 1.2 * P.debug_gain);
    } else if (P.view == VIEW_VORTICITY) {
      debug = fluid_diverging(fluid_vorticity(s) * 0.05 * P.debug_gain);
    } else {
      debug = fluid_diverging(fluid_pressure(s) * 4.0 * P.debug_gain);
    }
    colour = vec3f(0.012) + debug;
  }

  // The shapes. The derivative is taken here, in uniform control flow, for the bevel below.
  let g = vec2f(dpdx(d.x), dpdy(d.x));
  let solid = sd_coverage(d.x - P.skin, aa);
  if (solid > 0.0) {
    // A bevel: the outline's own normal from the screen-space gradient of the distance, a rim light and
    // a darker core.
    let n = g / max(length(g), 1e-6);
    let depth = clamp(-(d.x - P.skin) / 0.02, 0.0, 1.0);
    let lit = 0.5 + 0.5 * dot(n, LIGHT);
    let body = mix(P.body_edge, P.body_core, smoothstep(0.0, 1.0, depth));
    let shaded = body * (0.85 + 0.35 * lit * (1.0 - depth));
    colour = mix(colour, shaded, solid);
  }
  let hidden = sd_coverage(d.y - P.skin, aa);
  colour = mix(colour, P.ground * 0.9, hidden);

  let r = length((uv - 0.5) * vec2f(P.aspect, 1.0));
  colour *= 1.0 - P.vignette * smoothstep(0.35, 1.1, r);
  return vec4f(linear_to_srgb(clamp(colour, vec3f(0.0), vec3f(1.0))), 1.0);
}

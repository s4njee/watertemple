// display.wgsl -- draws the fluid-dye effect: the solver's dye over a dark or light ground, with an
// optional liquid sheen, or one of the debug views of the solver's fields. Pure shading: all the physics
// happened in src/gpu/sim/fluid.ts.
//
// TWO WAYS TO MIX COLOUR. The dye texture holds whatever the effect injected (index.ts):
//   dark ground   rgb = colour * amount, an EMISSIVE dye. Light adds, so it glows on a dark page. Mixing
//                 is hue-preserving: the sum is scaled so its brightest channel goes through a soft
//                 shoulder 1 - exp(-peak), which keeps saturation where plain clipping would wash every
//                 bright blob out to white or, worse, to a grey mix of complementary colours.
//   light ground  rgb = optical density (absorbance), a SUBTRACTIVE ink. Density adds in the solver
//                 (so advection and blur average the density, which is the right thing for pigment) and
//                 the page is multiplied by exp(-density): blue over yellow is green, thin ink fades
//                 to the paper, and nothing ever turns to mud.
// a holds the amount in both cases; the sheen reads its gradient.
#include "fullscreen.wgsl"
#include "fluid.wgsl"
#include "color.wgsl"
#include "display_params.wgsl"

@group(0) @binding(0) var<uniform> P: DisplayParams;
@group(0) @binding(1) var dye_tex: texture_2d<f32>;
@group(0) @binding(2) var samp: sampler;
@group(0) @binding(3) var fields_tex: texture_2d<f32>;
@group(0) @binding(4) var div_tex: texture_2d<f32>;

const VIEW_DYE: u32 = 0u;
const VIEW_VELOCITY: u32 = 1u;
const VIEW_PRESSURE: u32 = 2u;
const VIEW_DIVERGENCE: u32 = 3u;
const VIEW_VORTICITY: u32 = 4u;

const LIGHT: vec3f = vec3f(-0.45, -0.55, 0.7);

fn emissive(d: vec4f) -> vec3f {
  let e = max(d.rgb, vec3f(0.0)) * P.gain;
  let peak = max(e.r, max(e.g, e.b));
  // Hue-preserving soft shoulder, then a little bleed to white in the very brightest cores.
  let mapped = e * ((1.0 - exp(-peak)) / max(peak, 1e-4));
  let core = smoothstep(1.2, 4.0, peak);
  // A toe: the faintest dye (spread thin by many steps of advection) fades out instead of tinting
  // the whole canvas.
  return (mapped + vec3f(core * 0.18)) * smoothstep(0.01, 0.1, peak);
}

fn subtractive(d: vec4f) -> vec3f {
  let a = max(d.rgb, vec3f(0.0)) * P.gain;
  let toe = smoothstep(0.004, 0.06, max(a.r, max(a.g, a.b)));
  return P.ground * fluid_transmittance(a * toe);
}

@fragment fn fs(in: FullscreenVarying) -> @location(0) vec4f {
  let uv = in.uv;
  var colour = P.ground;

  if (P.view == VIEW_DYE) {
    let d = textureSampleLevel(dye_tex, samp, uv, 0.0);
    colour = select(P.ground + emissive(d), subtractive(d), P.mode == 1u);

    if (P.sheen > 0.0) {
      // The dye's density as a height field: its gradient tilts a normal and a glint catches the
      // slopes, like a film of oil or a skin on the ink. Fades out where there is no dye.
      let g = fluid_dye_gradient(dye_tex, samp, uv, P.dye_texel) * P.dye_scale;
      let n = normalize(vec3f(-g * P.sheen * 0.35, 1.0));
      let l = normalize(LIGHT);
      let h = normalize(l + vec3f(0.0, 0.0, 1.0));
      let glint = pow(max(dot(n, h), 0.0), 48.0) * smoothstep(0.02, 0.35, d.a);
      let shade = dot(n.xy, l.xy) * smoothstep(0.0, 0.4, d.a);
      colour = colour * (1.0 + 0.5 * P.sheen * shade) + vec3f(glint * P.sheen * 0.8);
    }
  } else {
    let s = fluid_sample_fields(fields_tex, samp, uv);
    var debug = vec3f(0.0);
    if (P.view == VIEW_VELOCITY) {
      debug = fluid_flow_color(fluid_velocity(s), 2.0 * P.debug_gain);
    } else if (P.view == VIEW_PRESSURE) {
      debug = fluid_diverging(fluid_pressure(s) * 40.0 * P.debug_gain);
    } else if (P.view == VIEW_VORTICITY) {
      debug = fluid_diverging(fluid_vorticity(s) * 0.04 * P.debug_gain);
    } else {
      let cell = min(vec2i(uv * P.grid), vec2i(P.grid) - vec2i(1));
      debug = fluid_diverging(textureLoad(div_tex, cell, 0).x * 6.0 * P.debug_gain);
    }
    colour = vec3f(0.012) + debug;
  }

  // A touch of vignette so a dark ground has depth and a light one has a paper feel.
  let r = length((uv - 0.5) * vec2f(P.aspect, 1.0));
  colour *= 1.0 - P.vignette * smoothstep(0.35, 1.1, r);
  return vec4f(linear_to_srgb(clamp(colour, vec3f(0.0), vec3f(1.0))), 1.0);
}

// display.wgsl -- draws the ink-water effect: the paper, the pigment in the water and the pigment in the
// paper, composited with Beer-Lambert. Pure shading; the physics is src/gpu/sim/pigment.ts.
//
// LAYERS (sampled with the dye grid's bilinear filter; the grid is coarser than the canvas)
//   dye    suspended pigment, rgb = optical density per channel (the solver's dye texture)
//   stain  deposited pigment, rgb = optical density, a = the shaped wetness of the paper
//   paper  the baked tile, one texel per dye texel: r grain height, g fibre
//
// SUBTRACTIVE COLOUR (light paper). The two layers' densities add, the paper is multiplied by
// exp(-density): optical density is the logarithm of transmittance, so adding densities is multiplying
// the transmittances. Blue over yellow gives green and never brighter than the paper, thin ink fades to
// the paper's own colour, and nothing turns to mud. fluid.wgsl's fluid_transmittance.
//
// LUMINOUS INK (dark paper). A pigment cannot darken a dark sheet, so on dark paper the density is
// read as opacity and the mixed hue shows as a light ink over it: coverage 1 - exp(-density) of the
// colour exp(-density / peak) lifted towards white. This is an opaque ink (gouache, metallic ink),
// not the same physics, and it is the honest way to keep the mixing rule (densities still add).
//
// WHAT THE PAPER ADDS. Granulation: the stain is deeper in the valleys (wash_granulate) as in
// pigment-step.wgsl; the grain and fibres tint the paper; a wet sheet is a little darker and its edge a
// little darker still, and its slopes catch a faint glint. The wetness comes from stain.a, which the
// pigment pass writes already shaped and filterable, so no extra texture is needed.
#include "fullscreen.wgsl"
#include "fluid.wgsl"
#include "wash.wgsl"
#include "sky.wgsl"
#include "color.wgsl"
#include "composite.wgsl"
#include "display_params.wgsl"

@group(0) @binding(0) var<uniform> P: DisplayParams;
@group(0) @binding(1) var dye_tex: texture_2d<f32>;
@group(0) @binding(2) var stain_tex: texture_2d<f32>;
@group(0) @binding(3) var paper_tex: texture_2d<f32>;
@group(0) @binding(4) var fields_tex: texture_2d<f32>;
@group(0) @binding(5) var clamp_samp: sampler;
@group(0) @binding(6) var repeat_samp: sampler;

const VIEW_PAINT: u32 = 0u;
const VIEW_WETNESS: u32 = 1u;
const VIEW_DEPOSITED: u32 = 2u;
const VIEW_SUSPENDED: u32 = 3u;
const VIEW_VELOCITY: u32 = 4u;

const LIGHT: vec3f = vec3f(-0.45, -0.55, 0.7);

fn wetness(uv: vec2f) -> f32 {
  return textureSampleLevel(stain_tex, clamp_samp, uv, 0.0).a;
}

@fragment fn fs(in: FullscreenVarying) -> @location(0) vec4f {
  let uv = in.uv;
  let dye = max(textureSampleLevel(dye_tex, clamp_samp, uv, 0.0).rgb, vec3f(0.0));
  let stain_sample = textureSampleLevel(stain_tex, clamp_samp, uv, 0.0);
  let stain = max(stain_sample.rgb, vec3f(0.0));
  let wet = stain_sample.a;
  let paper = textureSampleLevel(paper_tex, repeat_samp, uv * P.dye_size / P.tile, 0.0).xy;

  // The stain's own density, and how much deeper the paper's valleys make it.
  let dens = min((stain.r + stain.g + stain.b) / 3.0, 1.5);
  let granular = wash_granulate(dens, paper.x, P.gran * 0.6) / max(dens, 1e-4);
  let od_stain = P.strength * stain * granular;
  let od_dye = P.strength * dye;

  // The wet sheet: slightly darker, darker still along its edge, a glint on the slopes.
  let step_uv = 1.5 / P.dye_size;
  let g = vec2f(
    wetness(uv + vec2f(step_uv.x, 0.0)) - wetness(uv - vec2f(step_uv.x, 0.0)),
    wetness(uv + vec2f(0.0, step_uv.y)) - wetness(uv - vec2f(0.0, step_uv.y)),
  );
  let edge = smoothstep(0.04, 0.4, length(g));
  var base = P.paper * (1.0 + P.grain * (0.5 * (paper.x - 0.5) + 0.9 * paper.y) * 0.06);
  base *= 1.0 - P.wet_look * (0.025 * wet + 0.025 * edge);
  let n = normalize(vec3f(-g * P.wet_look * 1.6, 1.0));
  let h = normalize(normalize(LIGHT) + vec3f(0.0, 0.0, 1.0));
  let glint = max(pow(max(dot(n, h), 0.0), 24.0) - pow(h.z, 24.0), 0.0) * wet * P.wet_look;

  var colour = vec3f(0.0);
  if (P.view == VIEW_PAINT) {
    colour = ink_lay(base, od_stain + od_dye, P.mode == 1u) + vec3f(glint * 0.12);
  } else if (P.view == VIEW_WETNESS) {
    colour = mix(base * 0.6, vec3f(0.1, 0.45, 0.95), clamp(wet * P.debug_gain, 0.0, 1.0));
    colour += vec3f(edge * 0.4);
  } else if (P.view == VIEW_DEPOSITED) {
    colour = ink_lay(base, od_stain * P.debug_gain, P.mode == 1u);
  } else if (P.view == VIEW_SUSPENDED) {
    colour = ink_lay(base, od_dye * P.debug_gain, P.mode == 1u);
  } else {
    let s = fluid_sample_fields(fields_tex, clamp_samp, uv);
    colour = vec3f(0.01) + fluid_flow_color(fluid_velocity(s), 2.5 * P.debug_gain);
  }

  // A touch of vignette so the sheet has depth.
  let r = length((uv - 0.5) * vec2f(P.aspect, 1.0));
  colour *= 1.0 - P.vignette * smoothstep(0.35, 1.1, r);
  let dither = sky_dither(in.position.xy) / 255.0;
  return vec4f(clamp(linear_to_srgb(clamp(colour, vec3f(0.0), vec3f(1.0))) + vec3f(dither), vec3f(0.0), vec3f(1.0)), 1.0);
}

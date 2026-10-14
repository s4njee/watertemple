// display.wgsl -- draws the ink-orbit effect: the solver's dye as ink on paper, or the velocity field.
//
// On paper the dye is SUBTRACTIVE: index.ts injects optical density (absorbance) per channel, density
// adds in the solver, and the paper is multiplied by exp(-density). Overlapping colours filter each
// other the way real ink does (blue over yellow is green) and thin ink fades to the paper. On a dark
// ground the dye is emissive instead: rgb is colour times amount and light adds, through a hue-preserving
// shoulder. See src/effects/fluid-dye/display.wgsl, which this is cut down from.
//
// HDR. The canvas may be an extended-range one (index.ts asks; the host decides). Then `P.light` is
// above 1 and the whole picture, paper and ink, is that many times SDR white: the paper is brighter
// than anything the page's own CSS can draw, and the ink, which only ever takes light away from it,
// is as saturated as the display allows. On the dark ground RGB trails emit light with a soft halo,
// and HDR boosts that light above SDR white too. On an SDR canvas `P.light` is 1 and nothing changes.
//
// CUT-OUTS. A page whose elements have no background of their own (they cannot match a paper brighter
// than white) asks for them to be drawn here: inside each element's exact rounded rectangle the
// output is clean paper, with a one pixel edge. The solver keeps ink out of the elements already, but
// only to the nearest cell; this is to the pixel.
#include "fullscreen.wgsl"
#include "fluid.wgsl"
#include "color.wgsl"
#include "sdf.wgsl"
#include "display_params.wgsl"

@group(0) @binding(0) var<uniform> P: DisplayParams;
@group(0) @binding(1) var dye_tex: texture_2d<f32>;
@group(0) @binding(2) var samp: sampler;
@group(0) @binding(3) var fields_tex: texture_2d<f32>;

const LIGHT: vec3f = vec3f(-0.45, -0.55, 0.7);

fn emissive(d: vec4f) -> vec3f {
  let light = max(d.rgb, vec3f(0.0)) * P.gain;
  // Suppress the neutral light accumulated where all three colours overlap.
  // Subtracting the shared component retains the hue while keeping the mix saturated.
  let neutral = min(light.r, min(light.g, light.b));
  let e = light - vec3f(neutral * 0.7);
  let peak = max(e.r, max(e.g, e.b));
  // Leave headroom for the halo and background instead of clipping bright channels to white.
  let mapped = e * (0.75 * (1.0 - exp(-peak / 0.75)) / max(peak, 1e-4));
  return mapped * smoothstep(0.01, 0.1, peak);
}

fn glow(uv: vec2f) -> vec3f {
  // Two soft rings around the stored RGB light; older trails retain their own colours.
  var halo = vec3f(0.0);
  for (var ring = 0; ring < 2; ring++) {
    let radius = select(3.0, 9.0, ring == 1) * P.dye_texel;
    let weight = select(0.035, 0.015, ring == 1);
    halo += weight * (
      emissive(textureSampleLevel(dye_tex, samp, uv + vec2f(radius.x, 0.0), 0.0)) +
      emissive(textureSampleLevel(dye_tex, samp, uv - vec2f(radius.x, 0.0), 0.0)) +
      emissive(textureSampleLevel(dye_tex, samp, uv + vec2f(0.0, radius.y), 0.0)) +
      emissive(textureSampleLevel(dye_tex, samp, uv - vec2f(0.0, radius.y), 0.0))
    );
  }
  return halo;
}

// The most a pile of ink may absorb: of all three channels alike (the part that only darkens), and of
// its strongest channel beyond that (the part that colours).
const INK_GREY: f32 = 1.3;
const INK_DEPTH: f32 = 3.4;

fn subtractive(d: vec4f) -> vec3f {
  let a = max(d.rgb, vec3f(0.0)) * P.gain;
  // A toe: ink spread very thin by many steps of advection leaves the paper clean.
  let toe = smoothstep(0.004, 0.06, max(a.r, max(a.g, a.b)));
  // Trails cross and pile up, and honest subtractive mixing takes a pile of every hue to black. Two
  // soft shoulders keep it ink: the absorbance common to all channels saturates early, the rest on
  // its strongest channel with the hue kept. Thin ink is unchanged; red over green is a deep yellow,
  // not mud.
  let grey = min(a.r, min(a.g, a.b));
  let chroma = a - vec3f(grey);
  let peak = max(chroma.r, max(chroma.g, chroma.b));
  let kept = INK_DEPTH * (1.0 - exp(-peak / INK_DEPTH)) / max(peak, 1e-4);
  let ink = chroma * kept + vec3f(INK_GREY * (1.0 - exp(-grey / INK_GREY)));
  return P.ground * fluid_transmittance(ink * toe);
}

@fragment fn fs(in: FullscreenVarying) -> @location(0) vec4f {
  let uv = in.uv;
  var colour = P.ground;

  if (P.view == 0u) {
    // Colour is stored when the head lays its ink, so existing tails keep their colour.
    let d = textureSampleLevel(dye_tex, samp, uv, 0.0);
    colour = select(P.ground + emissive(d), subtractive(d), P.mode == 1u);
    if (P.mode == 0u) {
      colour += glow(uv);
    }
    if (P.sheen > 0.0) {
      // The ink's density as a height field: a glint on its slopes, like a wet edge.
      let g = fluid_dye_gradient(dye_tex, samp, uv, P.dye_texel) * P.dye_scale;
      let n = normalize(vec3f(-g * P.sheen * 0.35, 1.0));
      let l = normalize(LIGHT);
      let h = normalize(l + vec3f(0.0, 0.0, 1.0));
      let glint = pow(max(dot(n, h), 0.0), 48.0) * smoothstep(0.02, 0.35, d.a);
      let shade = dot(n.xy, l.xy) * smoothstep(0.0, 0.4, d.a);
      colour = colour * (1.0 + 0.5 * P.sheen * shade) + vec3f(glint * P.sheen * 0.8);
    }
  } else {
    // Direction as hue, speed as strength, over the ground.
    let v = fluid_velocity(fluid_sample_fields(fields_tex, samp, uv));
    let strength = 1.0 - exp(-length(v) * 6.0);
    let flow = fluid_flow_color(v, 6.0);
    colour = select(P.ground + flow, mix(P.ground, flow / max(strength, 1e-4), strength), P.mode == 1u);
  }

  if (P.rect_count > 0u) {
    let p = uv * vec2f(P.aspect, 1.0);
    var d = 1e9;
    for (var i = 0u; i < P.rect_count; i++) {
      let b = P.rects[i];
      d = min(d, sd_rounded_box_2d(p - b.xy, b.zw, P.radii[i].x));
    }
    colour = mix(colour, P.ground, clamp(0.5 - d / P.pixel, 0.0, 1.0));
  }

  let r = length((uv - 0.5) * vec2f(P.aspect, 1.0));
  colour *= 1.0 - P.vignette * smoothstep(0.35, 1.1, r);
  // sRGB-encoded even above 1: that is what an extended-range srgb canvas expects.
  return vec4f(linear_to_srgb(clamp(colour, vec3f(0.0), vec3f(1.0)) * P.light), 1.0);
}

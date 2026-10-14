// composite.wgsl -- how a density of pigment shows on paper: the one function the display pass and the
// tests share.
//
//   ink_lay(base: vec3f, od: vec3f, luminous: bool) -> vec3f
//       The colour of paper `base` (linear RGB) under pigment of optical density `od` per channel
//       (>= 0; the sum of every pigment's density, suspended and deposited alike).
//         light paper  (luminous false)   base * exp(-od): Beer-Lambert. Densities add, so
//                      transmittances multiply: blue over yellow is green, equal parts of two inks
//                      are the geometric mean of their colours, and the result is never brighter
//                      than the paper in any channel.
//         dark paper   (luminous true)    a pigment cannot darken a dark sheet, so the density is read
//                      as opacity and the ink glows: the pigment's own hue, exp(-2.4 od / peak)
//                      scaled to its brightest channel 1 (a saturated colour), covers the paper by
//                      (1 - exp(-0.8 peak)), faded in over the faintest densities. An opaque,
//                      luminous ink (gouache, neon, metallic ink); the mixing rule (densities add,
//                      so the hue of a mix is the hue of the summed density) is the same. Never
//                      brighter than 1.
#include "fluid.wgsl"

fn ink_lay(base: vec3f, od: vec3f, luminous: bool) -> vec3f {
  let density = max(od, vec3f(0.0));
  if (!luminous) {
    return base * fluid_transmittance(density);
  }
  let peak = max(density.r, max(density.g, density.b));
  let t = exp(-density * (2.4 / max(peak, 1e-3)));
  let hue = t / max(max(t.r, t.g), max(t.b, 1e-4));
  // A toe: the faintest density (the thin haze round a stroke) has no stable hue and must not draw a
  // coloured outline round it.
  let cover = (1.0 - exp(-peak * 0.8)) * smoothstep(0.03, 0.2, peak);
  return mix(base, hue * 0.95, cover);
}

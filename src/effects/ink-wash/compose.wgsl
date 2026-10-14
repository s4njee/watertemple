// Full-resolution pass: reconstruct the field, threshold it into blooms with pooled edges, add the
// paper, mix the pigments. Everything here is a texture fetch or a few ALU ops; there is no noise.
#include "fullscreen.wgsl"
#include "wash.wgsl"
#include "sky.wgsl"
#include "compose_params.wgsl"

@group(0) @binding(0) var<uniform> wash: ComposeParams;
@group(0) @binding(1) var field_tex: texture_2d<f32>;
@group(0) @binding(2) var paper_tex: texture_2d<f32>;
@group(0) @binding(3) var field_sampler: sampler;
@group(0) @binding(4) var paper_sampler: sampler;

// Signed distance in pixels to a level set of `value`, positive where value > level.
fn inside_px(value: f32, level: f32, pixel_change: f32) -> f32 {
  return (value - level) / max(pixel_change, 0.0007);
}

@fragment fn fs(in: FullscreenVarying) -> @location(0) vec4f {
  let f = wash_sample_bspline(field_tex, field_sampler, in.uv).xyz;
  // The paper repeats every `paper_tile` CSS pixels, so its grain keeps its size at any DPR.
  let paper = textureSampleLevel(paper_tex, paper_sampler, in.position.xy * wash.paper_scale, 0.0).xy;

  // Two blooms: the main pigment where the bloom field is high, a second where the tone field is.
  // The detail field roughens both outlines a little (`ragged`).
  let level_a = f.x + wash.ragged * f.z * 0.35;
  let level_b = f.y + wash.ragged * f.z * 0.5;
  let a_in = inside_px(level_a, wash.threshold, fwidth(level_a));
  let b_in = inside_px(level_b, wash.threshold_b, fwidth(level_b));

  // Edges differ: where the tone field is high the paint went on wet and feathers out, where it is low
  // it went on dry and stays crisp. `soft` is the edge width in pixels; the mottle thins the interior.
  let tone = clamp(0.5 + 0.6 * f.y, 0.0, 1.0);
  let soft_a = wash.edge_soft * mix(0.4, 3.2, tone);
  let soft_b = wash.edge_soft * mix(3.2, 0.4, tone);
  let mottle = 0.8 + 0.2 * clamp(0.5 + 0.9 * f.z, 0.0, 1.0);
  let rim_var = 0.55 + 0.45 * clamp(0.5 + 1.2 * f.z, 0.0, 1.0);

  // A faint wide wash under the blooms ties them together, as a first layer of water and pigment does.
  let wide = wash_body(a_in, 90.0) * 0.16;
  // Pigment is stronger toward the middle of a bloom than at its feathered outline.
  let deep = smoothstep(0.0, 0.45, level_a - wash.threshold);
  var dens_a = wash_body(a_in, soft_a) * mix(0.34, 0.6, deep) * mottle + wide
    + wash_pool(a_in, wash.pool_width, soft_a) * wash.edge * rim_var;
  var dens_b = (wash_body(b_in, soft_b) * 0.55 * mottle
    + wash_pool(b_in, wash.pool_width, soft_b) * wash.edge * rim_var) * 0.8;
  dens_a = wash_granulate(dens_a * wash.contrast, paper.x, wash.granulation);
  dens_b = wash_granulate(dens_b * wash.contrast, paper.x, wash.granulation);
  dens_a = clamp(dens_a, 0.0, 0.92);
  dens_b = clamp(dens_b, 0.0, 0.92);

  let dither = sky_dither(in.position.xy) / 255.0;
  if (wash.transparent != 0u) {
    // Premultiplied "over": pigment B over pigment A, over nothing. Fibres thin the ink a little.
    let fibre = 1.0 - wash.fibre * 0.5 * paper.y;
    let ka = dens_a * fibre;
    let kb = dens_b * fibre;
    let alpha = kb + ka * (1.0 - kb);
    let rgb = wash.ink_b * kb + wash.ink_a * ka * (1.0 - kb);
    let a = clamp(alpha + dither * step(0.0005, alpha), 0.0, 1.0);
    return vec4f(clamp(rgb + vec3f(dither) * a, vec3f(0.0), vec3f(a)), a);
  }

  // Opaque: the paper's own tone varies with its fibres, the pigments are mixed on top of it.
  var color = wash.paper * (1.0 + wash.fibre * (0.5 * (paper.x - 0.5) + 0.9 * paper.y) * 0.06);
  color = mix(color, wash.ink_a, dens_a);
  color = mix(color, wash.ink_b, dens_b);
  return vec4f(clamp(color + vec3f(dither), vec3f(0.0), vec3f(1.0)), 1.0);
}

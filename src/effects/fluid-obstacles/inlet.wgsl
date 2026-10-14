// inlet.wgsl -- the dye of the wind tunnel: a pass for the solver's `dye` hook that writes the whole
// dye texture, copying it everywhere except a band at the left edge where it REPLACES the dye with a
// pattern, so rows of dye are continually fed in and carried downstream (streaklines). Runs once per
// solver step. Dye convention as the fluid-dye effect: rgb = colour * amount (dark ground) or optical
// density (light ground), a = amount.
//
// PATTERNS (P.kind): 0 rows (thin lines, alternating hues), 1 bands (thick, alternating), 2 timelines
// (a full-height line released now and then: P.pulse is 1 on the step it is released; the lines are
// bent by the flow and show its velocity profile and wake), 3 none (clear water in).

#include "fullscreen.wgsl"
#include "color.wgsl"
#include "fluid.wgsl"
#include "inlet_params.wgsl"

@group(0) @binding(0) var<uniform> P: InletParams;
@group(0) @binding(1) var dye: texture_2d<f32>;

fn hue(t: f32) -> vec3f {
  return palette_cosine(t, vec3f(0.5), vec3f(0.5), vec3f(1.0), vec3f(0.0, 0.33, 0.67));
}

fn colour_of(row: f32) -> vec3f {
  // Two families alternate row by row: warm and cool, so neighbouring streaklines never merge to one colour.
  let warm = vec3f(1.0, 0.45, 0.2);
  let cool = vec3f(0.15, 0.7, 1.0);
  return select(cool, warm, (i32(row) & 1) == 0);
}

fn pattern(uv: vec2f) -> vec4f {
  if (P.kind == 3u) {
    return vec4f(0.0);
  }
  if (P.kind == 2u) {
    let c = hue(uv.y * 0.8 + P.phase);
    return select(vec4f(0.0), vec4f(c, 1.0), P.pulse > 0.5);
  }
  let rows = select(P.rows, P.rows * 0.4, P.kind == 1u);
  let y = uv.y * rows;
  let row = floor(y);
  let f = fract(y);
  let half_width = select(0.17, 0.4, P.kind == 1u);
  let m = 1.0 - smoothstep(half_width * 0.6, half_width, abs(f - 0.5));
  let c = colour_of(row);
  return vec4f(c, 1.0) * m;
}

@fragment fn fs(in: FullscreenVarying) -> @location(0) vec4f {
  let t = vec2i(in.position.xy);
  let old = textureLoad(dye, t, 0);
  let uv = in.uv;
  let x = uv.x * P.aspect;
  let w = 1.0 - smoothstep(P.width * 0.5, P.width, x);
  var p = pattern(uv);
  let amount = p.a * P.amount;
  // Dark ground: add light. Light ground: the same colour as optical density, which subtracts.
  let rgb = select(p.rgb, fluid_absorbance(p.rgb) * 0.6, P.mode == 1u) * P.amount;
  return mix(old, vec4f(rgb, amount), w);
}

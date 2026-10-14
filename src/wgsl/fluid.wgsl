// fluid.wgsl -- read the textures of the stable-fluids solver (src/gpu/sim/fluid.ts) without knowing
// how they are laid out, and a few helpers for drawing dye.
//
// CONVENTIONS. `uv` is the canvas uv of the fullscreen pass (0, 0 top-left, y down), and the solver's
// grid covers the canvas exactly, so the same uv addresses every texture below. Velocity is in IMAGE
// HEIGHTS PER SECOND (1.0 crosses the canvas height in a second; a horizontal 1.0 crosses a 16:9
// canvas in 1/1.78 s) with y DOWN. Vorticity is in 1 / s and is positive where the flow turns CLOCKWISE
// on screen (dv/dx > du/dy with y down). Pressure is the solver's velocity potential (see
// fluid-project.wgsl): a relative quantity, in the units of velocity. All of it is linear, unclamped.
//
// THE FIELDS TEXTURE (`fluid.fieldsView`, rgba16float, one texel per cell, filterable, stable between
// updates). The easiest thing to read: one filtered tap gives velocity, vorticity and pressure at
// the cell centres, bilinearly smooth in between. Solid cells are all zero.
//   fluid_velocity(s: vec4f) -> vec2f          velocity from a fields texel
//   fluid_speed(s: vec4f) -> f32               its length
//   fluid_vorticity(s: vec4f) -> f32           vorticity, 1 / s
//   fluid_pressure(s: vec4f) -> f32            pressure potential
//   fluid_sample_fields(tex, samp, uv) -> vec4f   one bilinear tap (needs a filtering sampler)
//
// THE STAGGERED VELOCITY TEXTURE (`fluid.velocity.view`, (W+1) x (H+1) texels, u in .x on the left face
// of each cell, v in .y on its top face), for passes that want the solver's own field rather than the
// cell-centred average:
//   fluid_velocity_at(tex, samp, uv, grid: vec2f) -> vec2f   velocity at uv by two filtered taps;
//                                              `grid` is [width, height] in cells (solver.grid)
//
// THE DYE TEXTURE (`fluid.dye.view`, rgba16float, dyeWidth x dyeHeight, filterable). What the channels
// mean is up to whoever injects the dye; the demo puts colour * amount in rgb and the amount in a.
//   fluid_dye_rgb(s: vec4f) -> vec3f           s.rgb
//   fluid_dye_density(s: vec4f) -> f32         s.a
//   fluid_dye_gradient(tex, samp, uv, texel: vec2f) -> vec2f
//                                              central-difference gradient of the density, per TEXEL
//                                              (texel = [1 / dyeWidth, 1 / dyeHeight]), x right, y down;
//                                              four taps. Feed it to a normal for a liquid sheen.
//
// COLOUR HELPERS
//   fluid_absorbance(colour: vec3f) -> vec3f   the per-channel optical density that makes light of
//                                              the given colour: -ln(colour), colour clamped to
//                                              [0.02, 1]. Inject this (times an amount) to make
//                                              inks that mix SUBTRACTIVELY, like paint
//   fluid_transmittance(absorbance: vec3f) -> vec3f   exp(-absorbance): multiply the page by it
//   fluid_flow_color(v: vec2f, scale: f32) -> vec3f   debug colour of a velocity: hue is direction
//                                              (right = red, down = green-cyan, left = cyan-blue, up = purple),
//                                              brightness 1 - exp(-|v| * scale); linear rgb
//   fluid_diverging(x: f32) -> vec3f           debug colour of a signed quantity in [-1, 1]: blue
//                                              negative, near-black at 0, orange positive; linear rgb
//
// Why subtractive? Adding light-coloured dyes gives grey mud (red + cyan = white) and averaging
// them in advection gives the same mud. Optical density adds, transmittance multiplies, so blue
// over yellow gives green and a dilute ink fades to the paper's own colour, which is how real ink
// behaves. Use additive (emissive) dye for glowing smoke on a dark ground.
//
// Cost: one tap per sample call (fluid_velocity_at two, fluid_dye_gradient four); the helpers are a
// few ALU ops.

#include "color.wgsl"
#include "math.wgsl"

fn fluid_velocity(s: vec4f) -> vec2f {
  return s.xy;
}

fn fluid_speed(s: vec4f) -> f32 {
  return length(s.xy);
}

fn fluid_vorticity(s: vec4f) -> f32 {
  return s.z;
}

fn fluid_pressure(s: vec4f) -> f32 {
  return s.w;
}

fn fluid_sample_fields(tex: texture_2d<f32>, samp: sampler, uv: vec2f) -> vec4f {
  return textureSampleLevel(tex, samp, uv, 0.0);
}

fn fluid_velocity_at(tex: texture_2d<f32>, samp: sampler, uv: vec2f, grid: vec2f) -> vec2f {
  let p = uv * grid;
  let size = grid + vec2f(1.0);
  let u = textureSampleLevel(tex, samp, vec2f(p.x + 0.5, p.y) / size, 0.0).x;
  let v = textureSampleLevel(tex, samp, vec2f(p.x, p.y + 0.5) / size, 0.0).y;
  return vec2f(u, v);
}

fn fluid_dye_rgb(s: vec4f) -> vec3f {
  return s.rgb;
}

fn fluid_dye_density(s: vec4f) -> f32 {
  return s.a;
}

fn fluid_dye_gradient(tex: texture_2d<f32>, samp: sampler, uv: vec2f, texel: vec2f) -> vec2f {
  let r = textureSampleLevel(tex, samp, uv + vec2f(texel.x, 0.0), 0.0).a;
  let l = textureSampleLevel(tex, samp, uv - vec2f(texel.x, 0.0), 0.0).a;
  let d = textureSampleLevel(tex, samp, uv + vec2f(0.0, texel.y), 0.0).a;
  let u = textureSampleLevel(tex, samp, uv - vec2f(0.0, texel.y), 0.0).a;
  return 0.5 * vec2f(r - l, d - u);
}

fn fluid_absorbance(colour: vec3f) -> vec3f {
  return -log(clamp(colour, vec3f(0.02), vec3f(1.0)));
}

fn fluid_transmittance(absorbance: vec3f) -> vec3f {
  return exp(-max(absorbance, vec3f(0.0)));
}

fn fluid_flow_color(v: vec2f, scale: f32) -> vec3f {
  if (dot(v, v) < 1e-12) {
    return vec3f(0.0);
  }
  let t = atan2(v.y, v.x) / TAU;
  let hue = palette_cosine(t, vec3f(0.5), vec3f(0.5), vec3f(1.0), vec3f(0.0, 0.33, 0.67));
  return hue * (1.0 - exp(-length(v) * scale));
}

fn fluid_diverging(x: f32) -> vec3f {
  let t = clamp(x, -1.0, 1.0);
  let cold = vec3f(0.02, 0.2, 0.9);
  let hot = vec3f(1.0, 0.35, 0.02);
  return abs(t) * select(cold, hot, t > 0.0);
}

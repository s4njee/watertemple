// motes.wgsl -- drifting particulates ("marine snow", dust in a sunbeam, plankton): soft dots that
// float, sway and twinkle, drawn procedurally with ONE hash per layer per pixel and no particle
// buffer. Stack two or three layers at different cell sizes and speeds for parallax.
//
// CONVENTIONS
//  * `p` is a screen position in aspect-corrected units (x in [0, aspect], y in [0, 1] growing down).
//  * `cell` is the grid cell size in the same units (0.04 to 0.2): at most one mote per cell, so it
//    sets the largest number of motes (aspect / cell * 1 / cell). `density` in [0, 1] is the share of
//    cells that hold one. `radius` is the mote radius as a share of `cell` (0.02 to 0.3, clamped);
//    the mote is kept inside its cell, which is why one cell lookup is enough (a mote never needs the
//    neighbours' hashes: cost is one pcg_4d per layer, not nine).
//  * `drift` is the layer's velocity in screen units per second (marine snow falls: positive y);
//    `time` in seconds. Use a larger cell, bigger radius and faster drift for the near layer.
//  * Output is a brightness factor in [0, 1]: 1 at the centre of a fully lit mote.
//
// Functions
//   motes_layer(p, time, cell, density, radius, drift, seed) -> f32
//       brightness of one layer at `p`. Each mote has its own size (0.6x to 1.2x radius), sway and
//       twinkle phase; `seed` (any float) gives a different layer. Pure function of its arguments, so
//       deterministic and loopable (everything periodic in `time` apart from the steady drift).
//
// Cost: pcg_4d (about 25 integer ops), one smoothstep and 2 sin/cos. About 60 ALU per layer.
// Pitfalls: the drift scrolls the whole lattice, so a layer never "ends"; `p` far from the origin
// loses precision after a few hundred seconds of drift (wrap `time` yourself in a long-running page).

#include "hash.wgsl"
#include "math.wgsl"

fn motes_layer(
  p: vec2f,
  time: f32,
  cell: f32,
  density: f32,
  radius: f32,
  drift: vec2f,
  seed: f32,
) -> f32 {
  let size = max(cell, 1e-3);
  let q = (p - drift * time) / size;
  let id = vec2i(floor(q));
  let f = fract(q);
  let h = to_unorm_4d(pcg_4d(vec4u(bitcast<vec2u>(id), u32(max(seed, 0.0) * 8.0) + 17u, 0u)));

  let r_cell = clamp(radius, 0.02, 0.3);
  let r = r_cell * (0.6 + 0.6 * h.w);
  // The mote's resting place, with room for its sway and its own radius inside the cell.
  let sway = 0.3 * r_cell;
  let margin = 1.5 * r_cell + sway;
  let rest = vec2f(margin) + (1.0 - 2.0 * margin) * h.yz;
  let phase = h.w * TAU;
  let centre = rest + sway * vec2f(sin(time * 0.7 + phase), cos(time * 0.55 + 2.0 * phase));

  let d = length(f - centre);
  let body = 1.0 - smoothstep(0.35 * r, r, d);
  let twinkle = 0.65 + 0.35 * sin(time * (0.8 + h.y) + phase * 3.0);
  let present = select(0.0, 1.0, h.x < saturate(density));
  return body * twinkle * present;
}

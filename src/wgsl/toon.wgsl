// toon.wgsl -- the building blocks of cel-shaded ("toon") water: hard-edged but anti-aliased steps,
// flat colour bands, outlines, and crisp cell-noise foam lines.
//
// The look comes from refusing smooth gradients: every shape has a clean edge, every colour is flat, and
// the only soft thing is the one pixel at an edge, which exists so the edge does not crawl. That makes
// anti-aliasing the whole game, and these functions all follow one convention:
//
//   `aa` is the width, in the units of the function's first argument, of the ramp that replaces a hard
//   edge. Pass the size of ONE PIXEL in those units (for a flat 2D scene it is a constant, `1 / height_px`;
//   for a shaded value use fwidth(x) in uniform control flow). aa = 0 gives a hard (aliased) edge.
//   The ramp is linear and exactly one pixel wide, which is what makes the average of the covered samples
//   equal the true covered area: lines keep a steady width as they move and never shimmer.
//
// Steps and bands (all return values in [0, 1])
//   toon_step(edge, x, aa) -> f32
//       anti-aliased step: 0 well below `edge`, 1 well above, linear ramp `aa` wide centred on the edge,
//       so toon_step(e, x, aa) is 0.5 at x == e. Its average over a pixel footprint equals the covered
//       fraction. monotonic in x.
//   toon_band(t, bands, aa) -> f32
//       quantises t in [0, 1] into `bands` (clamped to 1..64) flat levels and returns the level as a value in
//       [0, 1]: exactly k / (bands - 1), k = 0..bands-1, away from the band edges (t = k / bands, with `aa`
//       the ramp width in t units). Monotonic. bands = 1 returns 0. t outside [0, 1] clamps to the first or
//       last level. Feed the result to toon_ramp for flat colour bands.
//   toon_posterize(c, bands, aa) -> vec3f      toon_band on each channel of a colour
//   toon_ramp(level, c0, c1, c2, c3) -> vec3f
//       colour for a level in [0, 1]: linear between four stops at 0, 1/3, 2/3, 1. Fed with toon_band the
//       result is flat per band (a band count of 4 lands exactly on the stops; other counts take colours in
//       between, still flat). Stops in linear RGB.
//
// Lines and outlines (distance in any unit, `aa` in the same unit)
//   toon_line(d, width, aa) -> f32
//       coverage of a line `width` wide whose centre is d = 0, for d a distance (or |d|): 1 at the centre,
//       0 beyond width / 2 + aa / 2. Width 0 gives 0.
//   toon_ring(d, inner, outer, aa) -> f32
//       coverage of the band inner <= d <= outer of a SIGNED distance (sdf.wgsl): an outline hugging a
//       shape (inner 0, outer w), a foam ring, a shore line. 0 when outer <= inner.
//
// Cell foam (Voronoi cell borders, the "ocean network" of cartoon water)
//   toon_foam_cells(p, time, jitter) -> f32
//       F = f2 - f1 of an ANIMATED Voronoi (voronoi.wgsl): 0 on cell borders, growing to roughly 1 at cell
//       centres. p in cell units (one cell = 1.0). `time` in radians: the cell seeds orbit, so cells drift
//       and morph instead of sliding. jitter in [0, 1], about 0.8 looks organic.
//   toon_cell_edge(f, f_grad, width_px) -> f32
//       thresholds F into a line that is `width_px` pixels wide whatever the zoom: f / f_grad is the
//       distance to the nearest border in pixels, f_grad = the screen-space gradient length of F, i.e.
//       length(vec2f(dpdx(f), dpdy(f))) (call it in uniform control flow). Returns coverage in [0, 1]
//       with a one pixel ramp. width_px <= 0 gives 0.
//
// Glints (hard-edged highlights)
//   toon_sparkle(p, time, density, size, aa) -> f32
//       four-pointed white star glints at random cell positions that blink in and out with a pulse.
//       p in cell units (one cell = 1), density in [0, 1] the share of cells that hold a star, size the
//       star radius in cell units (up to 0.45, clamped), aa in cell units (one pixel). Coverage in [0, 1].
//       A star never leaves its cell, so one cell lookup is enough.
//
// Cost: toon_step / band / line / ring are 3 to 10 ALU; toon_foam_cells is one animated voronoi (about
// 9 hashes, 2 to 3 gnoise); toon_sparkle is two hashes and a few ALU. No textures, no derivatives inside.

#include "hash.wgsl"
#include "voronoi.wgsl"

fn toon_step(edge: f32, x: f32, aa: f32) -> f32 {
  return clamp((x - edge) / max(aa, 1e-6) + 0.5, 0.0, 1.0);
}

fn toon_band(t: f32, bands: f32, aa: f32) -> f32 {
  let n = clamp(floor(bands), 1.0, 64.0);
  let u = clamp(t, 0.0, 1.0) * n;
  let h = 0.5 * aa * n;                 // half width of the ramp, in band units
  let k = floor(u);
  // Level = (k - 1) + ramp across edge k + ramp across edge k + 1: a staircase whose risers are `aa` wide.
  let lower = toon_step(k, u, 2.0 * h);
  let upper = toon_step(k + 1.0, u, 2.0 * h);
  let level = clamp(k - 1.0 + lower + upper, 0.0, n - 1.0);
  return select(0.0, level / max(n - 1.0, 1.0), n > 1.0);
}

fn toon_posterize(c: vec3f, bands: f32, aa: f32) -> vec3f {
  return vec3f(toon_band(c.x, bands, aa), toon_band(c.y, bands, aa), toon_band(c.z, bands, aa));
}

fn toon_ramp(level: f32, c0: vec3f, c1: vec3f, c2: vec3f, c3: vec3f) -> vec3f {
  let x = clamp(level, 0.0, 1.0) * 3.0;
  let a = mix(c0, c1, clamp(x, 0.0, 1.0));
  let b = mix(c1, c2, clamp(x - 1.0, 0.0, 1.0));
  let c = mix(c2, c3, clamp(x - 2.0, 0.0, 1.0));
  return select(select(a, b, x >= 1.0), c, x >= 2.0);
}

fn toon_line(d: f32, width: f32, aa: f32) -> f32 {
  return select(0.0, 1.0 - toon_step(0.5 * width, abs(d), aa), width > 0.0);
}

fn toon_ring(d: f32, inner: f32, outer: f32, aa: f32) -> f32 {
  let ring = toon_step(inner, d, aa) - toon_step(outer, d, aa);
  return select(0.0, clamp(ring, 0.0, 1.0), outer > inner);
}

fn toon_foam_cells(p: vec2f, time: f32, jitter: f32) -> f32 {
  let v = voronoi_2d_anim(p, jitter, time);
  return v.f2 - v.f1;
}

fn toon_cell_edge(f: f32, f_grad: f32, width_px: f32) -> f32 {
  let distance_px = f / max(f_grad, 1e-4);
  return select(0.0, clamp(0.5 * width_px - distance_px + 0.5, 0.0, 1.0), width_px > 0.0);
}

fn toon_sparkle(p: vec2f, time: f32, density: f32, size: f32, aa: f32) -> f32 {
  let cell = floor(p);
  let h = hash_2to3(cell);
  let size_c = clamp(size, 0.0, 0.45);
  // Centre inside the cell, leaving room for the star; random pulse period and phase per cell.
  let centre = cell + vec2f(0.5) + (h.xy - vec2f(0.5)) * (1.0 - 2.0 * size_c);
  let period = 0.1 + 0.15 * fract(h.y * 17.17);
  let phase = fract(time * period + h.x * 7.31);
  // Blooms and fades within 70% of its cycle and is dark for the rest; zero at both ends, so no popping.
  let tri = clamp(1.0 - abs(phase * 2.0 - 0.7) / 0.7, 0.0, 1.0);
  let pulse = tri * tri * (3.0 - 2.0 * tri);
  let live = select(0.0, 1.0, h.z < density);
  let q = abs(p - centre);
  // Four-pointed star: points along the axes. |x|^0.5 + |y|^0.5 = r^0.5 is the astroid; its inside is the star.
  let r = size_c * pulse;
  let d = sqrt(q.x) + sqrt(q.y) - sqrt(max(r, 1e-6));
  // d is not a distance; convert the pixel ramp with the local slope of the astroid (about 1 / sqrt(r)).
  let aa_d = aa * 0.5 / sqrt(max(r, 1e-3));
  let inside = 1.0 - toon_step(0.0, d, aa_d);
  return select(0.0, live * clamp(inside, 0.0, 1.0), r > 1e-5);
}

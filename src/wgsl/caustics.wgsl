// caustics.wgsl -- procedural caustics: the bright, connected filaments that sunlight draws on a pool
// floor after it has been focused by a wavy surface. Two ways to build the pattern per pixel, plus the
// lookups for a baked caustic texture (the cheap one), all pure functions of position and time.
//
// CONVENTIONS (read once)
//  * TILE COORDINATES. `uv` is in tile units: the pattern repeats EXACTLY every 1.0 in x and in y
//    (f(uv + (1, 0)) == f(uv)), so it can be baked into a seamless texture or repeated. `cells` is
//    the integer number of pattern cells across one tile (a feature is about 1 / cells of a tile), so
//    it sets the feature size: for features about F in size on screen, use uv = world / (cells * F).
//    A bigger `cells` is also a longer period before the pattern repeats. `cells` < 1 counts as 1.
//  * TIME is in radians and every animated procedural function repeats after TAU (time + TAU == time),
//    so a bake of one loop is seamless. Advance it as phase += delta * speed.
//  * OUTPUT is light intensity, finite and >= 0, with a MEAN OF ABOUT 1 over the pattern for ANY
//    `sharp`. Caustics redistribute light rather than add it: turning `sharp` up concentrates the
//    same energy into thinner, brighter lines (peak value of 5 to 30) and leaves the average where it
//    was. Most pixels are therefore below 1 and the filaments far above it: expose with
//    caustics_expose(), or multiply into a lit scene and tone-map. The mean holds to about +-2% for
//    the procedural functions across sharp 0..1 and +-15% for the baked ones (tests/water-light.test.ts).
//  * `sharp` in [0, 1] (clamped): 0 is a broad soft net, 1 is thin hard filaments.
//
// The two per-pixel constructions (compared in src/effects/caustics/notes.md)
//   caustics_cellular(uv, cells, time, sharp) -> f32
//       Warped cellular: F2 - F1 of an animated, sine-warped tileable Voronoi, so the light collects
//       on the cell borders and the cell sites orbit. A polygonal net that reads instantly as caustics;
//       the cell structure shows at low `cells`. 9 cell tests (about 2.5 gnoise) + 4 sin.
//   caustics_sine(uv, cells, time, sharp) -> f32
//       Iterated sine warps, the "water turbulence" construction: four rounds of q = p + warp(q, time),
//       each adding a Lorentzian peak at the zeros of the warped phase. No hashes and no loops over
//       cells: 40 sin/cos and about 100 ALU. Long, smoothly flowing filaments, softer than cellular.
//       Uses cells / 2 phase periods per tile (cells is rounded down to an even number there).
//
//   caustics_eval(method, uv, cells, time, sharp) -> f32     dispatch on a uniform; method is one of
//       CAUSTICS_CELLULAR = 0, CAUSTICS_SINE = 1
//   caustics_stack(method, uv, cells, time, sharp) -> f32    two layers (the second turned 45 degrees,
//       1.41x finer, half a loop out of phase; still period 1) averaged: the mean stays 1 and two nets
//       crossing give the denser, more irregular look of a real pool. Twice the cost of caustics_eval.
//   caustics_rgb(method, layers, uv, cells, time, sharp, split) -> vec3f   chromatic fringing: the pattern
//       (layers = 1: caustics_eval, 2: caustics_stack) sampled at three positions `split` apart (tile
//       units; 0.002 to 0.01 is plenty), red shifted one way, blue the other, green in the middle.
//       Exactly 3x the cost of the grey pattern. split <= 0 returns the grey pattern in all channels
//       without the extra work.
//
// Baked variant: render a tileable caustic map once (src/gpu/bake-caustics.ts), then each frame is two
// drifting texture taps. The texture is RGBA: four time slices of one animation loop, one per channel.
//   CAUSTICS_BAKED_AVG = 0, CAUSTICS_BAKED_MIN = 1, CAUSTICS_BAKED_SINGLE = 2    how the taps are combined (`mode`)
//   caustics_baked_slice(slices: vec4f, phase: f32) -> f32      cyclic smooth blend of the four channels,
//       phase in radians, period TAU (this is what makes a baked texture boil instead of just slide)
//   caustics_baked(tex, samp, uv, time, lod, mode) -> f32       two taps (SINGLE: just the first, one texture
//       read, the cheapest and the cleanest look: it is the baked net itself, boiling through its slices;
//       the second 1.41x finer, turned
//       45 degrees, half a loop later, drifting the other way). AVG averages them: mean exactly 1 at
//       any lod. MIN takes the lower, the trick many shipped games use: only where both nets agree
//       does light survive, so you get sparse bright slivers and nodes; its mean is rescaled by a fit
//       to about +-15%. `lod` > 0 blurs by sampling a coarser mip (the cheap equivalent of lowering
//       `sharp`; the mean is unchanged because mips average). The sampler must repeat, ideally
//       trilinear. The drift is continuous (not periodic in `time`); only the slices loop.
//   caustics_baked_rgb(tex, samp, uv, time, lod, mode, split) -> vec3f   six taps with chromatic fringing
//
// Presentation
//   caustics_expose(c: f32, exposure: f32) -> f32     1 - exp(-c * exposure): maps the open-ended
//       intensity to [0, 1) without clipping the filaments. exposure 0.5 to 1 suits a mean-1 net.
//
// Cost summary (per pixel, rough): cellular ~ 130 ALU + 9 hashes; sine ~ 100 ALU + 40 sin/cos;
// baked ~ 2 texture taps + 20 ALU. stack = x2, rgb = x3, stack + rgb = x6.
// Credits: cellular distance F2 - F1 after S. Worley (1996); the iterated sine warp is the construction
// behind the classic "water turbulence" shaders, rewritten here with integer harmonics (so it tiles
// and loops), incommensurate per-round scales (so no lattice shows) and a low-frequency bend of the tile.

#include "math.wgsl"
#include "voronoi.wgsl"

const CAUSTICS_CELLULAR: u32 = 0u;
const CAUSTICS_SINE: u32 = 1u;
const CAUSTICS_BAKED_AVG: u32 = 0u;
const CAUSTICS_BAKED_MIN: u32 = 1u;
const CAUSTICS_BAKED_SINGLE: u32 = 2u;

// --- Shared profile -------------------------------------------------------------------------------

// Narrow line profile (1 - x)^2 for x in [0, 1) is the bright core of a filament.
fn _caustics_profile(x: f32) -> f32 {
  let t = saturate(1.0 - x);
  return t * t;
}

// --- Warped cellular ------------------------------------------------------------------------------

// Width of the line (in F2 - F1 units) for a given sharpness.
fn _caustics_cellular_width(sharp: f32) -> f32 {
  return mix(0.4, 0.05, saturate(sharp));
}

// Mean of the raw profile over the pattern, fitted to measurements (tests/water-light.test.ts): a line
// of width w covers a share of the plane about proportional to w, a little less for fat lines.
fn _caustics_cellular_mean(w: f32) -> f32 {
  return w * (0.86 - 0.3 * w);
}

fn caustics_cellular(uv: vec2f, cells: i32, time: f32, sharp: f32) -> f32 {
  let n = max(cells, 1);
  let a = TAU * uv;
  // Warp: low sine harmonics of the whole tile, integer frequencies in uv and in time, so the
  // displacement (in cell units) is exactly periodic and the cells bend like a wavy surface.
  let warp = 0.45 * vec2f(
    sin(a.y * 2.0 + time) + 0.5 * sin(a.x * 3.0 + a.y - 2.0 * time + 1.3),
    sin(a.x * 2.0 - time + 2.1) + 0.5 * sin(a.y * 3.0 - a.x + 2.0 * time),
  );
  let p = uv * f32(n) + warp;
  let v = voronoi_2d_tile_anim(p, 1.0, time, vec2i(n));
  let w = _caustics_cellular_width(sharp);
  // Some borders collect more light than others: modulate by the cell's id (mean 1).
  let strength = 0.6 + 0.8 * v.id;
  return strength * _caustics_profile((v.f2 - v.f1) / w) / _caustics_cellular_mean(w);
}

// --- Iterated sine warp ---------------------------------------------------------------------------

const _CAUSTICS_ITERS: i32 = 4;

fn _caustics_sine_width(sharp: f32) -> f32 {
  return mix(0.2, 0.025, saturate(sharp));
}

// Mean of the raw (squared) accumulation for line width w: a log-quadratic fit, within 1% over the
// widths used here (tests/water-light.test.ts re-measures the result).
fn _caustics_sine_mean(w: f32) -> f32 {
  let l = log(w);
  return exp(-0.0651 * l * l + 0.7645 * l - 0.4581);
}

fn caustics_sine(uv: vec2f, cells: i32, time: f32, sharp: f32) -> f32 {
  // One zero line of the phase per half cell, so a tile of `cells` cells has cells / 2 periods.
  let f = f32(max(cells / 2, 1));
  let a = TAU * uv;
  // A low-frequency bend of the whole tile (exactly periodic) so the lattice of the sine
  // construction, which repeats every period, does not show.
  let p = uv * (TAU * f) + (0.4 * f) * vec2f(
    sin(a.y * 2.0 + time) + 0.6 * sin(a.x * 3.0 - 2.0 * time),
    sin(a.x * 2.0 - time + 2.0) + 0.6 * sin(a.y * 3.0 + 2.0 * time),
  );
  var q = p;
  var sum = 0.0;
  let w = _caustics_sine_width(sharp);
  for (var n = 0; n < _CAUSTICS_ITERS; n++) {
    // Direction alternates so neighbouring iterations slide against each other. The per-iteration
    // scale factors (f + j) / f make the argument an integer number of periods per tile, so the
    // tile is exact, but the rounds are not commensurate: no regular lattice.
    let t = time * select(-1.0, 1.0, (n & 1) == 0);
    let sx = 1.0 + f32(n % 3 - 1) / f;
    let sy = 1.0 + f32((n + 1) % 3 - 1) / f;
    q = p + 0.55 * vec2f(cos(t - sx * q.x) + sin(t + sy * q.y), sin(t - sy * q.y) + cos(t + sx * q.x));
    // Distance to the nearest zero of the warped phase, in either axis: a warped grid of lines.
    let g = min(abs(sin(q.x + t)), abs(cos(q.y - t)));
    sum += w / (w + g);
  }
  let m = sum / f32(_CAUSTICS_ITERS);
  return m * m / _caustics_sine_mean(w);
}

// --- Dispatch and combinations --------------------------------------------------------------------

fn caustics_eval(method: u32, uv: vec2f, cells: i32, time: f32, sharp: f32) -> f32 {
  if (method == CAUSTICS_SINE) {
    return caustics_sine(uv, cells, time, sharp);
  }
  return caustics_cellular(uv, cells, time, sharp);
}

// The second layer's coordinates: (1, 1; -1, 1) * uv, an integer matrix, so it keeps period 1.
fn _caustics_second(uv: vec2f) -> vec2f {
  return vec2f(uv.x + uv.y, uv.y - uv.x);
}

fn caustics_stack(method: u32, uv: vec2f, cells: i32, time: f32, sharp: f32) -> f32 {
  let a = caustics_eval(method, uv, cells, time, sharp);
  let b = caustics_eval(method, _caustics_second(uv) + vec2f(0.37, 0.11), cells, time + PI, sharp);
  return 0.5 * (a + b);
}

fn _caustics_layers(method: u32, layers: u32, uv: vec2f, cells: i32, time: f32, sharp: f32) -> f32 {
  if (layers >= 2u) {
    return caustics_stack(method, uv, cells, time, sharp);
  }
  return caustics_eval(method, uv, cells, time, sharp);
}

fn caustics_rgb(
  method: u32,
  layers: u32,
  uv: vec2f,
  cells: i32,
  time: f32,
  sharp: f32,
  split: f32,
) -> vec3f {
  if (split <= 0.0) {
    return vec3f(_caustics_layers(method, layers, uv, cells, time, sharp));
  }
  let shift = split * vec2f(0.8, 0.6);
  return vec3f(
    _caustics_layers(method, layers, uv + shift, cells, time, sharp),
    _caustics_layers(method, layers, uv, cells, time, sharp),
    _caustics_layers(method, layers, uv - shift, cells, time, sharp),
  );
}

fn caustics_expose(c: f32, exposure: f32) -> f32 {
  return 1.0 - exp(-max(c, 0.0) * max(exposure, 0.0));
}

// --- Baked ----------------------------------------------------------------------------------------

fn caustics_baked_slice(slices: vec4f, phase: f32) -> f32 {
  let cycle = phase / TAU;
  let x = 4.0 * (cycle - floor(cycle));
  let i = u32(floor(x)) & 3u;
  let f = smoothstep(0.0, 1.0, x - floor(x));
  let a = select(select(select(slices.x, slices.y, i == 1u), slices.z, i == 2u), slices.w, i == 3u);
  let b = select(select(select(slices.y, slices.z, i == 1u), slices.w, i == 2u), slices.x, i == 3u);
  return mix(a, b, f);
}

// min of two mean-1 fields has a much lower mean (about 0.15 for sharp nets, rising toward 1 as the mips
// blur them); this restores it. A fit to measurements, 1 + 5.14 exp(-(lod / 2.39)^1.7), good to about 10%.
fn _caustics_min_gain(lod: f32) -> f32 {
  return 1.0 + 5.14 * exp(-pow(max(lod, 0.0) / 2.39, 1.7));
}

fn _caustics_baked_pair(
  tex: texture_2d<f32>,
  samp: sampler,
  uv: vec2f,
  time: f32,
  lod: f32,
  mode: u32,
) -> f32 {
  // Tap A drifts one way; tap B is 1.41x finer, turned 45 degrees (an integer matrix, so it still
  // tiles), drifts the other way and runs half a loop out of phase.
  let ua = uv + vec2f(0.021, 0.013) * time;
  let ub = _caustics_second(uv) + vec2f(0.37, 0.11) - vec2f(0.017, 0.009) * time;
  let a = caustics_baked_slice(textureSampleLevel(tex, samp, ua, lod), time);
  if (mode == CAUSTICS_BAKED_SINGLE) {
    return a;
  }
  let b = caustics_baked_slice(textureSampleLevel(tex, samp, ub, lod), time + PI);
  if (mode == CAUSTICS_BAKED_MIN) {
    return min(a, b) * _caustics_min_gain(lod);
  }
  return 0.5 * (a + b);
}

fn caustics_baked(
  tex: texture_2d<f32>,
  samp: sampler,
  uv: vec2f,
  time: f32,
  lod: f32,
  mode: u32,
) -> f32 {
  return _caustics_baked_pair(tex, samp, uv, time, lod, mode);
}

fn caustics_baked_rgb(
  tex: texture_2d<f32>,
  samp: sampler,
  uv: vec2f,
  time: f32,
  lod: f32,
  mode: u32,
  split: f32,
) -> vec3f {
  if (split <= 0.0) {
    return vec3f(_caustics_baked_pair(tex, samp, uv, time, lod, mode));
  }
  let shift = split * vec2f(0.8, 0.6);
  return vec3f(
    _caustics_baked_pair(tex, samp, uv + shift, time, lod, mode),
    _caustics_baked_pair(tex, samp, uv, time, lod, mode),
    _caustics_baked_pair(tex, samp, uv - shift, time, lod, mode),
  );
}

// ripples.wgsl -- rain ripple rings on a water surface: expanding, fading concentric rings from hashed
// drop positions and start times, with an analytic height and slope. No state, no textures.
//
// The surface is cut into square cells one unit wide. Each cell owns one drop per cycle: when it lands
// (a hashed time offset, so neighbouring cells are never in step), where in the cell it lands (always the
// middle half, so a ring never reaches further than half a cell past the cell border) and how big its
// ring grows come from an integer hash of (cell, cycle). A pixel therefore only needs the four cells
// nearest to it: its own and the neighbour on each axis it is closer to. Rings that cross a cell border
// stay continuous, because the ring is faded to exactly zero at distance RIPPLE_REACH from its drop.
//
// Conventions
//  * `p` is in CELL units: divide a position in metres or image heights by the cell size first. The
//    ring wavelength is RIPPLE_WAVELENGTH (0.1) cells, the largest ring has radius RIPPLE_MAX_RADIUS (0.4).
//    So a 0.4 m cell gives 4 cm wavelength rings that grow to 16 cm.
//  * `time` and `life` are in seconds. `life` is how long one ring lives. `density` is the share of the
//    time a cell has a ring on it (a duty cycle), [0, 1]; with 1 a cell is almost always rippling.
//    The cell's period is life / (0.9 * density), jittered by up to 35% per cell.
//  * Heights are NORMALISED: the crest of a fresh ring is 1, so one ring stays within [-1, 1] (a sum of
//    up to four overlapping rings can reach 4, in practice rarely past 1.5). Slopes are
//    d height / d p, per cell unit. To get a real surface, scale by the crest height in cell units:
//    ripple_normal does that. A crest of 0.01 cells is a gentle ripple, 0.03 a stormy one.
//  * A ring is a cosine wavelet riding on its front: a sharp leading crest followed by up to four
//    fading troughs and crests behind, in a Gaussian envelope (long behind the front, short ahead of
//    it). The front slows down as it spreads (r = R (1 - (1 - age)^2)); the height decays like
//    (1 - age)^2 and ramps in over the first 5% of the ring's life, so nothing pops.
//
//   ripple_ring(offset, age, size) -> vec3f
//        one ring. `offset` = pixel minus drop centre (cell units), `age` = fraction of the ring's life
//        gone (0 at impact, 1 when it has faded), `size` [0.5, 1] scales the maximum radius. Returns
//        (height, d/dx, d/dy). Exactly (0, 0, 0) for age <= 0, age >= 1 and beyond RIPPLE_REACH.
//        Radially symmetric: depends on |offset| only; the gradient points along offset.
//   ripple_rings(p, time, density, life, seed) -> vec3f
//        the sum of every ring that reaches p: (height, d/dx, d/dy). `seed` selects an independent
//        pattern, so two layers at different cell sizes do not coincide. 4 cell evaluations, 2 integer
//        hashes each, plus ring maths for the rings that are in range (about 60 ALU, one sqrt, one exp,
//        one sin, one cos per ring).
//   ripple_normal(rings, crest) -> vec3f
//        unit tangent-space normal (x right, y down, z up) for the sum from ripple_rings, with the
//        crest height `crest` in cell units
//   ripple_lod_fade(footprint) -> f32
//        [0, 1]; pass the size of one pixel on the surface in cell units. 1 where the rings are resolved
//        (footprint < 1/5 of a wavelength), 0 where they are not (> 5/8), so distant rings fade out
//        instead of shimmering
//
// Technique: the hashed-cell scheme used for rain ripples in many real-time shaders (the neighbourhood
// search is the point: rings are summed across cell borders), with the wavelet, the exact fade and the
// closed-form gradient written here. Euler-style cosine wavelets are not a wave simulation; they look
// right because the eye reads the expanding crest and the fading train behind it.

#include "hash.wgsl"
#include "math.wgsl"

const RIPPLE_WAVELENGTH: f32 = 0.1;
const RIPPLE_MAX_RADIUS: f32 = 0.4;
const RIPPLE_REACH: f32 = 0.5;

fn ripple_ring(offset: vec2f, age: f32, size: f32) -> vec3f {
  if (age <= 0.0 || age >= 1.0) {
    return vec3f(0.0);
  }
  let d = length(offset);
  if (d >= RIPPLE_REACH) {
    return vec3f(0.0);
  }
  let rest = 1.0 - age;
  let front = RIPPLE_MAX_RADIUS * clamp(size, 0.5, 1.0) * (1.0 - rest * rest);
  let x = d - front;
  let lam = RIPPLE_WAVELENGTH;
  let sigma = select(1.7 * lam, 0.3 * lam, x > 0.0);
  let g = exp(-(x * x) / (sigma * sigma));
  let phase = TAU * x / lam;
  let wave = cos(phase) * g;
  let dwave = -sin(phase) * (TAU / lam) * g - cos(phase) * g * (2.0 * x / (sigma * sigma));

  // Fade to exactly zero at the reach so rings from neighbouring cells meet without a seam.
  let edge_t = saturate((d - (RIPPLE_REACH - 0.06)) / 0.06);
  let window = 1.0 - edge_t * edge_t * (3.0 - 2.0 * edge_t);
  let dwindow = -6.0 * edge_t * (1.0 - edge_t) / 0.06;

  let ramp_t = saturate(age / 0.05);
  let amp = ramp_t * ramp_t * (3.0 - 2.0 * ramp_t) * rest * rest;

  let height = amp * window * wave;
  let slope = amp * (window * dwave + wave * dwindow);
  // The gradient of a function of |offset| is radial; the cone point at the centre has no direction.
  let radial = offset / max(d, 1e-5) * smoothstep(0.0, 0.004, d);
  return vec3f(height, slope * radial);
}

fn _ripple_cell(p: vec2f, cell: vec2i, time: f32, density: f32, life: f32, seed: i32) -> vec3f {
  let h = ihash_3to3(vec3i(cell, seed * 31 + 7));
  let period = life / (0.9 * clamp(density, 0.02, 1.0)) * mix(1.0, 1.35, h.x);
  let t = time / period + h.y;
  let k = floor(t);
  let age = (t - k) * period / max(life, 1e-4);
  if (age >= 1.0) {
    return vec3f(0.0);
  }
  let g = ihash_3to3(vec3i(cell.x * 2 + seed, cell.y * 3 + 1, i32(k)));
  let centre = vec2f(cell) + 0.25 + 0.5 * g.xy;
  return ripple_ring(p - centre, age, 0.55 + 0.45 * g.z);
}

fn ripple_rings(p: vec2f, time: f32, density: f32, life: f32, seed: i32) -> vec3f {
  let base = vec2i(floor(p));
  let f = p - floor(p);
  let step_x = select(-1, 1, f.x >= 0.5);
  let step_y = select(-1, 1, f.y >= 0.5);
  var sum = vec3f(0.0);
  sum += _ripple_cell(p, base, time, density, life, seed);
  sum += _ripple_cell(p, base + vec2i(step_x, 0), time, density, life, seed);
  sum += _ripple_cell(p, base + vec2i(0, step_y), time, density, life, seed);
  sum += _ripple_cell(p, base + vec2i(step_x, step_y), time, density, life, seed);
  return sum;
}

fn ripple_normal(rings: vec3f, crest: f32) -> vec3f {
  return normalize(vec3f(-rings.y * crest, -rings.z * crest, 1.0));
}

fn ripple_lod_fade(footprint: f32) -> f32 {
  return 1.0 - smoothstep(0.2 * RIPPLE_WAVELENGTH, 0.625 * RIPPLE_WAVELENGTH, footprint);
}

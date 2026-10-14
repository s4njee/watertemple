// backdrop.wgsl -- procedural things to look THROUGH water at: pool tiles, a test card, sandy bed.
//
// Refraction, absorption and foam all need a backdrop whose distortion is easy to judge. These are pure
// functions of a position, return LINEAR rgb (finite, in [0, 1]) and use no textures, so they can be
// drawn straight into a render target (the production pattern: the scene behind the water is a texture)
// or evaluated analytically per pixel. None of them uses derivatives; anti-aliasing is driven by an
// explicit `aa` argument (the size of one pixel in the function's own units), so they run in compute
// shaders (tests) as well as fragments. Pass aa = 0 for hard edges.
//
//   backdrop_pool_tiles(p: vec2f, aa: f32) -> vec3f
//       p in TILE units (one tile = 1.0). Aqua tiles with per-tile brightness jitter, a bevel, pale grout
//       lines and a dark lane stripe every 8 columns. The classic reference for judging refraction: every
//       line is straight, so any bend is the water's doing. aa in tile units.
//   backdrop_test_card(uv: vec2f, aspect: f32, aa: f32) -> vec3f
//       uv 0..1 over the whole image (top-left origin), aspect = width / height (images narrower than
//       1.5:1 get the whole card shrunk to fit their width). Mid-grey card with a
//       10% line grid (bold every 5th), colour bars on top, a grey ramp at the bottom, a black disc with
//       concentric rings (distortion becomes visible as ovals), a Siemens star, a checker patch and
//       fine stripes (dispersion fringes show on every black/white edge). aa in uv-height units
//       (1 / image height in pixels).
//   backdrop_sand(p: vec2f, pebbles: f32) -> vec3f
//       p in world units where a pebble is about 0.25-0.5 across and ripple marks repeat every ~0.8. Sand
//       grain, mottling, ripple marks and shaded pebbles with contact shadows. `pebbles` in [0, 1] is the
//       share of the maximum pebble count (0 = clean rippled sand). No aa needed (soft).
//   backdrop_sample(kind: i32, uv: vec2f, aspect: f32, aa: f32, density: f32) -> vec3f
//       one entry point: kind 0 pool tiles (12 tiles tall at density 1), 1 test card, 2 pebbly sand and
//       3 quiet sand with a few pebbles (both 6 world units tall at density 1). `density` scales the
//       feature count (tiles, pebbles); the test card ignores it. aa in uv-height units.
//
// Constants: BACKDROP_KIND_TILES = 0, BACKDROP_KIND_CARD = 1, BACKDROP_KIND_SAND = 2, BACKDROP_KIND_DUNE = 3.
//
// Cost: tiles ~30 ALU + one hash. Test card ~60 ALU, no hashes. Sand: one voronoi (9 cells), two gnoise
// and a few dozen ALU; bake it into a texture if you evaluate it every frame.

#include "hash.wgsl"
#include "noise.wgsl"
#include "voronoi.wgsl"
#include "color.wgsl"
#include "math.wgsl"

const BACKDROP_KIND_TILES: i32 = 0;
const BACKDROP_KIND_CARD: i32 = 1;
const BACKDROP_KIND_SAND: i32 = 2;
const BACKDROP_KIND_DUNE: i32 = 3;

// Coverage of a band of half-width `half_width` at signed distance `d` from its centre line.
fn _backdrop_cover(d: f32, half_width: f32, aa: f32) -> f32 {
  let w = max(aa, 1e-5);
  return 1.0 - smoothstep(half_width - w, half_width + w, d);
}

fn backdrop_pool_tiles(p: vec2f, aa: f32) -> vec3f {
  let cell = floor(p);
  let f = p - cell;
  let edge = min(f, 1.0 - f);
  let d = min(edge.x, edge.y); // distance to the nearest tile edge, 0..0.5
  let grout_half = 0.045;

  let jitter = hash_2to1(cell);
  var tile = srgb_to_linear(vec3f(0.22, 0.66, 0.74)) * (0.88 + 0.2 * jitter);
  // Bevel: tiles darken a little toward their edge and brighten at the inner lip.
  let bevel = smoothstep(grout_half, grout_half + 0.14, d);
  tile = tile * (0.78 + 0.22 * bevel) + vec3f(0.06) * smoothstep(0.1, 0.0, abs(d - grout_half - 0.1));

  // Dark lane stripe on column 4 of every 8, a tile wide but narrower than the tile.
  let column = ((i32(cell.x) % 8) + 8) % 8;
  let stripe = select(0.0, _backdrop_cover(abs(f.x - 0.5), 0.2, aa), column == 4);
  tile = mix(tile, srgb_to_linear(vec3f(0.04, 0.12, 0.3)), stripe);

  let grout = srgb_to_linear(vec3f(0.82, 0.84, 0.82));
  let tile_cover = 1.0 - _backdrop_cover(d, grout_half, aa);
  return clamp(mix(grout, tile, tile_cover), vec3f(0.0), vec3f(1.0));
}

fn _backdrop_bars(x: f32) -> vec3f {
  // Seven bars: white, yellow, cyan, green, magenta, red, blue (75% saturation, as on broadcast cards).
  let i = clamp(i32(floor(x * 7.0)), 0, 6);
  var c = vec3f(0.75);
  if (i == 1) { c = vec3f(0.75, 0.75, 0.0); }
  if (i == 2) { c = vec3f(0.0, 0.75, 0.75); }
  if (i == 3) { c = vec3f(0.0, 0.75, 0.0); }
  if (i == 4) { c = vec3f(0.75, 0.0, 0.75); }
  if (i == 5) { c = vec3f(0.75, 0.0, 0.0); }
  if (i == 6) { c = vec3f(0.0, 0.0, 0.75); }
  return srgb_to_linear(c);
}

fn backdrop_test_card(uv: vec2f, aspect_in: f32, aa_in: f32) -> vec3f {
  // The card is laid out for a frame at least 1.5 times as wide as it is tall. A narrower image (a phone
  // in portrait) shows the whole card shrunk to fit the width, centred, with plain grid above and below.
  let fit = max(1.5 / max(aspect_in, 0.1), 1.0);
  let aspect = aspect_in * fit;
  let aa = aa_in * fit;
  let q = vec2f(uv.x * aspect, (uv.y - 0.5) * fit + 0.5); // height 1, width `aspect`
  var col = srgb_to_linear(vec3f(0.62, 0.64, 0.66));

  // Grid every 0.1 of the height; every fifth line is bold.
  let g = q * 10.0;
  let gd = abs(g - round(g)) / 10.0; // distance to the nearest grid line, in height units
  let major = abs(g / 5.0 - round(g / 5.0)) / 2.0;
  let minor_line = max(_backdrop_cover(gd.x, 0.0016, aa), _backdrop_cover(gd.y, 0.0016, aa));
  let major_line = max(_backdrop_cover(major.x, 0.0035, aa), _backdrop_cover(major.y, 0.0035, aa));
  col = mix(col, vec3f(0.01, 0.015, 0.03), max(minor_line * 0.8, major_line));

  // Colour bars, then a grey ramp, at the top and bottom edges.
  let on_card = q.y >= 0.0 && q.y <= 1.0;
  if (on_card && q.y < 0.1) { col = _backdrop_bars(uv.x); }
  if (on_card && q.y > 0.93) { col = vec3f(srgb_to_linear_f(uv.x)); }

  // Centre target: black disc with white concentric rings and a white outline.
  let c = vec2f(aspect * 0.5, 0.52) - q;
  let r = length(c);
  let disc = _backdrop_cover(r, 0.22, aa);
  let rings = _backdrop_cover(abs(fract(r * 28.0) - 0.5) / 28.0, 0.0105, aa);
  let target_col = mix(vec3f(0.01), vec3f(0.9), rings);
  col = mix(col, target_col, disc);

  // Siemens star, upper left: 16 spokes, converging to a point where blur and distortion show.
  let s = q - vec2f(0.2 * aspect + 0.04, 0.3);
  let star_r = length(s);
  let spoke = step(0.0, sin(atan2(s.y, s.x) * 16.0));
  col = mix(col, vec3f(spoke * 0.9 + 0.01), _backdrop_cover(star_r, 0.14, aa));

  // Checker, lower left: eight 0.025 squares per side.
  let k = q - vec2f(0.06, 0.64);
  let in_checker = _backdrop_cover(max(abs(k.x - 0.1), abs(k.y - 0.1)), 0.1, aa);
  let ck = floor(k / 0.025);
  let checker = select(0.01, 0.9, (i32(ck.x) + i32(ck.y)) % 2 == 0);
  col = mix(col, vec3f(checker), in_checker);

  // Fine vertical stripes, lower right: pitch 0.012, the sharpest edges on the card.
  let v = q - vec2f(aspect - 0.28, 0.64);
  let in_stripes = _backdrop_cover(max(abs(v.x - 0.11), abs(v.y - 0.1)), 0.11, aa);
  let stripes = select(0.01, 0.9, fract(v.x / 0.012) < 0.5);
  col = mix(col, vec3f(stripes), in_stripes);

  return clamp(col, vec3f(0.0), vec3f(1.0));
}

fn backdrop_sand(p: vec2f, pebbles: f32) -> vec3f {
  let sand = srgb_to_linear(vec3f(0.86, 0.76, 0.56));
  // Large mottling, ripple marks bending with a slow noise, and fine grain.
  let mottle = gnoise_2d(p * 0.9 + 7.0);
  let bend = gnoise_2d(p * vec2f(0.6, 0.35));
  let ripple = 0.5 + 0.5 * sin((p.y + 1.5 * bend) * 7.5 + 2.0 * mottle);
  let grain = hash_2to1(floor(p * 110.0)) - 0.5;
  var col = sand * (0.9 + 0.12 * mottle + 0.07 * (ripple - 0.5) + 0.07 * grain);

  // Pebbles: about 60% of the cells hold a pebble of random radius, hue and brightness.
  let q = p * 1.6;
  let v = voronoi_2d(q, 0.8);
  let radius = 0.2 + 0.2 * fract(v.id * 7.31);
  let has_pebble = select(0.0, 1.0, v.id < 0.6 * clamp(pebbles, 0.0, 1.0));
  let offset = (q - v.site) / radius;
  let rr = length(offset);
  let inside = has_pebble * (1.0 - smoothstep(0.9, 1.0, rr));
  let z = sqrt(max(1.0 - rr * rr, 0.0));
  let normal = normalize(vec3f(offset.x, offset.y, max(z, 0.2)));
  let light = normalize(vec3f(-0.5, -0.6, 0.65));
  let lit = 0.45 + 0.7 * max(dot(normal, light), 0.0);
  let hue = fract(v.id * 13.7);
  var stone = mix(srgb_to_linear(vec3f(0.5, 0.47, 0.44)), srgb_to_linear(vec3f(0.78, 0.64, 0.46)), hue);
  stone = stone * (0.7 + 0.5 * fract(v.id * 91.3));
  // Contact shadow on the sand just outside the pebble.
  let shadow = has_pebble * 0.3 * (1.0 - smoothstep(1.0, 1.5, rr)) * (1.0 - inside);
  col = col * (1.0 - shadow);
  col = mix(col, stone * lit * (0.9 + 0.2 * mottle), inside);
  return clamp(col, vec3f(0.0), vec3f(1.0));
}

fn backdrop_sample(kind: i32, uv: vec2f, aspect: f32, aa: f32, density: f32) -> vec3f {
  let d = max(density, 0.05);
  let q = vec2f(uv.x * aspect, uv.y);
  if (kind == BACKDROP_KIND_CARD) {
    return backdrop_test_card(uv, aspect, aa);
  }
  if (kind == BACKDROP_KIND_SAND) {
    return backdrop_sand(q * 6.0 * d, 1.0);
  }
  if (kind == BACKDROP_KIND_DUNE) {
    return backdrop_sand(q * 6.0 * d, 0.12);
  }
  return backdrop_pool_tiles(q * 12.0 * d, aa * 12.0 * d);
}

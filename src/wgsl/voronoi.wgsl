// voronoi.wgsl -- 2D cellular (Worley) noise: distances to the nearest feature points.
//
//   struct Voronoi2D {
//     f1: f32,      distance to the nearest feature point (exact), in cell units
//     f2: f32,      distance to the second nearest. f2 - f1 >= 0 is 0 on cell borders: use it as a
//                   cheap edge mask for foam lines / cracks (smoothstep(0.0, w, f2 - f1)). Its
//                   width varies a little with cell shape; it is not a true border distance.
//     id: f32,      random value in [0, 1) that is constant per cell (colour / height / phase per cell)
//     cell: vec2i,  lattice cell that owns the nearest point (wrapped into [0, period) by _tile forms)
//     site: vec2f,  position of the nearest feature point in p's own coordinates; p - site points
//                   from the cell's seed to the sample (radial shading, per-cell uv)
//   }
//
//   voronoi_2d(p, jitter) -> Voronoi2D                       static points
//   voronoi_2d_anim(p, jitter, time) -> Voronoi2D            points orbit inside their cells: no popping,
//                                                            flows like water. `time` in radians (one orbit
//                                                            per 2 pi); the orbit is +-0.4 * jitter wide
//   voronoi_2d_tile(p, jitter, period: vec2i) -> Voronoi2D   repeats every `period` cells per axis (>= 1)
//   voronoi_2d_tile_anim(p, jitter, time, period: vec2i) -> Voronoi2D   both
//
// `p` is in lattice units (one feature point per unit cell). `jitter` in [0, 1] (clamped): 0 is a regular
// grid of seeds at cell centres, 1 is fully random inside each cell.
//
// Accuracy: the search covers the 3x3 cells around p. For jitter <= 0.65 f1 and f2 are exact everywhere
// (provable); in practice f1 stays exact up to jitter 1 and f2 up to ~0.8. At jitter 1 f2 can be too large
// in a vanishing fraction of points (~1e-4 of the area), always deep inside a cell and far from any border,
// so f2 - f1 masks along borders are unaffected. The animated forms orbit +-0.4 (not +-0.5) per unit jitter
// for the same reason.
// Cost: 9 hashes + 9 distance tests; about 2-3x a gnoise_2d. Cells are random per cell, deterministic.
//
// Technique: S. Worley, "A Cellular Texture Basis Function" (1996), 3x3 neighbour search as in
// Gustavson's and Quilez's implementations.

#include "hash.wgsl"
#include "math.wgsl"

struct Voronoi2D {
  f1: f32,
  f2: f32,
  id: f32,
  cell: vec2i,
  site: vec2f,
}

// x, y: jitter of the feature point inside its cell, z: the cell's id.
fn _voronoi_hash(c: vec2i) -> vec3f {
  return to_unorm_3d(pcg_3d(vec3u(bitcast<vec2u>(c), 0u)));
}

fn _voronoi_2d(
  p: vec2f,
  jitter: f32,
  time: f32,
  animate: bool,
  period: vec2i,
  tiled: bool,
) -> Voronoi2D {
  let j = clamp(jitter, 0.0, 1.0);
  let n = max(period, vec2i(1));
  let base = vec2i(floor(p));
  let hash_base = select(base, wrap_2d(base, n), tiled);
  var d1 = 1e9;
  var d2 = 1e9;
  var best = Voronoi2D(0.0, 0.0, 0.0, vec2i(0), vec2f(0.0));
  for (var y = -1; y <= 1; y++) {
    for (var x = -1; x <= 1; x++) {
      let offset = vec2i(x, y);
      // The cell whose hash we use. Wrapping once up front means a compare per cell, not a modulo.
      var w = hash_base + offset;
      if (tiled) {
        w += select(vec2i(0), n, w < vec2i(0)) - select(vec2i(0), n, w >= n);
      }
      let h = _voronoi_hash(w);
      var shift = j * (h.xy - 0.5);
      if (animate) {
        shift = 0.4 * j * sin(time + TAU * h.xy);
      }
      // The feature point is placed around the UNWRAPPED cell so distances stay local.
      let site = vec2f(base + offset) + 0.5 + shift;
      let d = site - p;
      let dd = dot(d, d);
      if (dd < d1) {
        d2 = d1;
        d1 = dd;
        best.id = h.z;
        best.cell = w;
        best.site = site;
      } else if (dd < d2) {
        d2 = dd;
      }
    }
  }
  best.f1 = sqrt(d1);
  best.f2 = sqrt(d2);
  return best;
}

fn voronoi_2d(p: vec2f, jitter: f32) -> Voronoi2D {
  return _voronoi_2d(p, jitter, 0.0, false, vec2i(1), false);
}

fn voronoi_2d_anim(p: vec2f, jitter: f32, time: f32) -> Voronoi2D {
  return _voronoi_2d(p, jitter, time, true, vec2i(1), false);
}

fn voronoi_2d_tile(p: vec2f, jitter: f32, period: vec2i) -> Voronoi2D {
  return _voronoi_2d(p, jitter, 0.0, false, period, true);
}

fn voronoi_2d_tile_anim(p: vec2f, jitter: f32, time: f32, period: vec2i) -> Voronoi2D {
  return _voronoi_2d(p, jitter, time, true, period, true);
}

// droplets.wgsl -- stateless water drops on a pane of glass: static beads, large drops that run
// down and pause and leave a trail of shrinking beads, and the clean streak each one wipes.
//
// No simulation and no textures. Everything is a pure function of (position, time): the glass is cut
// into a grid of cells, each cell owns one drop (or one run) per cycle, and the cycle's random numbers
// come from an integer hash of (cell, cycle). Nothing is stored, so it costs a few dozen ALU ops per
// layer per pixel, resizes for free and can be drawn at any aspect ratio.
//
// Conventions
//  * `p` is a position on the glass in IMAGE-HEIGHT units with y pointing DOWN the screen (the uv
//    direction): p = vec2f(uv.x * aspect, uv.y). Gravity is +y. `time` is a phase in seconds: pass an
//    accumulated clock (time += delta * speed), not time * speed, so a speed slider never jumps.
//  * `cell` is the width of one grid column in the same units. A running drop is a fifth to a quarter
//    of it wide, so cell 0.08 gives drops about 3% of the image height in diameter.
//  * `aa` is the size of one pixel in p units (1 / image height in pixels); 0 gives hard edges.
//  * A drop is a lens. The functions return where in the drop the pixel lies (`local`, the unit disc
//    of the drop, y down) and how big the drop is; droplet_lens and droplet_normal turn that into the
//    inverted, magnified-minus-one picture a real drop shows and into a shading normal.
//  * The returned `clear` is the "wiped" amount: 1 inside a drop and along its fresh trail, falling
//    off behind it. Use it to take the fog off the glass; it needs no state because the run's whole
//    path is a function of the cell.
//
// struct DropletSample
//     mask    f32      coverage of the drop body, [0, 1], anti-aliased over `aa`
//     local   vec2f    position in the drop's own frame; |local| <= 1 inside (a little over at the
//                      anti-aliased rim); (0, 0) where there is no drop
//     radius  f32      radius of the drop in p units (0 where there is no drop)
//     clear   f32      [0, 1], how much the glass is wiped here (trail of a running drop)
//
// Layers (each one grid-cell evaluation: 2 integer hashes, about 100 ALU)
//   droplet_none() -> DropletSample                  empty sample (no drop, glass untouched)
//   droplet_static(p, time, cell, density, seed, aa) -> DropletSample
//        Small beads that sit still, grow in, and slowly shrink away over 20-50 s before a new one forms
//        elsewhere in the cell. `density` [0, 1] is the share of cells holding a bead. A bead is up to
//        0.28 cells in radius.
//   droplet_running(p, time, cell, density, seed, aa) -> DropletSample
//        Big drops. Each column of cells is shifted vertically by its own random amount, so the cell
//        borders never line up. Each cell is 3.2 cells tall and runs one drop per cycle (6-14 s): it
//        forms at the top, runs down a gently wobbling path with pauses (it never goes backwards),
//        stretches while it moves, sheds beads that shrink with their age, and shrinks away at the
//        bottom. Drop radius 0.13-0.22 cells. `density` [0, 1] is the share of cycles that hold a drop.
//   droplet_merge(a, b) -> DropletSample             the nearer-to-opaque drop wins, the wipes add (max)
//   droplet_field(p, time, rain, size, layers, aa) -> DropletSample
//        The standard stack. rain [0, 1] sets all densities; size is the width of the big drops' column
//        in p units (0.05-0.2); rain 0 gives clean glass; layers 1 = beads + big runners, 2 = + a second, smaller runner layer,
//        3 = + a fine mist of tiny beads. Beads inside a wiped trail are removed.
//
// Turning a sample into pixels
//   droplet_lens(s, strength) -> vec2f     offset to ADD to p to find the point of the world this pixel of the
//        drop shows: -local * radius * strength, so the picture is inverted (strength > 1 reaches past the
//        centre) and every drop shows a differently sized slice of the world. 0 outside drops. Divide x by
//        the aspect ratio to get a uv offset.
//   droplet_normal(s, flatness) -> vec3f   unit normal of a spherical cap in the glass frame (x right, y down,
//        z toward the viewer): flatness 1 is a hemisphere, 0.5 a flat lens. (0, 0, 1) outside drops.
//   droplet_rim(s) -> f32                  [0, 1], 0 in the middle of a drop, 1 at its edge: darken or tint
//   droplet_highlight(normal, light, sharpness) -> f32   Blinn-Phong glint of a drop normal; `light` is a
//        unit vector from the glass toward the light in the glass frame. [0, 1]
//
// Time continuity: a drop's radius ramps in from 0 and back to 0 at both ends of its cycle, and its
// beads and wipe fade with it, so the picture changes by no more than the motion implies, also across
// a cycle boundary. Changing `density` or `cell` while running re-rolls drops (that does jump).
//
// Technique: the grid-of-cycling-drops construction that rain-on-window shaders use, written from
// scratch: per-cell hashed phase, a monotone stop-go progress curve (a line minus two sines whose
// amplitudes sum below 1, so its slope never goes negative), a path wobble that depends only on height
// so the trail can be rebuilt from the head's position, and a nearest-bead search along the path.

#include "hash.wgsl"
#include "math.wgsl"
#include "normals.wgsl"

struct DropletSample {
  mask: f32,
  local: vec2f,
  radius: f32,
  clear: f32,
}

const DROPLET_CELL_H: f32 = 3.2;   // run cell height, in column widths
const DROPLET_Y_TOP: f32 = 0.42;   // where a drop forms
const DROPLET_Y_BOTTOM: f32 = 2.78; // where it comes to rest
const _DROPLET_BEAD_SPACING: f32 = 0.46;

fn droplet_none() -> DropletSample {
  return DropletSample(0.0, vec2f(0.0), 0.0, 0.0);
}

// Anti-aliased coverage from a signed distance and a radius, both in cell units. A drop smaller than
// half a pixel fades out with its radius instead of keeping the 0.5 coverage of its centre pixel, so a
// drop that grows from nothing never pops in.
fn _droplet_cover(sd: f32, radius: f32, cell: f32, aa: f32) -> f32 {
  let a = max(aa, 1e-6);
  return clamp(0.5 - sd * cell / a, 0.0, 1.0) * smoothstep(0.0, 0.5 * a, radius * cell);
}

// Monotone stop-and-go progress: (progress, speed) for s in [0, 1]. progress(0) = 0, progress(1) = 1
// when n1 and n2 are integers; speed (the derivative) is >= 0.05, so the drop dwells but never reverses.
fn _droplet_stopgo(s: f32, n1: f32, c1: f32, n2: f32, c2: f32) -> vec2f {
  let w1 = TAU * n1;
  let w2 = TAU * n2;
  let progress = s - 0.55 * (sin(w1 * s + c1) - sin(c1)) / w1 - 0.4 * (sin(w2 * s + c2) - sin(c2)) / w2;
  let speed = 1.0 - 0.55 * cos(w1 * s + c1) - 0.4 * cos(w2 * s + c2);
  return vec2f(progress, speed);
}

// Horizontal position of a run's path at height y (cell units). Depends on y only, never on time.
fn _droplet_path_x(y: f32, x0: f32, phase: vec2f) -> f32 {
  return x0 + 0.06 * (0.65 * sin(y * 2.1 + phase.x) + 0.35 * sin(y * 4.7 + phase.y));
}

fn droplet_static(p: vec2f, time: f32, cell: f32, density: f32, seed: i32, aa: f32) -> DropletSample {
  let u = p / max(cell, 1e-5) + vec2f(f32(seed) * 5.17, f32(seed) * 3.71);
  let c = vec2i(floor(u));
  let l = fract(u);
  let h = ihash_3to3(vec3i(c, seed));
  if (h.z >= density) {
    return droplet_none();
  }
  let period = mix(20.0, 50.0, h.x);
  let t = time / period + h.y;
  let k = floor(t);
  let s = t - k;
  let g = ihash_3to3(vec3i(c.x * 3 + seed, c.y * 5 + 1, i32(k)));
  let grow = smoothstep(0.0, 0.06, s);
  let shrink = 1.0 - smoothstep(0.45, 1.0, s);
  let r = mix(0.1, 0.28, g.z * g.z) * grow * shrink;
  if (r <= 0.0) {
    return droplet_none();
  }
  let centre = vec2f(0.3 + 0.4 * g.x, 0.3 + 0.4 * g.y);
  let d = l - centre;
  let sd = length(d) - r;
  let mask = _droplet_cover(sd, r, cell, aa);
  if (mask <= 0.0) {
    return droplet_none();
  }
  return DropletSample(mask, d / r, r * cell, 0.0);
}

fn droplet_running(p: vec2f, time: f32, cell: f32, density: f32, seed: i32, aa: f32) -> DropletSample {
  let u = p / max(cell, 1e-5) + vec2f(f32(seed) * 17.31, 0.0);
  let col = i32(floor(u.x));
  let lx = u.x - floor(u.x);
  // Each column starts at its own height, so cells never line up into rows.
  let column = ihash_2to2(vec2i(col, seed));
  let v = u.y + column.x * DROPLET_CELL_H;
  let row = i32(floor(v / DROPLET_CELL_H));
  let ly = v - f32(row) * DROPLET_CELL_H;

  let h = ihash_3to3(vec3i(col, row, seed));
  let period = mix(6.0, 14.0, h.x);
  let t = time / period + h.y;
  let k = floor(t);
  let s = t - k;
  let g = ihash_3to3(vec3i(col * 3 + seed, row * 5 + 1, i32(k)));
  if (g.z >= density) {
    return droplet_none();
  }

  let grow = smoothstep(0.0, 0.08, s);
  let fade = 1.0 - smoothstep(0.84, 1.0, s);
  let env = grow * fade;
  if (env <= 0.0) {
    return droplet_none();
  }

  // Where along its path the head is: a stop-go curve over the first 78% of the cycle, then it rests.
  let n1 = 2.0 + floor(g.x * 3.0);
  let n2 = 5.0 + floor(h.z * 3.0);
  let travel = _droplet_stopgo(saturate(s / 0.78), n1, TAU * g.y, n2, TAU * h.x);
  let y_head = mix(DROPLET_Y_TOP, DROPLET_Y_BOTTOM, travel.x);
  let x0 = 0.5 + (g.x - 0.5) * 0.3;
  let phase = TAU * vec2f(h.z, fract(g.y * 7.31 + 0.13));
  let rest = mix(0.13, 0.22, g.y);
  let radius = rest * (1.0 - 0.3 * travel.x);       // loses water as it runs
  let r_head = radius * env;

  var out = droplet_none();

  // Head: an egg, stretched along the path while it moves, with a longer tail above.
  let stretch = 1.0 + 0.35 * saturate(travel.y - 0.4);
  let d = vec2f(lx - _droplet_path_x(y_head, x0, phase), ly - y_head);
  let dy = d.y / (stretch * select(1.0, 1.3, d.y < 0.0));
  let head_local = vec2f(d.x, dy);
  var best_mask = _droplet_cover(length(head_local) - r_head, r_head, cell, aa);
  var best_local = head_local / max(r_head, 1e-4);
  var best_radius = r_head;

  // Beads: one candidate per spacing along the path, thinning out and shrinking with distance
  // behind the head. Only the nearest one can touch this pixel, so that is all we test.
  let bi = floor((ly - DROPLET_Y_TOP) / _DROPLET_BEAD_SPACING + 0.5);
  let b = ihash_3to3(vec3i(col * 13 + i32(bi) + seed, row * 7 + 3, i32(k)));
  let by = DROPLET_Y_TOP + bi * _DROPLET_BEAD_SPACING + (b.y - 0.5) * 0.12;
  let behind = y_head - by;
  if (b.z < 0.72 && behind > 0.2 && by > DROPLET_Y_TOP - 0.05) {
    let age = saturate(behind / (DROPLET_Y_BOTTOM - DROPLET_Y_TOP));
    let rb = (0.045 + 0.05 * b.x) * (1.0 - 0.6 * age) * smoothstep(0.2, 0.7, behind) * env;
    let bx = _droplet_path_x(by, x0, phase) + (b.x - 0.5) * 0.1;
    let bd = vec2f(lx - bx, ly - by);
    // Whichever covers more of the pixel wins (not whichever is nearer: a speck would beat a bead).
    let bead_mask = _droplet_cover(length(bd) - rb, rb, cell, aa);
    if (bead_mask > best_mask) {
      best_mask = bead_mask;
      best_local = bd / max(rb, 1e-4);
      best_radius = rb;
    }
  }

  if (best_mask > 0.0 && best_radius > 0.0) {
    out.mask = best_mask;
    out.local = best_local;
    out.radius = best_radius * cell;
  }

  // The wipe: a streak along the path from where the drop formed to its head. It is narrower than the
  // drop and tapers away from the head (the old end re-fogs from its edges), clearest at the head.
  let path_dx = abs(lx - _droplet_path_x(ly, x0, phase));
  let recency = exp(-max(y_head - ly, 0.0) * 0.45);
  let width = 0.5 * rest * (1.0 - 0.3 * travel.x) * (0.45 + 0.55 * recency);
  let lateral = 1.0 - smoothstep(0.7 * width, 1.5 * width, path_dx);
  let along = smoothstep(DROPLET_Y_TOP - 0.15, DROPLET_Y_TOP + 0.25, ly)
    * (1.0 - smoothstep(y_head, y_head + r_head, ly));
  out.clear = max(lateral * along * (0.25 + 0.75 * recency) * env, out.mask);
  return out;
}

fn droplet_merge(a: DropletSample, b: DropletSample) -> DropletSample {
  var out = a;
  if (b.mask > a.mask) {
    out = b;
  }
  out.clear = max(a.clear, b.clear);
  return out;
}

fn droplet_field(p: vec2f, time: f32, rain: f32, size: f32, layers: i32, aa: f32) -> DropletSample {
  let r = saturate(rain);
  var s = droplet_running(p, time, size, 0.95 * r, 1, aa);
  if (layers >= 2) {
    s = droplet_merge(s, droplet_running(p, time * 1.15, size * 0.62, 0.8 * r, 2, aa));
  }
  var beads = droplet_static(p, time, size * 0.3, saturate(0.4 * sqrt(r) + 0.6 * r), 3, aa);
  if (layers >= 3) {
    let mist = droplet_static(p, time, size * 0.12, 0.7 * r, 4, aa);
    if (mist.mask > beads.mask) {
      beads = mist;
    }
  }
  // A passing drop sweeps up the beads on its path.
  beads.mask *= 1.0 - s.clear;
  return droplet_merge(s, beads);
}

fn droplet_lens(s: DropletSample, strength: f32) -> vec2f {
  return -s.local * s.radius * strength;
}

fn droplet_normal(s: DropletSample, flatness: f32) -> vec3f {
  return normal_from_xy(s.local * flatness);
}

fn droplet_rim(s: DropletSample) -> f32 {
  let r = min(length(s.local), 1.0);
  return r * r * r * r;
}

fn droplet_highlight(normal: vec3f, light: vec3f, sharpness: f32) -> f32 {
  let h = normalize(light + vec3f(0.0, 0.0, 1.0));
  return pow(max(dot(normal, h), 0.0), max(sharpness, 1.0));
}

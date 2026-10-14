// The heads of the ink trails: a handful of points that wander the canvas on the CPU and are drawn
// into loose orbits round the page's elements. index.ts lays ink and momentum into the fluid along the
// path each one takes; the fluid does the rest (the curl, the spreading, the fading).
//
// THE FORCE MODEL is the one of hologpu's DOM attractors, per head instead of per particle. Each
// element acts through the signed distance d to its outline and the outward normal n there:
//
//     attract   (-n * clamp((d - shell) * 5, -1.5, 1) + spin * tangent * swirl) * exp(-5 d)
//     repel       n * 1.8 * exp(-13 d)
//
// a spring towards a shell a little outside the edge, plus a constant push along the edge. It is not
// gravity (no 1 / r^2, nothing to soften) and on its own it gives a tidy band of circling points. Drag
// (v *= exp(-drag dt)) and a speed limit keep it bounded.
//
// WHAT MAKES IT UNTIDY. A slow field of sines pushes every head about (`wander`). No two heads are
// alike: each has its own shell distance, swirl, drag and top speed, so they do not share a track. And
// each one is RESTLESS: every few seconds it lets go, is pushed off its element for a moment and is
// caught by whichever element is nearest when it settles again. With a dozen elements pulling at once
// the paths never repeat.
//
// Space is image heights (x in 0..aspect, y in 0..1, y down), speeds are image heights per second.
// Everything here is pure and takes its random numbers from the caller, so it can be tested.

/** A rounded box and what it does to a head. */
export interface HeadBox {
  cx: number;
  cy: number;
  hx: number;
  hy: number;
  radius: number;
  /** +1 swings heads clockwise on screen, -1 counter-clockwise; 0 only pushes them away. */
  spin: number;
}

export interface Head {
  x: number;
  y: number;
  vx: number;
  vy: number;
  /** Seconds since it was (re)born, and the age it is reborn at. */
  age: number;
  life: number;
  /** Position in the palette; drifts as it goes, so a trail changes colour along its length. */
  hue: number;
  /** Which ink of a swatch palette it carries: its number, plus one for every time it was reborn. */
  colour: number;
  /** Its own character, each a multiplier round 1 (see `createHead`). */
  shell: number;
  swirl: number;
  drag: number;
  pace: number;
  /** Width of its trail, a multiplier. */
  size: number;
  /** Offsets the wander field in time, so two heads in one place are pushed differently. */
  offset: number;
  /** Seconds until it next lets go, and seconds of letting go that remain. */
  calm: number;
  loose: number;
}

export interface HeadField {
  boxes: readonly HeadBox[];
  /** Canvas width / height. */
  aspect: number;
  /** Seconds, for the wander field. */
  time: number;
  /** Strength of the elements' forces; 0 leaves only the wander. */
  pull: number;
  /** Tangential push round an attracting element. hologpu: 0.48. */
  swirl: number;
  /** Distance outside the edge the spring settles at, image heights. hologpu: 0.028. */
  shell: number;
  /** Strength of the wander field. 1 is hologpu's ambient drift. */
  wander: number;
  /** How restless the heads are: 0 never lets go, 1 every few seconds. */
  chaos: number;
  /** Top speed, image heights per second. hologpu: 0.5. */
  maxSpeed: number;
  /** How far outside every element a head of size 1 is kept, image heights. */
  clearance: number;
}

const clamp = (x: number, lo: number, hi: number) => Math.min(hi, Math.max(lo, x));

/** Signed distance from (x, y) to the box's outline (negative inside) and the outward normal there. */
export function boxDistance(
  x: number,
  y: number,
  box: HeadBox,
): { d: number; nx: number; ny: number } {
  const r = Math.min(box.radius, box.hx, box.hy);
  const lx = x - box.cx;
  const ly = y - box.cy;
  const qx = Math.abs(lx) - (box.hx - r);
  const qy = Math.abs(ly) - (box.hy - r);
  const ox = Math.max(qx, 0);
  const oy = Math.max(qy, 0);
  const outside = Math.hypot(ox, oy);
  const d = outside + Math.min(Math.max(qx, qy), 0) - r;
  const sx = lx >= 0 ? 1 : -1;
  const sy = ly >= 0 ? 1 : -1;
  if (outside > 1e-6) return { d, nx: (ox / outside) * sx, ny: (oy / outside) * sy };
  return qx > qy ? { d, nx: sx, ny: 0 } : { d, nx: 0, ny: sy };
}

/** The acceleration a head feels where it is. */
export function headForce(head: Head, field: HeadField): [number, number] {
  const { x, y } = head;
  // hologpu's ambient drift: a few sines of position that turn slowly.
  const t = field.time * 0.13 + head.offset;
  const gust = 0.09 * field.wander * (head.loose > 0 ? 1 + 2 * field.chaos : 1);
  let fx = (Math.sin(y * 9 + t) + Math.cos(x * 5 - y * 3 - t)) * gust;
  let fy = (Math.cos(x * 8 - t) - Math.sin(y * 6 + x * 2 + t)) * gust;

  // Never closer than the trail is wide: the spring must not fight the wall.
  const shell = Math.max(field.shell * head.shell, field.clearance * head.size * 1.2);
  for (const box of field.boxes) {
    const { d, nx, ny } = boxDistance(x, y, box);
    const out = Math.max(d, 0);
    if (box.spin === 0 || head.loose > 0) {
      // Pushed off: a repelling element always, any element while the head is letting go.
      const push = (box.spin === 0 ? 1.8 : 0.9) * Math.exp(-out * 13) * field.pull;
      fx += nx * push;
      fy += ny * push;
      continue;
    }
    const weight = Math.exp(-out * 5) * field.pull;
    const spring = -clamp((d - shell) * 5, -1.5, 1);
    const swing = field.swirl * head.swirl * box.spin;
    // Clockwise on screen (y down) is (-n.y, n.x).
    fx += (nx * spring - ny * swing) * weight;
    fy += (ny * spring + nx * swing) * weight;
  }

  // The canvas edge turns heads back before they reach it.
  const margin = 0.04;
  fx += 2 * (Math.exp(-x / margin) - Math.exp(-(field.aspect - x) / margin));
  fy += 2 * (Math.exp(-y / margin) - Math.exp(-(1 - y) / margin));
  return [fx, fy];
}

/** A point of open canvas: not inside (or hard against) any element. */
function openPoint(field: HeadField, random: () => number): [number, number] {
  let x = 0.5 * field.aspect;
  let y = 0.5;
  for (let attempt = 0; attempt < 12; attempt++) {
    x = (0.04 + 0.92 * random()) * field.aspect;
    y = 0.05 + 0.9 * random();
    if (field.boxes.every((box) => boxDistance(x, y, box).d > field.clearance * 1.5)) break;
  }
  return [x, y];
}

/** Puts a head somewhere new, at rest, with a new span of life. Its character and hue are kept. */
export function respawnHead(head: Head, field: HeadField, random: () => number): void {
  [head.x, head.y] = openPoint(field, random);
  head.vx = 0;
  head.vy = 0;
  head.age = 0;
  head.life = 12 + 12 * random();
  head.calm = 2 + 6 * random();
  head.loose = 0;
}

export function createHead(index: number, field: HeadField, random: () => number): Head {
  const head: Head = {
    x: 0,
    y: 0,
    vx: 0,
    vy: 0,
    age: 0,
    life: 0,
    // The golden angle: neighbours in the list are far apart in the palette.
    hue: index * 0.381966,
    colour: index,
    shell: 0.5 + 2 * random() * random(),
    swirl: 0.55 + 0.9 * random(),
    drag: 1.2 + 1.1 * random(),
    pace: 0.45 + 0.55 * random(),
    size: 0.7 + 0.7 * random(),
    offset: random() * 40,
    calm: 0,
    loose: 0,
  };
  respawnHead(head, field, random);
  // Born at a random point of its life, so the heads do not all die together.
  head.age = head.life * random() * 0.8;
  return head;
}

/** Longest step the integration takes; a longer `dt` is cut up. */
const MAX_STEP = 1 / 60;

/**
 * Moves a head on by `dt` seconds. Returns false when it did not travel there (it was reborn, or an
 * element moved over it and shoved it out): the trail must not be drawn across that jump.
 */
export function stepHead(head: Head, field: HeadField, dt: number, random: () => number): boolean {
  let travelled = true;
  const steps = Math.max(1, Math.ceil(dt / MAX_STEP));
  const h = dt / steps;
  const top = field.maxSpeed * head.pace;
  const clearance = field.clearance * head.size;
  for (let step = 0; step < steps; step++) {
    head.age += h;
    if (head.age > head.life) {
      head.colour++;
      respawnHead(head, field, random);
      travelled = false;
      continue;
    }

    // Restlessness: now and then let go for a moment, with a shove in a random direction.
    if (head.loose > 0) {
      head.loose -= h;
    } else if (field.chaos > 0) {
      head.calm -= h * field.chaos;
      if (head.calm <= 0) {
        head.calm = 2.5 + 7 * random();
        head.loose = 0.5 + 0.9 * random();
        const angle = random() * Math.PI * 2;
        head.vx += Math.cos(angle) * top * 0.8;
        head.vy += Math.sin(angle) * top * 0.8;
      }
    }

    const [fx, fy] = headForce(head, field);
    const keep = Math.exp(-head.drag * h);
    head.vx = (head.vx + fx * h) * keep;
    head.vy = (head.vy + fy * h) * keep;
    const speed = Math.hypot(head.vx, head.vy);
    if (speed > top) {
      head.vx *= top / speed;
      head.vy *= top / speed;
    }
    head.x += head.vx * h;
    head.y += head.vy * h;

    // Elements are solid: a head that ends up inside one is put back outside and loses the part of its
    // velocity that took it in. A big correction means the element moved onto it, not the reverse.
    for (const box of field.boxes) {
      const { d, nx, ny } = boxDistance(head.x, head.y, box);
      if (d >= clearance) continue;
      const shove = clearance - d;
      head.x += nx * shove;
      head.y += ny * shove;
      const inward = head.vx * nx + head.vy * ny;
      if (inward < 0) {
        head.vx -= nx * inward;
        head.vy -= ny * inward;
      }
      if (shove > clearance + 0.03) travelled = false;
    }
    head.x = clamp(head.x, 0.004, field.aspect - 0.004);
    head.y = clamp(head.y, 0.004, 1 - 0.004);
    head.hue += h * 0.05;
  }
  return travelled;
}

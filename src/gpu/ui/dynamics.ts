/** Exact damped spring response to a fixed target for one interval. Stable at any frame rate. */
export interface Spring {
  value: number;
  velocity: number;
}
export function advanceSpring(
  s: Spring,
  target: number,
  dt: number,
  frequency: number,
  damping: number,
): void {
  if (!(dt > 0)) return;
  const w = Math.max(0.01, frequency) * 2 * Math.PI;
  const z = Math.max(0.01, damping);
  const x = s.value - target;
  if (Math.abs(z - 1) < 0.001) {
    const b = s.velocity + w * x,
      e = Math.exp(-w * dt);
    s.value = target + (x + b * dt) * e;
    s.velocity = (s.velocity - w * b * dt) * e;
  } else if (z < 1) {
    const wd = w * Math.sqrt(1 - z * z),
      a = z * w;
    const b = (s.velocity + a * x) / wd,
      c = Math.cos(wd * dt),
      n = Math.sin(wd * dt),
      e = Math.exp(-a * dt);
    s.value = target + e * (x * c + b * n);
    s.velocity = e * ((b * wd - a * x) * c - (x * wd + a * b) * n);
  } else {
    const root = Math.sqrt(z * z - 1),
      r1 = -w * (z - root),
      r2 = -w * (z + root);
    const a = (s.velocity - r2 * x) / (r1 - r2),
      b = x - a;
    s.value = target + a * Math.exp(r1 * dt) + b * Math.exp(r2 * dt);
    s.velocity = a * r1 * Math.exp(r1 * dt) + b * r2 * Math.exp(r2 * dt);
  }
}
export interface Drop {
  x: number;
  y: number;
  radius: number;
  velocity: number;
}
/** Conserves spherical drop volume (r³) and its weighted centroid during coalescence. */
export function mergeDrops(a: Drop, b: Drop): Drop {
  const va = a.radius ** 3,
    vb = b.radius ** 3,
    v = va + vb;
  return {
    x: (a.x * va + b.x * vb) / v,
    y: (a.y * va + b.y * vb) / v,
    radius: Math.cbrt(v),
    velocity: (a.velocity * va + b.velocity * vb) / v,
  };
}
export const MAX_DROPS = 48;
export function seedDrops(): Drop[] {
  return Array.from({ length: 32 }, (_, i) => ({
    x: 0.16 + ((i * 0.618034) % 1) * 0.68,
    y: 0.16 + ((i * 0.414214) % 1) * 0.63,
    radius: 0.007 + (i % 5) * 0.003,
    velocity: 0,
  }));
}
/** Bounded CPU condensation: capillary pinning, gravity, swept collisions, replenishment at top. */
export function advanceDrops(drops: Drop[], dt: number, gravity: number, aspect = 1): void {
  if (!(dt > 0)) return;
  const steps = Math.max(1, Math.ceil(Math.min(dt, 0.1) * 120)),
    h = Math.min(dt, 0.1) / steps;
  for (let s = 0; s < steps; s++) {
    const previous = drops.map((d) => d.y);
    for (const d of drops) {
      const mobile = Math.max(0, (d.radius - 0.013) * 80);
      d.velocity = (d.velocity + gravity * mobile * h) * Math.exp(-3 * h);
      d.y += d.velocity * h;
      d.radius += d.radius < 0.05 ? h * 0.0008 : 0;
    }
    for (let i = 0; i < drops.length; i++)
      for (let j = i + 1; j < drops.length; j++) {
        const a = drops[i]!,
          b = drops[j]!;
        const dx = (a.x - b.x) * aspect,
          y0 = previous[i]! - previous[j]!,
          y1 = a.y - b.y;
        const dy = y0 * y1 <= 0 ? 0 : Math.min(Math.abs(y0), Math.abs(y1));
        if (Math.hypot(dx, dy) < a.radius + b.radius) {
          drops[i] = mergeDrops(a, b);
          drops.splice(j, 1);
          previous.splice(j, 1);
          j--;
        }
      }
    for (let i = drops.length - 1; i >= 0; i--)
      if (drops[i]!.y - drops[i]!.radius > 0.86) drops.splice(i, 1);
  }
}
/** Inclusive end points keep both section images intact at progress 0 and 1. */
export function transitionProgress(clock: number): number {
  return 0.5 - 0.5 * Math.cos(clock * 0.7);
}
export function segmentDistance(
  x: number,
  y: number,
  ax: number,
  ay: number,
  bx: number,
  by: number,
): number {
  const dx = bx - ax,
    dy = by - ay,
    h = Math.max(0, Math.min(1, ((x - ax) * dx + (y - ay) * dy) / (dx * dx + dy * dy || 1)));
  return Math.hypot(x - ax - h * dx, y - ay - h * dy);
}

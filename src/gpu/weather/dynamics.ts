/** Bounded CPU particles: no compute feature required for the Tier A weather effects. */
export type DropKind = 0 | 1 | 2 | 3;
export interface WeatherDrop {
  x: number;
  y: number;
  vx: number;
  vy: number;
  radius: number;
  age: number;
  life: number;
  depth: number;
  kind: DropKind;
}
export const MAX_WEATHER_DROPS = 2048;
export function seededRandom(seed: number) {
  let state = seed >>> 0 || 1;
  return () => {
    state ^= state << 13;
    state ^= state >>> 17;
    state ^= state << 5;
    return (state >>> 0) / 4294967296;
  };
}
export function newRain(
  random: () => number,
  aspect: number,
  wind: number,
  initial = false,
): WeatherDrop {
  const depth = 0.25 + 0.75 * random();
  return {
    x: random() * aspect,
    y: initial ? random() * 0.8 : -0.08,
    vx: wind * (0.12 + depth * 0.3),
    vy: 0.48 + depth * 1.45,
    radius: 0.0007 + depth * 0.0015,
    age: 0,
    life: 8,
    depth,
    kind: 0,
  };
}
export function newBubble(random: () => number, aspect: number, initial = false): WeatherDrop {
  const depth = 0.3 + random() * 0.7;
  return {
    x: random() * aspect,
    y: initial ? random() : 1.08,
    vx: 0,
    vy: -(0.055 + depth * 0.14),
    radius: 0.015 + random() ** 2 * 0.062,
    age: random() * 8,
    life: 100,
    depth,
    kind: 2,
  };
}
export function splashBurst(
  drops: WeatherDrop[],
  x: number,
  y: number,
  random: () => number,
  count: number,
  strength = 1,
  impact = false,
) {
  for (let i = 0; i < count && drops.length < MAX_WEATHER_DROPS; i++) {
    const angle = impact ? -Math.PI * (0.1 + random() * 0.8) : random() * Math.PI * 2;
    const speed = (0.08 + random() * 0.35) * strength;
    drops.push({
      x,
      y,
      vx: Math.cos(angle) * speed,
      vy: Math.sin(angle) * speed - (impact ? 0.08 : 0.06),
      radius: 0.0025 + random() * 0.006,
      age: 0,
      life: 0.4 + random() * 1.05,
      depth: 1,
      kind: 1,
    });
  }
}
export function advanceWeather(
  drops: WeatherDrop[],
  seconds: number,
  aspect: number,
  wind: number,
  gravity: number,
  random: () => number,
) {
  const impacts: { x: number; y: number }[] = [];
  for (const p of drops) {
    p.age += seconds;
    if (p.kind === 0) {
      p.vx = wind * (0.12 + p.depth * 0.3);
      p.x += p.vx * seconds;
      p.y += p.vy * seconds;
      const ground = 0.79 + p.depth * 0.15;
      if (p.y >= ground) {
        impacts.push({ x: p.x, y: ground });
        p.life = 0;
      }
    } else if (p.kind === 1) {
      p.x += p.vx * seconds;
      p.y += p.vy * seconds + 0.5 * gravity * seconds * seconds;
      p.vy += gravity * seconds;
    } else if (p.kind === 2) {
      // Use the exact integral of the wobble velocity, so motion is independent of frame partition.
      p.x +=
        0.015 *
        (Math.sin(p.age * 2 + p.depth * 19) - Math.sin((p.age - seconds) * 2 + p.depth * 19));
      p.y += p.vy * seconds;
    }
    p.x = ((p.x % aspect) + aspect) % aspect;
  }
  for (let i = drops.length - 1; i >= 0; i--) {
    const p = drops[i]!;
    if (p.age >= p.life || p.y < -0.12 || p.y > 1.2) drops.splice(i, 1);
  }
  for (const hit of impacts) {
    if (drops.length < MAX_WEATHER_DROPS)
      drops.push({
        x: hit.x,
        y: hit.y,
        vx: 0,
        vy: 0,
        radius: 0.015,
        age: 0,
        life: 0.6,
        depth: 1,
        kind: 3,
      });
    splashBurst(drops, hit.x, hit.y, random, 3, 0.55, true);
  }
  return impacts.length;
}
export function packWeather(drops: readonly WeatherDrop[]) {
  const data = new Float32Array(drops.length * 12);
  drops.forEach((p, i) =>
    data.set([p.x, p.y, p.vx, p.vy, p.radius, p.age, p.life, p.depth, p.kind, 0, 0, 0], i * 12),
  );
  return data;
}

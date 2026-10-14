/** A turn or a nearby pointer excites a head; light decays smoothly after the encounter. */
export function biolumePulse(
  previous: number,
  before: readonly number[],
  after: readonly number[],
  dt: number,
  proximity: number,
): number {
  const speed = Math.hypot(before[0]!, before[1]!) * Math.hypot(after[0]!, after[1]!);
  const turn = speed > 1e-4 ? Math.abs(before[0]! * after[1]! - before[1]! * after[0]!) / speed : 0;
  return Math.max(
    previous * Math.exp(-dt * 3.5),
    Math.min(1, (turn / Math.max(dt, 1e-4)) * 0.1),
    Math.max(0, Math.min(1, proximity)),
  );
}

export interface EmberSpark {
  x: number;
  y: number;
  vx: number;
  vy: number;
  age: number;
  life: number;
  radius: number;
}

/** Detached sparks coast away from the head, rise and cool. False means the spark has expired. */
export function stepSpark(spark: EmberSpark, dt: number): boolean {
  spark.age += dt;
  const drag = Math.exp(-dt * 1.2);
  spark.vx *= drag;
  spark.vy = spark.vy * drag - 0.025 * dt;
  spark.x += spark.vx * dt;
  spark.y += spark.vy * dt;
  return spark.age < spark.life;
}

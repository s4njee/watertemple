import type { ObstacleShape } from '../../gpu/sim/obstacle-shapes.ts';

// The shapes of each scene of the wind tunnel, in canvas uv with sizes in image heights. The wind
// blows from the left; `size` is the characteristic length (the diameter of the cylinder, half the
// height of the plate) and `angle` the angle of attack of the wing, in degrees.

export const SCENES = ['auto', 'cylinder', 'plates', 'wing', 'slalom', 'page elements'] as const;
export type Scene = (typeof SCENES)[number];

export interface SceneOptions {
  size: number;
  /** Degrees; positive lifts the nose (the angle of attack). */
  angle: number;
  aspect: number;
}

/** The analytic shapes of a scene (empty for the page elements). */
export function sceneShapes(scene: Exclude<Scene, 'auto'>, o: SceneOptions): ObstacleShape[] {
  const { size, aspect } = o;
  // Keep the pieces inside a narrow canvas: positions are fractions of the width, sizes are heights.
  const x = (fraction: number) => Math.min(fraction, 0.9);
  switch (scene) {
    case 'cylinder':
      return [
        { kind: 'circle', x: x(aspect > 1.3 ? 0.3 : 0.4), y: 0.5, size: [size / 2, size / 2] },
      ];
    case 'plates':
      return [
        { kind: 'box', x: x(0.28), y: 0.5, size: [0.012, size * 0.6] },
        { kind: 'box', x: x(0.62), y: 0.27, size: [size * 0.5, 0.012], rotation: 0.45 },
        { kind: 'box', x: x(0.62), y: 0.73, size: [size * 0.5, 0.012], rotation: -0.45 },
      ];
    case 'wing':
      // The wing's own +x is its tail, pointing downstream. A positive angle of attack lifts the nose,
      // which drops the tail: a clockwise turn on the screen, a positive rotation.
      return [
        {
          kind: 'wing',
          x: x(0.4),
          y: 0.5,
          size: [size * 1.5, size * 0.3],
          rotation: (o.angle * Math.PI) / 180,
        },
      ];
    case 'slalom': {
      const pegs: ObstacleShape[] = [];
      const columns = aspect > 1.3 ? 5 : 3;
      for (let i = 0; i < columns; i++) {
        pegs.push({
          kind: 'circle',
          x: 0.2 + (0.65 * i) / (columns - 1),
          y: i % 2 === 0 ? 0.42 : 0.6,
          size: [size * 0.22, size * 0.22],
        });
      }
      return pegs;
    }
    case 'page elements':
      return [];
  }
}

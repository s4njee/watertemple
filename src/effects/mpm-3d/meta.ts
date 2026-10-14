import type { EffectMeta } from '../types.ts';

export const meta: EffectMeta = {
  id: 'mpm-3d',
  title: 'MLS-MPM water tank',
  epic: '7',
  tier: 'C',
  tags: ['particles', 'simulation', 'compute', 'fluid', 'interactive', '3d', 'showpiece'],
  summary:
    'A tank of water simulated with the moving-least-squares material point method: 100 000 particles carry the fluid, a background grid does the physics, and there is no neighbour search. A wall pushes in and out and sends waves down the tank to break on a beach; or let a dam burst around pillars, or pour a stream into a pool. Drag through the water to push it, tap to drop a blob, drag empty space to orbit. Drawn as a refracting liquid by the screen-space fluid surface, or as the spheres, speed or density it is made of.',
  scorecard: {
    gpuMs: [
      { device: 'Apple M1 Max · Dawn/Metal (workload slope)', ms: 4.81, resolution: '1920 × 1080' },
    ],
    bundleKb: 20.06,
    // Every simulation pass is a compute dispatch with integer atomics; there is no render-pass equivalent.
    compute: true,
    // Not run on a phone; expected to fail the budget (see "When to use it" in the notes).
    mobile: 'untested',
    features: { required: [], optional: ['timestamp-query'] },
    fallback:
      'A looping video or poster of the tank. Nothing cheaper reproduces it: the fluid needs compute shaders and atomics, so a WebGL2 version would be a different effect (a shallow-water heightfield, see the shallow-water effect, or a ray-marched blob, Epic 9).',
    useCases: [
      'A dedicated showpiece page where a body of water sloshing, breaking and splashing is the product: a tank game, a drink, a water toy, a landing experience',
      'The reference for a particle fluid with no neighbour search, at counts the neighbour-search solvers (SPH, position-based fluids) cannot reach',
      'Not a background or a card, and not for phones: it needs a discrete or Apple Silicon GPU and most of a Tier C frame budget',
    ],
  },
};

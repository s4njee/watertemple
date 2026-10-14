import type { EffectMeta } from '../types.ts';

export const meta: EffectMeta = {
  id: 'pbf-2d',
  title: '2D position-based fluids',
  epic: '7',
  tier: 'B',
  tags: ['particles', 'simulation', 'compute', 'fluid', 'interactive', '2d'],
  summary:
    'A tank of water as particles solved with position-based fluids (Macklin and Mueller): incompressibility is a constraint on positions, relaxed by a few parallel iterations per sub-step, so a big time step cannot blow the water up. A dam break that collapses and sloshes, a stream pouring onto a shelf, a wave-maker, drops falling into a calm pool. Drag to push the water, tap to drop a blob. Colour the particles by speed, by density error (how well the constraint holds) or by neighbour count; change the particle spacing and the same parameters give the same water.',
  scorecard: {
    gpuMs: [
      { device: 'Apple M1 Max · Dawn/Metal (workload slope)', ms: 1.7, resolution: '1920 × 1080' },
    ],
    bundleKb: 4.7,
    // Every pass but the drawing is a compute dispatch: the neighbour grid, the density and correction iterations, viscosity.
    compute: true,
    mobile: 'untested',
    features: { required: [], optional: ['timestamp-query'] },
    fallback:
      'Without WebGPU or compute, show a looping video of the same scene (the dam break and the pour loop cleanly), or a heightfield water surface (Epic 4) if the point is only a sloshing top surface; the particle solver has no fragment-shader equivalent.',
    useCases: [
      'A hero or set piece where a body of water sloshes, splashes and can be pushed around, on desktop-class GPUs at around ten thousand particles',
      'A loading or empty-state illustration: a stream filling a vessel, or drops falling into a pool',
      'The 2D stepping stone to the 3D position-based fluids: the same solver, the same parameters, a quarter of the particles per unit of detail',
    ],
  },
};

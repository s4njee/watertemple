import type { EffectMeta } from '../types.ts';

export const meta: EffectMeta = {
  id: 'sph-2d',
  title: '2D SPH liquid',
  epic: '7',
  tier: 'B',
  tags: ['particles', 'simulation', 'compute', 'fluid', 'interactive', '2d'],
  summary:
    'A tank of water as thousands of particles: weakly compressible smoothed-particle hydrodynamics in compute shaders, drawn as one liquid body (soft blobs, a threshold, a rim, a highlight, refraction) instead of discs. A dam break that collapses and sloshes back, a stream pouring onto a shelf and an obstacle, two zero-gravity blobs that merge under surface tension. Drag to push the water, tap to drop a block of it.',
  scorecard: {
    gpuMs: [
      { device: 'Apple M1 Max · Dawn/Metal (workload slope)', ms: 2.08, resolution: '1920 × 1080' },
    ],
    bundleKb: 18.76,
    // Every pass but the drawing is a compute dispatch: grid build, density, forces and integration.
    compute: true,
    mobile: 'untested',
    features: { required: [], optional: ['timestamp-query'] },
    fallback:
      'Without WebGPU or compute, show a looping video of the same scene (the dam break and the pour loop cleanly), or a WebGL2 version of the drawing over a few thousand particles stepped on the CPU; the simulation has no fragment-shader equivalent.',
    useCases: [
      'A hero or set piece where a body of water sloshes, splashes and can be pushed around, on desktop-class GPUs at around ten thousand particles',
      'A loading or empty-state illustration: a stream filling a vessel, or a droplet that merges with another',
      'The reference for particle-based liquid in 2D: the same solver and renderer serve the position-based-fluids comparison',
    ],
  },
};

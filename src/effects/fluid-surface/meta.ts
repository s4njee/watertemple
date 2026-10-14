import type { EffectMeta } from '../types.ts';

export const meta: EffectMeta = {
  id: 'fluid-surface',
  title: 'Screen-space fluid surface',
  epic: '7',
  tier: 'C',
  tags: ['particles', 'rendering', '3d', 'refraction', 'showpiece'],
  summary:
    'A cloud of particles turned into a continuous, refracting liquid: sphere impostors, depth smoothing (bilateral against narrow-range), thickness, normals and a composite with absorption, reflection and a sun highlight. Three synthetic sources (a ball that splits and merges, a wave, a pouring stream) stand in for a solver; one control swaps between the finished surface, the bag of marbles it replaces and each intermediate image.',
  scorecard: {
    bundleKb: 3.94,
    gpuMs: [
      { device: 'Apple M1 Max · Dawn/Metal (workload slope)', ms: 2.73, resolution: '1920 × 1080' },
    ],
    // The renderer is vertex and fragment passes only. The demo's particle sources are a compute
    // pass (so a million analytic particles cost no CPU), and the solvers this will serve need it.
    compute: true,
    mobile: 'untested',
    features: { required: [], optional: ['timestamp-query'] },
    fallback:
      'A looping video or poster of the liquid. There is no useful cut-down without WebGPU: the particles come from a compute solver, so a WebGL2 version would be a different effect (a metaball SDF blob, see Epic 9).',
    useCases: [
      'A dedicated showpiece page where a pouring, sloshing or splashing liquid is the product (a drink, a pool toy, a tank game)',
      'The renderer behind any particle solver in Epic 7: SPH, position-based fluids or MLS-MPM hand over positions and get a liquid',
      'Not a background or a card: on a laptop it needs the whole GPU budget of a Tier C page',
    ],
  },
};

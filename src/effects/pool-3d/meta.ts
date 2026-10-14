import type { EffectMeta } from '../types.ts';

export const meta: EffectMeta = {
  id: 'pool-3d',
  title: '3D pool',
  epic: '5',
  tier: 'B',
  tags: ['interactive', 'simulation', 'caustics', '3d', 'heightfield', 'ray-tracing', 'showpiece'],
  summary:
    'A pool in perspective with an orbiting camera, a ball you can drag through the water, ripples that make real caustics on the floor and walls, and refraction and reflection ray-traced against the pool box and the ball.',
  scorecard: {
    // The default path is render passes only: solver, caustics and the scene.
    compute: false,
    mobile: 'untested',
    features: { required: [] },
    fallback:
      'A looping video of the pool with the ball bobbing (orbit and drag are lost), or the ripple-caustics floor effect, which needs no 3D scene.',
    useCases: [
      'A dedicated landing page, product configurator or game-like hero where the water is the subject',
      'A desktop showpiece that proves what the site can do; not a background behind text',
      'Reference for ray-tracing a few analytic shapes through a refracting surface inside a rasterised scene',
    ],
  },
};

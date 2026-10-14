import type { EffectMeta } from '../types.ts';

export const meta: EffectMeta = {
  id: 'pool-floor',
  title: 'Pool floor',
  epic: '4',
  tier: 'A',
  tags: ['background', 'hero', 'pointer', 'caustics', 'refraction', 'absorption', 'composition'],
  summary:
    'A swimming pool seen from above: tiles refracted through moving water, lit by caustics, tinted by depth. Every optics chunk in one cheap pass.',
  scorecard: {
    compute: false,
    mobile: 'untested',
    features: { required: [] },
    fallback:
      'A looping video or animated WebP of the same pool (it loops cleanly: the ripple and caustic phases are periodic), or a still photo of tiles with the baked caustic map as a CSS overlay.',
    useCases: [
      'A hero or section background for a pool, spa, travel, swimwear or hydration brand',
      'The ambient pointer-reactive water surface over any tiled or photographic floor',
      'A reference for how refraction, absorption and caustics compose in one pass',
    ],
  },
};

import type { EffectMeta } from '../types.ts';

export const meta: EffectMeta = {
  id: 'caustics',
  title: 'Procedural caustics',
  epic: '4',
  tier: 'A',
  tags: ['overlay', 'background', 'caustics', 'building-block', 'tileable', 'comparison'],
  summary:
    'The light pattern at the bottom of a pool as a transparent, tileable overlay. Two procedural constructions (warped Voronoi and iterated sine) and a baked one- or two-tap version, compared on cost and look.',
  scorecard: {
    compute: false,
    mobile: 'untested',
    features: { required: [] },
    fallback:
      'A looping WebP or MP4 of the baked caustic map as a CSS background with mix-blend-mode: screen (or a static tileable PNG), which is also what the baked method amounts to.',
    useCases: [
      'A shimmering light layer over a hero image, a card or a pool-coloured section',
      'The light source for pool floors, underwater scenes and seabeds (the other 4.x demos)',
      'Choosing between a per-pixel and a baked caustic for your own scene',
    ],
  },
};

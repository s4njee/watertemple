import type { EffectMeta } from '../types.ts';

export const meta: EffectMeta = {
  id: 'detail-normals',
  title: 'Detail normals',
  epic: '3',
  tier: 'A',
  tags: ['building-block', 'normals', 'shading', 'water'],
  summary:
    'Two or three scrolling layers of a baked, tileable ripple normal map, blended five ways so the classic flattening is visible.',
  scorecard: {
    compute: false,
    mobile: 'untested',
    features: { required: [] },
    fallback: 'A still frame of the lit surface; the ripples are decoration, not content.',
    useCases: [
      'Small-scale ripple detail on any water surface, from a calm lake to a pool floor',
      'The reference for how to layer normal maps without washing the detail out',
      'A cheap stand-in for per-pixel noise wherever a texture fetch is affordable',
    ],
  },
};

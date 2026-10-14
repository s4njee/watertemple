import type { EffectMeta } from '../types.ts';

export const meta: EffectMeta = {
  id: 'mist',
  title: 'Mist and steam layer',
  epic: '4',
  tier: 'A',
  tags: ['overlay', 'mist', 'steam', 'fog', 'scroll', 'parallax', 'fbm', 'transparent'],
  summary:
    'A translucent overlay of drifting mist or rising steam in three depth layers that shift at different rates as the page scrolls, with a hard cap on its opacity.',
  scorecard: {
    compute: false,
    mobile: 'untested',
    features: { required: [] },
    fallback:
      'Two or three stacked, pre-rendered transparent WebP or PNG wisps with a slow CSS translate animation and a mask-image gradient; or a plain CSS linear-gradient fade to the page colour.',
    useCases: [
      'Fog hugging the bottom of a hero image, or steam over a photograph of a hot spring',
      'A soft veil that thins toward one edge, laid over imagery to calm it before text',
      'Depth for a long page: layers that move at their own rates as the visitor scrolls',
    ],
  },
};

import type { EffectMeta } from '../types.ts';

export const meta: EffectMeta = {
  id: 'ink-wash',
  title: 'Ink wash background',
  epic: '4',
  tier: 'A',
  tags: ['background', 'watercolour', 'ink', 'fbm', 'low-contrast', 'paper', 'transparent'],
  summary:
    'A very slow watercolour field: soft blooms of pigment with darker pooled edges, granulation and paper fibres, rendered at a quarter of the resolution and upsampled.',
  scorecard: {
    compute: false,
    mobile: 'untested',
    features: { required: [] },
    fallback:
      'A tiled, pre-rendered WebP or SVG feTurbulence texture in the same palette on a CSS background; or two or three large blurred radial gradients if you do not need the paper grain.',
    useCases: [
      'A calm, low-contrast background behind long-form text, documentation or a portfolio',
      'A hero background that reads as paper and pigment rather than as a gradient',
      'Transparent paper mode: an ink layer over a page colour or an image, so the section keeps its own background',
    ],
  },
};

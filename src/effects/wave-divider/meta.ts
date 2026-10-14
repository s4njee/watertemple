import type { EffectMeta } from '../types.ts';

export const meta: EffectMeta = {
  id: 'wave-divider',
  title: 'Layered wave divider',
  epic: '4',
  tier: 'A',
  tags: ['divider', 'overlay', 'scroll', 'parallax', 'sdf', 'section-break', 'transparent'],
  summary:
    'Two to five layers of wave silhouettes with scroll parallax and a foam line, transparent above and filled with the next section’s colour below.',
  scorecard: {
    compute: false,
    mobile: 'untested',
    features: { required: [] },
    fallback:
      'An inline SVG with the same layered paths (a few hundred bytes each, animated with SMIL or CSS transforms if motion is wanted), or a single static SVG on a CSS background.',
    useCases: [
      'The break between a hero and the next section, or the top edge of a footer',
      'Capping the bottom of a coloured section (flip) so it ends in water rather than a ruler line',
      'Anywhere an SVG wave would do but you want per-pixel foam, five or more layers, or a response to scroll speed',
    ],
  },
};

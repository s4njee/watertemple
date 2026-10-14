import type { EffectMeta } from '../types.ts';

export const meta: EffectMeta = {
  id: 'toon-water',
  title: 'Toon water',
  epic: '4',
  tier: 'A',
  tags: ['stylised', 'foam', 'voronoi', 'illustrative', 'cheap'],
  summary:
    'Cel-shaded water in one cheap pass: flat depth bands, crisp animated cell-noise foam, a breathing shore ring, ink outlines and hard-edged glints around a little island.',
  scorecard: {
    compute: false,
    mobile: 'untested',
    features: { required: [] },
    fallback:
      'A static illustration or an SVG/CSS-animated wave pattern in the same palette: flat colours and hard edges survive being a still image better than any realistic water does.',
    useCases: [
      'Playful or illustrative sites, games, kids and education products',
      'A hero or footer background that is meant to look drawn, not photographed',
      'Anywhere a cheap, resolution-independent look is wanted over a photoreal one',
    ],
  },
};

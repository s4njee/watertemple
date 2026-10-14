import type { EffectMeta } from '../types.ts';

export const meta: EffectMeta = {
  id: 'depth-absorption',
  title: 'Depth absorption',
  epic: '3',
  tier: 'A',
  tags: ['building-block', 'absorption', 'beer-lambert', 'depth', 'color'],
  summary:
    'Beer-Lambert tint and scattering fog by water depth: a sloping basin with a drop-off and submerged objects, shown through colour alone.',
  scorecard: {
    compute: false,
    mobile: 'untested',
    features: { required: [] },
    fallback:
      'A static render of the basin, or a CSS gradient from the shallow colour to the deep colour behind the content.',
    useCases: [
      'Depth-dependent water colour for pool floors, lakes, shorelines and underwater scenes',
      'A designer-facing way to set water colour: pick the colour at a depth, get coefficients',
      'The absorption building block that later 3D effects feed with real depth-buffer values',
    ],
  },
};

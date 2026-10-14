import type { EffectMeta } from '../types.ts';
export const meta: EffectMeta = {
  id: 'frost-melt',
  title: 'Frost and melt',
  epic: '10',
  tier: 'B',
  tags: ['weather', 'water', 'interactive', 'glass'],
  summary: 'Dendritic ice spreads over glass, thaws into refracting beads and drains downward.',
  scorecard: {
    compute: false,
    bundleKb: 9.85,
    gpuMs: [
      {
        device: 'Apple M1 Max (Dawn Metal)',
        ms: 0.59,
        resolution: '1920×1080, conservative measured configurations',
      },
    ],
    mobile: 'untested',
    features: { required: [] },
    fallback: 'The captured still poster; pause the animation under reduced motion.',
    useCases: ['Hero windows, cards and interactive glass panels'],
  },
};

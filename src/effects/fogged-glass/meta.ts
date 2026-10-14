import type { EffectMeta } from '../types.ts';
export const meta: EffectMeta = {
  id: 'fogged-glass',
  title: 'Fogged glass',
  epic: '10',
  tier: 'B',
  tags: ['weather', 'water', 'interactive', 'glass'],
  summary: 'Persistent condensation can be wiped with a pointer and slowly refogs.',
  scorecard: {
    compute: false,
    bundleKb: 9.85,
    gpuMs: [
      {
        device: 'Apple M1 Max (Dawn Metal)',
        ms: 1.442,
        resolution: '1920×1080, conservative measured configurations',
      },
    ],
    mobile: 'untested',
    features: { required: [] },
    fallback: 'The captured still poster; pause the animation under reduced motion.',
    useCases: ['Hero windows, cards and interactive glass panels'],
  },
};

import type { EffectMeta } from '../types.ts';
export const meta: EffectMeta = {
  id: 'liquid-blob',
  title: 'Ray-marched liquid blob',
  epic: '9',
  tier: 'B',
  tags: ['ui', 'web-native', 'interactive'],
  summary:
    'A three-dimensional smooth metaball lens refracts the environment through an exit surface.',
  scorecard: {
    compute: false,
    mobile: 'untested',
    bundleKb: 54.0,
    gpuMs: [{ device: 'Apple M1 Max (Metal)', ms: 0.437, resolution: '1920 × 1080' }],
    features: { optional: ['timestamp-query'] },
    fallback: 'A static poster with semantic HTML for the component and its content.',
    useCases: ['Website components', 'Liquid interface research'],
  },
};

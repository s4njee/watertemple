import type { EffectMeta } from '../types.ts';
export const meta: EffectMeta = {
  id: 'water-cursor',
  title: 'Water cursor trail',
  epic: '9',
  tier: 'A',
  tags: ['ui', 'web-native', 'interactive'],
  summary: 'Pointer movement leaves a persistent wet streak that spreads and dries.',
  scorecard: {
    compute: false,
    mobile: 'untested',
    bundleKb: 54.0,
    gpuMs: [{ device: 'Apple M1 Max (Metal)', ms: 0.175, resolution: '1920 × 1080' }],
    features: { optional: ['timestamp-query'] },
    fallback: 'A static poster with semantic HTML for the component and its content.',
    useCases: ['Website components', 'Liquid interface research'],
  },
};

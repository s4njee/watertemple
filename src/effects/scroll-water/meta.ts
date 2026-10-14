import type { EffectMeta } from '../types.ts';
export const meta: EffectMeta = {
  id: 'scroll-water',
  title: 'Scroll-linked water level',
  epic: '9',
  tier: 'B',
  tags: ['ui', 'web-native', 'interactive'],
  summary: 'The backdrop submerges with page or nested-container scrolling.',
  scorecard: {
    compute: false,
    mobile: 'untested',
    bundleKb: 54.0,
    gpuMs: [{ device: 'Apple M1 Max (Metal)', ms: 0.153, resolution: '1920 × 1080' }],
    features: { optional: ['timestamp-query'] },
    fallback: 'A static poster with semantic HTML for the component and its content.',
    useCases: ['Website components', 'Liquid interface research'],
  },
};

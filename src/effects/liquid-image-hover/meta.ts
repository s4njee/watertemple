import type { EffectMeta } from '../types.ts';
export const meta: EffectMeta = {
  id: 'liquid-image-hover',
  title: 'Liquid image hover',
  epic: '9',
  tier: 'A',
  tags: ['ui', 'web-native', 'interactive'],
  summary: 'Move across a coastal image to bend it with ripples and a subtle colour split.',
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

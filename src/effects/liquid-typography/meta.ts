import type { EffectMeta } from '../types.ts';
export const meta: EffectMeta = {
  id: 'liquid-typography',
  title: 'Liquid typography',
  epic: '9',
  tier: 'B',
  tags: ['ui', 'web-native', 'interactive'],
  summary:
    'An Anton MSDF atlas keeps liquid lettering sharp as it ripples, melts and fills with water.',
  scorecard: {
    compute: false,
    mobile: 'untested',
    bundleKb: 54.0,
    gpuMs: [{ device: 'Apple M1 Max (Metal)', ms: 0.262, resolution: '1920 × 1080' }],
    features: { optional: ['timestamp-query'] },
    fallback: 'A static poster with semantic HTML for the component and its content.',
    useCases: ['Website components', 'Liquid interface research'],
  },
};

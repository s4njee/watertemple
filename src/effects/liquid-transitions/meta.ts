import type { EffectMeta } from '../types.ts';
export const meta: EffectMeta = {
  id: 'liquid-transitions',
  title: 'Liquid transitions',
  epic: '9',
  tier: 'A',
  tags: ['ui', 'web-native', 'interactive'],
  summary:
    'Drip wipes, wave reveals and displacement dissolves switch between two designed sections.',
  scorecard: {
    compute: false,
    mobile: 'untested',
    bundleKb: 54.0,
    gpuMs: [{ device: 'Apple M1 Max (Metal)', ms: 0.109, resolution: '1920 × 1080' }],
    features: { optional: ['timestamp-query'] },
    fallback: 'A static poster with semantic HTML for the component and its content.',
    useCases: ['Website components', 'Liquid interface research'],
  },
};

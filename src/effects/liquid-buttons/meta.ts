import type { EffectMeta } from '../types.ts';
export const meta: EffectMeta = {
  id: 'liquid-buttons',
  title: 'Liquid buttons',
  epic: '9',
  tier: 'A',
  tags: ['ui', 'web-native', 'interactive'],
  summary: 'Hover or focus to stretch a spring-driven button; press it to launch a surface ripple.',
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

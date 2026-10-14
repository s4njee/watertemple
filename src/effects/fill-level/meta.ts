import type { EffectMeta } from '../types.ts';
export const meta: EffectMeta = {
  id: 'fill-level',
  title: 'Fill-level components',
  epic: '9',
  tier: 'A',
  tags: ['ui', 'web-native', 'interactive'],
  summary: 'A tank, progress bar and round loader share a sloshing liquid level.',
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

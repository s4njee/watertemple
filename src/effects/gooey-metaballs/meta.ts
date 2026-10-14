import type { EffectMeta } from '../types.ts';
export const meta: EffectMeta = {
  id: 'gooey-metaballs',
  title: 'Gooey metaballs',
  epic: '9',
  tier: 'A',
  tags: ['ui', 'web-native', 'interactive'],
  summary:
    'Smooth SDF unions join navigation pills to an animated or pointer-driven liquid follower.',
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

import type { EffectMeta } from '../types.ts';
export const meta: EffectMeta = {
  id: 'river',
  title: 'River current',
  epic: '8',
  tier: 'B',
  tags: ['3d', 'open-water', 'shader'],
  summary:
    'A signed flow map diverts two-phase water detail around rocks and trails foam downstream.',
  scorecard: {
    compute: false,
    mobile: 'untested',
    bundleKb: 13.29,
    gpuMs: [{ device: 'Apple M1 Max (Metal)', ms: 0.546, resolution: '1920 × 1080' }],
    features: { optional: ['timestamp-query'] },
    fallback: 'A static poster of this scene with a matching sky and water palette.',
    useCases: ['Immersive water hero scenes', 'Water rendering and performance research'],
  },
};

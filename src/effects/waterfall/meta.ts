import type { EffectMeta } from '../types.ts';
export const meta: EffectMeta = {
  id: 'waterfall',
  title: 'Waterfall and mist',
  epic: '8',
  tier: 'B',
  tags: ['3d', 'open-water', 'compute'],
  summary: 'Scrolling water sheets, a foamy plunge pool and bounded GPU particle spray.',
  scorecard: {
    compute: true,
    mobile: 'untested',
    bundleKb: 13.28,
    gpuMs: [{ device: 'Apple M1 Max (Metal)', ms: 0.786, resolution: '1920 × 1080' }],
    features: { optional: ['timestamp-query'] },
    fallback: 'A static poster of this scene with a matching sky and water palette.',
    useCases: ['Immersive water hero scenes', 'Water rendering and performance research'],
  },
};

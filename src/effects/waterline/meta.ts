import type { EffectMeta } from '../types.ts';
export const meta: EffectMeta = {
  id: 'waterline',
  title: 'Across the waterline',
  epic: '8',
  tier: 'B',
  tags: ['3d', 'open-water', 'shader'],
  summary: 'A moving wave meniscus separates air and water with underwater tint and attenuation.',
  scorecard: {
    compute: false,
    mobile: 'untested',
    bundleKb: 13.27,
    gpuMs: [{ device: 'Apple M1 Max (Metal)', ms: 1.223, resolution: '1920 × 1080' }],
    features: { optional: ['timestamp-query'] },
    fallback: 'A static poster of this scene with a matching sky and water palette.',
    useCases: ['Immersive water hero scenes', 'Water rendering and performance research'],
  },
};

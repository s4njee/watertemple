import type { EffectMeta } from '../types.ts';
export const meta: EffectMeta = {
  id: 'calm-lake',
  title: 'Calm lake',
  epic: '8',
  tier: 'B',
  tags: ['3d', 'open-water', 'shader'],
  summary: 'A mirrored camera pass reflects distant ridges and trees through light wind ripples.',
  scorecard: {
    compute: false,
    mobile: 'untested',
    bundleKb: 13.28,
    gpuMs: [{ device: 'Apple M1 Max (Metal)', ms: 0.524, resolution: '1920 × 1080' }],
    features: { optional: ['timestamp-query'] },
    fallback: 'A static poster of this scene with a matching sky and water palette.',
    useCases: ['Immersive water hero scenes', 'Water rendering and performance research'],
  },
};

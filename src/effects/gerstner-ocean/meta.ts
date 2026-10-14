import type { EffectMeta } from '../types.ts';
export const meta: EffectMeta = {
  id: 'gerstner-ocean',
  title: 'Gerstner ocean',
  epic: '8',
  tier: 'B',
  tags: ['3d', 'open-water', 'shader'],
  summary:
    'Six analytic waves, projected and clipmap meshes, and floating buoys with CPU or GPU sampling.',
  scorecard: {
    compute: false,
    mobile: 'untested',
    bundleKb: 13.3,
    gpuMs: [{ device: 'Apple M1 Max (Metal)', ms: 0.393, resolution: '1920 × 1080' }],
    features: { optional: ['timestamp-query'] },
    fallback: 'A static poster of this scene with a matching sky and water palette.',
    useCases: ['Immersive water hero scenes', 'Water rendering and performance research'],
  },
};

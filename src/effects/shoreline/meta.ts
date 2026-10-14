import type { EffectMeta } from '../types.ts';
export const meta: EffectMeta = {
  id: 'shoreline',
  title: 'Breaking shoreline',
  epic: '8',
  tier: 'B',
  tags: ['3d', 'open-water', 'shader'],
  summary:
    'Shoaling wave crests wash across a sloping beach with depth foam and reflective wet sand.',
  scorecard: {
    compute: false,
    mobile: 'untested',
    bundleKb: 13.29,
    gpuMs: [{ device: 'Apple M1 Max (Metal)', ms: 0.721, resolution: '1920 × 1080' }],
    features: { optional: ['timestamp-query'] },
    fallback: 'A static poster of this scene with a matching sky and water palette.',
    useCases: ['Immersive water hero scenes', 'Water rendering and performance research'],
  },
};

import type { EffectMeta } from '../types.ts';
export const meta: EffectMeta = {
  id: 'fft-ocean',
  title: 'Spectral ocean',
  epic: '8',
  tier: 'C',
  tags: ['3d', 'open-water', 'compute'],
  summary: 'Three band-limited Phillips cascades, compute inverse FFT and persistent folding foam.',
  scorecard: {
    compute: true,
    mobile: 'untested',
    bundleKb: 13.28,
    gpuMs: [{ device: 'Apple M1 Max (Metal)', ms: 0.83, resolution: '1920 × 1080' }],
    features: { optional: ['timestamp-query'] },
    fallback: 'A static poster of this scene with a matching sky and water palette.',
    useCases: ['Immersive water hero scenes', 'Water rendering and performance research'],
  },
};

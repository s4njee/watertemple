import type { EffectMeta } from '../types.ts';

export const meta: EffectMeta = {
  id: 'hello',
  title: 'Hello, water',
  epic: '1',
  tier: 'A',
  tags: ['foundation'],
  summary: 'Domain-warped noise. Exists to exercise the foundation end to end.',
  scorecard: {
    compute: false,
    mobile: 'untested',
    features: { required: [] },
    fallback: 'A static image in the same palette, or a plain CSS background colour.',
    useCases: [
      'Smoke test for the shader pipeline, shell and shared WGSL library',
      'A quiet abstract backdrop; nothing about it reads as water yet',
    ],
  },
};

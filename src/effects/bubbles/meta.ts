import type { EffectMeta } from '../types.ts';
export const meta: EffectMeta = {
  id: 'bubbles',
  title: 'Bubbles',
  epic: '10',
  tier: 'A',
  tags: ['weather', 'water', 'ambient', 'particles'],
  summary: 'Rising, wobbling underwater lenses and iridescent soap-film bubbles.',
  scorecard: {
    compute: false,
    bundleKb: 9.85,
    gpuMs: [
      {
        device: 'Apple M1 Max (Dawn Metal)',
        ms: 0.328,
        resolution: '1920×1080, conservative measured configurations',
      },
    ],
    mobile: 'untested',
    features: { required: [] },
    fallback: 'The captured still poster; pause the animation under reduced motion.',
    useCases: ['Weather backdrops, hero decoration and interactive feedback'],
  },
};

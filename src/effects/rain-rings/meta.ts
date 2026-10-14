import type { EffectMeta } from '../types.ts';
export const meta: EffectMeta = {
  id: 'rain-rings',
  title: 'Rain ripple rings',
  epic: '10',
  tier: 'A',
  tags: ['weather', 'water', 'ambient', 'ripples'],
  summary: 'Expanding wave rings refract a tiled water surface after procedural rain impacts.',
  scorecard: {
    compute: false,
    bundleKb: 9.86,
    gpuMs: [
      {
        device: 'Apple M1 Max (Dawn Metal)',
        ms: 0.786,
        resolution: '1920×1080, conservative measured configurations',
      },
    ],
    mobile: 'untested',
    features: { required: [] },
    fallback: 'The captured still poster; pause the animation under reduced motion.',
    useCases: ['Weather backdrops, hero decoration and interactive feedback'],
  },
};

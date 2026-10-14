import type { EffectMeta } from '../types.ts';
export const meta: EffectMeta = {
  id: 'splash-burst',
  title: 'Splash burst',
  epic: '10',
  tier: 'A',
  tags: ['weather', 'water', 'interactive', 'particles'],
  summary: 'Bounded ballistic droplet bursts on click or tap, with optional ambient fountains.',
  scorecard: {
    compute: false,
    bundleKb: 9.86,
    gpuMs: [
      {
        device: 'Apple M1 Max (Dawn Metal)',
        ms: 0.197,
        resolution: '1920×1080, conservative measured configurations',
      },
    ],
    mobile: 'untested',
    features: { required: [] },
    fallback: 'The captured still poster; pause the animation under reduced motion.',
    useCases: ['Weather backdrops, hero decoration and interactive feedback'],
  },
};

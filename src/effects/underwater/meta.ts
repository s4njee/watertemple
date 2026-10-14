import type { EffectMeta } from '../types.ts';

export const meta: EffectMeta = {
  id: 'underwater',
  title: 'Underwater ambience',
  epic: '4',
  tier: 'A',
  tags: ['background', 'hero', 'ambient', 'caustics', 'depth-fog', 'particles'],
  summary:
    'Looking up from below the surface: slow light shafts, drifting marine snow in two layers, depth fog and a gentle wobble, with an optional caustic seabed. Calm enough to put text over.',
  scorecard: {
    compute: false,
    mobile: 'untested',
    features: { required: [] },
    fallback:
      'A CSS gradient in the fog colour with a blurred, slowly panning image of the shafts, or a looping muted video of the same scene.',
    useCases: [
      'A hero or section background for ocean, diving, aquarium, wellness or water-tech pages',
      'A calm full-bleed layer behind headings and body text',
      'The seabed caustic lighting from the baked map, in a scene instead of an overlay',
    ],
  },
};

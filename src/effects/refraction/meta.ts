import type { EffectMeta } from '../types.ts';

export const meta: EffectMeta = {
  id: 'refraction',
  title: 'Screen-space refraction',
  epic: '3',
  tier: 'A',
  tags: ['building-block', 'refraction', 'dispersion', 'texture', 'pointer'],
  summary:
    'Looks through animated water at a backdrop texture: a normal-driven uv offset (Snell or the cheap form), chromatic dispersion, mip blur and safe edges.',
  scorecard: {
    compute: false,
    mobile: 'untested',
    features: { required: [] },
    fallback:
      'A static image of the backdrop (or the backdrop itself) with a CSS blur or SVG feDisplacementMap filter for a rough imitation.',
    useCases: [
      'Water or glass distortion over an image, video or any texture you already have',
      'The refraction building block for pool floors, liquid glass and heightfield ripples',
      'Judging a refraction model: the tile lines must bend coherently, not wobble',
    ],
  },
};

import type { EffectMeta } from '../types.ts';

export const meta: EffectMeta = {
  id: 'fresnel-sky',
  title: 'Fresnel and sky reflection',
  epic: '3',
  tier: 'A',
  tags: ['building-block', 'optics', 'reflection', 'ocean'],
  summary:
    'Open water reflecting a procedural sky, weighted by Schlick Fresnel over a deep-water body colour. Proves fresnel.wgsl, sky.wgsl and the cubemap helper.',
  scorecard: {
    compute: false,
    mobile: 'untested',
    features: { required: [] },
    fallback:
      'A static image of a horizon: a sky-to-water gradient in the same palette, or a plain CSS gradient.',
    useCases: [
      'The reflection model for any larger water scene (lake, sea, pool) that wants a sky',
      'A calm, cheap water backdrop with no geometry and no textures',
      'Reference for choosing between a procedural sky and a baked cubemap',
    ],
  },
};

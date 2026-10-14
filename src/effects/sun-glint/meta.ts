import type { EffectMeta } from '../types.ts';

export const meta: EffectMeta = {
  id: 'sun-glint',
  title: 'Sun glint and sparkle',
  epic: '3',
  tier: 'A',
  tags: ['building-block', 'optics', 'specular', 'ocean'],
  summary:
    'A low sun over open water: a tight GGX sun lobe plus hashed-cell glitter that twinkles with the waves and stays stable under camera motion. Proves specular.wgsl.',
  scorecard: {
    compute: false,
    mobile: 'untested',
    features: { required: [] },
    fallback:
      'A still photo or render of a sunlit water surface; the glitter does not survive being frozen, so a short looped video is the better fallback where it matters.',
    useCases: [
      'The sun reflection for a sea or lake scene (feeds the ocean shading story)',
      'Sparkle on any water-like surface in a hero image, where flat specular alone looks dead',
      'Reference for stable, non-aliasing glitter on any rough reflective surface',
    ],
  },
};

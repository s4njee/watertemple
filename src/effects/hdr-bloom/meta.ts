import type { EffectMeta } from '../types.ts';

export const meta: EffectMeta = {
  id: 'hdr-bloom',
  title: 'HDR bloom and tonemapping',
  epic: '3',
  tier: 'B',
  tags: ['building-block', 'hdr', 'bloom', 'tonemapping', 'post-process'],
  summary:
    'Dark water with sun glints, lantern reflections and caustic lines whose radiance runs far above 1, rendered into rgba16float, bloomed and tonemapped. Proves the post chain and the opt-in HDR canvas.',
  scorecard: {
    compute: false,
    mobile: 'untested',
    features: { required: [] },
    fallback:
      'A still frame of the same scene with the bloom baked in, or a CSS gradient in the same palette. The effect needs rgba16float render targets, which every WebGPU device has, so there is no reduced-quality path to fall back to inside WebGPU other than turning bloom off.',
    useCases: [
      'Any water scene with highlights above white (sun glints, lanterns, lava, neon) that should glow instead of clipping',
      'The shared final pass for the ocean and showpiece effects: exposure, tonemapping and dithering in one place',
      'Hero sections on HDR displays, as an opt-in extra over the same SDR picture',
    ],
  },
};

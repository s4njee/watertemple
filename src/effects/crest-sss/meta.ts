import type { EffectMeta } from '../types.ts';

export const meta: EffectMeta = {
  id: 'crest-sss',
  title: 'Wave-crest subsurface scattering',
  epic: '3',
  tier: 'A',
  tags: ['building-block', 'optics', 'subsurface', 'ocean'],
  summary:
    'Big, backlit waves where the tall, thin crests glow green-blue as the sun shines through them. One view- and light-dependent term; proves sss.wgsl.',
  scorecard: {
    compute: false,
    mobile: 'untested',
    features: { required: [] },
    fallback:
      'A still render of backlit swell with the glow baked in; the term is view dependent but a single frame reads fine as a poster.',
    useCases: [
      'The translucent glow on wave crests in an ocean shader (feeds the ocean shading story)',
      'Backlit water, waterfalls and sheets of liquid that need a cheap "light through it" cue',
      'A drop-in term for any thin translucent surface with a height or thickness estimate',
    ],
  },
};

import type { EffectMeta } from '../types.ts';

export const meta: EffectMeta = {
  id: 'puddles',
  title: 'Puddles and wet surfaces',
  epic: '4',
  tier: 'A',
  tags: ['background', 'weather', 'rain', 'reflection', 'ripples', 'wetness', 'ground'],
  summary:
    'Rain-soaked ground seen at an angle: puddles fill the low spots and mirror a dusk city through rain ripple rings, and the damp stone around them is darker and glossier.',
  scorecard: {
    compute: false,
    mobile: 'untested',
    features: { required: [] },
    fallback:
      'A still photograph or render of the wet street, or a looping video of the rain with the reflections baked in.',
    useCases: [
      'A wet-street or courtyard backdrop for a hero, a game menu or a weather app',
      'The wet-material model (darker, glossier, mirror in the hollows) for any ground you already render',
      'The rain ripple rings that story 10.2 turns into a standalone effect',
    ],
  },
};

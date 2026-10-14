import type { EffectMeta } from '../types.ts';
export const meta: EffectMeta = {
  id: 'game-of-life',
  title: 'Game of Life',
  epic: '6',
  tier: 'A',
  tags: ['cellular automata', 'compute', 'background', 'interactive'],
  summary: 'Conway’s B3/S23 automaton, with luminous cells and fading echoes behind a real page.',
  scorecard: {
    compute: true,
    mobile: 'untested',
    fallback: 'A still cellular pattern.',
    useCases: ['Interactive backgrounds and playful editorial pages'],
  },
};

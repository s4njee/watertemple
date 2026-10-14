import type { EffectMeta } from '../types.ts';
export const meta: EffectMeta = {
  id: 'silk-embers',
  title: 'Silk embers',
  epic: '6',
  tier: 'B',
  tags: ['interactive', 'simulation', 'compute', 'fluid', 'dom', 'background'],
  summary: 'Copper, crimson and gold filaments that cool as detached sparks drift upward.',
  scorecard: {
    compute: true,
    mobile: 'untested',
    features: { required: [] },
    fallback: 'A still gradient matching the material over a dark background.',
    useCases: ['An interactive hero or atmospheric reading page'],
  },
};

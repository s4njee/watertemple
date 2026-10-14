import type { EffectMeta } from '../types.ts';
export const meta: EffectMeta = {
  id: 'bioluminescent-currents',
  title: 'Bioluminescent currents',
  epic: '6',
  tier: 'B',
  tags: ['interactive', 'simulation', 'compute', 'fluid', 'dom', 'background'],
  summary:
    'Cyan and green organisms flash on sharp turns or pointer proximity, leaving delicate wakes.',
  scorecard: {
    compute: true,
    mobile: 'untested',
    features: { required: [] },
    fallback: 'A still gradient matching the material over a dark background.',
    useCases: ['An interactive hero or atmospheric reading page'],
  },
};

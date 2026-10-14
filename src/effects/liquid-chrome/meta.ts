import type { EffectMeta } from '../types.ts';
export const meta: EffectMeta = {
  id: 'liquid-chrome',
  title: 'Liquid chrome',
  epic: '6',
  tier: 'B',
  tags: ['interactive', 'simulation', 'compute', 'fluid', 'dom', 'background'],
  summary:
    'Black and silver fluid ribbons reflecting moving studio lights with tiny spectral glints.',
  scorecard: {
    compute: true,
    mobile: 'untested',
    features: { required: [] },
    fallback: 'A still gradient matching the material over a dark background.',
    useCases: ['An interactive hero or atmospheric reading page'],
  },
};

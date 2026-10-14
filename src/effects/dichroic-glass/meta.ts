import type { EffectMeta } from '../types.ts';
export const meta: EffectMeta = {
  id: 'dichroic-glass',
  title: 'Dichroic glass',
  epic: '6',
  tier: 'B',
  tags: ['interactive', 'simulation', 'compute', 'fluid', 'dom', 'background'],
  summary:
    'Refractive teal and magenta ribbons with thin-film colour, crisp highlights and overlapping depth.',
  scorecard: {
    compute: true,
    mobile: 'untested',
    features: { required: [] },
    fallback: 'A still gradient matching the material over a dark background.',
    useCases: ['An interactive hero or atmospheric reading page'],
  },
};

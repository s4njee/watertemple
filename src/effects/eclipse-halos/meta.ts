import type { EffectMeta } from '../types.ts';
export const meta: EffectMeta = {
  id: 'eclipse-halos',
  title: 'Eclipse halos',
  epic: '6',
  tier: 'B',
  tags: ['interactive', 'simulation', 'compute', 'fluid', 'dom', 'background'],
  summary:
    'Narrow moving rings around page outlines, with quiet fluid arcs escaping into the dark.',
  scorecard: {
    compute: true,
    mobile: 'untested',
    features: { required: [] },
    fallback: 'A still gradient matching the material over a dark background.',
    useCases: ['An interactive hero or atmospheric reading page'],
  },
};

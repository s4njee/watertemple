import type { EffectMeta } from '../types.ts';

export const meta: EffectMeta = {
  id: 'aurora',
  title: 'Aurora ribbons',
  epic: '6',
  tier: 'B',
  tags: ['interactive', 'simulation', 'compute', 'fluid', 'dom', 'aurora', 'background'],
  summary:
    'Slow emerald, cyan and violet ribbons flowing around page elements, with fine luminous folds and soft rising curtains. A separate variation of ink orbit using the same obstacle-aware fluid simulation.',
  scorecard: {
    compute: true,
    mobile: 'untested',
    features: { required: [] },
    fallback: 'A still gradient of emerald, cyan and violet over a midnight background.',
    useCases: [
      'An atmospheric hero or reading page',
      'A calm interactive background with luminous ribbons',
    ],
  },
};

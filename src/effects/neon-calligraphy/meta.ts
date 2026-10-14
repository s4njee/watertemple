import type { EffectMeta } from '../types.ts';
export const meta: EffectMeta = {
  id: 'neon-calligraphy',
  title: 'Neon calligraphy',
  epic: '6',
  tier: 'B',
  tags: ['interactive', 'simulation', 'compute', 'fluid', 'dom', 'background'],
  summary: 'Glowing ink blots, tapering luminous lines and faint afterimages.',
  scorecard: {
    compute: true,
    mobile: 'untested',
    features: { required: [] },
    fallback: 'A still gradient matching the material over a dark background.',
    useCases: ['An interactive hero or atmospheric reading page'],
  },
};

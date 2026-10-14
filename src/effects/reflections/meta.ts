import type { EffectMeta } from '../types.ts';

export const meta: EffectMeta = {
  id: 'reflections',
  title: 'Reflection strategies',
  epic: '3',
  tier: 'B',
  tags: ['optics', '3d', 'reflection', 'multi-pass', 'comparison'],
  summary:
    'Planar reflection against screen-space reflection on a calm pool, side by side: what each costs and where each one breaks.',
  scorecard: {
    compute: false,
    mobile: 'untested',
    features: { required: [], optional: ['timestamp-query'] },
    fallback:
      'A poster of the scene with a pre-rendered reflection, or the sky-only strategy, which is a single cheap pass.',
    useCases: [
      'A pool, pond or puddle in a 3D hero scene where a flat mirror is enough: planar, at half or quarter resolution',
      'A scene with many water surfaces or an unknown layout of them: screen-space reflection, with a sky fallback',
      'Ambient water behind UI: the sky-only strategy and a light normal wobble',
    ],
  },
};

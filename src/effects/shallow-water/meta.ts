import type { EffectMeta } from '../types.ts';

export const meta: EffectMeta = {
  id: 'shallow-water',
  title: 'Shallow water: dam break and flow',
  epic: '6',
  tier: 'B',
  tags: ['interactive', 'simulation', 'fluid', 'terrain', '3d', 'compute', 'showpiece'],
  summary:
    'Water that actually flows: a shallow-water solver on the GPU moves a depth and a velocity over terrain. A dam breaks and the surge rushes around pillars, a spring runs down a hillside into a lake, a tide fills and drains rock pools. Pour water, push it, dig channels.',
  scorecard: {
    // The default solver is the compute-pass one (a render-pass variant of the same maths exists
    // and can be chosen, but the default path needs compute).
    compute: true,
    mobile: 'untested',
    features: { required: [] },
    fallback:
      'A looping video of the dam break (the surge, the pillars, the settling pond) or a still of the terrain with a CSS-animated water level; the picking and sculpting are lost. The `solver: render` variant has no WebGL2 port yet but needs no compute shaders.',
    useCases: [
      'A feature page where flowing water is the subject: a landing hero for a water utility, a flood-risk explainer, a game or toy about rivers and dams',
      'An interactive explainer of why a flood spreads the way it does: pour, dig, build a dam, watch where the water goes',
      'Reference for a conservative, well-balanced shallow-water solver: wet/dry fronts, lakes at rest, mass bookkeeping',
    ],
  },
};

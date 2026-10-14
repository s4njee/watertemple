import type { EffectMeta } from '../types.ts';

export const meta: EffectMeta = {
  id: 'flow-map',
  title: 'Flow maps',
  epic: '3',
  tier: 'A',
  tags: ['building-block', 'flow', 'river', 'water'],
  summary:
    'A top-down river: a baked flow field steers two cross-faded phases of a ripple normal map around the rocks.',
  scorecard: {
    compute: false,
    mobile: 'untested',
    features: { required: [] },
    fallback:
      'A still frame of the river; or a short looping video of it, since the flow is periodic.',
    useCases: [
      'Rivers, streams, canals and spillways seen from above or at an angle',
      'Any directed current on a surface: drains, tidal inlets, waterfall pools',
      'Foam, oil or leaves carried along a baked flow (swap the normal map for any texture)',
    ],
  },
};

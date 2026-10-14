import type { EffectMeta } from '../types.ts';

export const meta: EffectMeta = {
  id: 'shore-foam',
  title: 'Shore foam and edge fade',
  epic: '3',
  tier: 'A',
  tags: ['building-block', 'foam', 'shoreline', 'depth', 'soft-edge'],
  summary:
    'Top-down beach: waves lap in and out, foam dissolves along the waterline and around rocks, and the water fades in softly where it meets the sand.',
  scorecard: {
    compute: false,
    mobile: 'untested',
    features: { required: [] },
    fallback:
      'A looping video or a static render of the beach, or a CSS gradient with an SVG turbulence mask for the foam line.',
    useCases: [
      'Shoreline, pond-edge and river-bank water in a hero or a section background',
      'The foam and soft-edge building block for the 3D shoreline (8.7) and river (8.8)',
      'Any water surface that meets geometry and needs the hard contact line hidden',
    ],
  },
};

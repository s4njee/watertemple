import type { EffectMeta } from '../types.ts';

export const meta: EffectMeta = {
  id: 'ripple-reactive',
  title: 'Ripples driven by scroll and sound',
  epic: '5',
  tier: 'A',
  tags: ['interactive', 'simulation', 'scroll', 'audio', 'heightfield', 'refraction', 'background'],
  summary:
    'A heightfield surface that answers the page instead of the pointer: scrolling feeds waves in from the leading edge and sloshes on a hard stop, and a built-in synthesised beat (or a dropped audio or video file) turns bass into broad swells, mids and highs into fine rain and onsets into drops. Silent and calm until the first click.',
  scorecard: {
    compute: false,
    mobile: 'untested',
    features: { required: [] },
    fallback:
      'The still backdrop image, or a looping video of the pond; scroll response can be a CSS parallax on that image. The audio input has no static equivalent: without Web Audio the surface just runs on scroll and its idle swell.',
    useCases: [
      'A background that reacts to scrolling the page, subtle at reading speed',
      'A music or podcast page where the surface follows the track that is playing',
      'A visualiser for a dropped audio or video file that never leaves the browser',
    ],
  },
};

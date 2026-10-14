import type { EffectMeta } from '../types.ts';

export const meta: EffectMeta = {
  id: 'media-distortion',
  title: 'Water over image or video',
  epic: '4',
  tier: 'A',
  tags: ['image', 'video', 'refraction', 'dispersion', 'pointer', 'hero', 'texture'],
  summary:
    'Looks at a picture or a video through moving water: baked-normal refraction, a light highlight, optional colour fringing, safe edges and a ripple under the pointer. Drop your own file on it.',
  scorecard: {
    compute: false,
    mobile: 'untested',
    features: { required: [] },
    fallback:
      'The plain <img> or <video> underneath the canvas. Nothing is lost but the motion; an SVG feDisplacementMap filter is a rough but heavier substitute.',
    useCases: [
      'A hero image that ripples gently, or wobbles when the pointer crosses it',
      'A product shot that reacts on hover',
      'Water or heat shimmer over a looping background video',
    ],
  },
};

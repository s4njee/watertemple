import type { EffectMeta } from '../types.ts';

export const meta: EffectMeta = {
  id: 'ripples',
  title: 'Interactive ripples',
  epic: '5',
  tier: 'A',
  tags: ['interactive', 'simulation', 'pointer', 'heightfield', 'refraction', 'background'],
  summary:
    'The classic touch-the-water surface: a 2D wave equation on the GPU, stirred by mouse, pen and fingers, falling rain and a faint idle swell, refracting a backdrop. Runs as render passes or compute passes, same maths.',
  scorecard: {
    // The default solver is the render-pass one, which needs no compute; `solver: compute` is an
    // optional second implementation of the same maths.
    compute: false,
    mobile: 'untested',
    features: { required: [] },
    fallback:
      'A looping video or animated WebP of the same pond under rain (the ambient rain makes a convincing loop), or the still backdrop with a CSS ripple on click.',
    useCases: [
      'A hero or full-page background that answers the cursor and settles when left alone',
      'Tap-to-ripple feedback on an image, a pool photograph or a brand surface',
      'The reference surface for the caustics, pool and obstacle effects built on the same solver',
    ],
  },
};

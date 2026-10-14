import type { EffectMeta } from '../types.ts';

export const meta: EffectMeta = {
  id: 'ripple-dom',
  title: 'Ripples around page elements',
  epic: '5',
  tier: 'A',
  tags: ['interactive', 'simulation', 'dom', 'pointer', 'heightfield', 'refraction', 'background'],
  summary:
    'Water behind a real page where the page itself is solid: elements marked data-wt-obstacle are measured every frame and become walls of the wave solver, so ripples bend around a card, bounce off a headline and are pushed by a card that moves.',
  scorecard: {
    compute: false,
    mobile: 'untested',
    features: { required: [] },
    fallback:
      'The static backdrop image behind the content with a CSS ripple on click. Nothing about the layout depends on the water: the page works unchanged, because the elements are only measured.',
    useCases: [
      'A hero or section background whose water respects the cards and headings laid over it',
      'A pool behind a sticky article column that laps at the text plate and is pushed by it as it scrolls',
      'Any backdrop where a draggable or animated element should visibly disturb the water',
    ],
  },
};

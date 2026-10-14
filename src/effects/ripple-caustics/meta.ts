import type { EffectMeta } from '../types.ts';

export const meta: EffectMeta = {
  id: 'ripple-caustics',
  title: 'Ripple caustics',
  epic: '5',
  tier: 'B',
  tags: ['interactive', 'simulation', 'caustics', 'heightfield', 'refraction', 'comparison'],
  summary:
    'Real caustics computed from the live ripple surface: a drop sends a ring of focused light sweeping across a tiled pool floor. Two methods (a projected light mesh and a per-pixel Jacobian estimate) and a side-by-side against the procedural fake.',
  scorecard: {
    // Both caustics methods and the default render-pass solver are plain render passes.
    compute: false,
    mobile: 'untested',
    features: { required: [] },
    fallback:
      'The procedural pool floor effect (baked caustic map, no simulation), or a looping video of the same pool under rain.',
    useCases: [
      'A hero or pool-themed section where light should visibly respond to the cursor and to rain',
      'The reference for how much a simulated caustic differs from the baked or procedural fake',
      'Any shallow-water floor (pond, fountain basin, aquarium) lit by the sun through a stirred surface',
    ],
  },
};

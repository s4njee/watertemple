import type { EffectMeta } from '../types.ts';

export const meta: EffectMeta = {
  id: 'fluid-3d',
  title: 'Ink in a tank (3D fluid)',
  epic: '6',
  tier: 'C',
  tags: ['interactive', 'simulation', 'compute', 'fluid', '3d', 'volumetric', 'showpiece', 'ink'],
  summary:
    'A drop of ink released in a glass tank of water sinks, rolls up into a vortex ring and breaks into tendrils. A volumetric stable-fluids solve (48 to 96 cells, multigrid pressure, vorticity confinement, MacCormack ink) in compute shaders, drawn by a ray march with coloured absorption and a precomputed light volume. Tap to release a drop, drag to push the water, drag the background to orbit.',
  scorecard: {
    // Every pass is a compute dispatch writing 3D storage textures: there is no WebGL2 path short of
    // a rewrite as render passes into 3D slices, which nobody should attempt for this.
    compute: true,
    // Not measured on a phone. By arithmetic it does not fit: the default is a 64-cell solve plus a ray
    // march; a phone GPU is a fraction of the laptop that measured it, so the answer is "no" until a
    // device says otherwise. The `low-cost` preset is the attempt.
    mobile: 'untested',
    features: { required: [] },
    fallback:
      'Without WebGPU or compute, show a looping video or animated WebP of the tank (a drop sinking, rolling into a ring and branching) with the orbit and the drops lost, or the 2D fluid-dye effect, which keeps the ink and the interaction on a flat canvas for a thirtieth of the work.',
    useCases: [
      'A dedicated landing page or full-screen experience for a drink, a dye, a paint or a water brand where the ink in water is the subject',
      'A desktop showpiece that proves what the site can do; not a background behind text and not for phones',
      'The reference for a 3D grid solver, a multigrid pressure solve and a ray-marched volume on this site',
    ],
  },
};

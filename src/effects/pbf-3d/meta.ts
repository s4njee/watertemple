import type { EffectMeta } from '../types.ts';

export const meta: EffectMeta = {
  id: 'pbf-3d',
  title: 'Position-based fluid tank (3D)',
  epic: '7',
  tier: 'C',
  tags: ['particles', 'simulation', 'compute', 'fluid', 'interactive', '3d', 'showpiece'],
  summary:
    'Water as ten thousand particles by default in a glass tank, solved with position-based fluids (Macklin and Mueller 2013) on the GPU and drawn as a refracting liquid surface. Three restartable scenes: a dam break with two pillars, a pour that fills the tank, and a wave tank with an oscillating wall. Drag empty space to orbit, drag through the water to push it, tap to drop a blob. Secondary spray, foam and bubbles follow the live water. Choose container mode to drag or shake the vessel, or enable device tilt. The particle count, solver iterations and sub-steps, viscosity, vorticity and the renderer quality are live.',
  scorecard: {
    gpuMs: [
      {
        device: 'Apple M1 Max · Dawn/Metal (workload slope)',
        ms: 4.33,
        resolution: '1920 × 1080',
      },
    ],
    bundleKb: 6.67,
    // Every solver pass is a compute dispatch (predict, counting sort, scan, density, correction, viscosity).
    compute: true,
    // Not measured on a phone. A desktop discrete or Apple Silicon GPU is the target; see the notes for the cut-down.
    mobile: 'untested',
    features: { required: [], optional: ['timestamp-query'] },
    fallback:
      'A captured poster of the tank (orbit, push and tilt are lost). There is no cheaper live version: a heightfield water surface (the 3D pool effect) loses the splashes, and a WebGL2 particle solver would need transform feedback and no neighbour search.',
    useCases: [
      'A dedicated landing page or product experience where pouring, sloshing and splashing water is the subject, on a desktop-class GPU',
      'A game or toy where the player pushes real water around inside a container (drag, tilt, shake and wave-maker)',
      'The reference for the stable, large-time-step particle fluid: the solver, the neighbour grid and the screen-space renderer working together',
    ],
  },
};

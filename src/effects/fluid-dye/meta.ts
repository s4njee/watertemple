import type { EffectMeta } from '../types.ts';

export const meta: EffectMeta = {
  id: 'fluid-dye',
  title: 'Ink in a stable fluid',
  epic: '6',
  tier: 'B',
  tags: ['interactive', 'simulation', 'compute', 'fluid', 'pointer', 'ink', 'background'],
  summary:
    'Drag to push a real incompressible fluid and paint dye that curls into vortices and fades. A stable-fluids solver in compute shaders (advection, pressure projection, vorticity confinement, MacCormack or BFECC), a coarse velocity grid carrying a fine dye grid.',
  scorecard: {
    // The solver is compute passes writing storage textures: no WebGL2 port without rewriting it as
    // render passes over float targets (which is how the classic web fluids do it).
    compute: true,
    mobile: 'untested',
    features: { required: [] },
    fallback:
      'Without WebGPU, or without compute shaders, show a looping video or animated WebP of the same ink in water on a CSS background (the idle autonomous strokes loop well), with a plain CSS gradient behind it; the interaction is the part that does not survive, so keep the page working without it.',
    useCases: [
      'An interactive hero or set piece where the visitor stirs coloured ink with the cursor',
      'A slow ambient background of drifting smoke or lava-lamp blobs that never repeats',
      'The reference for fluid-driven page distortion, obstacles and ink-in-water built on the same solver',
    ],
  },
};

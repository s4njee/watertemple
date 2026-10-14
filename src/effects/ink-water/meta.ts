import type { EffectMeta } from '../types.ts';

export const meta: EffectMeta = {
  id: 'ink-water',
  title: 'Ink in water and on paper',
  epic: '6',
  tier: 'B',
  tags: ['interactive', 'simulation', 'compute', 'fluid', 'ink', 'watercolour', 'paper'],
  summary:
    'Paint pigment into a stable-fluids solver and watch it bloom, feather along the paper fibres, pool darker at the edge of a wash and dry. Diffusion, a deposited layer that stays when the water stops, wetness, granulation and subtractive colour on top of the dye solver.',
  scorecard: {
    // The solver and the pigment layer are compute passes writing storage textures: no WebGL2 port
    // without rewriting both as render passes over float targets.
    compute: true,
    mobile: 'untested',
    features: { required: [] },
    fallback:
      'Without WebGPU, or without compute shaders, show the non-simulated `ink-wash` background (a fragment-only effect with a WebP or SVG fallback of its own), or a looping video of the idle drops blooming on a CSS paper colour; the painting is the part that does not survive, so keep the page working without it.',
    useCases: [
      'An interactive hero or set piece where the visitor paints pigment that blooms, feathers and dries on paper',
      'A slow ambient background of ink drops landing in water that never repeats, with a pointer that stirs it',
      'The reference for pigment-on-paper behaviour (diffusion, absorption, edge darkening, back-runs) built as a layer on the fluid solver',
    ],
  },
};

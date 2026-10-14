import type { EffectMeta } from '../types.ts';

export const meta: EffectMeta = {
  id: 'ink-orbit',
  title: 'Ink orbiting page elements',
  epic: '6',
  tier: 'B',
  tags: ['interactive', 'simulation', 'compute', 'fluid', 'dom', 'pointer', 'ink', 'background'],
  summary:
    'Luminous RGB trails in a stable fluid on a dark background, with heads smoothly cycling through red, green and blue together while their tails keep the colours they were laid with. Drawn into loose orbits round the page: elements marked data-wt-obstacle are measured every frame and become solid, and the heads of the trails are pulled to each outline and swung round it, never quite settling. Drag a card and the trails follow. On an HDR display the glow is brighter than SDR white.',
  scorecard: {
    // The solver and the current are compute passes writing storage textures.
    compute: true,
    mobile: 'untested',
    features: { required: [] },
    fallback:
      'Without WebGPU the page is unchanged, because the elements are only measured: show the plain paper colour, or a few soft radial gradients of the same colours behind the content for a hint of the look.',
    useCases: [
      'A landing page whose headline, cards and buttons are circled by a few broad trails of coloured ink',
      'A draggable or animated card that visibly stirs the page behind it',
      'A light, paper-coloured background that stays readable because the ink keeps to the gaps between the content',
    ],
  },
};

import type { EffectMeta } from '../types.ts';

export const meta: EffectMeta = {
  id: 'fluid-warp',
  title: 'Fluid-pushed image and type',
  epic: '6',
  tier: 'B',
  tags: [
    'interactive',
    'simulation',
    'compute',
    'fluid',
    'pointer',
    'image',
    'video',
    'text',
    'hero',
    'texture',
  ],
  summary:
    'Drag across a picture, a video or a headline and the content is pushed around by a real incompressible fluid: it smears and swirls like wet paint, then springs back at a rate you choose. A displacement field is carried by the flow of the stable-fluids solver and the content is sampled through it.',
  scorecard: {
    // The flow is the compute solver; the warp itself is two small render passes. There is no WebGL2
    // port without rewriting the solver as render passes over float targets.
    compute: true,
    mobile: 'untested',
    features: { required: [] },
    fallback:
      'The plain <img>, <video> or live headline underneath the canvas, with a CSS filter: blur() or a short transform on hover for a hint of response. Nothing is lost but the smear; for a headline, per-letter CSS transforms on pointer movement are a cheap substitute that keeps the text selectable.',
    useCases: [
      'A hero image that smears and settles like wet paint when the visitor drags across it',
      'A playful headline whose letters are dragged along and reassemble, drawn into the canvas from a 2D canvas',
      'A page-transition or hover smear over an image or a looping video',
    ],
  },
};

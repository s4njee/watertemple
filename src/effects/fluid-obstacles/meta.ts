import type { EffectMeta } from '../types.ts';

export const meta: EffectMeta = {
  id: 'fluid-obstacles',
  title: 'Wind tunnel: flow round obstacles',
  epic: '6',
  tier: 'B',
  tags: ['interactive', 'simulation', 'compute', 'fluid', 'pointer', 'dom', 'background', 'ink'],
  summary:
    'A steady wind carries rows of dye past solid shapes drawn from signed distance fields, or past the real cards and headline of the page. The stable-fluids solver makes the flow go round them: a bluff body sheds an alternating vortex street. Drag a shape to push the water, stir where there is none.',
  scorecard: {
    // The solver is compute; the shape rasteriser, the inflow and the shading are render passes.
    compute: true,
    mobile: 'untested',
    features: { required: [] },
    fallback:
      'A looping video or animated WebP of the same flow as a background, with the cards laid over it in plain CSS. The flow cannot react to the layout without the solver, so the fallback is a picture that happens to fit the page it was made for.',
    useCases: [
      'A hero where ink or smoke streams round the real cards and headline instead of through them',
      'A data-viz or product explainer showing flow past a shape (a wing, a pipe, a bottle)',
      'A playful page where visitors drag objects through a current',
    ],
  },
};

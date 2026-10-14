import { createOrbitEffect } from '../ink-orbit/index.ts';
import { defineEffect } from '../types.ts';
import displayShader from './display.wgsl?raw';
import { meta } from './meta.ts';

const orbit = createOrbitEffect({
  meta,
  displayShader,
  // #050917, matching the separate demo's midnight surface.
  darkGround: [0.001518, 0.002732, 0.008568],
});

export const effect = defineEffect({
  ...orbit,
  params: {
    ...orbit.params,
    heads: { ...orbit.params.heads, default: 7 },
    palette: { ...orbit.params.palette, default: 'aurora' },
    ink: { ...orbit.params.ink, default: 0.42 },
    inkRadius: { ...orbit.params.inkRadius, default: 0.03 },
    headSpeed: { ...orbit.params.headSpeed, default: 0.28 },
    chaos: { ...orbit.params.chaos, default: 0.15 },
    wander: { ...orbit.params.wander, default: 0.35 },
    swirl: { ...orbit.params.swirl, default: 0.35 },
    current: { ...orbit.params.current, default: 0.07 },
    wake: { ...orbit.params.wake, default: 0.28 },
    stir: { ...orbit.params.stir, default: 0.16 },
    vorticity: { ...orbit.params.vorticity, default: 2.5 },
    dyeDissipation: { ...orbit.params.dyeDissipation, default: 0.19 },
    gain: { ...orbit.params.gain, default: 1.2 },
    paperLight: { ...orbit.params.paperLight, default: 1.25 },
  },
  presets: { 'aurora ribbons': {}, 'quiet sky': { heads: 5, chaos: 0, wander: 0.2, ink: 0.3 } },
  async create(context) {
    return orbit.create(context);
  },
});

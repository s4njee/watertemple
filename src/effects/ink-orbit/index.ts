// M3.6: the orbit engine lives in _shared/orbit; this effect is a thin wrapper so importing the
// Ink Orbit effect no longer drags the shared engine (or vice versa) into another effect.
import { createOrbitEffect } from '../_shared/orbit/engine.ts';
import displayShader from './display.wgsl?raw';
import { meta } from './meta.ts';

export { createOrbitEffect, paletteColour, ORBIT_ATTRIBUTE } from '../_shared/orbit/engine.ts';

export const effect = createOrbitEffect({ meta, displayShader });

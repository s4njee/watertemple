import { createOrbitEffect } from './engine.ts';
import { defineEffect, type EffectMeta, type ParamValue } from '../../types.ts';
import { linearColour, ORBIT_VARIATIONS, type OrbitVariationId } from './catalog.ts';
import displayShader from '../../orbit-variants/display.wgsl?raw';

export function createMaterialVariation(meta: EffectMeta, id: OrbitVariationId) {
  const variation = ORBIT_VARIATIONS[id];
  const orbit = createOrbitEffect({
    meta,
    displayShader,
    style: variation.style,
    darkGround: linearColour(variation.background),
  });
  type Schema = typeof orbit.params;
  type AdjustableSchema = {
    [K in keyof Schema]: Omit<Schema[K], 'default'> & { default: ParamValue<Schema[K]> };
  };
  const values: Record<string, string | number | boolean> = {
    ...variation.params,
    inkRadius: variation.width / 800,
    shell: variation.shell / 800,
    paperLight: variation.light,
    cutout: true,
  };
  const params = Object.fromEntries(
    Object.entries(orbit.params).map(([key, definition]) => [
      key,
      { ...definition, default: values[key] ?? definition.default },
    ]),
  ) as unknown as AdjustableSchema;
  return defineEffect({
    ...orbit,
    params,
    presets: { [variation.title.toLowerCase()]: {} },
    async create(context) {
      return orbit.create(context);
    },
  });
}

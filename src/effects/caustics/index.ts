import { createCausticsMap, createCausticsSampler } from '../../gpu/bake-caustics.ts';
import { createFullscreenPipeline, renderFullscreen } from '../../gpu/fullscreen.ts';
import { createUniformBuffer, defineUniformStruct } from '../../gpu/uniforms.ts';
import { defineEffect } from '../types.ts';
import { meta } from './meta.ts';
import overlayShader from './overlay.wgsl?raw';

const CausticsParams = defineUniformStruct('CausticsParams', {
  tint: 'vec3f',
  aspect: 'f32',
  deep: 'vec3f',
  time: 'f32',
  scale: 'f32',
  sharp: 'f32',
  split: 'f32',
  intensity: 'f32',
  opacity: 'f32',
  lod: 'f32',
  method: 'u32',
  layers: 'u32',
  combine: 'u32',
  view: 'u32',
});

/** Index = the CAUSTICS_* constant in caustics.wgsl; `baked` is the demo's own extra method. */
const METHODS = ['cellular', 'sine', 'baked'] as const;
const LAYERS = ['1 layer', '2 layers'] as const;
const COMBINES = ['average', 'min', 'single tap'] as const;
const BAKE_SOURCES = ['cellular', 'sine'] as const;
const VIEWS = ['overlay', 'opaque', 'seam check'] as const;

/** Pattern cells per tile. Must match CELLS in overlay.wgsl. */
const CELLS = 12;
/** Fringing is in tile units: 1 on the slider = a tenth of a cell. */
const SPLIT_PER_UNIT = 0.1 / CELLS;

const indexOf = <T extends readonly string[]>(list: T, value: string) =>
  Math.max(0, list.indexOf(value));

export const effect = defineEffect({
  meta,
  params: {
    method: { type: 'select', default: 'cellular', options: METHODS },
    layers: {
      type: 'select',
      default: '2 layers',
      options: LAYERS,
      label: 'Layers (procedural)',
    },
    bakedFrom: {
      type: 'select',
      default: 'sine',
      options: BAKE_SOURCES,
      label: 'Baked: map built from',
    },
    combine: {
      type: 'select',
      default: 'single tap',
      options: COMBINES,
      label: 'Baked: tap combine',
    },
    scale: { type: 'number', default: 5.5, min: 2, max: 40, step: 0.1, label: 'Cells per height' },
    speed: { type: 'number', default: 0.8, min: 0, max: 3, step: 0.01 },
    sharpness: { type: 'number', default: 0.6, min: 0, max: 1, step: 0.01 },
    fringing: {
      type: 'number',
      default: 0,
      min: 0,
      max: 1,
      step: 0.01,
      label: 'Chromatic fringing (3x cost)',
    },
    intensity: { type: 'number', default: 0.55, min: 0.1, max: 5, step: 0.01 },
    opacity: { type: 'number', default: 1, min: 0, max: 1, step: 0.01 },
    tint: { type: 'color', default: [0.8, 0.96, 1] },
    view: { type: 'select', default: 'opaque', options: VIEWS },
  },
  presets: {
    pool: {
      method: 'cellular',
      layers: '2 layers',
      scale: 5.5,
      speed: 0.8,
      sharpness: 0.6,
      fringing: 0.25,
      intensity: 0.55,
      opacity: 1,
      tint: [0.8, 0.96, 1],
      view: 'opaque',
    },
    deep: {
      method: 'sine',
      layers: '1 layer',
      scale: 4,
      speed: 0.45,
      sharpness: 0.35,
      fringing: 0,
      intensity: 0.45,
      opacity: 1,
      tint: [0.3, 0.75, 1],
      view: 'opaque',
    },
    'subtle overlay': {
      method: 'baked',
      bakedFrom: 'sine',
      combine: 'single tap',
      scale: 4.5,
      speed: 0.6,
      sharpness: 0.55,
      fringing: 0,
      intensity: 0.5,
      opacity: 0.45,
      tint: [0.7, 0.92, 1],
      view: 'overlay',
    },
    // A pale tint cannot lighten a light page, so this one darkens it with saturated blue.
    'light page': {
      method: 'baked',
      bakedFrom: 'sine',
      combine: 'single tap',
      scale: 4.5,
      speed: 0.6,
      sharpness: 0.8,
      fringing: 0,
      intensity: 0.7,
      opacity: 0.55,
      tint: [0.03, 0.4, 0.75],
      view: 'overlay',
    },
    'baked, min and fringed': {
      method: 'baked',
      bakedFrom: 'cellular',
      combine: 'min',
      scale: 6,
      speed: 0.9,
      sharpness: 0.85,
      fringing: 0.5,
      intensity: 0.3,
      opacity: 1,
      tint: [0.85, 0.97, 1],
      view: 'opaque',
    },
  },

  async create({ device, format, params, profiler }) {
    const uniforms = createUniformBuffer(device, CausticsParams, 'caustics params');
    // One baked map per construction, so the baked method can be compared with either. Each is
    // 2.7 MB with its mips and costs a fraction of a millisecond to render, once.
    const maps = BAKE_SOURCES.map((method) =>
      createCausticsMap(device, {
        cells: CELLS,
        method,
        sharpness: 0.85,
        label: `caustics demo map (${method})`,
      }),
    );
    const sampler = createCausticsSampler(device);
    const pipeline = createFullscreenPipeline(device, {
      fragment: {
        label: 'caustics/overlay.wgsl',
        code: overlayShader,
        chunks: { 'caustics_params.wgsl': CausticsParams.wgsl },
      },
      format,
    });
    const bindGroups = maps.map((map) =>
      device.createBindGroup({
        label: 'caustics',
        layout: pipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: uniforms.buffer } },
          { binding: 1, resource: map.view },
          { binding: 2, resource: sampler },
        ],
      }),
    );

    let aspect = 1;
    // Accumulated, so changing the speed never makes the pattern jump.
    let phase = 0;

    return {
      resize(width, height) {
        aspect = Math.max(1, width) / Math.max(1, height);
      },

      frame({ delta }, encoder, target) {
        phase += delta * params.speed;
        uniforms.set({
          tint: params.tint,
          aspect,
          deep: [0.01, 0.075, 0.13],
          time: phase,
          scale: params.scale,
          sharp: params.sharpness,
          split: params.fringing * SPLIT_PER_UNIT,
          intensity: params.intensity,
          opacity: params.opacity,
          // The baked map is sharp; coarser mips are the cheap way to soften it.
          lod: (1 - params.sharpness) * 3.2,
          method: indexOf(METHODS, params.method),
          layers: params.layers === LAYERS[1] ? 2 : 1,
          combine: indexOf(COMBINES, params.combine),
          view: indexOf(VIEWS, params.view),
        });
        uniforms.upload();
        renderFullscreen(
          encoder,
          target,
          pipeline,
          [bindGroups[indexOf(BAKE_SOURCES, params.bakedFrom)] ?? bindGroups[0]!],
          {
            timestampWrites: profiler.pass('caustics'),
          },
        );
      },

      destroy() {
        for (const map of maps) map.destroy();
        uniforms.destroy();
      },
    };
  },
});

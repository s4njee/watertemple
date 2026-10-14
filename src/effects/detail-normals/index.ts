import { createWaterNormalMap } from '../../gpu/bake.ts';
import { createFullscreenPipeline, renderFullscreen } from '../../gpu/fullscreen.ts';
import { createUniformBuffer, defineUniformStruct } from '../../gpu/uniforms.ts';
import { defineEffect } from '../types.ts';
import shader from './detail-normals.wgsl?raw';
import { meta } from './meta.ts';

const Params = defineUniformStruct('Params', {
  phase: 'f32',
  aspect: 'f32',
  strength: 'f32',
  ratio: 'f32',
  scale: 'f32',
  spread: 'f32',
  sun_height: 'f32',
  layers: 'i32',
  blend: 'i32',
  debug: 'u32',
  source: 'u32',
  compare: 'u32',
  water: 'vec3f',
});

const BLENDS = ['whiteout', 'rnm', 'udn', 'pd', 'linear'] as const;
const FILTERS = ['anisotropic', 'trilinear', 'none'] as const;

export const effect = defineEffect({
  meta,
  params: {
    blend: {
      type: 'select',
      default: 'whiteout',
      options: BLENDS,
      label: 'Blend (linear = naive)',
    },
    layers: { type: 'number', default: 3, min: 1, max: 3, step: 1, label: 'Layers' },
    ratio: { type: 'number', default: 2.3, min: 1, max: 4, step: 0.01, label: 'Layer scale ratio' },
    scale: { type: 'number', default: 0.45, min: 0.1, max: 2, step: 0.01, label: 'Tiles per unit' },
    spread: {
      type: 'number',
      default: 70,
      min: 0,
      max: 180,
      step: 1,
      label: 'Direction spread (deg)',
    },
    speed: { type: 'number', default: 1, min: 0, max: 4, step: 0.01 },
    strength: { type: 'number', default: 1, min: 0, max: 3, step: 0.01 },
    sunHeight: { type: 'number', default: 14, min: 3, max: 70, step: 1, label: 'Sun height (deg)' },
    compare: { type: 'boolean', default: false, label: 'Compare (left half: linear)' },
    source: { type: 'select', default: 'texture', options: ['texture', 'procedural'] },
    filter: { type: 'select', default: 'anisotropic', options: FILTERS },
    debug: {
      type: 'select',
      default: 'off',
      options: ['off', 'normal', 'topdown'],
      label: 'Debug view',
    },
    water: { type: 'color', default: [0.01, 0.075, 0.1], label: 'Body colour' },
  },
  presets: {
    'calm lake': { strength: 0.6, spread: 45, speed: 0.5, ratio: 2.3, sunHeight: 10 },
    choppy: { strength: 1.8, spread: 85, speed: 2.2, ratio: 2.0, scale: 0.7, sunHeight: 22 },
    'naive vs correct': {
      compare: true,
      blend: 'rnm',
      strength: 1.6,
      layers: 2,
      sunHeight: 16,
    },
    'sliding layers': { spread: 180, ratio: 1, layers: 2, strength: 1.4, speed: 2 },
    'no mipmaps': { filter: 'none', strength: 1.6, sunHeight: 12 },
  },

  async create({ device, format, params, profiler }) {
    const uniforms = createUniformBuffer(device, Params, 'detail-normals params');
    // Baked once: 512 x 512 rg16float with a full mip chain, about 1.4 MB.
    const normals = createWaterNormalMap(device, { size: 512, seed: 3 });
    const samplers = {
      anisotropic: device.createSampler({
        label: 'detail anisotropic',
        addressModeU: 'repeat',
        addressModeV: 'repeat',
        magFilter: 'linear',
        minFilter: 'linear',
        mipmapFilter: 'linear',
        maxAnisotropy: 16,
      }),
      trilinear: device.createSampler({
        label: 'detail trilinear',
        addressModeU: 'repeat',
        addressModeV: 'repeat',
        magFilter: 'linear',
        minFilter: 'linear',
        mipmapFilter: 'linear',
      }),
      // Locked to mip 0: what a texture without mipmaps does.
      none: device.createSampler({
        label: 'detail no mips',
        addressModeU: 'repeat',
        addressModeV: 'repeat',
        magFilter: 'linear',
        minFilter: 'linear',
        lodMinClamp: 0,
        lodMaxClamp: 0,
      }),
    };

    const pipeline = createFullscreenPipeline(device, {
      fragment: {
        label: 'detail-normals/detail-normals.wgsl',
        code: shader,
        chunks: { 'params.wgsl': Params.wgsl },
      },
      format,
    });
    const bindGroups = Object.fromEntries(
      FILTERS.map((name) => [
        name,
        device.createBindGroup({
          label: `detail-normals ${name}`,
          layout: pipeline.getBindGroupLayout(0),
          entries: [
            { binding: 0, resource: { buffer: uniforms.buffer } },
            { binding: 1, resource: normals.view },
            { binding: 2, resource: samplers[name] },
          ],
        }),
      ]),
    ) as Record<(typeof FILTERS)[number], GPUBindGroup>;

    // Accumulated so moving the speed slider never makes the water jump.
    let phase = 0;

    return {
      resize(width, height) {
        uniforms.set({ aspect: width / height });
      },
      frame({ delta }, encoder, target) {
        phase += delta * params.speed;
        uniforms.set({
          phase,
          strength: params.strength,
          ratio: params.ratio,
          scale: params.scale,
          spread: params.spread,
          sun_height: params.sunHeight,
          layers: Math.round(params.layers),
          blend: Math.max(0, BLENDS.indexOf(params.blend as (typeof BLENDS)[number])),
          debug: params.debug === 'normal' ? 1 : params.debug === 'topdown' ? 2 : 0,
          source: params.source === 'procedural' ? 1 : 0,
          compare: params.compare ? 1 : 0,
          water: params.water,
        });
        uniforms.upload();
        const filter = FILTERS.includes(params.filter as (typeof FILTERS)[number])
          ? (params.filter as (typeof FILTERS)[number])
          : 'anisotropic';
        renderFullscreen(encoder, target, pipeline, [bindGroups[filter]], {
          timestampWrites: profiler.pass('water surface'),
        });
      },
      destroy() {
        uniforms.destroy();
        normals.destroy();
      },
    };
  },
});

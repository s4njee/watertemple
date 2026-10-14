import {
  createFlowMap,
  createFlowSampler,
  createWaterNormalMap,
  createWaterSampler,
} from '../../gpu/bake.ts';
import { createFullscreenPipeline, renderFullscreen } from '../../gpu/fullscreen.ts';
import { createUniformBuffer, defineUniformStruct } from '../../gpu/uniforms.ts';
import { defineEffect } from '../types.ts';
import shader from './flow-map.wgsl';
import { meta } from './meta.ts';

const Params = defineUniformStruct('Params', {
  time: 'f32',
  aspect: 'f32',
  speed: 'f32',
  cycle: 'f32',
  phase_noise: 'f32',
  strength: 'f32',
  tile: 'f32',
  zoom: 'f32',
  foam: 'f32',
  sun_height: 'f32',
  debug: 'u32',
  shallow: 'vec3f',
  deep: 'vec3f',
});

export const effect = defineEffect({
  meta,
  params: {
    speed: { type: 'number', default: 1, min: 0, max: 3, step: 0.01, label: 'Flow speed' },
    cycle: {
      type: 'number',
      default: 1.6,
      min: 0.2,
      max: 8,
      step: 0.01,
      label: 'Cycle length (s)',
    },
    phaseNoise: { type: 'number', default: 0.85, min: 0, max: 1, step: 0.01, label: 'Phase noise' },
    strength: {
      type: 'number',
      default: 1.2,
      min: 0,
      max: 3,
      step: 0.01,
      label: 'Normal strength',
    },
    tile: {
      type: 'number',
      default: 2.2,
      min: 0.5,
      max: 8,
      step: 0.01,
      label: 'Ripple tiles per unit',
    },
    zoom: { type: 'number', default: 1, min: 0.6, max: 2.5, step: 0.01 },
    foam: { type: 'number', default: 0.8, min: 0, max: 1, step: 0.01 },
    sunHeight: {
      type: 'number',
      default: 34,
      min: 15,
      max: 70,
      step: 1,
      label: 'Sun height (deg)',
    },
    debug: {
      type: 'select',
      default: 'off',
      options: ['off', 'flow', 'phase'],
      label: 'Debug view',
    },
    shallow: { type: 'color', default: [0.045, 0.1, 0.07], label: 'Shallow colour' },
    deep: { type: 'color', default: [0.004, 0.05, 0.07], label: 'Deep colour' },
  },
  presets: {
    'lazy river': { speed: 0.45, cycle: 2.6, strength: 0.8, foam: 0.5 },
    'fast stream': { speed: 2.2, cycle: 1.0, strength: 1.6, foam: 1, tile: 2.8 },
    'cycle too long': { cycle: 8, speed: 1.6, phaseNoise: 0.85 },
    'no phase noise': { phaseNoise: 0, cycle: 1.4 },
    'flow field': { debug: 'flow' },
  },

  async create({ device, format, params, profiler }) {
    const uniforms = createUniformBuffer(device, Params, 'flow-map params');
    // Baked once. Normal map: 512 x 512 rgba16float with mips and a height channel, about 2.8 MB.
    // Flow map: 1024 x 512 rgba8unorm, 2 MB, no mips.
    const normals = createWaterNormalMap(device, { size: 512, seed: 7, height: true });
    const flowMap = createFlowMap(device, { width: 1024, seed: 2, rocks: 6 });
    const flowSampler = createFlowSampler(device);
    const normalSampler = createWaterSampler(device, { anisotropy: 4 });

    const pipeline = createFullscreenPipeline(device, {
      fragment: {
        label: 'flow-map/flow-map.wgsl',
        code: shader.code,
        chunks: { ...shader.chunks, 'params.wgsl': Params.wgsl },
      },
      format,
    });
    const bindGroup = device.createBindGroup({
      label: 'flow-map',
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: uniforms.buffer } },
        { binding: 1, resource: flowMap.view },
        { binding: 2, resource: flowSampler },
        { binding: 3, resource: normals.view },
        { binding: 4, resource: normalSampler },
      ],
    });

    // Accumulated, so moving the cycle slider (or pausing) never rewinds the animation.
    let time = 0;

    return {
      resize(width, height) {
        uniforms.set({ aspect: width / height });
      },
      frame({ delta }, encoder, target) {
        time += delta;
        uniforms.set({
          time,
          speed: params.speed,
          cycle: params.cycle,
          phase_noise: params.phaseNoise,
          strength: params.strength,
          tile: params.tile,
          zoom: params.zoom,
          foam: params.foam,
          sun_height: params.sunHeight,
          debug: params.debug === 'flow' ? 1 : params.debug === 'phase' ? 2 : 0,
          shallow: params.shallow,
          deep: params.deep,
        });
        uniforms.upload();
        renderFullscreen(encoder, target, pipeline, [bindGroup], {
          timestampWrites: profiler.pass('river surface'),
        });
      },
      destroy() {
        uniforms.destroy();
        normals.destroy();
        flowMap.destroy();
      },
    };
  },
});

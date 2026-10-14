import {
  CAUSTICS_MAP_CELLS,
  createCausticsMap,
  createCausticsSampler,
} from '../../gpu/bake-caustics.ts';
import { createFullscreenPipeline, renderFullscreen } from '../../gpu/fullscreen.ts';
import { createUniformBuffer, defineUniformStruct } from '../../gpu/uniforms.ts';
import { defineEffect } from '../types.ts';
import { meta } from './meta.ts';
import sceneShader from './scene.wgsl?raw';

const SceneParams = defineUniformStruct('SceneParams', {
  deep: 'vec3f',
  aspect: 'f32',
  time: 'f32',
  wobble: 'f32',
  fog: 'f32',
  shafts: 'f32',
  slant: 'f32',
  source_height: 'f32',
  shaft_frequency: 'f32',
  shaft_contrast: 'f32',
  shaft_reach: 'f32',
  particles: 'f32',
  seabed: 'f32',
  caustic_scale: 'f32',
});

export const effect = defineEffect({
  meta,
  params: {
    shaftIntensity: {
      type: 'number',
      default: 0.7,
      min: 0,
      max: 1.5,
      step: 0.01,
      label: 'Light shafts',
    },
    shaftAngle: {
      type: 'number',
      default: 12,
      min: -45,
      max: 45,
      step: 0.5,
      label: 'Shaft angle (deg)',
    },
    shaftSpread: {
      type: 'number',
      default: 0.5,
      min: 0,
      max: 1,
      step: 0.01,
      label: 'Shaft fan (0 parallel)',
    },
    particles: { type: 'number', default: 0.5, min: 0, max: 1, step: 0.01, label: 'Marine snow' },
    depth: { type: 'number', default: 1.2, min: 0.2, max: 4, step: 0.01, label: 'Fog density' },
    fogColor: { type: 'color', default: [0.0, 0.1, 0.2], label: 'Deep colour' },
    wobble: { type: 'number', default: 0.5, min: 0, max: 1.5, step: 0.01 },
    seabed: { type: 'number', default: 0, min: 0, max: 1, step: 0.01, label: 'Seabed' },
    speed: { type: 'number', default: 0.6, min: 0, max: 2, step: 0.01 },
  },
  presets: {
    'shallow reef': {
      shaftIntensity: 0.9,
      shaftAngle: 14,
      shaftSpread: 0.35,
      particles: 0.4,
      depth: 0.9,
      fogColor: [0.0, 0.16, 0.22],
      wobble: 0.6,
      seabed: 0.85,
      speed: 0.7,
    },
    'open blue': {
      shaftIntensity: 0.6,
      shaftAngle: 8,
      shaftSpread: 0.7,
      particles: 0.35,
      depth: 0.7,
      fogColor: [0.0, 0.05, 0.18],
      wobble: 0.3,
      seabed: 0,
      speed: 0.5,
    },
    murky: {
      shaftIntensity: 0.35,
      shaftAngle: -10,
      shaftSpread: 0.6,
      particles: 0.9,
      depth: 2.2,
      fogColor: [0.03, 0.09, 0.04],
      wobble: 0.8,
      seabed: 0,
      speed: 0.4,
    },
  },

  async create({ device, format, params, profiler }) {
    const uniforms = createUniformBuffer(device, SceneParams, 'underwater params');
    // Only the seabed reads it, but it is tiny and baked once; the pass skips the taps when
    // `seabed` is 0.
    const caustics = createCausticsMap(device, {
      cells: CAUSTICS_MAP_CELLS,
      size: 256,
      method: 'sine',
      sharpness: 0.7,
      label: 'underwater caustics',
    });
    const sampler = createCausticsSampler(device);
    const pipeline = createFullscreenPipeline(device, {
      fragment: {
        label: 'underwater/scene.wgsl',
        code: sceneShader,
        chunks: { 'scene_params.wgsl': SceneParams.wgsl },
      },
      format,
    });
    const bindGroup = device.createBindGroup({
      label: 'underwater',
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: uniforms.buffer } },
        { binding: 1, resource: caustics.view },
        { binding: 2, resource: sampler },
      ],
    });

    let aspect = 1;
    let phase = 0;

    return {
      resize(width, height) {
        aspect = Math.max(1, width) / Math.max(1, height);
      },

      frame({ delta }, encoder, target) {
        phase += delta * params.speed;
        // The fan: a source far away makes parallel shafts, a near one fans them out.
        const sourceHeight = 0.12 + (1 - params.shaftSpread) * 4;
        uniforms.set({
          deep: params.fogColor,
          aspect,
          time: phase,
          wobble: params.wobble,
          fog: params.depth,
          shafts: params.shaftIntensity,
          slant: Math.tan((params.shaftAngle * Math.PI) / 180),
          source_height: sourceHeight,
          shaft_frequency: 3 + 5 * (1 - 0.3 * params.shaftSpread),
          shaft_contrast: 0.55,
          shaft_reach: 0.9 / Math.max(params.depth, 0.2) + 0.35,
          particles: params.particles,
          seabed: params.seabed,
          caustic_scale: 0.9,
        });
        uniforms.upload();
        renderFullscreen(encoder, target, pipeline, [bindGroup], {
          timestampWrites: profiler.pass('underwater'),
        });
      },

      destroy() {
        caustics.destroy();
        uniforms.destroy();
      },
    };
  },
});

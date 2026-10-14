import { createFullscreenPipeline, renderFullscreen } from '../../gpu/fullscreen.ts';
import {
  createPostChain,
  POST_VIEWS,
  TONEMAPPERS,
  type Tonemapper,
  type PostView,
} from '../../gpu/post/index.ts';
import { createUniformBuffer, defineUniformStruct } from '../../gpu/uniforms.ts';
import { defineEffect } from '../types.ts';
import { meta } from './meta.ts';
import sceneShader from './scene.wgsl?raw';

const SceneParams = defineUniformStruct('SceneParams', {
  time: 'f32',
  aspect: 'f32',
  sun: 'f32',
  lights: 'f32',
  caustics: 'f32',
});

const HDR_LABEL = 'HDR output';

export const effect = defineEffect({
  meta,
  // Ask for an HDR canvas where the display and browser have one; ctx.hdr says what we got.
  hdr: true,
  params: {
    exposure: { type: 'number', default: 1, min: 0.1, max: 6, step: 0.01 },
    tonemap: { type: 'select', default: 'aces', options: TONEMAPPERS, label: 'Tonemapper' },
    bloomIntensity: {
      type: 'number',
      default: 0.6,
      min: 0,
      max: 2,
      step: 0.01,
      label: 'Bloom intensity',
    },
    bloomThreshold: {
      type: 'number',
      default: 1,
      min: 0,
      max: 8,
      step: 0.01,
      label: 'Bloom threshold',
    },
    bloomKnee: { type: 'number', default: 0.5, min: 0, max: 2, step: 0.01, label: 'Bloom knee' },
    bloomRadius: {
      type: 'number',
      default: 0.7,
      min: 0,
      max: 1,
      step: 0.01,
      label: 'Bloom radius',
    },
    bloomLevels: { type: 'number', default: 5, min: 1, max: 8, step: 1, label: 'Bloom levels' },
    fireflyFilter: { type: 'boolean', default: true, label: 'Firefly filter' },
    dither: {
      type: 'number',
      default: 1,
      min: 0,
      max: 4,
      step: 0.05,
      label: 'Dither (8-bit steps)',
    },
    hdrOutput: { type: 'boolean', default: true, label: HDR_LABEL },
    headroom: { type: 'number', default: 2.5, min: 1, max: 6, step: 0.05, label: 'HDR headroom' },
    view: { type: 'select', default: 'final', options: POST_VIEWS, label: 'Debug view' },
    speed: { type: 'number', default: 1, min: 0, max: 3, step: 0.01 },
    sun: { type: 'number', default: 1, min: 0, max: 3, step: 0.01, label: 'Sun glints' },
    lights: { type: 'number', default: 1, min: 0, max: 3, step: 0.01, label: 'Lantern radiance' },
    caustics: { type: 'number', default: 1, min: 0, max: 3, step: 0.01 },
  },
  presets: {
    subtle: {
      bloomIntensity: 0.25,
      bloomThreshold: 1.5,
      bloomKnee: 0.6,
      bloomRadius: 0.5,
      bloomLevels: 4,
      tonemap: 'aces',
      exposure: 1,
    },
    dreamy: {
      bloomIntensity: 1.3,
      bloomThreshold: 0.4,
      bloomKnee: 0.8,
      bloomRadius: 0.85,
      bloomLevels: 7,
      tonemap: 'reinhard',
      exposure: 1.2,
    },
    off: { bloomIntensity: 0, tonemap: 'none', exposure: 1 },
  },

  async create({ device, format, params, profiler, hdr = false }) {
    // The label is the only place the control panel can say this; it is rewritten on every build.
    (effect.params.hdrOutput as { label?: string }).label = hdr
      ? HDR_LABEL
      : `${HDR_LABEL} (unavailable on this display or browser)`;

    const uniforms = createUniformBuffer(device, SceneParams, 'hdr-bloom scene params');
    const pipeline = createFullscreenPipeline(device, {
      fragment: {
        label: 'hdr-bloom/scene.wgsl',
        code: sceneShader,
        chunks: { 'scene-params.wgsl': SceneParams.wgsl },
      },
      format: 'rgba16float',
    });
    const bindGroup = device.createBindGroup({
      label: 'hdr-bloom scene',
      layout: pipeline.getBindGroupLayout(0),
      entries: [{ binding: 0, resource: { buffer: uniforms.buffer } }],
    });
    const chain = createPostChain(device, { canvasFormat: format, width: 1, height: 1 });
    let phase = 0;

    return {
      resize(width, height) {
        chain.resize(width, height);
        uniforms.set({ aspect: width / height });
      },
      frame({ delta, frame }, encoder, target) {
        phase += delta * params.speed;
        uniforms.set({
          time: phase,
          sun: params.sun,
          lights: params.lights,
          caustics: params.caustics,
        });
        uniforms.upload();
        renderFullscreen(encoder, chain.scene.view, pipeline, [bindGroup], {
          timestampWrites: profiler.pass('scene'),
        });
        chain.encode(
          encoder,
          target,
          {
            bloom: {
              intensity: params.bloomIntensity,
              threshold: params.bloomThreshold,
              knee: params.bloomKnee,
              radius: params.bloomRadius,
              levels: params.bloomLevels,
              fireflyFilter: params.fireflyFilter,
            },
            output: {
              exposure: params.exposure,
              tonemap: params.tonemap as Tonemapper,
              dither: params.dither,
              frame,
              hdr: hdr && params.hdrOutput,
              headroom: params.headroom,
              view: params.view as PostView,
            },
          },
          profiler,
        );
      },
      destroy() {
        chain.destroy();
        uniforms.destroy();
      },
    };
  },
});

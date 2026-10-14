import { createFullscreenPipeline, renderFullscreen } from '../../gpu/fullscreen.ts';
import { createUniformBuffer, defineUniformStruct } from '../../gpu/uniforms.ts';
import { defineEffect } from '../types.ts';
import shader from './hello.wgsl?raw';
import { meta } from './meta.ts';

const Params = defineUniformStruct('Params', {
  time: 'f32',
  aspect: 'f32',
  scale: 'f32',
  warp: 'f32',
  tint: 'vec3f',
});

export const effect = defineEffect({
  meta,
  params: {
    speed: { type: 'number', default: 1, min: 0, max: 3, step: 0.01 },
    scale: { type: 'number', default: 4, min: 1, max: 12, step: 0.1 },
    warp: { type: 'number', default: 0.6, min: 0, max: 1.5, step: 0.01 },
    tint: { type: 'color', default: [0.05, 0.25, 0.4] },
  },
  presets: {
    calm: { speed: 0.4, warp: 0.3 },
    choppy: { speed: 1.8, scale: 7, warp: 1.1 },
  },

  async create({ device, format, params, profiler }) {
    const uniforms = createUniformBuffer(device, Params, 'hello params');
    const pipeline = createFullscreenPipeline(device, {
      fragment: { label: 'hello/hello.wgsl', code: shader, chunks: { 'params.wgsl': Params.wgsl } },
      format,
    });
    const bindGroup = device.createBindGroup({
      label: 'hello',
      layout: pipeline.getBindGroupLayout(0),
      entries: [{ binding: 0, resource: { buffer: uniforms.buffer } }],
    });

    // Accumulated rather than `time * speed`, so changing speed never makes the pattern jump.
    let phase = 0;

    return {
      resize(width, height) {
        uniforms.set({ aspect: width / height });
      },
      frame({ delta }, encoder, target) {
        phase += delta * params.speed;
        uniforms.set({ time: phase, scale: params.scale, warp: params.warp, tint: params.tint });
        uniforms.upload();
        renderFullscreen(encoder, target, pipeline, [bindGroup], {
          timestampWrites: profiler.pass('noise'),
        });
      },
      destroy() {
        uniforms.destroy();
      },
    };
  },
});

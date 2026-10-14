import { createFullscreenPipeline, renderFullscreen } from '../fullscreen.ts';
import { createPingPongTarget } from '../targets.ts';
import { createUniformBuffer } from '../uniforms.ts';
import { WeatherParams } from './params.ts';
import code from './memory.wgsl';
export function createWeatherMemory(device: GPUDevice, width: number, height: number) {
  const pair = createPingPongTarget(device, {
    label: 'weather glass memory',
    format: 'rgba16float',
    width,
    height,
    usage:
      GPUTextureUsage.RENDER_ATTACHMENT |
      GPUTextureUsage.TEXTURE_BINDING |
      GPUTextureUsage.COPY_SRC,
  });
  const uniforms = createUniformBuffer(device, WeatherParams, 'weather memory params');
  const sampler = device.createSampler({
    magFilter: 'linear',
    minFilter: 'linear',
    addressModeV: 'clamp-to-edge',
  });
  const pipeline = createFullscreenPipeline(device, {
    format: 'rgba16float',
    fragment: {
      label: 'weather/memory.wgsl',
      code: code.code,
      chunks: { 'weather_params.wgsl': WeatherParams.wgsl },
    },
  });
  const groups = [pair.read, pair.write].map((target) =>
    device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: uniforms.buffer } },
        { binding: 1, resource: target.view },
        { binding: 2, resource: sampler },
      ],
    }),
  );
  const views = [pair.read.view, pair.write.view];
  let current = 0,
    initial = true;
  return {
    views,
    get index() {
      return current;
    },
    get texture() {
      return pair.read.texture;
    },
    get view() {
      return pair.read.view;
    },
    update(
      encoder: GPUCommandEncoder,
      values: Parameters<typeof uniforms.set>[0],
      timestampWrites?: GPURenderPassTimestampWrites,
    ) {
      uniforms.set({ ...values, initial: initial ? 1 : 0 });
      uniforms.upload();
      renderFullscreen(encoder, pair.write.view, pipeline, [groups[current]!], { timestampWrites });
      pair.swap();
      current = 1 - current;
      initial = false;
    },
    destroy() {
      pair.destroy();
      uniforms.destroy();
    },
  };
}

import { createShader } from '../../gpu/shader.ts';
import { defineUniformStruct, createUniformBuffer } from '../../gpu/uniforms.ts';
import { createFullscreenPipeline, renderFullscreen } from '../../gpu/fullscreen.ts';
import code from './life.wgsl';
import display from './display.wgsl';
const Params = defineUniformStruct('LifeParams', {
  grid: 'vec2u',
  plant: 'vec2f',
  time: 'f32',
  seed: 'u32',
  patches: 'u32',
  light: 'f32',
});
export function createLifeSimulation(
  device: GPUDevice,
  format: GPUTextureFormat,
  width: number,
  height: number,
) {
  const uniforms = createUniformBuffer(device, Params, 'Life params');
  uniforms.set({ grid: [width, height], plant: [0, 0], time: 0, seed: 1, patches: 3, light: 1 });
  uniforms.upload();
  const textures = [0, 1].map((i) =>
    device.createTexture({
      label: `Life cells ${i}`,
      size: [width, height],
      format: 'r32uint',
      usage:
        GPUTextureUsage.STORAGE_BINDING |
        GPUTextureUsage.TEXTURE_BINDING |
        GPUTextureUsage.COPY_SRC |
        GPUTextureUsage.COPY_DST,
    }),
  );
  const module = createShader(device, {
    label: 'game-of-life/life.wgsl',
    code: code.code,
    chunks: { 'life_params.wgsl': Params.wgsl },
  });
  const pipelines = ['advance', 'plant', 'sprinkle'].map((entryPoint) =>
    device.createComputePipeline({ layout: 'auto', compute: { module, entryPoint } }),
  );
  const shade = createFullscreenPipeline(device, {
    format,
    fragment: {
      label: 'game-of-life/display.wgsl',
      code: display.code,
      chunks: { ...display.chunks, 'life_params.wgsl': Params.wgsl },
    },
  });
  const groups = pipelines.map((pipeline) =>
    textures.map((texture, i) =>
      device.createBindGroup({
        layout: pipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: uniforms.buffer } },
          { binding: 1, resource: texture.createView() },
          { binding: 2, resource: textures[1 - i]!.createView() },
        ],
      }),
    ),
  );
  const shades = textures.map((texture) =>
    device.createBindGroup({
      layout: shade.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: uniforms.buffer } },
        { binding: 1, resource: texture.createView() },
      ],
    }),
  );
  let current = 0;
  function dispatch(encoder: GPUCommandEncoder, kernel: number) {
    const pass = encoder.beginComputePass({
      label: ['Conway B3/S23', 'plant glider', 'random seed patches'][kernel],
    });
    pass.setPipeline(pipelines[kernel]!);
    pass.setBindGroup(0, groups[kernel]![current]!);
    pass.dispatchWorkgroups(Math.ceil(width / 8), Math.ceil(height / 8));
    pass.end();
    current = 1 - current;
  }
  return {
    get cells() {
      return textures[current]!;
    },
    seed(data: Uint32Array) {
      if (data.length !== width * height) throw new Error('Life seed dimensions do not match');
      current = 0;
      device.queue.writeTexture(
        { texture: textures[0]! },
        data as Uint32Array<ArrayBuffer>,
        { bytesPerRow: width * 4 },
        { width, height },
      );
    },
    step(encoder: GPUCommandEncoder, iterations = 1) {
      for (let i = 0; i < iterations; i++) dispatch(encoder, 0);
    },
    plant(encoder: GPUCommandEncoder, x: number, y: number) {
      uniforms.set({ plant: [x, y] });
      uniforms.upload();
      dispatch(encoder, 1);
    },
    sprinkle(encoder: GPUCommandEncoder, seed: number) {
      uniforms.set({
        seed: seed >>> 0,
        patches: Math.min(8, Math.max(3, Math.round((width * height) / 4000))),
      });
      uniforms.upload();
      dispatch(encoder, 2);
    },
    setTime(time: number, light = 1) {
      uniforms.set({ time, light });
      uniforms.upload();
    },
    draw(
      encoder: GPUCommandEncoder,
      target: GPUTextureView,
      timestampWrites?: GPURenderPassTimestampWrites,
    ) {
      renderFullscreen(encoder, target, shade, [shades[current]!], { timestampWrites });
    },
    destroy() {
      textures.forEach((texture) => texture.destroy());
      uniforms.destroy();
    },
  };
}

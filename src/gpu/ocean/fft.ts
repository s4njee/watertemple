import { createShader } from '../shader.ts';
import { createUniformBuffer, defineUniformStruct } from '../uniforms.ts';
import { noProfiler, type PassProfiler } from '../profiler.ts';
import { oceanSpectrum, OCEAN_LENGTHS } from './spectrum.ts';
import code from './fft.wgsl';
const Params = defineUniformStruct('FftParams', {
  size: 'u32',
  bits: 'u32',
  time: 'f32',
  delta: 'f32',
  lengths: 'vec3f',
  strength: 'f32',
  chop: 'f32',
  foam_threshold: 'f32',
  foam_lifetime: 'f32',
});
const Step = defineUniformStruct('FftStep', { span: 'u32', axis: 'u32' });
export interface SpectralOcean {
  readonly size: number;
  readonly displacement: GPUTexture;
  readonly derivative: GPUTextureView;
  readonly derivativeTexture: GPUTexture;
  /** Normalized height and horizontal displacement remain available for numeric validation. */
  readonly spatial: GPUBuffer;
  run(
    encoder: GPUCommandEncoder,
    time: number,
    delta: number,
    options?: { strength?: number; chop?: number; foamThreshold?: number; foamLifetime?: number },
  ): void;
  destroy(): void;
}
export function createSpectralOcean(
  device: GPUDevice,
  size = 64,
  profiler: PassProfiler = noProfiler,
  wind = 12,
): SpectralOcean {
  const seeds = oceanSpectrum({ size, wind });
  const bits = Math.log2(size);
  const params = createUniformBuffer(device, Params, 'ocean FFT parameters');
  const seed = device.createBuffer({
    label: 'ocean Phillips seeds',
    size: seeds.byteLength,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(seed, 0, seeds);
  const buffers = [0, 1].map(() =>
    device.createBuffer({
      label: 'ocean FFT ping pong',
      size: 6 * size * size * 16,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
    }),
  );
  const texture = () =>
    device.createTexture({
      label: 'ocean displacement / derivatives',
      size: [size, size, 3],
      format: 'rgba16float',
      usage:
        GPUTextureUsage.STORAGE_BINDING |
        GPUTextureUsage.TEXTURE_BINDING |
        GPUTextureUsage.COPY_SRC,
    });
  const displacement = texture(),
    derivatives = [texture(), texture()];
  const views = derivatives.map((t) => t.createView({ dimension: '2d-array' }));
  const module = createShader(device, {
    label: 'ocean/fft.wgsl',
    code: code.code,
    chunks: { 'fft-params.wgsl': Params.wgsl, 'fft-step.wgsl': Step.wgsl },
  });
  const pipeline = (entryPoint: string) =>
    device.createComputePipeline({
      label: `ocean ${entryPoint}`,
      layout: 'auto',
      compute: { module, entryPoint },
    });
  const evolve = pipeline('evolve'),
    butterfly = pipeline('butterfly'),
    pack = pipeline('pack');
  const group = (pipeline: GPUComputePipeline, entries: GPUBindGroupEntry[]) =>
    device.createBindGroup({ layout: pipeline.getBindGroupLayout(0), entries });
  const uniform = { binding: 0, resource: { buffer: params.buffer } };
  const evolveGroup = group(evolve, [
    uniform,
    { binding: 1, resource: { buffer: seed } },
    { binding: 2, resource: { buffer: buffers[0]! } },
  ]);
  const steps: {
    u: ReturnType<typeof createUniformBuffer<typeof Step.fields>>;
    bind: GPUBindGroup;
  }[] = [];
  let current = 0;
  for (let axis = 0; axis < 2; axis++)
    for (let s = 1; s <= bits; s++) {
      const u = createUniformBuffer(device, Step, `FFT ${axis}:${s}`);
      u.set({ span: 2 ** s, axis });
      u.upload();
      const bind = group(butterfly, [
        uniform,
        { binding: 1, resource: { buffer: buffers[current]! } },
        { binding: 2, resource: { buffer: buffers[1 - current]! } },
        { binding: 3, resource: { buffer: u.buffer } },
      ]);
      steps.push({ u, bind });
      current = 1 - current;
    }
  const packGroups = [0, 1].map((i) =>
    group(pack, [
      uniform,
      { binding: 1, resource: { buffer: buffers[current]! } },
      { binding: 4, resource: displacement.createView({ dimension: '2d-array' }) },
      { binding: 5, resource: views[i]! },
      { binding: 6, resource: views[1 - i]! },
    ]),
  );
  let active = 0;
  return {
    size,
    displacement,
    get derivative() {
      return views[active]!;
    },
    get derivativeTexture() {
      return derivatives[active]!;
    },
    spatial: buffers[current]!,
    run(encoder, time, delta, options = {}) {
      params.set({
        size,
        bits,
        time,
        delta: Math.min(Math.max(delta, 0), 0.1),
        lengths: OCEAN_LENGTHS,
        strength: options.strength ?? 1,
        chop: options.chop ?? 1,
        foam_threshold: options.foamThreshold ?? 0.75,
        foam_lifetime: Math.max(0.1, options.foamLifetime ?? 3),
      });
      params.upload();
      const run = (p: GPUComputePipeline, b: GPUBindGroup, label: string, layers: number) => {
        const pass = encoder.beginComputePass({ label, timestampWrites: profiler.pass(label) });
        pass.setPipeline(p);
        pass.setBindGroup(0, b);
        pass.dispatchWorkgroups(Math.ceil(size / 8), Math.ceil(size / 8), layers);
        pass.end();
      };
      run(evolve, evolveGroup, 'ocean spectrum', 3);
      for (const s of steps) run(butterfly, s.bind, 'ocean IFFT', 6);
      active = 1 - active;
      run(pack, packGroups[active]!, 'ocean derivatives and foam', 3);
    },
    destroy() {
      params.destroy();
      seed.destroy();
      buffers.forEach((b) => b.destroy());
      displacement.destroy();
      derivatives.forEach((t) => t.destroy());
      steps.forEach((s) => s.u.destroy());
    },
  };
}

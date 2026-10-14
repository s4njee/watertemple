import { createShader } from '../../gpu/shader.ts';
import {
  createUniformBuffer,
  defineUniformStruct,
  uniformArray,
  type UniformValues,
} from '../../gpu/uniforms.ts';
import { createFullscreenPipeline, renderFullscreen } from '../../gpu/fullscreen.ts';
import physicsCode from './physarum.wgsl';
import displayCode from './display.wgsl';

export const MAX_ROOTS = 24;
const fields = {
  grid: 'vec2u',
  count: 'u32',
  rect_count: 'u32',
  aspect: 'f32',
  pixel: 'f32',
  sensor_distance: 'f32',
  sensor_angle: 'f32',
  speed: 'f32',
  turn: 'f32',
  diffusion: 'f32',
  decay: 'f32',
  deposit: 'f32',
  shell: 'f32',
  pull: 'f32',
  light: 'f32',
  exposure: 'f32',
  pointer: 'vec4f',
  colours: uniformArray('vec4f', 3),
  boxes: uniformArray('vec4f', MAX_ROOTS),
  extras: uniformArray('vec4f', MAX_ROOTS),
} as const;
const Params = defineUniformStruct('MyceliumParams', fields);
export type MyceliumValues = UniformValues<typeof fields>;

/** The actual Physarum simulation, exposed independently for geometry/pixel integration tests. */
export function createMyceliumSimulation(
  device: GPUDevice,
  format: GPUTextureFormat,
  width: number,
  height: number,
  count: number,
) {
  const uniforms = createUniformBuffer(device, Params, 'mycelium params');
  uniforms.set({
    grid: [width, height],
    count,
    rect_count: 0,
    aspect: width / height,
    pixel: 1 / height,
    sensor_distance: 12,
    sensor_angle: 0.55,
    speed: 1.4,
    turn: 0.25,
    diffusion: 0.6,
    decay: 1.4,
    deposit: 0.35,
    shell: 0.025,
    pull: 1,
    light: 1,
    exposure: 1.35,
    pointer: [0.5, 0.5, 0, 0],
    colours: [0.02, 0.65, 1, 0, 0.7, 0.025, 0.55, 0, 0.25, 0.03, 1, 0],
  });
  const texture = (label: string) =>
    device.createTexture({
      label,
      size: [width, height],
      format: 'rgba16float',
      usage:
        GPUTextureUsage.TEXTURE_BINDING |
        GPUTextureUsage.STORAGE_BINDING |
        GPUTextureUsage.COPY_SRC,
    });
  const trails = [texture('mycelium trail A'), texture('mycelium trail B')] as const;
  const walls = texture('mycelium page boundaries');
  const agents = device.createBuffer({
    label: 'mycelium agents',
    size: count * 16,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
  });
  const deposits = device.createBuffer({
    label: 'mycelium atomic deposits',
    size: width * height * 16,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  });
  const module = createShader(device, {
    label: 'neon-mycelium/physarum.wgsl',
    code: physicsCode.code,
    chunks: { ...physicsCode.chunks, 'mycelium_params.wgsl': Params.wgsl },
  });
  const pipeline = (entryPoint: string) =>
    device.createComputePipeline({
      label: `mycelium ${entryPoint}`,
      layout: 'auto',
      compute: { module, entryPoint },
    });
  const pipelines = {
    mask: pipeline('mask'),
    initialize: pipeline('initialize'),
    advance: pipeline('advance'),
    diffuse: pipeline('diffuse'),
  };
  // Separate layouts expose only the resources each kernel actually uses.
  const bindings = {
    mask: [0, 6],
    initialize: [0, 3, 5],
    advance: [0, 1, 3, 4, 5],
    diffuse: [0, 1, 2, 4, 5],
  } as const;
  function group(name: keyof typeof pipelines, side: number) {
    const resources: Record<number, GPUBindingResource> = {
      0: { buffer: uniforms.buffer },
      1: trails[side]!.createView(),
      2: trails[1 - side]!.createView(),
      3: { buffer: agents },
      4: { buffer: deposits },
      5: walls.createView(),
      6: walls.createView(),
    };
    return device.createBindGroup({
      layout: pipelines[name].getBindGroupLayout(0),
      entries: bindings[name].map((binding) => ({ binding, resource: resources[binding]! })),
    });
  }
  const groups = [0, 1].map((side) => ({
    initialize: group('initialize', side),
    advance: group('advance', side),
    diffuse: group('diffuse', side),
  }));
  const maskGroup = group('mask', 0);
  const display = createFullscreenPipeline(device, {
    fragment: {
      label: 'neon-mycelium/display.wgsl',
      code: displayCode.code,
      chunks: { ...displayCode.chunks, 'mycelium_params.wgsl': Params.wgsl },
    },
    format,
  });
  const sampler = device.createSampler({ minFilter: 'linear', magFilter: 'linear' });
  const displayGroups = trails.map((trail) =>
    device.createBindGroup({
      layout: display.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: uniforms.buffer } },
        { binding: 1, resource: trail.createView() },
        { binding: 2, resource: sampler },
      ],
    }),
  );
  let side = 0;
  function dispatch(
    encoder: GPUCommandEncoder,
    name: keyof typeof pipelines,
    group: GPUBindGroup,
    x: number,
    y = 1,
  ) {
    const pass = encoder.beginComputePass({ label: `mycelium ${name}` });
    pass.setPipeline(pipelines[name]);
    pass.setBindGroup(0, group);
    pass.dispatchWorkgroups(x, y);
    pass.end();
  }
  return {
    agents,
    walls,
    get trail() {
      return trails[side]!;
    },
    configure(values: MyceliumValues) {
      uniforms.set(values);
      uniforms.upload();
    },
    updateWalls(encoder: GPUCommandEncoder) {
      dispatch(encoder, 'mask', maskGroup, Math.ceil(width / 8), Math.ceil(height / 8));
    },
    initialize(encoder: GPUCommandEncoder) {
      dispatch(encoder, 'initialize', groups[side]!.initialize, Math.ceil(count / 64));
    },
    step(encoder: GPUCommandEncoder, iterations: number) {
      for (let i = 0; i < iterations; i++) {
        encoder.clearBuffer(deposits);
        dispatch(encoder, 'advance', groups[side]!.advance, Math.ceil(count / 64));
        dispatch(
          encoder,
          'diffuse',
          groups[side]!.diffuse,
          Math.ceil(width / 8),
          Math.ceil(height / 8),
        );
        side = 1 - side;
      }
    },
    draw(
      encoder: GPUCommandEncoder,
      target: GPUTextureView,
      timestampWrites?: GPURenderPassTimestampWrites,
    ) {
      renderFullscreen(encoder, target, display, [displayGroups[side]!], { timestampWrites });
    },
    destroy() {
      uniforms.destroy();
      agents.destroy();
      deposits.destroy();
      walls.destroy();
      trails.forEach((t) => t.destroy());
    },
  };
}

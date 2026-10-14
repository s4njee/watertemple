import { describe, expect, it } from 'vitest';
import {
  createFullscreenPipeline,
  drawFullscreen,
  renderFullscreen,
} from '../src/gpu/fullscreen.ts';
import { createUniformBuffer, defineUniformStruct } from '../src/gpu/uniforms.ts';
import { captureValidationError, expectValid, getTestDevice, readTexture } from './gpu.ts';

function createTarget(device: GPUDevice, width: number, height: number, format: GPUTextureFormat) {
  return device.createTexture({
    size: [width, height],
    format,
    usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC,
  });
}

const header = '#include "fullscreen.wgsl"\n';

describe('fullscreen.wgsl', () => {
  it('puts uv (0, 0) at the top-left, matching texture coordinates', async () => {
    const device = await getTestDevice();
    const target = createTarget(device, 4, 2, 'rgba32float');
    await expectValid(device, () => {
      const pipeline = createFullscreenPipeline(device, {
        fragment: {
          label: 'test/uv.wgsl',
          code:
            header +
            '@fragment fn fs(in: FullscreenVarying) -> @location(0) vec4f { return vec4f(in.uv, 0.0, 1.0); }',
        },
        format: 'rgba32float',
      });
      const encoder = device.createCommandEncoder();
      renderFullscreen(encoder, target.createView(), pipeline);
      device.queue.submit([encoder.finish()]);
    });

    const texels = new Float32Array((await readTexture(device, target, 16)).buffer);
    const at = (x: number, y: number) => [texels[(y * 4 + x) * 4], texels[(y * 4 + x) * 4 + 1]];
    // Pixel centres: u = (x + 0.5) / 4, v = (y + 0.5) / 2, with row 0 at the top.
    const expected = [
      [0, 0, 0.125, 0.25],
      [3, 0, 0.875, 0.25],
      [0, 1, 0.125, 0.75],
      [3, 1, 0.875, 0.75],
    ] as const;
    for (const [x, y, u, v] of expected) {
      const [gotU, gotV] = at(x, y);
      expect(gotU).toBeCloseTo(u, 5);
      expect(gotV).toBeCloseTo(v, 5);
    }
  });

  it('covers every pixel of any target size', async () => {
    const device = await getTestDevice();
    const pipeline = await expectValid(device, () =>
      createFullscreenPipeline(device, {
        fragment: {
          label: 'test/white.wgsl',
          code: header + '@fragment fn fs() -> @location(0) vec4f { return vec4f(1.0); }',
        },
        format: 'rgba8unorm',
      }),
    );
    for (const [width, height] of [
      [1, 1],
      [7, 3],
      [64, 64],
    ] as const) {
      const target = createTarget(device, width, height, 'rgba8unorm');
      await expectValid(device, () => {
        const encoder = device.createCommandEncoder();
        renderFullscreen(encoder, target.createView(), pipeline);
        device.queue.submit([encoder.finish()]);
      });
      const pixels = await readTexture(device, target);
      expect(pixels.every((byte) => byte === 255)).toBe(true);
    }
  });
});

describe('createFullscreenPipeline', () => {
  it('applies the blend state to every target, and load keeps existing contents', async () => {
    const device = await getTestDevice();
    const target = createTarget(device, 2, 2, 'rgba8unorm');
    const additive: GPUBlendComponent = { srcFactor: 'one', dstFactor: 'one', operation: 'add' };
    await expectValid(device, () => {
      const pipeline = createFullscreenPipeline(device, {
        fragment: {
          label: 'test/quarter.wgsl',
          code:
            header +
            '@fragment fn fs() -> @location(0) vec4f { return vec4f(0.25, 0.0, 0.0, 1.0); }',
        },
        format: 'rgba8unorm',
        blend: { color: additive, alpha: { ...additive, dstFactor: 'zero' } },
      });
      const encoder = device.createCommandEncoder();
      const view = target.createView();
      renderFullscreen(encoder, view, pipeline);
      renderFullscreen(encoder, view, pipeline, [], { load: true });
      renderFullscreen(encoder, view, pipeline, [], { load: true });
      device.queue.submit([encoder.finish()]);
    });
    const pixels = await readTexture(device, target);
    expect(pixels[0]).toBeGreaterThanOrEqual(190);
    expect(pixels[0]).toBeLessThanOrEqual(194);
    expect(pixels[3]).toBe(255);
  });

  it('accepts explicit bind group layouts', async () => {
    const device = await getTestDevice();
    const target = createTarget(device, 1, 1, 'rgba8unorm');
    await expectValid(device, () => {
      const layout = device.createBindGroupLayout({
        entries: [{ binding: 0, visibility: GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } }],
      });
      const buffer = device.createBuffer({
        size: 16,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      });
      device.queue.writeBuffer(buffer, 0, new Float32Array([0, 1, 0, 1]));
      const group = device.createBindGroup({
        layout,
        entries: [{ binding: 0, resource: { buffer } }],
      });
      const pipeline = createFullscreenPipeline(device, {
        fragment: {
          label: 'test/uniform-colour.wgsl',
          code:
            header +
            '@group(0) @binding(0) var<uniform> colour: vec4f;\n' +
            '@fragment fn fs() -> @location(0) vec4f { return colour; }',
        },
        format: 'rgba8unorm',
        layout: [layout],
      });
      const encoder = device.createCommandEncoder();
      renderFullscreen(encoder, target.createView(), pipeline, [group]);
      device.queue.submit([encoder.finish()]);
    });
    expect([...(await readTexture(device, target))]).toEqual([0, 255, 0, 255]);
  });

  it('supports several colour targets and a custom entry point, and labels the pipeline', async () => {
    const device = await getTestDevice();
    const first = createTarget(device, 1, 1, 'rgba8unorm');
    const second = createTarget(device, 1, 1, 'rgba8unorm');
    const pipeline = await expectValid(device, () => {
      const pipeline = createFullscreenPipeline(device, {
        fragment: {
          label: 'test/mrt.wgsl',
          code:
            header +
            'struct Out { @location(0) a: vec4f, @location(1) b: vec4f }\n' +
            '@fragment fn custom() -> Out { return Out(vec4f(1.0, 0.0, 0.0, 1.0), vec4f(0.0, 0.0, 1.0, 1.0)); }',
        },
        entryPoint: 'custom',
        format: ['rgba8unorm', 'rgba8unorm'],
      });
      const encoder = device.createCommandEncoder();
      renderFullscreen(encoder, [first.createView(), second.createView()], pipeline);
      device.queue.submit([encoder.finish()]);
      return pipeline;
    });
    expect(pipeline.label).toBe('test/mrt.wgsl');
    expect([...(await readTexture(device, first))]).toEqual([255, 0, 0, 255]);
    expect([...(await readTexture(device, second))]).toEqual([0, 0, 255, 255]);
  });

  it('drawFullscreen assigns bind groups to consecutive group indices', async () => {
    const device = await getTestDevice();
    const target = createTarget(device, 1, 1, 'rgba8unorm');
    await expectValid(device, () => {
      const pipeline = createFullscreenPipeline(device, {
        fragment: {
          label: 'test/two-groups.wgsl',
          code:
            header +
            '@group(0) @binding(0) var<uniform> a: vec4f;\n' +
            '@group(1) @binding(0) var<uniform> b: vec4f;\n' +
            '@fragment fn fs() -> @location(0) vec4f { return a + b; }',
        },
        format: 'rgba8unorm',
      });
      const group = (index: number, value: number[]) => {
        const buffer = device.createBuffer({
          size: 16,
          usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });
        device.queue.writeBuffer(buffer, 0, new Float32Array(value));
        return device.createBindGroup({
          layout: pipeline.getBindGroupLayout(index),
          entries: [{ binding: 0, resource: { buffer } }],
        });
      };
      const encoder = device.createCommandEncoder();
      const pass = encoder.beginRenderPass({
        colorAttachments: [{ view: target.createView(), loadOp: 'clear', storeOp: 'store' }],
      });
      drawFullscreen(pass, pipeline, group(0, [1, 0, 0, 0]), group(1, [0, 0, 1, 1]));
      pass.end();
      device.queue.submit([encoder.finish()]);
    });
    expect([...(await readTexture(device, target))]).toEqual([255, 0, 255, 255]);
  });

  it('reports a shader that forgot the include as a validation error', async () => {
    const device = await getTestDevice();
    const message = await captureValidationError(device, () =>
      createFullscreenPipeline(device, {
        fragment: {
          label: 'test/no-include.wgsl',
          code: '@fragment fn fs() -> @location(0) vec4f { return vec4f(1.0); }',
        },
        format: 'rgba8unorm',
      }),
    );
    expect(message).not.toBeNull();
  });
});

describe('a minimal effect built from the helpers', () => {
  const Globals = defineUniformStruct('Globals', { time: 'f32', resolution: 'vec2f' });

  async function createEffect(device: GPUDevice, format: GPUTextureFormat) {
    const uniforms = createUniformBuffer(device, Globals);
    const pipeline = createFullscreenPipeline(device, {
      fragment: {
        label: 'test/effect.wgsl',
        chunks: { 'globals.wgsl': Globals.wgsl },
        code: `#include "fullscreen.wgsl"
#include "globals.wgsl"
@group(0) @binding(0) var<uniform> globals: Globals;
@fragment fn fs(in: FullscreenVarying) -> @location(0) vec4f {
  let pixel = in.uv * globals.resolution;
  return vec4f(pixel.x / globals.resolution.x, globals.time, pixel.y / globals.resolution.y, 1.0);
}`,
      },
      format,
    });
    const group = device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [{ binding: 0, resource: { buffer: uniforms.buffer } }],
    });
    return {
      resize: (width: number, height: number) => uniforms.set({ resolution: [width, height] }),
      frame(time: number, encoder: GPUCommandEncoder, target: GPUTextureView) {
        uniforms.set({ time });
        uniforms.upload();
        renderFullscreen(encoder, target, pipeline, [group]);
      },
      destroy: () => uniforms.destroy(),
    };
  }

  it('renders time and resolution through the generated struct', async () => {
    const device = await getTestDevice();
    const target = createTarget(device, 4, 2, 'rgba32float');
    const effect = await expectValid(device, () => createEffect(device, 'rgba32float'));
    effect.resize(4, 2);
    await expectValid(device, () => {
      const encoder = device.createCommandEncoder();
      effect.frame(0.5, encoder, target.createView());
      device.queue.submit([encoder.finish()]);
    });
    const texels = new Float32Array((await readTexture(device, target, 16)).buffer);
    const rgb = (x: number, y: number) =>
      Array.from(texels.subarray((y * 4 + x) * 4, (y * 4 + x) * 4 + 3));
    expect(rgb(0, 0)).toEqual([0.125, 0.5, 0.25]);
    expect(rgb(3, 1)).toEqual([0.875, 0.5, 0.75]);
    effect.destroy();
  });
});

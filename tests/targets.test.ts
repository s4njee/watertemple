import { describe, expect, it, vi } from 'vitest';
import { createFullscreenPipeline, renderFullscreen } from '../src/gpu/fullscreen.ts';
import { blit } from '../src/gpu/passes/blit.ts';
import { createPingPongTarget, createRenderTarget } from '../src/gpu/targets.ts';
import { expectValid, getTestDevice, readTexture } from './gpu.ts';

const READBACK =
  GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_SRC;

async function fill(device: GPUDevice, view: GPUTextureView, r: number) {
  const pipeline = createFullscreenPipeline(device, {
    fragment: {
      label: 'test/fill.wgsl',
      code: `#include "fullscreen.wgsl"
@fragment fn fs() -> @location(0) vec4f { return vec4f(${r}, 0.0, 0.0, 1.0); }`,
    },
    format: 'rgba8unorm',
  });
  const encoder = device.createCommandEncoder();
  renderFullscreen(encoder, view, pipeline);
  device.queue.submit([encoder.finish()]);
}

describe('createRenderTarget', () => {
  it('creates a texture and default view that can be rendered to and sampled', async () => {
    const device = await getTestDevice();
    const target = await expectValid(device, () =>
      createRenderTarget(device, { label: 'scene', format: 'rgba16float', width: 32, height: 16 }),
    );
    expect(target.texture.label).toBe('scene');
    expect(target.texture.width).toBe(32);
    expect(target.texture.height).toBe(16);
    expect(target.texture.format).toBe('rgba16float');
    expect(target.format).toBe('rgba16float');
    expect(target.usage & GPUTextureUsage.RENDER_ATTACHMENT).toBeTruthy();
    expect(target.usage & GPUTextureUsage.TEXTURE_BINDING).toBeTruthy();
    expect(target.usage & GPUTextureUsage.STORAGE_BINDING).toBeFalsy();
    target.destroy();
  });

  it('accepts custom usage, including storage binding', async () => {
    const device = await getTestDevice();
    const target = await expectValid(device, () =>
      createRenderTarget(device, {
        label: 'storage',
        format: 'rgba8unorm',
        width: 8,
        height: 8,
        usage: READBACK | GPUTextureUsage.STORAGE_BINDING,
      }),
    );
    expect(target.texture.usage & GPUTextureUsage.STORAGE_BINDING).toBeTruthy();
    expect(target.texture.usage & GPUTextureUsage.COPY_SRC).toBeTruthy();
    target.destroy();
  });

  it('reallocates on resize only when the size changes', async () => {
    const device = await getTestDevice();
    const target = createRenderTarget(device, {
      label: 'resizable',
      format: 'rgba8unorm',
      width: 8,
      height: 8,
    });
    const { texture, view } = target;
    const destroy = vi.spyOn(texture, 'destroy');

    expect(target.resize(8, 8)).toBe(false);
    expect(target.texture).toBe(texture);
    expect(target.view).toBe(view);
    expect(destroy).not.toHaveBeenCalled();

    expect(target.resize(16, 4)).toBe(true);
    expect(target.texture).not.toBe(texture);
    expect(target.view).not.toBe(view);
    expect(target.width).toBe(16);
    expect(target.height).toBe(4);
    expect(target.texture.width).toBe(16);
    expect(destroy).toHaveBeenCalledOnce();
    target.destroy();
  });

  it('loses contents on resize', async () => {
    const device = await getTestDevice();
    const target = createRenderTarget(device, {
      label: 'volatile',
      format: 'rgba8unorm',
      width: 2,
      height: 2,
      usage: READBACK,
    });
    await fill(device, target.view, 1);
    expect((await readTexture(device, target.texture))[0]).toBe(255);

    target.resize(3, 3);
    expect([...(await readTexture(device, target.texture))].every((byte) => byte === 0)).toBe(true);
    target.destroy();
  });

  it('rounds sizes, keeps them at least 1, and rejects non-finite ones', async () => {
    const device = await getTestDevice();
    const target = createRenderTarget(device, {
      label: 'clamped',
      format: 'rgba8unorm',
      width: 0,
      height: 0.4,
    });
    expect([target.width, target.height]).toEqual([1, 1]);
    expect(target.resize(10.6, 3.2)).toBe(true);
    expect([target.width, target.height]).toEqual([11, 3]);
    expect(() => target.resize(Number.NaN, 4)).toThrow(RangeError);
    await expectValid(device, () => target.destroy());
  });

  it('ignores resize after destroy instead of leaking a texture', async () => {
    const device = await getTestDevice();
    const target = createRenderTarget(device, {
      label: 'dead',
      format: 'rgba8unorm',
      width: 2,
      height: 2,
    });
    target.destroy();
    expect(target.resize(4, 4)).toBe(false);
    expect(target.width).toBe(2);
  });
});

describe('createPingPongTarget', () => {
  it('exposes distinct read and write targets that exchange on swap', async () => {
    const device = await getTestDevice();
    const pair = await expectValid(device, () =>
      createPingPongTarget(device, { label: 'state', format: 'rgba16float', width: 4, height: 4 }),
    );
    const { read, write } = pair;
    expect(read).not.toBe(write);
    expect(read.texture).not.toBe(write.texture);
    expect(read.texture.label).toBe('state A');
    expect(write.texture.label).toBe('state B');

    pair.swap();
    expect(pair.read).toBe(write);
    expect(pair.write).toBe(read);
    pair.swap();
    expect(pair.read).toBe(read);
    pair.destroy();
  });

  it('carries a render from write to read across a swap', async () => {
    const device = await getTestDevice();
    const pair = createPingPongTarget(device, {
      label: 'state',
      format: 'rgba8unorm',
      width: 2,
      height: 2,
      usage: READBACK,
    });
    await fill(device, pair.write.view, 1);
    pair.swap();
    expect((await readTexture(device, pair.read.texture))[0]).toBe(255);
    expect((await readTexture(device, pair.write.texture))[0]).toBe(0);
    pair.destroy();
  });

  it('resizes both targets, keeps the swap state, and destroys both', async () => {
    const device = await getTestDevice();
    const pair = createPingPongTarget(device, {
      label: 'state',
      format: 'rgba8unorm',
      width: 4,
      height: 4,
    });
    pair.swap();
    const { read, write } = pair;

    expect(pair.resize(4, 4)).toBe(false);
    expect(pair.resize(8, 2)).toBe(true);
    expect(pair.read).toBe(read);
    expect(pair.write).toBe(write);
    expect([read.width, read.height, write.width, write.height]).toEqual([8, 2, 8, 2]);

    const destroyRead = vi.spyOn(read.texture, 'destroy');
    const destroyWrite = vi.spyOn(write.texture, 'destroy');
    pair.destroy();
    expect(destroyRead).toHaveBeenCalledOnce();
    expect(destroyWrite).toHaveBeenCalledOnce();
  });

  it('lets a compute pass write the pair while render passes read it', async () => {
    const device = await getTestDevice();
    const pair = createPingPongTarget(device, {
      label: 'sim',
      format: 'rgba8unorm',
      width: 4,
      height: 4,
      usage: READBACK | GPUTextureUsage.STORAGE_BINDING,
    });
    const output = device.createTexture({
      size: [4, 4],
      format: 'rgba8unorm',
      usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC,
    });

    await expectValid(device, () => {
      const pipeline = device.createComputePipeline({
        layout: 'auto',
        compute: {
          module: device.createShaderModule({
            code: `@group(0) @binding(0) var dst: texture_storage_2d<rgba8unorm, write>;
@compute @workgroup_size(4, 4) fn main(@builtin(local_invocation_id) id: vec3u) {
  textureStore(dst, id.xy, vec4f(0.0, 1.0, 0.0, 1.0));
}`,
          }),
        },
      });
      const group = device.createBindGroup({
        layout: pipeline.getBindGroupLayout(0),
        entries: [{ binding: 0, resource: pair.write.view }],
      });
      const encoder = device.createCommandEncoder();
      const pass = encoder.beginComputePass();
      pass.setPipeline(pipeline);
      pass.setBindGroup(0, group);
      pass.dispatchWorkgroups(1);
      pass.end();
      device.queue.submit([encoder.finish()]);

      pair.swap();
      const copy = device.createCommandEncoder();
      blit(device, copy, pair.read.view, { view: output.createView(), format: 'rgba8unorm' });
      device.queue.submit([copy.finish()]);
    });
    expect([...(await readTexture(device, output)).subarray(0, 4)]).toEqual([0, 255, 0, 255]);
    pair.destroy();
  });
});

import { describe, expect, it } from 'vitest';
import { createShader } from '../src/gpu/shader.ts';
import { expectValid, getTestDevice, readTexture } from './gpu.ts';

describe('test harness', () => {
  it('renders through a real WebGPU implementation', async () => {
    const device = await getTestDevice();
    const texture = device.createTexture({
      size: [4, 4],
      format: 'rgba8unorm',
      usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC,
    });

    await expectValid(device, () => {
      const module = createShader(device, {
        label: 'smoke.wgsl',
        code: `
          #include "tint.wgsl"
          @vertex fn vs(@builtin(vertex_index) i: u32) -> @builtin(position) vec4f {
            let p = vec2f(f32((i << 1u) & 2u), f32(i & 2u));
            return vec4f(p * 2.0 - 1.0, 0.0, 1.0);
          }
          @fragment fn fs() -> @location(0) vec4f { return tint(); }
        `,
        chunks: { 'tint.wgsl': 'fn tint() -> vec4f { return vec4f(1.0, 0.0, 0.0, 1.0); }' },
      });
      const pipeline = device.createRenderPipeline({
        layout: 'auto',
        vertex: { module },
        fragment: { module, targets: [{ format: 'rgba8unorm' }] },
      });
      const encoder = device.createCommandEncoder();
      const pass = encoder.beginRenderPass({
        colorAttachments: [{ view: texture.createView(), loadOp: 'clear', storeOp: 'store' }],
      });
      pass.setPipeline(pipeline);
      pass.draw(3);
      pass.end();
      device.queue.submit([encoder.finish()]);
    });

    const pixels = await readTexture(device, texture);
    expect([...pixels.subarray(0, 4)]).toEqual([255, 0, 0, 255]);
  });
});

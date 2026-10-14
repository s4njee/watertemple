import { createShader } from '../../../gpu/shader.ts';
import code from './strokes.wgsl';

interface Stroke {
  from: readonly [number, number];
  to: readonly [number, number];
  colour: readonly [number, number, number];
  radius: number;
  strength: number;
  born: number;
}
const CAPACITY = 2048;

/** Precise vector strokes retain a short head history, over the diffuse fluid afterimage. */
export function createStrokeLayer(
  device: GPUDevice,
  format: GPUTextureFormat,
  uniforms: GPUBuffer,
  wgsl: string,
) {
  const strokes: Stroke[] = [];
  const data = new Float32Array(CAPACITY * 12);
  const buffer = device.createBuffer({
    label: 'neon stroke instances',
    size: data.byteLength,
    usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
  });
  const module = createShader(device, {
    label: 'neon strokes',
    code: code.code,
    chunks: { 'display_params.wgsl': wgsl },
  });
  const pipeline = device.createRenderPipeline({
    label: 'neon strokes',
    layout: 'auto',
    vertex: {
      module,
      entryPoint: 'vs',
      buffers: [
        {
          arrayStride: 48,
          stepMode: 'instance',
          attributes: [
            { shaderLocation: 0, offset: 0, format: 'float32x4' },
            { shaderLocation: 1, offset: 16, format: 'float32x4' },
            { shaderLocation: 2, offset: 32, format: 'float32x4' },
          ],
        },
      ],
    },
    fragment: {
      module,
      entryPoint: 'fs',
      targets: [
        {
          format,
          blend: {
            color: { srcFactor: 'src-alpha', dstFactor: 'one-minus-src-alpha', operation: 'add' },
            alpha: { srcFactor: 'one', dstFactor: 'one-minus-src-alpha', operation: 'add' },
          },
        },
      ],
    },
  });
  const group = device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: [{ binding: 0, resource: { buffer: uniforms } }],
  });
  return {
    add(stroke: Stroke) {
      if (Math.hypot(stroke.to[0] - stroke.from[0], stroke.to[1] - stroke.from[1]) < 1e-6) return;
      strokes.push(stroke);
      if (strokes.length > CAPACITY) strokes.shift();
    },
    clear() {
      strokes.length = 0;
    },
    render(encoder: GPUCommandEncoder, target: GPUTextureView, time: number) {
      let count = 0;
      for (let i = strokes.length - 1; i >= 0; i--) {
        if (time - strokes[i]!.born > 1.7) strokes.splice(i, 1);
      }
      for (const stroke of strokes) {
        const remaining = Math.max(0, 1 - (time - stroke.born) / 1.7);
        data.set(
          [
            ...stroke.from,
            ...stroke.to,
            ...stroke.colour,
            stroke.strength * remaining * remaining,
            stroke.radius * (0.2 + 0.8 * remaining),
            0,
            0,
            0,
          ],
          count++ * 12,
        );
      }
      if (!count) return;
      device.queue.writeBuffer(buffer, 0, data, 0, count * 12);
      const pass = encoder.beginRenderPass({
        label: 'neon strokes',
        colorAttachments: [{ view: target, loadOp: 'load', storeOp: 'store' }],
      });
      pass.setPipeline(pipeline);
      pass.setBindGroup(0, group);
      pass.setVertexBuffer(0, buffer);
      pass.draw(6, count);
      pass.end();
    },
    destroy() {
      buffer.destroy();
    },
  };
}

import { createShader } from '../shader.ts';
import { createUniformBuffer, defineUniformStruct } from '../uniforms.ts';
import type { PassProfiler } from '../profiler.ts';
const Params = defineUniformStruct('BuoyancyParams', {
  time: 'f32',
  strength: 'f32',
  chop: 'f32',
  count: 'u32',
});
export const BUOY_POINTS = new Float32Array([
  -4, 0, -8, 0, -3, 0, -8, 0, -4, 0, -7, 0, 5, 0, -15, 0, 6, 0, -15, 0, 5, 0, -14, 0,
]);
/** Nonblocking readback: map on the next frame, deferred until its encoder is submitted. One in flight. */
export function createBuoyancyReadback(
  device: GPUDevice,
  profiler: PassProfiler,
  points = BUOY_POINTS,
) {
  const input = device.createBuffer({
    size: points.byteLength,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(input, 0, points);
  const output = device.createBuffer({
    size: points.byteLength,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
  });
  const staging = device.createBuffer({
    size: points.byteLength,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  });
  const uniforms = createUniformBuffer(device, Params, 'buoyancy query');
  const module = createShader(device, {
    label: 'ocean/buoyancy.wgsl',
    code: `#include "ocean-waves.wgsl"\n#include "buoyancy-params.wgsl"\n@group(0) @binding(0) var<uniform> p:BuoyancyParams;\n@group(0) @binding(1) var<storage,read> points:array<vec4f>;\n@group(0) @binding(2) var<storage,read_write> results:array<vec4f>;\n@compute @workgroup_size(32) fn cs(@builtin(global_invocation_id) id:vec3u){if(id.x>=p.count){return;}let w=ocean_world_wave(points[id.x].xz,p.time,p.strength,p.chop);results[id.x]=vec4f(w.position,p.time);}`,
    chunks: { 'buoyancy-params.wgsl': Params.wgsl },
  });
  const pipeline = device.createComputePipeline({
    layout: 'auto',
    compute: { module, entryPoint: 'cs' },
  });
  const bind = device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: uniforms.buffer } },
      { binding: 1, resource: { buffer: input } },
      { binding: 2, resource: { buffer: output } },
    ],
  });
  let state: 'idle' | 'encoded' | 'mapping' = 'idle',
    disposed = false,
    latest: Float32Array | undefined;
  let latencyMs = 0;
  let started = 0;
  const encode = (encoder: GPUCommandEncoder, time: number, strength: number, chop: number) => {
    uniforms.set({ time, strength, chop, count: points.length / 4 });
    uniforms.upload();
    const pass = encoder.beginComputePass({
      label: 'buoyancy queries',
      timestampWrites: profiler.pass('buoyancy queries'),
    });
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bind);
    pass.dispatchWorkgroups(Math.ceil(points.length / 128));
    pass.end();
  };
  return {
    get latest() {
      return latest;
    },
    get latencyMs() {
      return latencyMs;
    },
    output,
    encode,
    run(encoder: GPUCommandEncoder, time: number, strength: number, chop: number) {
      if (state === 'encoded') {
        state = 'mapping';
        void Promise.resolve()
          .then(() => staging.mapAsync(GPUMapMode.READ))
          .then(() => {
            if (disposed) return;
            latest = new Float32Array(staging.getMappedRange().slice(0));
            latencyMs = performance.now() - started;
            staging.unmap();
            state = 'idle';
          })
          .catch(() => {
            if (!disposed) state = 'idle';
          });
      }
      if (state !== 'idle' || disposed) return;
      encode(encoder, time, strength, chop);
      encoder.copyBufferToBuffer(output, 0, staging, 0, points.byteLength);
      state = 'encoded';
      started = performance.now();
    },
    destroy() {
      disposed = true;
      uniforms.destroy();
      input.destroy();
      output.destroy();
      staging.destroy();
    },
  };
}

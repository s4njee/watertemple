import { createShader } from '../shader.ts';
import { createUniformBuffer, defineUniformStruct } from '../uniforms.ts';
import type { PassProfiler } from '../profiler.ts';
import type { ParticleSet } from '../particles/types.ts';
const Params = defineUniformStruct('SprayParams', { delta: 'f32', time: 'f32', count: 'u32' });
export function createWaterfallSpray(device: GPUDevice, profiler: PassProfiler, count = 1536) {
  const positions = device.createBuffer({
    size: count * 16,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_SRC,
  });
  const velocities = device.createBuffer({ size: count * 16, usage: GPUBufferUsage.STORAGE });
  const u = createUniformBuffer(device, Params, 'waterfall spray');
  const module = createShader(device, {
    label: 'ocean/spray.wgsl',
    chunks: { 'spray-params.wgsl': Params.wgsl },
    code: `#include "hash.wgsl"\n#include "spray-params.wgsl"\n@group(0) @binding(0) var<uniform> p:SprayParams;\n@group(0) @binding(1) var<storage,read_write> pos:array<vec4f>;\n@group(0) @binding(2) var<storage,read_write> vel:array<vec4f>;\n@compute @workgroup_size(64) fn cs(@builtin(global_invocation_id) id:vec3u){if(id.x>=p.count){return;}var a=pos[id.x];var v=vel[id.x];let dt=min(p.delta,.033);v.w-=dt;if(v.w<=0 || a.y<-.8){let seed=id.x+u32(p.time*47)*197u;let r=hash_3to3(vec3f(f32(seed),19,47));a=vec4f((r.x-.5)*7,.15,-7.5,1);v=vec4f((r.x-.5)*3,1+r.y*3,r.z*3,1+r.y*2);}v.y-=2.5*dt;v.xyz*=exp(-dt*.35);a.xyz+=v.xyz*dt;a.w=clamp(v.w*.6,0.,1.);pos[id.x]=a;vel[id.x]=v;}`,
  });
  const pipeline = device.createComputePipeline({
    layout: 'auto',
    compute: { module, entryPoint: 'cs' },
  });
  const bind = device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: u.buffer } },
      { binding: 1, resource: { buffer: positions } },
      { binding: 2, resource: { buffer: velocities } },
    ],
  });
  const set: ParticleSet = {
    positions,
    velocities,
    count,
    capacity: count,
    radius: 0.16,
    restVolume: 0.004,
    generation: 0,
  };
  return {
    set,
    run(encoder: GPUCommandEncoder, time: number, delta: number) {
      u.set({ time, delta, count });
      u.upload();
      const pass = encoder.beginComputePass({
        timestampWrites: profiler.pass('waterfall spray integration'),
      });
      pass.setPipeline(pipeline);
      pass.setBindGroup(0, bind);
      pass.dispatchWorkgroups(Math.ceil(count / 64));
      pass.end();
    },
    destroy() {
      u.destroy();
      positions.destroy();
      velocities.destroy();
    },
  };
}

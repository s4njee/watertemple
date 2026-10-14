import { checkShader, formatDiagnostic } from '../src/gpu/shader.ts';
import { getTestDevice } from './gpu.ts';

// --- GPU harness ---------------------------------------------------------------------------------
//
// A test supplies WGSL that defines `fn f(x: vec4f) -> vec4f` (or vec4u -> vec4u) and may `#include`
// library chunks. The harness runs f once per input row in a compute shader and reads the results back,
// so every assertion below is about numbers the real WebGPU implementation produced.

let harnessCounter = 0;

async function dispatch(
  code: string,
  input: Float32Array | Uint32Array,
  scalar: 'f32' | 'u32',
): Promise<ArrayBuffer> {
  const device = await getTestDevice();
  const count = input.length / 4;
  const vec = scalar === 'f32' ? 'vec4f' : 'vec4u';
  const source = `${code}
@group(0) @binding(0) var<storage, read> test_inputs: array<${vec}>;
@group(0) @binding(1) var<storage, read_write> test_outputs: array<${vec}>;
@compute @workgroup_size(64)
fn test_main(@builtin(global_invocation_id) id: vec3u) {
  if (id.x < ${count}u) {
    test_outputs[id.x] = f(test_inputs[id.x]);
  }
}`;
  const { module, diagnostics } = await checkShader(device, {
    label: `harness-${harnessCounter++}.wgsl`,
    code: source,
  });
  const errors = diagnostics.filter((d) => d.severity === 'error');
  if (errors.length > 0) throw new Error(errors.map(formatDiagnostic).join('\n\n'));

  device.pushErrorScope('validation');
  const pipeline = device.createComputePipeline({
    layout: 'auto',
    compute: { module, entryPoint: 'test_main' },
  });
  const inputBuffer = device.createBuffer({
    size: input.byteLength,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(inputBuffer, 0, input);
  const outputBuffer = device.createBuffer({
    size: input.byteLength,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
  });
  const readBuffer = device.createBuffer({
    size: input.byteLength,
    usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
  });
  const bindGroup = device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: inputBuffer } },
      { binding: 1, resource: { buffer: outputBuffer } },
    ],
  });
  const encoder = device.createCommandEncoder();
  const pass = encoder.beginComputePass();
  pass.setPipeline(pipeline);
  pass.setBindGroup(0, bindGroup);
  pass.dispatchWorkgroups(Math.ceil(count / 64));
  pass.end();
  encoder.copyBufferToBuffer(outputBuffer, 0, readBuffer, 0, input.byteLength);
  device.queue.submit([encoder.finish()]);
  const validationError = await device.popErrorScope();
  if (validationError) throw new Error(`WebGPU validation error: ${validationError.message}`);

  await readBuffer.mapAsync(GPUMapMode.READ);
  const result = readBuffer.getMappedRange().slice(0);
  readBuffer.unmap();
  for (const buffer of [inputBuffer, outputBuffer, readBuffer]) buffer.destroy();
  return result;
}

/** Runs `fn f(x: vec4f) -> vec4f` over the rows of `input` (4 floats per row). */
export async function gpuF32(code: string, input: Float32Array): Promise<Float32Array> {
  return new Float32Array(await dispatch(code, input, 'f32'));
}

/** Runs `fn f(x: vec4u) -> vec4u` over the rows of `input` (4 words per row). */
export async function gpuU32(code: string, input: Uint32Array): Promise<Uint32Array> {
  return new Uint32Array(await dispatch(code, input, 'u32'));
}

/** Rows from a list of tuples, padded with zeros. */
export function rowsOf(tuples: number[][]): Float32Array {
  const rows = new Float32Array(tuples.length * 4);
  tuples.forEach((tuple, i) => rows.set(tuple, i * 4));
  return rows;
}

/** Lane `lane` of every row of `data`. */
export function lane(data: Float32Array | Uint32Array, laneIndex: number): number[] {
  const values: number[] = [];
  for (let i = laneIndex; i < data.length; i += 4) values.push(data[i] ?? Number.NaN);
  return values;
}

/** Evaluates each WGSL f32 expression once on the GPU (the expressions can use anything `include` brings in). */
export async function scalars(include: string, expressions: string[]): Promise<number[]> {
  const branches = expressions.map((e, i) => `if (i == ${i}) { r = ${e}; }`).join('\n  ');
  const out = await gpuF32(
    `${include}
     fn f(x: vec4f) -> vec4f {
       let i = i32(x.x);
       var r = 0.0;
       ${branches}
       return vec4f(r, 0.0, 0.0, 0.0);
     }`,
    rowsOf(expressions.map((_, i) => [i])),
  );
  return lane(out, 0).slice(0, expressions.length);
}

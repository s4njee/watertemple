import { describe, expect, it, vi } from 'vitest';
import { createFullscreenPipeline, renderFullscreen } from '../src/gpu/fullscreen.ts';
import {
  createUniformBuffer,
  defineUniformStruct,
  uniformArray,
  type UniformField,
  type UniformStruct,
  type UniformType,
} from '../src/gpu/uniforms.ts';
import { expectValid, getTestDevice, readTexture } from './gpu.ts';

function offsetsOf(struct: UniformStruct): Record<string, number> {
  return Object.fromEntries(struct.layout.map((field) => [field.name, field.offset]));
}

describe('layout', () => {
  it('lays out the brief example: time then resolution', () => {
    const struct = defineUniformStruct('Globals', { time: 'f32', resolution: 'vec2f' });
    expect(offsetsOf(struct)).toEqual({ time: 0, resolution: 8 });
    expect([struct.size, struct.align]).toEqual([16, 8]);
  });

  it('keeps vec3f at 16-byte alignment, so a scalar after it packs into the padding', () => {
    const between = defineUniformStruct('A', { a: 'f32', b: 'vec3f', c: 'f32' });
    expect(offsetsOf(between)).toEqual({ a: 0, b: 16, c: 28 });
    expect(between.size).toBe(32);

    const packed = defineUniformStruct('B', { v: 'vec3f', w: 'f32' });
    expect(offsetsOf(packed)).toEqual({ v: 0, w: 12 });
    expect(packed.size).toBe(16);
  });

  it('pads every mat3x3f column to 16 bytes', () => {
    const struct = defineUniformStruct('C', { m: 'mat3x3f', k: 'f32' });
    expect(offsetsOf(struct)).toEqual({ m: 0, k: 48 });
    expect(struct.size).toBe(64);
  });

  it('aligns a matrix after a vec2 and rounds the struct up to its alignment', () => {
    const struct = defineUniformStruct('D', { a: 'vec2f', m: 'mat4x4f', n: 'u32' });
    expect(offsetsOf(struct)).toEqual({ a: 0, m: 16, n: 80 });
    expect(struct.size).toBe(96);
  });

  it('handles mixed scalar, vec2 and integer vector alignments', () => {
    const struct = defineUniformStruct('E', {
      a: 'f32',
      b: 'vec2f',
      c: 'f32',
      d: 'vec2i',
      e: 'vec3u',
      f: 'i32',
      g: 'vec4i',
    });
    expect(offsetsOf(struct)).toEqual({ a: 0, b: 8, c: 16, d: 24, e: 32, f: 44, g: 48 });
    expect(struct.size).toBe(64);
  });

  it('sizes small structs by their own alignment', () => {
    expect(defineUniformStruct('F', { a: 'f32' }).size).toBe(4);
    expect(defineUniformStruct('G', { a: 'vec3f' }).size).toBe(16);
    expect(defineUniformStruct('H', { a: 'vec2f', b: 'f32' }).size).toBe(16);
  });

  it('strides arrays by 16 (64 for mat4x4f) and aligns them to 16', () => {
    const waves = defineUniformStruct('I', {
      n: 'u32',
      waves: uniformArray('vec4f', 3),
      tail: 'f32',
    });
    expect(offsetsOf(waves)).toEqual({ n: 0, waves: 16, tail: 64 });
    expect(waves.size).toBe(80);

    const points = defineUniformStruct('J', {
      a: 'f32',
      pts: uniformArray('vec3f', 2),
      b: 'vec2f',
    });
    expect(offsetsOf(points)).toEqual({ a: 0, pts: 16, b: 48 });
    expect(points.size).toBe(64);

    const matrices = defineUniformStruct('K', { a: 'f32', m: uniformArray('mat4x4f', 2) });
    expect(offsetsOf(matrices)).toEqual({ a: 0, m: 16 });
    expect(matrices.size).toBe(144);
  });

  it('rejects array elements whose stride is not a multiple of 16', () => {
    // @ts-expect-error f32 elements have stride 4, which the uniform address space forbids
    expect(() => defineUniformStruct('L', { x: uniformArray('f32', 4) })).toThrow(/stride/);
    // @ts-expect-error vec2f elements have stride 8
    expect(() => defineUniformStruct('L', { x: uniformArray('vec2f', 4) })).toThrow(/stride/);
    expect(() => defineUniformStruct('L', { x: uniformArray('vec4f', 0) })).toThrow(RangeError);
  });

  it('rejects names WGSL would not accept', () => {
    expect(() => defineUniformStruct('bad name', { a: 'f32' })).toThrow(TypeError);
    expect(() => defineUniformStruct('Ok', { 'bad-field': 'f32' })).toThrow(TypeError);
    expect(() => defineUniformStruct('Ok', {})).toThrow(RangeError);
    // @ts-expect-error not a WGSL type
    expect(() => defineUniformStruct('Ok', { a: 'bool' })).toThrow(/unsupported/);
  });

  it('generates the matching WGSL declaration', () => {
    const struct = defineUniformStruct('Globals', {
      time: 'f32',
      resolution: 'vec2f',
      waves: uniformArray('vec4f', 2),
      view: 'mat4x4f',
    });
    expect(struct.wgsl).toBe(
      [
        'struct Globals {',
        '  time: f32,',
        '  resolution: vec2f,',
        '  waves: array<vec4f, 2>,',
        '  view: mat4x4f,',
        '}',
        '',
      ].join('\n'),
    );
  });
});

describe('UniformBuffer', () => {
  const Struct = defineUniformStruct('Params', {
    time: 'f32',
    resolution: 'vec2f',
    tint: 'vec3f',
    count: 'u32',
    offset: 'vec2i',
    basis: 'mat3x3f',
    waves: uniformArray('vec4f', 2),
  });

  async function setup() {
    const device = await getTestDevice();
    const writes: ArrayBuffer[] = [];
    const spy = vi.spyOn(device.queue, 'writeBuffer').mockImplementation((...args) => {
      writes.push((args[2] as ArrayBuffer).slice(0));
    });
    const uniforms = createUniformBuffer(device, Struct);
    return { device, uniforms, writes, spy };
  }

  it('allocates a uniform buffer rounded to 16 bytes, labelled', async () => {
    const device = await getTestDevice();
    const odd = defineUniformStruct('Odd', { a: 'f32' });
    const uniforms = await expectValid(device, () => createUniformBuffer(device, odd, 'my params'));
    expect(uniforms.buffer.size).toBe(16);
    expect(uniforms.buffer.label).toBe('my params');
    expect(uniforms.buffer.usage & GPUBufferUsage.UNIFORM).toBeTruthy();
    expect(uniforms.buffer.usage & GPUBufferUsage.COPY_DST).toBeTruthy();
    expect(createUniformBuffer(device, odd).buffer.label).toBe('Odd');
    expect(createUniformBuffer(device, Struct).buffer.size).toBe(Struct.size);
  });

  it('uploads only when a set changed something', async () => {
    const { uniforms, writes, spy } = await setup();
    try {
      expect(uniforms.upload()).toBe(false);
      expect(spy).not.toHaveBeenCalled();

      uniforms.set({ time: 1.5 });
      expect(uniforms.upload()).toBe(true);
      expect(spy).toHaveBeenCalledOnce();
      expect(spy.mock.calls[0]?.[0]).toBe(uniforms.buffer);
      expect(spy.mock.calls[0]?.[1]).toBe(0);
      expect(writes[0]?.byteLength).toBe(uniforms.buffer.size);

      expect(uniforms.upload()).toBe(false);
      uniforms.set({ time: 1.5, resolution: [0, 0] });
      expect(uniforms.upload()).toBe(false);
      uniforms.set({ time: 2 });
      expect(uniforms.upload()).toBe(true);
      expect(spy).toHaveBeenCalledTimes(2);
    } finally {
      spy.mockRestore();
    }
  });

  it('writes each field at its offset and leaves unmentioned fields alone', async () => {
    const { uniforms, writes, spy } = await setup();
    try {
      uniforms.set({ time: 1, resolution: [2, 3], tint: new Float32Array([4, 5, 6]), count: 7 });
      uniforms.set({ offset: [-8, 9] });
      uniforms.upload();
      const data = writes[0];
      if (!data) throw new Error('nothing was uploaded');
      const f32 = new Float32Array(data);
      const i32 = new Int32Array(data);
      const u32 = new Uint32Array(data);
      const at = (name: string) => (Struct.layout.find((f) => f.name === name)?.offset ?? NaN) / 4;
      expect(f32[at('time')]).toBe(1);
      expect(Array.from(f32.slice(at('resolution'), at('resolution') + 2))).toEqual([2, 3]);
      expect(Array.from(f32.slice(at('tint'), at('tint') + 3))).toEqual([4, 5, 6]);
      expect(u32[at('count')]).toBe(7);
      expect(Array.from(i32.slice(at('offset'), at('offset') + 2))).toEqual([-8, 9]);
      // Never set, so still zero.
      expect(Array.from(f32.slice(at('basis'))).every((value) => value === 0)).toBe(true);
    } finally {
      spy.mockRestore();
    }
  });

  it('accepts mat3x3f as 9 tight or 12 padded numbers and leaves the padding at zero', async () => {
    const { uniforms, writes, spy } = await setup();
    try {
      uniforms.set({ basis: [1, 2, 3, 4, 5, 6, 7, 8, 9] });
      uniforms.upload();
      uniforms.set({ basis: [0, 0, 0, 0, 0, 0, 0, 0, 0] });
      uniforms.set({ basis: [1, 2, 3, 99, 4, 5, 6, 99, 7, 8, 9, 99] });
      uniforms.upload();
      const base = Struct.layout.find((f) => f.name === 'basis')!.offset / 4;
      const columns = (data: ArrayBuffer | undefined) =>
        Array.from(new Float32Array(data ?? new ArrayBuffer(0)).slice(base, base + 12));
      const expected = [1, 2, 3, 0, 4, 5, 6, 0, 7, 8, 9, 0];
      expect(columns(writes[0])).toEqual(expected);
      expect(columns(writes[1])).toEqual(expected);
    } finally {
      spy.mockRestore();
    }
  });

  it('zeroes array elements that a shorter set does not cover', async () => {
    const { uniforms, writes, spy } = await setup();
    try {
      uniforms.set({ waves: [1, 2, 3, 4, 5, 6, 7, 8] });
      uniforms.upload();
      uniforms.set({ waves: [9, 9, 9, 9] });
      uniforms.upload();
      const base = Struct.layout.find((f) => f.name === 'waves')!.offset / 4;
      const second = new Float32Array(writes[1] ?? new ArrayBuffer(0)).slice(base, base + 8);
      expect(Array.from(second)).toEqual([9, 9, 9, 9, 0, 0, 0, 0]);
    } finally {
      spy.mockRestore();
    }
  });

  it('rejects unknown fields and values of the wrong shape, as the types do', async () => {
    const { uniforms, spy } = await setup();
    try {
      // @ts-expect-error no such field
      expect(() => uniforms.set({ nope: 1 })).toThrow(/no field "nope"/);
      // @ts-expect-error vec2f needs exactly two numbers
      expect(() => uniforms.set({ resolution: [1] })).toThrow(/takes 2 numbers/);
      // @ts-expect-error f32 takes a number
      expect(() => uniforms.set({ time: [1] })).toThrow(/expected a number/);
      // @ts-expect-error vec2f takes an array
      expect(() => uniforms.set({ resolution: 1 })).toThrow(TypeError);
      expect(() => uniforms.set({ basis: [1, 2, 3] })).toThrow(/mat3x3f takes 9/);
      expect(() => uniforms.set({ waves: [1, 2, 3] })).toThrow(/multiples of 4/);
      expect(() => uniforms.set({ waves: new Array(12).fill(0) })).toThrow(/up to 8/);
      uniforms.set({ resolution: [1, 2], time: 3, count: 4 });
      expect(uniforms.struct).toBe(Struct);
    } finally {
      spy.mockRestore();
    }
  });

  it('destroys its buffer', async () => {
    const { uniforms, spy } = await setup();
    try {
      const destroy = vi.spyOn(uniforms.buffer, 'destroy');
      uniforms.destroy();
      expect(destroy).toHaveBeenCalledOnce();
    } finally {
      spy.mockRestore();
    }
  });
});

// Dawn is the reference: each field is read back through a shader that declares the struct, so a
// layout rule this module gets wrong shows up as a mismatched value, not as a matching bug.

const scalarOf = (type: UniformType): 'f32' | 'i32' | 'u32' =>
  type === 'i32' || type.endsWith('i')
    ? 'i32'
    : type === 'u32' || type.endsWith('u')
      ? 'u32'
      : 'f32';

function componentCount(type: UniformType): number {
  if (type === 'mat3x3f') return 9;
  if (type === 'mat4x4f') return 16;
  const match = /^vec(\d)/.exec(type);
  return match ? Number(match[1]) : 1;
}

/** WGSL expressions, one per output pixel, that reinterpret a field's bits as unsigned ints. */
function readExpressions(access: string, type: UniformField): string[] {
  if (typeof type !== 'string') {
    return Array.from({ length: type.count }, (_, i) =>
      readExpressions(`${access}[${i}]`, type.array),
    ).flat();
  }
  if (type === 'mat3x3f') {
    return [0, 1, 2].map((c) => `vec4u(bitcast<vec3u>(${access}[${c}]), 0u)`);
  }
  if (type === 'mat4x4f') return [0, 1, 2, 3].map((c) => `bitcast<vec4u>(${access}[${c}])`);
  switch (componentCount(type)) {
    case 1:
      return [`vec4u(bitcast<u32>(${access}), 0u, 0u, 0u)`];
    case 2:
      return [`vec4u(bitcast<vec2u>(${access}), 0u, 0u)`];
    case 3:
      return [`vec4u(bitcast<vec3u>(${access}), 0u)`];
    default:
      return [`bitcast<vec4u>(${access})`];
  }
}

function bitsOf(kind: 'f32' | 'i32' | 'u32', numbers: readonly number[]): number[] {
  const Array_ = kind === 'f32' ? Float32Array : kind === 'i32' ? Int32Array : Uint32Array;
  return Array.from(new Uint32Array(new Array_(numbers).buffer));
}

function expectedPixels(type: UniformField, flat: readonly number[]): number[][] {
  if (typeof type !== 'string') {
    const per = componentCount(type.array);
    return Array.from({ length: type.count }, (_, i) =>
      expectedPixels(type.array, flat.slice(i * per, (i + 1) * per)),
    ).flat();
  }
  const width = type === 'mat3x3f' ? 3 : type === 'mat4x4f' ? 4 : componentCount(type);
  const bits = bitsOf(scalarOf(type), flat);
  const pixels: number[][] = [];
  for (let i = 0; i < bits.length; i += width) {
    const pixel = bits.slice(i, i + width);
    while (pixel.length < 4) pixel.push(0);
    pixels.push(pixel);
  }
  return pixels;
}

function mulberry32(seed: number): () => number {
  let state = seed;
  return () => {
    state = (state + 0x6d2b79f5) | 0;
    let t = Math.imul(state ^ (state >>> 15), 1 | state);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function randomNumber(kind: 'f32' | 'i32' | 'u32', rng: () => number): number {
  if (kind === 'f32') return Math.fround((rng() - 0.5) * 2000);
  const unsigned = Math.floor(rng() * 4294967296);
  return kind === 'u32' ? unsigned : unsigned - 2147483648;
}

/** A random value for a field, flat as `set` takes it. */
function randomFlat(type: UniformField, rng: () => number): number[] {
  if (typeof type !== 'string') {
    return Array.from({ length: type.count }, () => randomFlat(type.array, rng)).flat();
  }
  return Array.from({ length: componentCount(type) }, () => randomNumber(scalarOf(type), rng));
}

const asSetValue = (type: UniformField, flat: number[]) =>
  typeof type === 'string' && componentCount(type) === 1 ? flat[0] : flat;

async function probe(
  device: GPUDevice,
  struct: UniformStruct,
  buffer: GPUBuffer,
  declaration: string,
): Promise<number[][]> {
  const expressions = struct.layout.flatMap((field) =>
    readExpressions(`g.${field.name}`, field.type),
  );
  const cases = expressions.map((expression, i) => `    case ${i}u: { return ${expression}; }`);
  const code = `
#include "fullscreen.wgsl"
#include "uniforms.wgsl"
@group(0) @binding(0) var<uniform> g: ${struct.name};
fn probe(i: u32) -> vec4u {
  switch i {
${cases.join('\n')}
    default: { return vec4u(0xdeadbeefu); }
  }
}
@fragment fn fs(in: FullscreenVarying) -> @location(0) vec4u { return probe(u32(in.position.x)); }
`;
  const target = device.createTexture({
    size: [expressions.length, 1],
    format: 'rgba32uint',
    usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC,
  });
  await expectValid(device, () => {
    const pipeline = createFullscreenPipeline(device, {
      fragment: {
        label: `test/probe-${struct.name}.wgsl`,
        code,
        chunks: { 'uniforms.wgsl': declaration },
      },
      format: 'rgba32uint',
    });
    const group = device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [{ binding: 0, resource: { buffer } }],
    });
    const encoder = device.createCommandEncoder();
    renderFullscreen(encoder, target.createView(), pipeline, [group]);
    device.queue.submit([encoder.finish()]);
  });
  const words = new Uint32Array((await readTexture(device, target, 16)).buffer);
  target.destroy();
  return Array.from({ length: expressions.length }, (_, i) =>
    Array.from(words.subarray(i * 4, i * 4 + 4)),
  );
}

async function roundTrip(
  device: GPUDevice,
  struct: UniformStruct,
  rng: () => number,
  declaration = struct.wgsl,
): Promise<void> {
  const uniforms = createUniformBuffer(device, struct);
  const values: Record<string, number | number[]> = {};
  const expected: number[][] = [];
  for (const field of struct.layout) {
    const flat = randomFlat(field.type, rng);
    const value = asSetValue(field.type, flat);
    if (value !== undefined) values[field.name] = value;
    expected.push(...expectedPixels(field.type, flat));
  }
  uniforms.set(values);
  uniforms.upload();
  expect(await probe(device, struct, uniforms.buffer, declaration)).toEqual(expected);
  uniforms.destroy();
}

describe('layout against Dawn', () => {
  const cases: Record<string, UniformStruct> = {
    'f32, vec3f, f32': defineUniformStruct('Tricky', { a: 'f32', b: 'vec3f', c: 'f32' }),
    'vec3f, f32 (packs into the padding)': defineUniformStruct('Tricky', { v: 'vec3f', w: 'f32' }),
    'mat3x3f then f32': defineUniformStruct('Tricky', { m: 'mat3x3f', k: 'f32' }),
    'vec2f, mat4x4f, u32': defineUniformStruct('Tricky', { a: 'vec2f', m: 'mat4x4f', n: 'u32' }),
    'time and resolution': defineUniformStruct('Tricky', { time: 'f32', resolution: 'vec2f' }),
    'mixed scalars and integer vectors': defineUniformStruct('Tricky', {
      a: 'f32',
      b: 'vec2f',
      c: 'f32',
      d: 'vec2i',
      e: 'vec3u',
      f: 'i32',
      g: 'vec4i',
      h: 'u32',
    }),
    'vec4f array between scalars': defineUniformStruct('Tricky', {
      n: 'u32',
      waves: uniformArray('vec4f', 3),
      tail: 'f32',
    }),
    'vec3f array then vec2f': defineUniformStruct('Tricky', {
      a: 'f32',
      pts: uniformArray('vec3f', 3),
      b: 'vec2f',
    }),
    'mat4x4f array then vec3u': defineUniformStruct('Tricky', {
      a: 'f32',
      views: uniformArray('mat4x4f', 2),
      b: 'vec3u',
    }),
  };

  for (const [name, struct] of Object.entries(cases)) {
    it(`round-trips ${name}`, async () => {
      const device = await getTestDevice();
      await roundTrip(device, struct, mulberry32(struct.size));
    });
  }

  it('also works with a hand-written struct of the same order and types', async () => {
    const device = await getTestDevice();
    const rng = mulberry32(7);
    const handWritten: [UniformStruct, string][] = [
      [cases['f32, vec3f, f32']!, 'struct Tricky { a: f32, b: vec3<f32>, c: f32 }'],
      [cases['vec3f, f32 (packs into the padding)']!, 'struct Tricky { v: vec3f, w: f32, }'],
      [cases['mat3x3f then f32']!, 'struct Tricky {\n  m: mat3x3<f32>,\n  k: f32,\n}'],
      [cases['vec2f, mat4x4f, u32']!, 'struct Tricky { a: vec2<f32>, m: mat4x4<f32>, n: u32 }'],
      [
        cases['vec4f array between scalars']!,
        'struct Tricky { n: u32, waves: array<vec4<f32>, 3>, tail: f32 }',
      ],
    ];
    for (const [struct, declaration] of handWritten) {
      await roundTrip(device, struct, rng, declaration);
    }
  });

  it('would catch a wrong layout: tightly packed bytes do not read back', async () => {
    const device = await getTestDevice();
    const struct = cases['f32, vec3f, f32']!;
    const buffer = device.createBuffer({
      size: struct.size,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    // a, b.xyz, c packed back to back, as a C struct of floats would be.
    device.queue.writeBuffer(buffer, 0, new Float32Array([1, 2, 3, 4, 5, 0, 0, 0]));
    const pixels = await probe(device, struct, buffer, struct.wgsl);
    const asFloats = pixels.map((pixel) =>
      Array.from(new Float32Array(new Uint32Array(pixel).buffer)),
    );
    expect(asFloats[0]?.[0]).toBe(1);
    expect(asFloats[1]?.slice(0, 3)).not.toEqual([2, 3, 4]);
    expect(asFloats[2]?.[0]).not.toBe(5);
  });

  it('round-trips randomly generated structs', async () => {
    const device = await getTestDevice();
    const rng = mulberry32(20240607);
    const scalars: UniformType[] = [
      'f32',
      'i32',
      'u32',
      'vec2f',
      'vec3f',
      'vec4f',
      'vec2i',
      'vec3i',
      'vec4i',
      'vec2u',
      'vec3u',
      'vec4u',
      'mat3x3f',
      'mat4x4f',
    ];
    const elements = ['vec3f', 'vec4f', 'vec3i', 'vec4i', 'vec3u', 'vec4u', 'mat4x4f'] as const;
    const pick = <T>(items: readonly T[]): T => items[Math.floor(rng() * items.length)]!;

    for (let n = 0; n < 40; n++) {
      const fields: Record<string, UniformField> = {};
      const fieldCount = 1 + Math.floor(rng() * 7);
      for (let i = 0; i < fieldCount; i++) {
        fields[`f${i}`] =
          rng() < 0.2 ? uniformArray(pick(elements), 1 + Math.floor(rng() * 3)) : pick(scalars);
      }
      const struct = defineUniformStruct(`Fuzz${n}`, fields);
      await roundTrip(device, struct, rng);
    }
  });
});

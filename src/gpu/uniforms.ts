export type UniformScalarType = 'f32' | 'i32' | 'u32';
export type UniformVectorType =
  'vec2f' | 'vec3f' | 'vec4f' | 'vec2i' | 'vec3i' | 'vec4i' | 'vec2u' | 'vec3u' | 'vec4u';
export type UniformMatrixType = 'mat3x3f' | 'mat4x4f';
export type UniformType = UniformScalarType | UniformVectorType | UniformMatrixType;

/**
 * Array elements need a stride that is a multiple of 16 in the uniform address space, which rules
 * out f32, i32, u32 and vec2. Pack those into a vec4 (or a vec3 plus a scalar) instead.
 */
export type UniformArrayElement =
  'vec3f' | 'vec4f' | 'vec3i' | 'vec4i' | 'vec3u' | 'vec4u' | 'mat4x4f';

export interface UniformArray<E extends UniformArrayElement = UniformArrayElement> {
  readonly array: E;
  readonly count: number;
}

export function uniformArray<const E extends UniformArrayElement>(
  element: E,
  count: number,
): UniformArray<E> {
  return { array: element, count };
}

export type UniformField = UniformType | UniformArray;

/** Declaration order is memory order. WGSL has no bool in uniform buffers; use u32. */
export type UniformFields = Readonly<Record<string, UniformField>>;

type Packed = Float32Array | Int32Array | Uint32Array;
type Vec2 = readonly [number, number] | Packed;
type Vec3 = readonly [number, number, number] | Packed;
type Vec4 = readonly [number, number, number, number] | Packed;

/**
 * What `set` accepts for a field. Matrices are column-major: 9 numbers for mat3x3f, or the 12
 * padded floats wgpu-matrix produces; 16 for mat4x4f. Arrays are flat, element after element, and
 * may be shorter than `count` (the rest is zeroed).
 */
export type UniformValue<T extends UniformField> = T extends UniformArray
  ? ArrayLike<number>
  : T extends UniformScalarType
    ? number
    : T extends 'vec2f' | 'vec2i' | 'vec2u'
      ? Vec2
      : T extends 'vec3f' | 'vec3i' | 'vec3u'
        ? Vec3
        : T extends 'vec4f' | 'vec4i' | 'vec4u'
          ? Vec4
          : ArrayLike<number>;

export type UniformValues<F extends UniformFields> = {
  -readonly [K in keyof F]?: UniformValue<F[K]>;
};

interface TypeInfo {
  scalar: UniformScalarType;
  /** Numbers in one value as the author supplies it. */
  length: number;
  align: number;
  size: number;
  /** Matrices only. Columns start 16 bytes apart whether or not the last row is padding. */
  columns?: number;
  rows?: number;
}

// WGSL spec, "Alignment and Size": vec3 is align 16 / size 12, so a following scalar packs into
// its padding; matCxR is C columns of vecR, each padded to the column vector's alignment.
const TYPES: Record<UniformType, TypeInfo> = {
  f32: { scalar: 'f32', length: 1, align: 4, size: 4 },
  i32: { scalar: 'i32', length: 1, align: 4, size: 4 },
  u32: { scalar: 'u32', length: 1, align: 4, size: 4 },
  vec2f: { scalar: 'f32', length: 2, align: 8, size: 8 },
  vec3f: { scalar: 'f32', length: 3, align: 16, size: 12 },
  vec4f: { scalar: 'f32', length: 4, align: 16, size: 16 },
  vec2i: { scalar: 'i32', length: 2, align: 8, size: 8 },
  vec3i: { scalar: 'i32', length: 3, align: 16, size: 12 },
  vec4i: { scalar: 'i32', length: 4, align: 16, size: 16 },
  vec2u: { scalar: 'u32', length: 2, align: 8, size: 8 },
  vec3u: { scalar: 'u32', length: 3, align: 16, size: 12 },
  vec4u: { scalar: 'u32', length: 4, align: 16, size: 16 },
  mat3x3f: { scalar: 'f32', length: 9, align: 16, size: 48, columns: 3, rows: 3 },
  mat4x4f: { scalar: 'f32', length: 16, align: 16, size: 64, columns: 4, rows: 4 },
};

const ARRAY_ELEMENTS = new Set<string>([
  'vec3f',
  'vec4f',
  'vec3i',
  'vec4i',
  'vec3u',
  'vec4u',
  'mat4x4f',
]);

const IDENTIFIER = /^[A-Za-z_][A-Za-z0-9_]*$/;

const roundUp = (multiple: number, value: number) => Math.ceil(value / multiple) * multiple;

export interface UniformFieldLayout {
  readonly name: string;
  readonly type: UniformField;
  /** Bytes from the start of the struct. */
  readonly offset: number;
  readonly size: number;
}

export interface UniformStruct<F extends UniformFields = UniformFields> {
  readonly name: string;
  readonly layout: readonly UniformFieldLayout[];
  /** Struct size in bytes, rounded up to `align` as WGSL requires. */
  readonly size: number;
  readonly align: number;
  /**
   * The matching WGSL declaration. Inject it as an include chunk
   * (`chunks: { 'globals.wgsl': Globals.wgsl }`) or copy the same fields, in the same order and
   * with the same types, into a hand-written struct: the layout depends only on those.
   */
  readonly wgsl: string;
  /** The declaration this was built from; carries the field types that `set` checks against. */
  readonly fields: F;
}

function describe(name: string, type: UniformField): { align: number; size: number; wgsl: string } {
  if (typeof type === 'string') {
    if (!Object.hasOwn(TYPES, type)) throw new TypeError(`${name}: unsupported type "${type}"`);
    const info = TYPES[type];
    return { align: info.align, size: info.size, wgsl: type };
  }
  if (!ARRAY_ELEMENTS.has(type.array)) {
    throw new TypeError(
      `${name}: array<${type.array}> has a stride that is not a multiple of 16, which the uniform address space requires; pack the data into vec4f`,
    );
  }
  if (!Number.isInteger(type.count) || type.count < 1) {
    throw new RangeError(`${name}: array count must be a positive integer, got ${type.count}`);
  }
  const element = TYPES[type.array];
  return {
    align: 16,
    size: type.count * roundUp(element.align, element.size),
    wgsl: `array<${type.array}, ${type.count}>`,
  };
}

export function defineUniformStruct<const F extends UniformFields>(
  name: string,
  fields: F,
): UniformStruct<F> {
  if (!IDENTIFIER.test(name)) throw new TypeError(`"${name}" is not a valid WGSL identifier`);
  const entries = Object.entries(fields);
  if (entries.length === 0) throw new RangeError(`${name}: a uniform struct needs a field`);

  const layout: UniformFieldLayout[] = [];
  const lines: string[] = [];
  let end = 0;
  let align = 1;
  for (const [fieldName, type] of entries) {
    if (!IDENTIFIER.test(fieldName)) {
      throw new TypeError(`${name}: "${fieldName}" is not a valid WGSL identifier`);
    }
    const described = describe(`${name}.${fieldName}`, type);
    const offset = roundUp(described.align, end);
    layout.push({ name: fieldName, type, offset, size: described.size });
    lines.push(`  ${fieldName}: ${described.wgsl},`);
    end = offset + described.size;
    align = Math.max(align, described.align);
  }

  return {
    name,
    layout,
    size: roundUp(align, end),
    align,
    wgsl: `struct ${name} {\n${lines.join('\n')}\n}\n`,
    fields,
  };
}

interface Views {
  f32: Float32Array;
  i32: Int32Array;
  u32: Uint32Array;
}

const isArrayLike = (value: unknown): value is ArrayLike<number> =>
  typeof value === 'object' && value !== null && 'length' in value;

const ZEROS = new Float32Array(16);

/** Stores one number; returns whether any bit of the destination changed. */
function put(
  views: Views,
  scalar: UniformScalarType,
  index: number,
  value: number | undefined,
): boolean {
  const before = views.u32[index];
  views[scalar][index] = value ?? 0;
  return views.u32[index] !== before;
}

/** Writes one value of `info` at `byteOffset`; returns whether any bit changed. */
function writeValue(
  views: Views,
  info: TypeInfo,
  byteOffset: number,
  source: ArrayLike<number>,
  start: number,
): boolean {
  const base = byteOffset >> 2;
  let changed = false;
  if (info.columns !== undefined && info.rows !== undefined) {
    // wgpu-matrix pads mat3 columns to 4 floats; accept that layout as well as the tight one.
    const sourceRows = source.length === info.columns * 4 ? 4 : info.rows;
    for (let column = 0; column < info.columns; column++) {
      for (let row = 0; row < info.rows; row++) {
        const value = source[start + column * sourceRows + row];
        changed = put(views, info.scalar, base + column * 4 + row, value) || changed;
      }
    }
  } else {
    for (let i = 0; i < info.length; i++) {
      changed = put(views, info.scalar, base + i, source[start + i]) || changed;
    }
  }
  return changed;
}

function writeField(
  views: Views,
  structName: string,
  field: UniformFieldLayout,
  value: unknown,
): boolean {
  const where = `${structName}.${field.name}`;
  const { type } = field;

  if (typeof type === 'string') {
    const info = TYPES[type];
    if (info.length === 1) {
      if (typeof value !== 'number') throw new TypeError(`${where}: expected a number`);
      return put(views, info.scalar, field.offset >> 2, value);
    }
    if (!isArrayLike(value))
      throw new TypeError(`${where}: expected ${type} as an array of numbers`);
    const padded = info.columns !== undefined && value.length === info.columns * 4;
    if (value.length !== info.length && !padded) {
      throw new RangeError(`${where}: ${type} takes ${info.length} numbers, got ${value.length}`);
    }
    return writeValue(views, info, field.offset, value, 0);
  }

  const info = TYPES[type.array];
  const stride = roundUp(info.align, info.size);
  if (!isArrayLike(value)) throw new TypeError(`${where}: expected a flat array of numbers`);
  const given = value.length / info.length;
  if (!Number.isInteger(given) || given > type.count) {
    throw new RangeError(
      `${where}: array<${type.array}, ${type.count}> takes up to ${type.count * info.length} numbers in multiples of ${info.length}, got ${value.length}`,
    );
  }
  let changed = false;
  for (let element = 0; element < type.count; element++) {
    const present = element < given;
    const offset = field.offset + element * stride;
    const wrote = present
      ? writeValue(views, info, offset, value, element * info.length)
      : writeValue(views, info, offset, ZEROS, 0);
    changed = wrote || changed;
  }
  return changed;
}

export interface UniformBuffer<F extends UniformFields = UniformFields> {
  readonly struct: UniformStruct<F>;
  readonly buffer: GPUBuffer;
  /** Writes into the CPU copy only; call `upload()` to send it. Unmentioned fields keep their value. */
  set(values: UniformValues<F>): void;
  /**
   * Sends the CPU copy with `queue.writeBuffer`, but only if a `set` changed a bit since the last
   * upload. Returns whether it wrote. The write lands in queue order, before any command buffer
   * submitted after this call, so two uploads before one submit leave only the second visible.
   */
  upload(): boolean;
  destroy(): void;
}

export function createUniformBuffer<F extends UniformFields>(
  device: GPUDevice,
  struct: UniformStruct<F>,
  label = struct.name,
): UniformBuffer<F> {
  const data = new ArrayBuffer(roundUp(16, struct.size));
  const views: Views = {
    f32: new Float32Array(data),
    i32: new Int32Array(data),
    u32: new Uint32Array(data),
  };
  const fieldsByName = new Map(struct.layout.map((field) => [field.name, field]));
  const buffer = device.createBuffer({
    label,
    size: data.byteLength,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });
  let dirty = false;

  return {
    struct,
    buffer,
    set(values) {
      for (const [name, value] of Object.entries(values)) {
        const field = fieldsByName.get(name);
        if (!field) throw new TypeError(`${struct.name} has no field "${name}"`);
        if (value === undefined) continue;
        if (writeField(views, struct.name, field, value)) dirty = true;
      }
    },
    upload() {
      if (!dirty) return false;
      device.queue.writeBuffer(buffer, 0, data);
      dirty = false;
      return true;
    },
    destroy() {
      buffer.destroy();
    },
  };
}

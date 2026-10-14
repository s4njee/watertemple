/*
 * Live texture memory, estimated by wrapping `device.createTexture` and `texture.destroy` on one
 * device. WebGPU has no API for this, and wrapping sees every texture made on the device by any
 * effect without asking them to cooperate. It is an estimate: drivers pad, tile, and compress; the
 * canvas's own swap-chain textures never pass through createTexture; and a texture dropped
 * without destroy() is only subtracted once the garbage collector reports it.
 */

interface FormatLayout {
  /** Texel block size in pixels (1x1 for uncompressed formats). */
  blockWidth: number;
  blockHeight: number;
  bytesPerBlock: number;
}

const px = (bytes: number): FormatLayout => ({
  blockWidth: 1,
  blockHeight: 1,
  bytesPerBlock: bytes,
});

const FORMATS = new Map<string, FormatLayout>();
function add(names: string, layout: FormatLayout) {
  for (const name of names.split(' ')) FORMATS.set(name, layout);
}
add('r8unorm r8snorm r8uint r8sint stencil8', px(1));
add('rg8unorm rg8snorm rg8uint rg8sint r16unorm r16snorm r16uint r16sint r16float', px(2));
add('depth16unorm', px(2));
add('rgba8unorm rgba8unorm-srgb rgba8snorm rgba8uint rgba8sint bgra8unorm bgra8unorm-srgb', px(4));
add('rg16uint rg16sint rg16float rg16unorm rg16snorm r32uint r32sint r32float', px(4));
add('rgb10a2uint rgb10a2unorm rg11b10ufloat rgb9e5ufloat', px(4));
// Implementation-defined; 4 is the usual packed layout and 8 is what depth32float-stencil8 pads to.
add('depth24plus depth24plus-stencil8 depth32float', px(4));
add('depth32float-stencil8', px(8));
add('rgba16uint rgba16sint rgba16float rgba16unorm rgba16snorm rg32uint rg32sint rg32float', px(8));
add('rgba32uint rgba32sint rgba32float', px(16));

const block4 = (bytes: number): FormatLayout => ({
  blockWidth: 4,
  blockHeight: 4,
  bytesPerBlock: bytes,
});
add('bc1-rgba-unorm bc1-rgba-unorm-srgb bc4-r-unorm bc4-r-snorm', block4(8));
add(
  'bc2-rgba-unorm bc2-rgba-unorm-srgb bc3-rgba-unorm bc3-rgba-unorm-srgb bc5-rg-unorm bc5-rg-snorm ' +
    'bc6h-rgb-ufloat bc6h-rgb-float bc7-rgba-unorm bc7-rgba-unorm-srgb',
  block4(16),
);
add(
  'etc2-rgb8unorm etc2-rgb8unorm-srgb etc2-rgb8a1unorm etc2-rgb8a1unorm-srgb ' +
    'eac-r11unorm eac-r11snorm',
  block4(8),
);
add('etc2-rgba8unorm etc2-rgba8unorm-srgb eac-rg11unorm eac-rg11snorm', block4(16));

function layoutOf(format: string): FormatLayout | undefined {
  const known = FORMATS.get(format);
  if (known) return known;
  const astc = /^astc-(\d+)x(\d+)-/.exec(format);
  if (astc) {
    return { blockWidth: Number(astc[1]), blockHeight: Number(astc[2]), bytesPerBlock: 16 };
  }
  return undefined;
}

export interface TextureSizeInput {
  size: GPUExtent3DStrict | Iterable<number>;
  format: string;
  mipLevelCount?: number;
  sampleCount?: number;
  dimension?: GPUTextureDimension;
}

export interface TextureEstimate {
  bytes: number;
  /** False when the format is not in the table and was counted as 4 bytes per pixel. */
  known: boolean;
}

function extent(size: TextureSizeInput['size']): [number, number, number] {
  if (Symbol.iterator in Object(size)) {
    const [width = 1, height = 1, depth = 1] = size as Iterable<number>;
    return [width, height, depth];
  }
  const dict = size as GPUExtent3DDictStrict;
  return [dict.width, dict.height ?? 1, dict.depthOrArrayLayers ?? 1];
}

/** Bytes a texture occupies across all its mip levels, layers and samples. */
export function estimateTextureBytes(input: TextureSizeInput): TextureEstimate {
  const [width, height, depth] = extent(input.size);
  const dimension = input.dimension ?? '2d';
  const layout = layoutOf(input.format);
  const { blockWidth, blockHeight, bytesPerBlock } = layout ?? px(4);
  const maxLevels =
    Math.floor(
      Math.log2(Math.max(width, dimension === '1d' ? 1 : height, dimension === '3d' ? depth : 1)),
    ) + 1;
  const levels = Math.min(Math.max(1, input.mipLevelCount ?? 1), maxLevels);

  let bytes = 0;
  for (let level = 0; level < levels; level++) {
    const w = Math.max(1, width >> level);
    const h = Math.max(1, height >> level);
    // Only 3D textures shrink in depth; array layers and cube faces keep every mip.
    const d = dimension === '3d' ? Math.max(1, depth >> level) : depth;
    bytes += Math.ceil(w / blockWidth) * Math.ceil(h / blockHeight) * d * bytesPerBlock;
  }
  return { bytes: bytes * Math.max(1, input.sampleCount ?? 1), known: layout !== undefined };
}

export interface TextureMemory {
  /** Estimated bytes of live textures created through the device. */
  bytes: number;
  textures: number;
  /** Formats missing from the size table, each counted as 4 bytes per pixel. */
  unknownFormats: readonly string[];
}

interface Tracker {
  bytes: number;
  textures: number;
  unknownFormats: Set<string>;
  removed: boolean;
  restore(): void;
}

interface TextureRecord {
  bytes: number;
  live: boolean;
}

const trackers = new WeakMap<GPUDevice, Tracker>();

/**
 * Starts counting textures created on `device` from now on; textures that already exist are not
 * seen, so call it before anything allocates. Safe to call repeatedly. A replacement device after
 * a loss is a different object and needs its own call.
 */
export function trackTextureMemory(device: GPUDevice): void {
  if (trackers.has(device)) return;

  const tracker: Tracker = {
    bytes: 0,
    textures: 0,
    unknownFormats: new Set(),
    removed: false,
    restore() {},
  };
  const release = (record: TextureRecord) => {
    if (!record.live) return;
    record.live = false;
    if (tracker.removed) return;
    tracker.bytes -= record.bytes;
    tracker.textures--;
  };
  // Textures nobody destroyed disappear when the collector frees them. The held value is a plain
  // record, so it does not keep the texture alive.
  const collected =
    typeof FinalizationRegistry === 'undefined'
      ? undefined
      : new FinalizationRegistry<TextureRecord>(release);

  const hadOwn = Object.prototype.hasOwnProperty.call(device, 'createTexture');
  const original = device.createTexture;
  device.createTexture = function (this: GPUDevice, descriptor: GPUTextureDescriptor) {
    const texture = original.call(this, descriptor);
    if (tracker.removed) return texture;
    const estimate = estimateTextureBytes(descriptor);
    if (!estimate.known) tracker.unknownFormats.add(descriptor.format);
    const record: TextureRecord = { bytes: estimate.bytes, live: true };
    tracker.bytes += record.bytes;
    tracker.textures++;
    const destroy = texture.destroy;
    texture.destroy = function (this: GPUTexture) {
      release(record);
      return destroy.call(this);
    };
    collected?.register(texture, record);
    return texture;
  };
  tracker.restore = () => {
    if (hadOwn) device.createTexture = original;
    else delete (device as { createTexture?: unknown }).createTexture;
  };
  trackers.set(device, tracker);
}

/** Undefined when the device is not tracked. */
export function getTextureMemory(device: GPUDevice): TextureMemory | undefined {
  const tracker = trackers.get(device);
  if (!tracker) return undefined;
  return {
    bytes: tracker.bytes,
    textures: tracker.textures,
    unknownFormats: [...tracker.unknownFormats],
  };
}

/** Stops counting and restores the device's own `createTexture`. Textures already wrapped keep working. */
export function untrackTextureMemory(device: GPUDevice): void {
  const tracker = trackers.get(device);
  if (!tracker) return;
  tracker.removed = true;
  tracker.restore();
  trackers.delete(device);
}

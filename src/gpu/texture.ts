import { generateMipmaps, mipLevelCount } from './mipmap.ts';
import { assertLinearRenderable } from './passes/common.ts';

export interface TextureOptions {
  label?: string;
  /** Allocate a full mip chain and fill it. Needs a filterable, renderable format. */
  mipmaps?: boolean;
  /** Added to the usage the helper always sets, e.g. `COPY_SRC` for readback. */
  extraUsage?: GPUTextureUsageFlags;
}

export interface DataTextureOptions extends TextureOptions {
  width: number;
  height: number;
  /**
   * Tightly packed rows, exactly `width * height` texels. For half-float formats pass the raw
   * 16-bit patterns in a Uint16Array.
   */
  data: ArrayBufferView;
  /** Defaults to `rgba8unorm`. */
  format?: GPUTextureFormat;
}

export interface SourceTextureOptions extends TextureOptions {
  /**
   * `true` (default) stores `rgba8unorm-srgb`, so sampling returns linear light: right for photos
   * and anything shown as colour. `false` stores `rgba8unorm`: right for normal maps, noise,
   * flow maps and masks, which are data rather than colour.
   */
  srgb?: boolean;
  /** Texture coordinates start at the top-left, so images are upright without this. */
  flipY?: boolean;
  /** Store RGB multiplied by alpha. */
  premultipliedAlpha?: boolean;
}

const BYTES_PER_TEXEL: Partial<Record<GPUTextureFormat, number>> = {
  r8unorm: 1,
  r8snorm: 1,
  r8uint: 1,
  r8sint: 1,
  rg8unorm: 2,
  rg8snorm: 2,
  rg8uint: 2,
  rg8sint: 2,
  r16float: 2,
  r16uint: 2,
  r16sint: 2,
  rgba8unorm: 4,
  'rgba8unorm-srgb': 4,
  rgba8snorm: 4,
  rgba8uint: 4,
  rgba8sint: 4,
  bgra8unorm: 4,
  'bgra8unorm-srgb': 4,
  rgb10a2unorm: 4,
  rg11b10ufloat: 4,
  rg16float: 4,
  rg16uint: 4,
  rg16sint: 4,
  r32float: 4,
  r32uint: 4,
  r32sint: 4,
  rgba16float: 8,
  rgba16uint: 8,
  rgba16sint: 8,
  rg32float: 8,
  rg32uint: 8,
  rg32sint: 8,
  rgba32float: 16,
  rgba32uint: 16,
  rgba32sint: 16,
};

function assertFits(device: GPUDevice, width: number, height: number, what: string): void {
  const max = device.limits.maxTextureDimension2D;
  if (!(width >= 1 && height >= 1) || !Number.isInteger(width) || !Number.isInteger(height)) {
    throw new RangeError(`${what}: ${width}x${height} is not a valid texture size`);
  }
  if (width > max || height > max) {
    throw new RangeError(
      `${what}: ${width}x${height} exceeds this device's maxTextureDimension2D (${max}); downscale it first`,
    );
  }
}

export function createTextureFromData(device: GPUDevice, options: DataTextureOptions): GPUTexture {
  const { width, height, data, format = 'rgba8unorm' } = options;
  const label = options.label ?? 'data texture';
  const bytesPerTexel = BYTES_PER_TEXEL[format];
  if (bytesPerTexel === undefined) {
    throw new Error(
      `createTextureFromData: ${format} is not supported for uploads from typed arrays`,
    );
  }
  assertFits(device, width, height, 'createTextureFromData');
  if (data.byteLength !== width * height * bytesPerTexel) {
    throw new RangeError(
      `createTextureFromData: ${width}x${height} ${format} needs ${width * height * bytesPerTexel} bytes, got ${data.byteLength}`,
    );
  }
  if (options.mipmaps) assertLinearRenderable(device, format, 'createTextureFromData');

  const levels = options.mipmaps ? mipLevelCount(width, height) : 1;
  const texture = device.createTexture({
    label,
    size: [width, height],
    format,
    mipLevelCount: levels,
    usage:
      GPUTextureUsage.TEXTURE_BINDING |
      GPUTextureUsage.COPY_DST |
      (levels > 1 ? GPUTextureUsage.RENDER_ATTACHMENT : 0) |
      (options.extraUsage ?? 0),
  });
  device.queue.writeTexture(
    { texture },
    data,
    { bytesPerRow: width * bytesPerTexel, rowsPerImage: height },
    [width, height],
  );
  if (levels > 1) generateMipmaps(device, texture);
  return texture;
}

function sourceSize(source: GPUCopyExternalImageSource): { width: number; height: number } {
  if ('videoWidth' in source) return { width: source.videoWidth, height: source.videoHeight };
  if ('naturalWidth' in source) return { width: source.naturalWidth, height: source.naturalHeight };
  if ('displayWidth' in source) return { width: source.displayWidth, height: source.displayHeight };
  return { width: source.width, height: source.height };
}

function copySource(
  device: GPUDevice,
  texture: GPUTexture,
  source: GPUCopyExternalImageSource,
  options: SourceTextureOptions,
  encoder?: GPUCommandEncoder,
): void {
  device.queue.copyExternalImageToTexture(
    { source, flipY: options.flipY ?? false },
    { texture, premultipliedAlpha: options.premultipliedAlpha ?? false },
    [texture.width, texture.height],
  );
  if (texture.mipLevelCount > 1) generateMipmaps(device, texture, encoder);
}

/**
 * Uploads an image, canvas, bitmap or video frame as an `rgba8unorm[-srgb]` texture sized to the
 * source. Image elements must already be decoded (`await img.decode()`).
 */
export function createTextureFromSource(
  device: GPUDevice,
  source: GPUCopyExternalImageSource,
  options: SourceTextureOptions = {},
): GPUTexture {
  const { width, height } = sourceSize(source);
  if (width === 0 || height === 0) {
    throw new Error(
      'createTextureFromSource: the source has no pixels yet; wait for the image to decode or the video metadata to load',
    );
  }
  assertFits(device, width, height, 'createTextureFromSource');
  const format = (options.srgb ?? true) ? 'rgba8unorm-srgb' : 'rgba8unorm';
  const levels = options.mipmaps ? mipLevelCount(width, height) : 1;

  const texture = device.createTexture({
    label: options.label ?? 'source texture',
    size: [width, height],
    format,
    mipLevelCount: levels,
    // copyExternalImageToTexture requires COPY_DST and RENDER_ATTACHMENT on its destination.
    usage:
      GPUTextureUsage.TEXTURE_BINDING |
      GPUTextureUsage.COPY_DST |
      GPUTextureUsage.RENDER_ATTACHMENT |
      (options.extraUsage ?? 0),
  });
  copySource(device, texture, source, options);
  return texture;
}

/**
 * Re-uploads `source` into a texture made by `createTextureFromSource`, e.g. once per frame for a
 * video, and refreshes its mips. The source must still be the size the texture was made for.
 *
 * The copy itself is a queue operation that happens at the moment of the call, before any command
 * buffer submitted later. Pass the frame's `encoder` and the mip passes are recorded into it
 * instead of a command buffer of their own, so a per-frame video update costs no extra submit.
 */
export function updateTextureFromSource(
  device: GPUDevice,
  texture: GPUTexture,
  source: GPUCopyExternalImageSource,
  options: Pick<SourceTextureOptions, 'flipY' | 'premultipliedAlpha'> & {
    encoder?: GPUCommandEncoder;
  } = {},
): void {
  const { width, height } = sourceSize(source);
  if (width !== texture.width || height !== texture.height) {
    throw new RangeError(
      `updateTextureFromSource: source is ${width}x${height} but the texture is ${texture.width}x${texture.height}`,
    );
  }
  copySource(device, texture, source, options, options.encoder);
}

/**
 * Fetches and decodes an image into a texture. Alpha is decoded without premultiplying, so that
 * `premultipliedAlpha` alone decides how alpha is stored. Colour: data textures (`srgb: false`:
 * normal maps, masks, flow maps) are decoded with `colorSpaceConversion: 'none'`, so the values
 * reach the GPU exactly as stored in the file. Colour textures (`srgb`, the default) use the
 * browser's default conversion, which maps a tagged wide-gamut file (Display P3 and other ICC
 * profiles) into sRGB; with 'none' such a photo is read in the wrong space and looks washed out
 * (checked in Chrome: P3 (0.8, 0.3, 0.3) came out as 204,77,77 instead of 221,64,71).
 */
export async function loadTexture(
  device: GPUDevice,
  url: string,
  options: SourceTextureOptions = {},
): Promise<GPUTexture> {
  const response = await fetch(url);
  if (!response.ok) {
    throw new Error(`loadTexture: ${url} responded ${response.status} ${response.statusText}`);
  }
  const bitmap = await createImageBitmap(await response.blob(), {
    colorSpaceConversion: (options.srgb ?? true) ? 'default' : 'none',
    premultiplyAlpha: 'none',
  });
  return createTextureFromSource(device, bitmap, { label: url, ...options });
}

import { assertLinearRenderable } from './passes/common.ts';
import { blit } from './passes/blit.ts';

/** Levels in a full mip chain: floor(log2(max(width, height))) + 1. */
export function mipLevelCount(width: number, height: number): number {
  return 32 - Math.clz32(Math.max(width, height));
}

// One view per (layer, level), shared as the destination of a level and the source of the next.
// Cached so regenerating every frame (video) creates no GPU objects.
const levelViews = new WeakMap<GPUTexture, GPUTextureView[][]>();

function viewsFor(texture: GPUTexture): GPUTextureView[][] {
  let layers = levelViews.get(texture);
  if (!layers) {
    layers = Array.from({ length: texture.depthOrArrayLayers }, (_, layer) =>
      Array.from({ length: texture.mipLevelCount }, (_, level) =>
        texture.createView({
          label: `${texture.label} mip ${level} layer ${layer}`,
          dimension: '2d',
          baseMipLevel: level,
          mipLevelCount: 1,
          baseArrayLayer: layer,
          arrayLayerCount: 1,
        }),
      ),
    );
    levelViews.set(texture, layers);
  }
  return layers;
}

/**
 * Fills mip levels 1.. from level 0 with a linear downsample per level, for every array layer.
 * The texture needs `TEXTURE_BINDING | RENDER_ATTACHMENT`, more than one level, and a format that
 * can be filtered and rendered to; anything else throws here rather than failing at draw time.
 * Records into `encoder` if given, otherwise submits its own. sRGB formats filter in linear space.
 */
export function generateMipmaps(
  device: GPUDevice,
  texture: GPUTexture,
  encoder?: GPUCommandEncoder,
): void {
  if (texture.dimension !== '2d' || texture.sampleCount !== 1) {
    throw new Error('generateMipmaps: only single-sampled 2D textures are supported');
  }
  if (texture.mipLevelCount === 1) {
    if (texture.width === 1 && texture.height === 1) return;
    throw new Error(
      `generateMipmaps: ${texture.width}x${texture.height} texture has a single mip level; create it with mipLevelCount: mipLevelCount(width, height)`,
    );
  }
  const required = GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.RENDER_ATTACHMENT;
  if ((texture.usage & required) !== required) {
    throw new Error(
      'generateMipmaps: the texture needs TEXTURE_BINDING and RENDER_ATTACHMENT usage',
    );
  }
  assertLinearRenderable(device, texture.format, 'generateMipmaps');

  const recorder = encoder ?? device.createCommandEncoder({ label: `${texture.label} mipmaps` });
  for (const levels of viewsFor(texture)) {
    for (let level = 1; level < levels.length; level++) {
      const source = levels[level - 1];
      const view = levels[level];
      if (!source || !view) continue;
      blit(device, recorder, source, { view, format: texture.format }, { filter: 'linear' });
    }
  }
  if (!encoder) device.queue.submit([recorder.finish()]);
}

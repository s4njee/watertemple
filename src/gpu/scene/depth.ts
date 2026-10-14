import { createRenderTarget, type RenderTarget } from '../targets.ts';

export type DepthFormat = 'depth32float' | 'depth24plus';

export interface DepthTargetOptions {
  label: string;
  width: number;
  height: number;
  /**
   * `depth32float` (default) is what a screen-space pass should read: it is exact and sampleable
   * with `textureLoad` as `texture_depth_2d`. `depth24plus` is enough for a pass nobody samples.
   */
  format?: DepthFormat;
  /** Add TEXTURE_BINDING so shaders can read it. Default true. */
  sampled?: boolean;
  /** Add COPY_SRC so a test or tool can read the depth back. Default false. */
  readback?: boolean;
}

/**
 * A depth attachment as a resizable `RenderTarget` (the view and texture change on `resize`).
 * Binding it as a texture gives `texture_depth_2d`; use `textureLoad`, which needs no sampler.
 */
export function createDepthTarget(device: GPUDevice, options: DepthTargetOptions): RenderTarget {
  const { sampled = true } = options;
  return createRenderTarget(device, {
    label: options.label,
    format: options.format ?? 'depth32float',
    width: options.width,
    height: options.height,
    usage:
      GPUTextureUsage.RENDER_ATTACHMENT |
      (sampled ? GPUTextureUsage.TEXTURE_BINDING : 0) |
      (options.readback ? GPUTextureUsage.COPY_SRC : 0),
  });
}

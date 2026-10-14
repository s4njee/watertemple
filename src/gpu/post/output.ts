import { createFullscreenPipeline, renderFullscreen } from '../fullscreen.ts';
import { linearSampler, perDevice } from '../passes/common.ts';
import { createUniformBuffer, defineUniformStruct } from '../uniforms.ts';
import outputSource from './output.wgsl?raw';

const OutputParams = defineUniformStruct('OutputParams', {
  exposure: 'f32',
  tonemap: 'u32',
  view: 'u32',
  hdr: 'u32',
  headroom: 'f32',
  dither_lsb: 'f32',
  frame: 'f32',
});

/** The tonemappers of `color.wgsl`; `none` clamps to [0, 1] on an SDR canvas. */
export const TONEMAPPERS = ['aces', 'reinhard', 'none'] as const;
export type Tonemapper = (typeof TONEMAPPERS)[number];

/** What the output pass shows: the normal result, or one input on its own for debugging. */
export const POST_VIEWS = ['final', 'scene', 'bloom'] as const;
export type PostView = (typeof POST_VIEWS)[number];

const TONEMAP_ID: Record<Tonemapper, number> = { none: 0, aces: 1, reinhard: 2 };
const VIEW_ID: Record<PostView, number> = { final: 0, scene: 1, bloom: 2 };

export interface OutputSettings {
  /** Linear multiplier applied to scene + bloom before tonemapping. Default 1. */
  exposure?: number;
  /** Default 'aces'. */
  tonemap?: Tonemapper;
  /**
   * Strength of the dither in 8-bit steps (default 1; 0 turns it off). The noise is triangular
   * interleaved-gradient noise added after the sRGB encode and before the canvas quantises. On an
   * HDR canvas it is scaled to a 10-bit step instead.
   */
  dither?: number;
  /** Changes every frame when the caller passes the frame counter, so the pattern does not stick. */
  frame?: number;
  /**
   * Write extended-range values for a canvas configured with `toneMapping: { mode: 'extended' }`.
   * Content at or below scene white (1.0 after exposure) looks the same as in SDR; brighter content
   * keeps rising instead of clipping, rolling off towards `headroom`.
   */
  hdr?: boolean;
  /** Peak output in multiples of SDR white when `hdr` is on. Default 3; keep it modest (see notes). */
  headroom?: number;
  /** Show only the scene, or only the bloom. Both still get exposure and tonemapping. */
  view?: PostView;
}

export interface OutputPassOptions {
  /** Format of the target views passed to `encode` (the canvas format, e.g. bgra8unorm or rgba16float). */
  format: GPUTextureFormat;
  label?: string;
}

export interface OutputPass {
  /**
   * Draws `scene` (+ `bloom`) into `target`. `scene` and `bloom` may be different sizes: both are
   * sampled bilinearly. Without `bloom` a shared black texel stands in for it.
   */
  encode(
    encoder: GPUCommandEncoder,
    target: GPUTextureView,
    scene: GPUTextureView,
    bloom: GPUTextureView | undefined,
    settings?: OutputSettings,
    timestampWrites?: GPURenderPassTimestampWrites,
  ): void;
  destroy(): void;
}

interface OutputState {
  layout: GPUBindGroupLayout;
  pipelines: Map<GPUTextureFormat, GPURenderPipeline>;
  black: GPUTexture;
  blackView: GPUTextureView;
}

const stateFor = perDevice((device): OutputState => {
  const black = device.createTexture({
    label: 'post black',
    size: [1, 1],
    format: 'rgba16float',
    usage: GPUTextureUsage.TEXTURE_BINDING,
  });
  return {
    layout: device.createBindGroupLayout({
      label: 'post output layout',
      entries: [
        { binding: 0, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'float' } },
        { binding: 1, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'float' } },
        { binding: 2, visibility: GPUShaderStage.FRAGMENT, sampler: { type: 'filtering' } },
        { binding: 3, visibility: GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } },
      ],
    }),
    pipelines: new Map(),
    black,
    // A new texture is zero-filled, which is exactly "no bloom".
    blackView: black.createView(),
  };
});

export function createOutputPass(device: GPUDevice, options: OutputPassOptions): OutputPass {
  const label = options.label ?? 'post output';
  const state = stateFor(device);
  const uniforms = createUniformBuffer(device, OutputParams, `${label} params`);

  let pipeline = state.pipelines.get(options.format);
  if (!pipeline) {
    pipeline = createFullscreenPipeline(device, {
      label: `post output ${options.format}`,
      fragment: {
        label: 'post/output.wgsl',
        code: outputSource,
        chunks: { 'output-params.wgsl': OutputParams.wgsl },
      },
      format: options.format,
      layout: [state.layout],
    });
    state.pipelines.set(options.format, pipeline);
  }

  let group: GPUBindGroup | undefined;
  let groupKey: [GPUTextureView, GPUTextureView] | undefined;

  return {
    encode(encoder, target, scene, bloom, settings = {}, timestampWrites) {
      const glow = bloom ?? state.blackView;
      if (!group || groupKey?.[0] !== scene || groupKey[1] !== glow) {
        group = device.createBindGroup({
          label,
          layout: state.layout,
          entries: [
            { binding: 0, resource: scene },
            { binding: 1, resource: glow },
            { binding: 2, resource: linearSampler(device) },
            { binding: 3, resource: { buffer: uniforms.buffer } },
          ],
        });
        groupKey = [scene, glow];
      }
      const hdr = settings.hdr === true;
      uniforms.set({
        exposure: Math.max(0, settings.exposure ?? 1),
        tonemap: TONEMAP_ID[settings.tonemap ?? 'aces'],
        view: VIEW_ID[settings.view ?? 'final'],
        hdr: hdr ? 1 : 0,
        headroom: Math.max(1, settings.headroom ?? 3),
        // One 8-bit step is 1/255; an HDR canvas is presented at 10 bits at best, so a quarter of it.
        dither_lsb: ((settings.dither ?? 1) / 255) * (hdr ? 0.25 : 1),
        frame: (settings.frame ?? 0) % 64,
      });
      uniforms.upload();
      renderFullscreen(encoder, target, pipeline, [group], { label, timestampWrites });
    },
    destroy() {
      uniforms.destroy();
      group = undefined;
      groupKey = undefined;
    },
  };
}

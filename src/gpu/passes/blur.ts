import { createFullscreenPipeline, renderFullscreen } from '../fullscreen.ts';
import { createRenderTarget } from '../targets.ts';
import { createUniformBuffer, defineUniformStruct, type UniformBuffer } from '../uniforms.ts';
import { assertLinearRenderable, linearSampler, perDevice } from './common.ts';
import blurSource from './blur.wgsl?raw';

const BlurParams = defineUniformStruct('BlurParams', {
  direction: 'vec2f',
  radius: 'i32',
  sigma: 'f32',
});

/** Caps the loop so a stray value cannot hang the GPU. */
export const MAX_BLUR_RADIUS = 64;

export interface BlurOptions {
  label: string;
  /** Format of the intermediate target and of every target passed to `encode`. Must be filterable. */
  format: GPUTextureFormat;
  width: number;
  height: number;
}

export interface BlurParameters {
  /** Taps on each side of the centre, rounded and clamped to 0..MAX_BLUR_RADIUS. 0 copies. */
  radius: number;
  /** Standard deviation in pixels. Defaults to (radius + 1) / 3, which keeps ~99% of the weight inside. */
  sigma?: number;
}

export interface Blur {
  /**
   * Blurs `source` into `target`: horizontal into the owned intermediate, then vertical into
   * `target`. Source, target and the blur should all be the same size; `source` and `target` may
   * be the same view. `target` must have the format the blur was created with.
   *
   * The parameters go out through `queue.writeBuffer` here, so two `encode` calls on one Blur
   * before a single submit both run with the last parameters. Submit in between, or use one Blur
   * per set of parameters.
   */
  encode(
    encoder: GPUCommandEncoder,
    source: GPUTextureView,
    target: GPUTextureView,
    parameters: BlurParameters,
  ): void;
  /** Resizes the intermediate. Returns whether it changed (its contents are scratch anyway). */
  resize(width: number, height: number): boolean;
  destroy(): void;
}

interface BlurState {
  layout: GPUBindGroupLayout;
  pipelines: Map<GPUTextureFormat, GPURenderPipeline>;
}

const stateFor = perDevice((device): BlurState => ({
  layout: device.createBindGroupLayout({
    label: 'blur layout',
    entries: [
      { binding: 0, visibility: GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } },
      { binding: 1, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'float' } },
      { binding: 2, visibility: GPUShaderStage.FRAGMENT, sampler: { type: 'filtering' } },
    ],
  }),
  pipelines: new Map(),
}));

function pipelineFor(device: GPUDevice, format: GPUTextureFormat): GPURenderPipeline {
  const state = stateFor(device);
  let pipeline = state.pipelines.get(format);
  if (!pipeline) {
    pipeline = createFullscreenPipeline(device, {
      label: `blur ${format}`,
      fragment: {
        label: 'passes/blur.wgsl',
        code: blurSource,
        chunks: { 'blur_params.wgsl': BlurParams.wgsl },
      },
      format,
      layout: [state.layout],
    });
    state.pipelines.set(format, pipeline);
  }
  return pipeline;
}

export function createBlur(device: GPUDevice, options: BlurOptions): Blur {
  const { label, format } = options;
  assertLinearRenderable(device, format, 'createBlur');
  const { layout } = stateFor(device);
  const pipeline = pipelineFor(device, format);

  const intermediate = createRenderTarget(device, { ...options, label: `${label} intermediate` });
  const horizontal = createUniformBuffer(device, BlurParams, `${label} horizontal params`);
  const vertical = createUniformBuffer(device, BlurParams, `${label} vertical params`);
  horizontal.set({ direction: [1, 0] });
  vertical.set({ direction: [0, 1] });

  const horizontalGroups = new WeakMap<GPUTextureView, GPUBindGroup>();
  const verticalGroups = new WeakMap<GPUTextureView, GPUBindGroup>();
  const bind = (
    groups: WeakMap<GPUTextureView, GPUBindGroup>,
    params: UniformBuffer<typeof BlurParams.fields>,
    view: GPUTextureView,
  ) => {
    let group = groups.get(view);
    if (!group) {
      group = device.createBindGroup({
        label: `${label} bind group`,
        layout,
        entries: [
          { binding: 0, resource: { buffer: params.buffer } },
          { binding: 1, resource: view },
          { binding: 2, resource: linearSampler(device) },
        ],
      });
      groups.set(view, group);
    }
    return group;
  };

  return {
    encode(encoder, source, target, parameters) {
      const radius = Math.min(MAX_BLUR_RADIUS, Math.max(0, Math.round(parameters.radius)));
      const sigma = Math.max(parameters.sigma ?? (radius + 1) / 3, 0.1);
      for (const params of [horizontal, vertical]) {
        params.set({ radius, sigma });
        params.upload();
      }
      renderFullscreen(
        encoder,
        intermediate.view,
        pipeline,
        [bind(horizontalGroups, horizontal, source)],
        { label: `${label} horizontal` },
      );
      renderFullscreen(
        encoder,
        target,
        pipeline,
        [bind(verticalGroups, vertical, intermediate.view)],
        { label: `${label} vertical` },
      );
    },
    resize: (width, height) => intermediate.resize(width, height),
    destroy() {
      intermediate.destroy();
      horizontal.destroy();
      vertical.destroy();
    },
  };
}

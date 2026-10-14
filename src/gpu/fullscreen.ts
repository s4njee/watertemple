import { createShader, type ShaderSource } from './shader.ts';

export interface FullscreenPipelineOptions {
  /** Must `#include "fullscreen.wgsl"`, which supplies the vertex stage and `FullscreenVarying`. */
  fragment: ShaderSource;
  /** Fragment entry point. Defaults to `fs`. */
  entryPoint?: string;
  /** One format per colour attachment. */
  format: GPUTextureFormat | readonly GPUTextureFormat[];
  /** Applied to every colour attachment. Pair with `load: true` when drawing over existing content. */
  blend?: GPUBlendState;
  /** `'auto'` (default) derives bind group layouts from the shader. */
  layout?: 'auto' | readonly GPUBindGroupLayout[];
  /** Defaults to the fragment shader's label. */
  label?: string;
}

export function createFullscreenPipeline(
  device: GPUDevice,
  options: FullscreenPipelineOptions,
): GPURenderPipeline {
  const { fragment, blend, layout = 'auto' } = options;
  const label = options.label ?? fragment.label;
  const formats = typeof options.format === 'string' ? [options.format] : options.format;
  const module = createShader(device, fragment);

  return device.createRenderPipeline({
    label,
    layout:
      layout === 'auto'
        ? 'auto'
        : device.createPipelineLayout({ label: `${label} layout`, bindGroupLayouts: [...layout] }),
    vertex: { module, entryPoint: 'fullscreen_vs' },
    fragment: {
      module,
      entryPoint: options.entryPoint ?? 'fs',
      targets: formats.map((format) => ({ format, blend })),
    },
  });
}

/** Draws the fullscreen triangle. Bind groups are assigned to groups 0, 1, 2... in order. */
export function drawFullscreen(
  pass: GPURenderPassEncoder,
  pipeline: GPURenderPipeline,
  ...bindGroups: GPUBindGroup[]
): void {
  pass.setPipeline(pipeline);
  bindGroups.forEach((group, index) => pass.setBindGroup(index, group));
  pass.draw(3);
}

export interface RenderFullscreenOptions {
  /** Defaults to the pipeline's label. */
  label?: string;
  /** Keep the target's existing contents (needed for blending). Otherwise it is cleared. */
  load?: boolean;
  /** GPU timing for this pass, from `profiler.pass(label)`. */
  timestampWrites?: GPURenderPassTimestampWrites;
}

/** Begins a pass on `views`, draws the pipeline across it, and ends the pass. */
export function renderFullscreen(
  encoder: GPUCommandEncoder,
  views: GPUTextureView | readonly GPUTextureView[],
  pipeline: GPURenderPipeline,
  bindGroups: readonly GPUBindGroup[] = [],
  options: RenderFullscreenOptions = {},
): void {
  const loadOp = options.load ? 'load' : 'clear';
  const pass = encoder.beginRenderPass({
    label: options.label ?? pipeline.label,
    colorAttachments: ('length' in views ? views : [views]).map((view) => ({
      view,
      loadOp,
      storeOp: 'store',
    })),
    timestampWrites: options.timestampWrites,
  });
  drawFullscreen(pass, pipeline, ...bindGroups);
  pass.end();
}

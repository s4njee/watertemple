import { createFullscreenPipeline, renderFullscreen } from '../fullscreen.ts';
import type { TargetView } from '../targets.ts';
import { assertLinearRenderable, linearSampler, perDevice } from './common.ts';
import downsampleSource from './downsample.wgsl?raw';

interface DownsampleState {
  layout: GPUBindGroupLayout;
  pipelines: Map<GPUTextureFormat, GPURenderPipeline>;
  groups: WeakMap<GPUTextureView, GPUBindGroup>;
}

const stateFor = perDevice((device): DownsampleState => ({
  layout: device.createBindGroupLayout({
    label: 'downsample layout',
    entries: [
      {
        binding: 0,
        visibility: GPUShaderStage.FRAGMENT,
        texture: { sampleType: 'float' },
      },
      { binding: 1, visibility: GPUShaderStage.FRAGMENT, sampler: { type: 'filtering' } },
    ],
  }),
  pipelines: new Map(),
  groups: new WeakMap(),
}));

/**
 * Halves `source` into `target` (which should be about half its size) with the 13-tap filter from
 * Call of Duty: Advanced Warfare's bloom. It overlaps five 2x2 boxes, so every source texel
 * contributes the same total weight whatever its phase; a plain 4-tap box skips texels and
 * shimmers when the image moves. Edges clamp. Needs a filterable source and target format.
 */
export function downsample(
  device: GPUDevice,
  encoder: GPUCommandEncoder,
  source: GPUTextureView,
  target: TargetView,
): void {
  assertLinearRenderable(device, target.format, 'downsample');
  const state = stateFor(device);

  let pipeline = state.pipelines.get(target.format);
  if (!pipeline) {
    pipeline = createFullscreenPipeline(device, {
      label: `downsample ${target.format}`,
      fragment: { label: 'passes/downsample.wgsl', code: downsampleSource },
      format: target.format,
      layout: [state.layout],
    });
    state.pipelines.set(target.format, pipeline);
  }

  let group = state.groups.get(source);
  if (!group) {
    group = device.createBindGroup({
      label: 'downsample source',
      layout: state.layout,
      entries: [
        { binding: 0, resource: source },
        { binding: 1, resource: linearSampler(device) },
      ],
    });
    state.groups.set(source, group);
  }

  renderFullscreen(encoder, target.view, pipeline, [group]);
}

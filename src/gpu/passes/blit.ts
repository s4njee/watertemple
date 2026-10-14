import { createFullscreenPipeline, renderFullscreen } from '../fullscreen.ts';
import type { TargetView } from '../targets.ts';
import blitSource from './blit.wgsl?raw';
import { linearSampler, perDevice } from './common.ts';

export type BlitFilter = 'nearest' | 'linear';

export interface BlitOptions {
  /**
   * `nearest` copies texels exactly and accepts any float format. `linear` smooths when scaling
   * and needs a filterable source (not r32float / rgba32float without `float32-filterable`).
   * Defaults to `linear`.
   */
  filter?: BlitFilter;
}

interface BlitState {
  layouts: Record<BlitFilter, GPUBindGroupLayout>;
  pipelines: Map<string, GPURenderPipeline>;
  groups: Record<BlitFilter, WeakMap<GPUTextureView, GPUBindGroup>>;
}

const stateFor = perDevice((device): BlitState => {
  const texture = (sampleType: GPUTextureSampleType): GPUBindGroupLayoutEntry => ({
    binding: 0,
    visibility: GPUShaderStage.FRAGMENT,
    texture: { sampleType },
  });
  return {
    layouts: {
      // 'unfilterable-float' also accepts filterable textures.
      nearest: device.createBindGroupLayout({
        label: 'blit nearest layout',
        entries: [texture('unfilterable-float')],
      }),
      linear: device.createBindGroupLayout({
        label: 'blit linear layout',
        entries: [
          texture('float'),
          { binding: 1, visibility: GPUShaderStage.FRAGMENT, sampler: { type: 'filtering' } },
        ],
      }),
    },
    pipelines: new Map(),
    groups: { nearest: new WeakMap(), linear: new WeakMap() },
  };
});

/** Draws `source` across all of `target`, scaling if the sizes differ. Replaces the target's contents. */
export function blit(
  device: GPUDevice,
  encoder: GPUCommandEncoder,
  source: GPUTextureView,
  target: TargetView,
  options: BlitOptions = {},
): void {
  const filter = options.filter ?? 'linear';
  const state = stateFor(device);

  const key = `${filter}:${target.format}`;
  let pipeline = state.pipelines.get(key);
  if (!pipeline) {
    pipeline = createFullscreenPipeline(device, {
      label: `blit ${filter} ${target.format}`,
      fragment: { label: 'passes/blit.wgsl', code: blitSource },
      entryPoint: filter === 'nearest' ? 'fs_nearest' : 'fs_linear',
      format: target.format,
      layout: [state.layouts[filter]],
    });
    state.pipelines.set(key, pipeline);
  }

  const groups = state.groups[filter];
  let group = groups.get(source);
  if (!group) {
    group = device.createBindGroup({
      label: `blit ${filter} source`,
      layout: state.layouts[filter],
      entries: [
        { binding: 0, resource: source },
        ...(filter === 'linear' ? [{ binding: 1, resource: linearSampler(device) }] : []),
      ],
    });
    groups.set(source, group);
  }

  renderFullscreen(encoder, target.view, pipeline, [group]);
}

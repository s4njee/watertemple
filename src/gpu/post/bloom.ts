import { createFullscreenPipeline, drawFullscreen, renderFullscreen } from '../fullscreen.ts';
import { noProfiler, type PassProfiler } from '../profiler.ts';
import { createRenderTarget, type RenderTarget } from '../targets.ts';
import { createUniformBuffer, defineUniformStruct } from '../uniforms.ts';
import { linearSampler, perDevice } from '../passes/common.ts';
import downsampleSource from '../passes/downsample.wgsl?raw';
import downSource from './bloom-down.wgsl?raw';
import upSource from './bloom-up.wgsl?raw';

/** Bloom levels are always this format: HDR energy has to survive the chain. */
export const BLOOM_FORMAT: GPUTextureFormat = 'rgba16float';
export const MAX_BLOOM_LEVELS = 8;
/** The smallest level keeps at least this many texels on its shorter side. */
export const MIN_BLOOM_LEVEL_SIZE = 4;

const BloomParams = defineUniformStruct('BloomParams', {
  threshold: 'f32',
  knee: 'f32',
  intensity: 'f32',
  karis: 'f32',
});

export interface BloomSettings {
  /**
   * Strength of the glow, as a multiple of the energy that passes the threshold. 0 turns bloom off
   * and `encode` then does no GPU work at all.
   */
  intensity: number;
  /** Scene-linear brightness (brightest channel, before exposure) below which nothing glows. */
  threshold: number;
  /** Width of the soft transition below the threshold, in the same units. 0 is a hard edge. */
  knee: number;
  /**
   * 0..1: how much of each level comes from the coarser one on the way up. 0 keeps only the finest
   * (tight) glow; towards 1 the wide, faint levels dominate. Each step is a convex blend, so this
   * moves energy between scales and never adds any.
   */
  radius: number;
  /** Requested number of mip levels; clamped to 1..8 and to what the image size allows. */
  levels: number;
  /** Karis-average the first downsample so single hot pixels cannot flicker. Costs nothing measurable. */
  fireflyFilter: boolean;
}

export const DEFAULT_BLOOM: Readonly<BloomSettings> = {
  intensity: 0.6,
  threshold: 1,
  knee: 0.5,
  radius: 0.7,
  levels: 5,
  fireflyFilter: true,
};

export interface BloomOptions {
  label?: string;
  /** Size of the image that will be passed to `encode`. */
  width: number;
  height: number;
  /**
   * Start the chain at half the source size (default), which is what keeps the cost low: the first
   * pass reads the full-size image once and everything after it is a quarter of that or less.
   * `false` starts at full size, which is sharper and about four times dearer.
   */
  halfResolution?: boolean;
  /** Adds COPY_SRC to the level textures so tests and tools can read them back. */
  readable?: boolean;
}

export interface Bloom {
  readonly format: GPUTextureFormat;
  /**
   * Records the chain for `source` (any filterable colour texture) and returns the finest level,
   * which holds the glow already multiplied by `intensity`, or `undefined` when intensity is 0.
   * Add it to the scene (the output pass does). The view is valid until the next `resize`.
   *
   * The parameters go out through `queue.writeBuffer`, so two `encode` calls on one Bloom before a
   * single submit both run with the last settings.
   */
  encode(
    encoder: GPUCommandEncoder,
    source: GPUTextureView,
    settings?: Partial<BloomSettings>,
    profiler?: PassProfiler,
  ): GPUTextureView | undefined;
  /** Size of the image `encode` will read. Reallocates the levels and returns true if it changed. */
  resize(width: number, height: number): boolean;
  /** Sizes of the levels currently allocated, finest first. */
  levelSizes(): readonly (readonly [number, number])[];
  /** The levels (finest first), for tests and debugging. */
  readonly levels: readonly RenderTarget[];
  /** How many levels a source of the current size supports. */
  maxLevels(): number;
  destroy(): void;
}

interface BloomState {
  downLayout: GPUBindGroupLayout;
  prefilterLayout: GPUBindGroupLayout;
  down: GPURenderPipeline;
  prefilter: GPURenderPipeline;
  up: GPURenderPipeline;
}

// Created once per device; the constant-factor blend is what turns the upsample into a mix.
const stateFor = perDevice((device): BloomState => {
  const entry = (
    binding: number,
    kind: 'texture' | 'sampler' | 'uniform',
  ): GPUBindGroupLayoutEntry => ({
    binding,
    visibility: GPUShaderStage.FRAGMENT,
    ...(kind === 'texture'
      ? { texture: { sampleType: 'float' as const } }
      : kind === 'sampler'
        ? { sampler: { type: 'filtering' as const } }
        : { buffer: { type: 'uniform' as const } }),
  });
  const downLayout = device.createBindGroupLayout({
    label: 'bloom down layout',
    entries: [entry(0, 'texture'), entry(1, 'sampler')],
  });
  const prefilterLayout = device.createBindGroupLayout({
    label: 'bloom prefilter layout',
    entries: [entry(0, 'texture'), entry(1, 'sampler'), entry(2, 'uniform')],
  });
  const chunks = { 'downsample13.wgsl': downsampleSource, 'bloom-params.wgsl': BloomParams.wgsl };
  const constant: GPUBlendComponent = {
    srcFactor: 'constant',
    dstFactor: 'one-minus-constant',
    operation: 'add',
  };
  return {
    downLayout,
    prefilterLayout,
    down: createFullscreenPipeline(device, {
      label: 'bloom down',
      fragment: { label: 'post/bloom-down.wgsl', code: downSource, chunks },
      entryPoint: 'fs',
      format: BLOOM_FORMAT,
      layout: [downLayout],
    }),
    prefilter: createFullscreenPipeline(device, {
      label: 'bloom prefilter',
      fragment: { label: 'post/bloom-down.wgsl', code: downSource, chunks },
      entryPoint: 'fs_prefilter',
      format: BLOOM_FORMAT,
      layout: [prefilterLayout],
    }),
    up: createFullscreenPipeline(device, {
      label: 'bloom up',
      fragment: { label: 'post/bloom-up.wgsl', code: upSource },
      format: BLOOM_FORMAT,
      blend: { color: constant, alpha: constant },
      layout: [downLayout],
    }),
  };
});

const halve = (size: number) => Math.max(1, (size + 1) >> 1);

/** Level sizes for a source of `width` x `height`, finest first, as many as fit. */
export function bloomLevelSizes(
  width: number,
  height: number,
  halfResolution = true,
): [number, number][] {
  let w = Math.max(1, Math.round(width));
  let h = Math.max(1, Math.round(height));
  if (halfResolution) [w, h] = [halve(w), halve(h)];
  const sizes: [number, number][] = [[w, h]];
  while (sizes.length < MAX_BLOOM_LEVELS) {
    const next: [number, number] = [halve(w), halve(h)];
    if (Math.min(next[0], next[1]) < MIN_BLOOM_LEVEL_SIZE) break;
    sizes.push(next);
    [w, h] = next;
  }
  return sizes;
}

export function createBloom(device: GPUDevice, options: BloomOptions): Bloom {
  const label = options.label ?? 'bloom';
  const halfResolution = options.halfResolution ?? true;
  const usage =
    GPUTextureUsage.RENDER_ATTACHMENT |
    GPUTextureUsage.TEXTURE_BINDING |
    (options.readable ? GPUTextureUsage.COPY_SRC : 0);
  const state = stateFor(device);
  const uniforms = createUniformBuffer(device, BloomParams, `${label} params`);
  const sampler = linearSampler(device);
  const levels: RenderTarget[] = [];
  const prefilterGroups = new WeakMap<GPUTextureView, GPUBindGroup>();
  const downGroups = new WeakMap<GPUTextureView, GPUBindGroup>();
  let sizes = bloomLevelSizes(options.width, options.height, halfResolution);
  let destroyed = false;

  // Bind groups are cached per source view; a level's view changes only when it is reallocated.
  const prefilterGroupFor = (view: GPUTextureView) => {
    let group = prefilterGroups.get(view);
    if (!group) {
      group = device.createBindGroup({
        label: `${label} prefilter source`,
        layout: state.prefilterLayout,
        entries: [
          { binding: 0, resource: view },
          { binding: 1, resource: sampler },
          { binding: 2, resource: { buffer: uniforms.buffer } },
        ],
      });
      prefilterGroups.set(view, group);
    }
    return group;
  };

  const groupForDown = (view: GPUTextureView) => {
    let group = downGroups.get(view);
    if (!group) {
      group = device.createBindGroup({
        label: `${label} level source`,
        layout: state.downLayout,
        entries: [
          { binding: 0, resource: view },
          { binding: 1, resource: sampler },
        ],
      });
      downGroups.set(view, group);
    }
    return group;
  };

  function ensure(count: number) {
    while (levels.length < count) {
      const [w, h] = sizes[levels.length] ?? [1, 1];
      levels.push(
        createRenderTarget(device, {
          label: `${label} level ${levels.length}`,
          format: BLOOM_FORMAT,
          width: w,
          height: h,
          usage,
        }),
      );
    }
  }

  return {
    format: BLOOM_FORMAT,
    levels,
    maxLevels: () => sizes.length,
    levelSizes: () => levels.map((level) => [level.width, level.height] as const),

    resize(width, height) {
      if (destroyed) return false;
      sizes = bloomLevelSizes(width, height, halfResolution);
      let changed = false;
      // Levels past the new count are dropped so a smaller image does not keep a big chain alive.
      for (const extra of levels.splice(sizes.length)) {
        extra.destroy();
        changed = true;
      }
      levels.forEach((level, index) => {
        const [w, h] = sizes[index] ?? [1, 1];
        changed = level.resize(w, h) || changed;
      });
      return changed;
    },

    encode(encoder, source, settings = {}, profiler = noProfiler) {
      const s = { ...DEFAULT_BLOOM, ...settings };
      if (destroyed || !(s.intensity > 0)) return undefined;
      const count = Math.min(
        sizes.length,
        Math.max(1, Math.min(MAX_BLOOM_LEVELS, Math.round(s.levels))),
      );
      ensure(count);

      uniforms.set({
        threshold: Math.max(0, s.threshold),
        knee: Math.max(0, s.knee),
        intensity: s.intensity,
        karis: s.fireflyFilter ? 1 : 0,
      });
      uniforms.upload();

      renderFullscreen(encoder, levels[0]!.view, state.prefilter, [prefilterGroupFor(source)], {
        label: `${label} prefilter`,
        timestampWrites: profiler.pass('bloom-down'),
      });
      for (let i = 1; i < count; i++) {
        renderFullscreen(
          encoder,
          levels[i]!.view,
          state.down,
          [groupForDown(levels[i - 1]!.view)],
          { label: `${label} down ${i}`, timestampWrites: profiler.pass('bloom-down') },
        );
      }

      // Back up: level i = mix(down[i], tent(level i+1), radius). The constant blend factor does the
      // mix in the ROP, so the finer level is read and written in place with no extra texture.
      const radius = Math.min(1, Math.max(0, s.radius));
      for (let i = count - 2; i >= 0; i--) {
        const pass = encoder.beginRenderPass({
          label: `${label} up ${i}`,
          colorAttachments: [{ view: levels[i]!.view, loadOp: 'load', storeOp: 'store' }],
          timestampWrites: profiler.pass('bloom-up'),
        });
        pass.setBlendConstant({ r: radius, g: radius, b: radius, a: radius });
        drawFullscreen(pass, state.up, groupForDown(levels[i + 1]!.view));
        pass.end();
      }
      return levels[0]!.view;
    },

    destroy() {
      destroyed = true;
      for (const level of levels.splice(0)) level.destroy();
      uniforms.destroy();
    },
  };
}

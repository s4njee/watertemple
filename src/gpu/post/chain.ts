import { noProfiler, type PassProfiler } from '../profiler.ts';
import { assertLinearRenderable } from '../passes/common.ts';
import { createRenderTarget, type RenderTarget } from '../targets.ts';
import { createBloom, type Bloom, type BloomSettings } from './bloom.ts';
import { createOutputPass, type OutputPass, type OutputSettings } from './output.ts';

export interface PostChainOptions {
  /** Format of the canvas view passed to `encode`. */
  canvasFormat: GPUTextureFormat;
  width: number;
  height: number;
  /** Format of the HDR scene target. Default rgba16float; anything else must be filterable and wide enough for HDR. */
  sceneFormat?: GPUTextureFormat;
  /** Start the bloom chain at half the scene size (default) or at full size. */
  bloomHalfResolution?: boolean;
  label?: string;
}

export interface PostSettings {
  /** `false` leaves bloom out. Omitted (or omitted fields) means `DEFAULT_BLOOM`; set `intensity: 0` to skip it too. */
  bloom?: Partial<BloomSettings> | false;
  output?: OutputSettings;
}

export interface PostChain {
  /**
   * The HDR target the effect renders its linear scene into. Its `view` changes when `resize`
   * reallocates, so read it every frame.
   */
  readonly scene: RenderTarget;
  readonly width: number;
  readonly height: number;
  /** The parts, for effects that want to drive them directly. */
  readonly bloom: Bloom;
  readonly output: OutputPass;
  /** Returns true when the scene target was reallocated (its contents are then undefined). */
  resize(width: number, height: number): boolean;
  /**
   * Records bloom (if enabled) and the output pass into `canvasTarget`. The scene must already be
   * drawn. Pass labels for the HUD: `bloom-down`, `bloom-up`, `output`.
   */
  encode(
    encoder: GPUCommandEncoder,
    canvasTarget: GPUTextureView,
    settings?: PostSettings,
    profiler?: PassProfiler,
  ): void;
  destroy(): void;
}

/**
 * The standard shape of an HDR effect: draw into `chain.scene`, then `chain.encode(...)` bloom and
 * the final tonemapped pass into the canvas. Each piece also works alone (`createBloom`,
 * `createOutputPass`) for an effect that only wants one of them.
 */
export function createPostChain(device: GPUDevice, options: PostChainOptions): PostChain {
  const label = options.label ?? 'post';
  const sceneFormat = options.sceneFormat ?? 'rgba16float';
  assertLinearRenderable(device, sceneFormat, 'createPostChain');
  const scene = createRenderTarget(device, {
    label: `${label} scene`,
    format: sceneFormat,
    width: options.width,
    height: options.height,
  });
  const bloom = createBloom(device, {
    label: `${label} bloom`,
    width: scene.width,
    height: scene.height,
    halfResolution: options.bloomHalfResolution,
  });
  const output = createOutputPass(device, {
    format: options.canvasFormat,
    label: `${label} output`,
  });

  return {
    scene,
    bloom,
    output,
    get width() {
      return scene.width;
    },
    get height() {
      return scene.height;
    },
    resize(width, height) {
      const changed = scene.resize(width, height);
      bloom.resize(scene.width, scene.height);
      return changed;
    },
    encode(encoder, canvasTarget, settings = {}, profiler = noProfiler) {
      const glow =
        settings.bloom === false || settings.output?.view === 'scene'
          ? undefined
          : bloom.encode(encoder, scene.view, settings.bloom ?? {}, profiler);
      output.encode(
        encoder,
        canvasTarget,
        scene.view,
        glow,
        settings.output,
        profiler.pass('output'),
      );
    },
    destroy() {
      bloom.destroy();
      output.destroy();
      scene.destroy();
    },
  };
}

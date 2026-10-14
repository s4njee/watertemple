import { effectPoster, loadEffect, onEffectsChanged } from '../effects/registry.ts';
import {
  defaultParams,
  type EffectInstance,
  type EffectModule,
  type ParamSchema,
  type ParamValues,
} from '../effects/types.ts';
import {
  configureCanvas,
  configureCanvasHdr,
  observeCanvasSize,
  type CanvasSizer,
} from '../gpu/canvas.ts';
import type { GpuSession } from '../gpu/device.ts';
import { showFallback } from '../gpu/fallback.ts';
import { createFrameLoop } from '../gpu/loop.ts';
import { createGpuProfiler, type GpuProfiler, type PassTiming } from '../gpu/profiler.ts';
import { trackTextureMemory } from '../gpu/texture-memory.ts';
import { onChunksChanged } from '../wgsl/index.ts';
import { observeMotionEligibility } from './motion.ts';
import './host.css';

export interface EffectHostOptions {
  /** Values to start from instead of the schema defaults, e.g. decoded from the URL. */
  initialParams?: Readonly<Record<string, unknown>>;
  /** Draw one frame and hold it; call `resume()` to animate. */
  paused?: boolean;
  /** Default 2. */
  maxDpr?: number;
  /** Resolution scale, 0..1. Default 1. */
  resolutionScale?: number;
  /**
   * Let an effect that asks for an HDR canvas have one. Off by default: an extended-range float
   * canvas takes a slower path through the browser's compositor (measured here: 120 fps down to
   * about 50 for the whole page), so it is something a visitor opts into, never a thumbnail.
   */
  allowHdr?: boolean;
}

export interface LoadedEffect {
  effect: EffectModule;
  /** Live values: mutate this object to change what the effect draws. */
  params: ParamValues<ParamSchema>;
}

export interface FrameStats {
  /** Main-thread time spent encoding and submitting the frame, in milliseconds. */
  cpuMs: number;
  /** Time since the previous frame, in milliseconds. */
  frameMs: number;
  /** Drawing-buffer size and the buffer pixels per CSS pixel in effect. */
  width: number;
  height: number;
  dpr: number;
  /** GPU time per pass, from an earlier frame; empty when timing is off or unsupported. */
  passes: readonly PassTiming[];
  /**
   * GPU time of the frame `passes` came from: first timed pass starting to last one ending.
   * Use this for totals; the per-pass figures overlap on tile-based GPUs and must not be summed.
   */
  gpuMs: number | undefined;
}

export interface EffectHost {
  /** Holds the canvas (and the fallback notice). Move it to re-parent the effect. */
  readonly element: HTMLElement;
  readonly canvas: HTMLCanvasElement;
  /** Undefined until the effect has loaded once. */
  readonly loaded: LoadedEffect | undefined;
  /** Null when there is no working device. */
  readonly profiler: GpuProfiler | null;
  readonly paused: boolean;
  readonly resolutionScale: number;
  /** Fires each time the effect is (re)built, and at once if it already has been. */
  onLoad(listener: (loaded: LoadedEffect) => void): () => void;
  /** Fires after every rendered frame. */
  onStats(listener: (stats: FrameStats) => void): () => void;
  /**
   * Turns on per-pass GPU timing until the returned function is called. Timing costs a little
   * GPU time, so it runs only while at least one panel is displaying it.
   */
  requestTiming(): () => void;
  pause(): void;
  resume(): void;
  setResolutionScale(scale: number): void;
  destroy(): void;
}

interface Mounted {
  device: GPUDevice;
  context: GPUCanvasContext;
  instance: EffectInstance;
  /** Format of the canvas textures this instance was built for. */
  format: GPUTextureFormat;
  sizer: CanvasSizer;
  profiler: GpuProfiler;
}

/** Keeps values already chosen when an effect is rebuilt, dropping any the schema no longer fits. */
function carryParams(
  schema: ParamSchema,
  previous: Readonly<Record<string, unknown>> | undefined,
): ParamValues<ParamSchema> {
  const next = defaultParams(schema);
  if (!previous) return next;
  for (const key of Object.keys(next)) {
    const old = previous[key];
    if (typeof old === typeof next[key] && Array.isArray(old) === Array.isArray(next[key])) {
      next[key] = old as ParamValues<ParamSchema>[string];
    }
  }
  return next;
}

/**
 * Runs one effect in a canvas inside `container`, and rebuilds it when its device is replaced
 * after a loss or, during development, when its code or shaders are edited.
 */
export function mountEffect(
  session: GpuSession,
  container: HTMLElement,
  id: string,
  options: EffectHostOptions = {},
): EffectHost {
  const element = document.createElement('div');
  element.className = 'wt-host';
  const canvas = document.createElement('canvas');
  canvas.className = 'wt-host__canvas';
  element.append(canvas);
  container.append(element);

  const loadListeners = new Set<(loaded: LoadedEffect) => void>();
  const statsListeners = new Set<(stats: FrameStats) => void>();
  let mounted: Mounted | undefined;
  let loaded: LoadedEffect | undefined;
  let hideFallback: (() => void) | undefined;
  let size = { width: 0, height: 0, dpr: 1 };
  let resolutionScale = options.resolutionScale ?? 1;
  let paused = options.paused ?? false;
  let suspended = false;
  let timingRequests = 0;
  let generation = 0;
  let destroyed = false;

  // One loop for the host's lifetime, so time keeps running across hot reloads.
  const loop = createFrameLoop((time) => {
    if (!mounted) return;
    const { device, context, instance, sizer, profiler } = mounted;
    const started = performance.now();
    sizer.apply();
    profiler.beginFrame();
    const encoder = device.createCommandEncoder({ label: `${id} frame` });
    instance.frame(time, encoder, context.getCurrentTexture().createView());
    profiler.endFrame(encoder);
    device.queue.submit([encoder.finish()]);
    profiler.afterSubmit();
    // A paused host still draws the frame that follows a (re)build, so it never shows blank.
    if (paused || suspended) loop.stop();

    if (statsListeners.size === 0) return;
    const stats: FrameStats = {
      cpuMs: performance.now() - started,
      frameMs: time.delta * 1000,
      ...size,
      passes: profiler.latest,
      gpuMs: profiler.latestFrameMs,
    };
    for (const listener of [...statsListeners]) listener(stats);
  });

  function unmount() {
    mounted?.sizer.disconnect();
    mounted?.instance.destroy();
    mounted?.profiler.destroy();
    mounted = undefined;
    hideFallback?.();
    hideFallback = undefined;
    canvas.hidden = false;
  }

  async function mount() {
    const current = ++generation;
    const state = session.state;
    if (!state.ok) {
      unmount();
      canvas.hidden = true;
      hideFallback = showFallback(element, { ...state, posterUrl: effectPoster(id) });
      return;
    }

    const { device, format } = state.gpu;
    // Before the effect allocates anything, so the HUD's texture estimate sees all of it.
    trackTextureMemory(device);
    const profiler = createGpuProfiler(device);
    let built:
      | {
          effect: EffectModule;
          context: GPUCanvasContext;
          instance: EffectInstance;
          format: GPUTextureFormat;
        }
      | undefined;
    let params: ParamValues<ParamSchema> | undefined;
    let failure: unknown;
    // A shader that fails to compile does not throw: it yields an invalid pipeline that draws
    // nothing. The error scope catches that, so a broken build is rejected before it replaces
    // what is on screen, and a typo mid-edit does not blank the canvas.
    device.pushErrorScope('validation');
    try {
      const effect = await loadEffect(id);
      params = carryParams(effect.params, loaded?.params ?? options.initialParams);
      // Effects that opt in get an HDR canvas where there is one; everyone else is unchanged.
      const configured =
        effect.hdr && options.allowHdr
          ? configureCanvasHdr(canvas, device, format)
          : { context: configureCanvas(canvas, device, format), format, hdr: false };
      const { context } = configured;
      // The canvas now hands out textures in a different format than the running instance was
      // built for (HDR switched on or off), so it must not draw another frame into them.
      if (mounted && mounted.format !== configured.format) unmount();
      const instance = await effect.create({
        device,
        format: configured.format,
        canvas,
        params,
        profiler,
        hdr: configured.hdr,
      });
      built = { effect, context, instance, format: configured.format };
    } catch (error) {
      failure = error;
    }
    const gpuError = await device.popErrorScope();
    failure ??= gpuError?.message;
    if (!built || !params || failure !== undefined) {
      built?.instance.destroy();
      profiler.destroy();
      console.error(`[watertemple] could not start effect "${id}":`, failure);
      return;
    }
    const { effect, context, instance, format: builtFormat } = built;
    if (current !== generation || destroyed) {
      instance.destroy();
      profiler.destroy();
      return;
    }

    unmount();
    const sizer = observeCanvasSize(canvas, device, {
      maxDpr: options.maxDpr,
      scale: resolutionScale,
      onPendingResize() {
        // Resize a static frame in the same task as its redraw; no continuous animation.
        if (mounted && !element.ownerDocument.hidden && !loop.running) loop.start();
      },
      onResize(width, height, dpr) {
        size = { width, height, dpr };
        instance.resize(width, height, dpr);
      },
    });
    mounted = { device, context, instance, format: builtFormat, sizer, profiler };
    // Set here, not when the profiler was created: a panel may have asked for timing while the
    // effect was still being built.
    profiler.enabled = timingRequests > 0;
    loaded = { effect, params };
    loop.start();
    for (const listener of [...loadListeners]) listener(loaded);
  }

  const stopMotion = observeMotionEligibility(element, (eligible) => {
    suspended = !eligible;
    if (suspended || paused) loop.stop();
    else if (mounted) loop.start();
  });

  const unsubscribe = [
    stopMotion,
    session.onChange(() => void mount()),
    onEffectsChanged(() => void mount()),
    onChunksChanged(() => void mount()),
  ];
  void mount();

  return {
    element,
    canvas,
    get loaded() {
      return loaded;
    },
    get profiler() {
      return mounted?.profiler ?? null;
    },
    get paused() {
      return paused || suspended;
    },
    get resolutionScale() {
      return resolutionScale;
    },
    onLoad(listener) {
      loadListeners.add(listener);
      if (loaded) listener(loaded);
      return () => loadListeners.delete(listener);
    },
    onStats(listener) {
      statsListeners.add(listener);
      return () => statsListeners.delete(listener);
    },
    requestTiming() {
      let released = false;
      timingRequests++;
      if (mounted) mounted.profiler.enabled = true;
      return () => {
        if (released) return;
        released = true;
        timingRequests--;
        if (mounted) mounted.profiler.enabled = timingRequests > 0;
      };
    },
    pause() {
      paused = true;
      loop.stop();
    },
    resume() {
      paused = false;
      if (mounted && !suspended) loop.start();
    },
    setResolutionScale(scale) {
      resolutionScale = Math.min(1, Math.max(0.1, scale));
      mounted?.sizer.setScale(resolutionScale);
    },
    destroy() {
      destroyed = true;
      for (const stop of unsubscribe) stop();
      loadListeners.clear();
      statsListeners.clear();
      loop.stop();
      unmount();
      element.remove();
    },
  };
}

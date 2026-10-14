import { effectPoster, loadEffect, onEffectsChanged } from '../effects/registry.ts';
import type { GpuSession } from '../gpu/device.ts';
import {
  createEffectHost,
  type EffectHostOptions,
  type FrameStats,
  type LoadedEffect,
} from '../lib/host.ts';
import { onChunksChanged } from '../wgsl/index.ts';
import './host.css';
import '../gpu/fallback.css';

export type { EffectHostOptions, FrameStats, LoadedEffect };

/**
 * The gallery wrapper: creates the host element and canvas, resolves effects through the Vite
 * registry and shared-WGSL chunk watcher (including hot reload), and delegates to the CSS-free
 * library host in `src/lib`.
 */
export interface EffectHost {
  /** Holds the canvas (and the fallback notice). Move it to re-parent the effect. */
  readonly element: HTMLElement;
  readonly canvas: HTMLCanvasElement;
  /** Undefined until the effect has loaded once. */
  readonly loaded: LoadedEffect | undefined;
  /** Null when there is no working device. */
  readonly profiler: ReturnType<typeof createEffectHost>['profiler'];
  readonly paused: boolean;
  readonly resolutionScale: number;
  onLoad(listener: (loaded: LoadedEffect) => void): () => void;
  onStats(listener: (stats: FrameStats) => void): () => void;
  requestTiming(): () => void;
  pause(): void;
  resume(): void;
  setResolutionScale(scale: number): void;
  destroy(): void;
}

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

  const core = createEffectHost(session, canvas, () => loadEffect(id), {
    ...options,
    id,
    container: element,
    poster: effectPoster(id),
  });
  const stopEffectsChanged = onEffectsChanged(() => core.reload());
  const stopChunksChanged = onChunksChanged(() => core.reload());

  return {
    element,
    canvas,
    get loaded() {
      return core.loaded;
    },
    get profiler() {
      return core.profiler;
    },
    get paused() {
      return core.paused;
    },
    get resolutionScale() {
      return core.resolutionScale;
    },
    onLoad: (listener) => core.onLoad(listener),
    onStats: (listener) => core.onStats(listener),
    requestTiming: () => core.requestTiming(),
    pause: () => core.pause(),
    resume: () => core.resume(),
    setResolutionScale: (scale) => core.setResolutionScale(scale),
    destroy() {
      stopEffectsChanged();
      stopChunksChanged();
      core.destroy();
      element.remove();
    },
  };
}

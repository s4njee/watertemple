import type { EffectModule } from '../effects/types.ts';
import type { GpuSession } from '../gpu/device.ts';
import { createEffectHost, type EffectHost, type EffectHostOptions } from './host.ts';

export { createGpuSession, type GpuSession, type GpuSessionOptions } from '../gpu/device.ts';

export {
  createEffectHost,
  type EffectHost,
  type EffectHostOptions,
  type EffectLoader,
  type FrameStats,
  type LoadedEffect,
} from './host.ts';
export {
  defaultParams,
  defineEffect,
  type EffectContext,
  type EffectInstance,
  type EffectMeta,
  type EffectModule,
  type ParamSchema,
  type ParamValues,
  type Scorecard,
} from '../effects/types.ts';

/** Mount an already-imported effect module on a canvas the caller owns. */
export function mount(
  session: GpuSession,
  canvas: HTMLCanvasElement,
  effect: EffectModule,
  options: EffectHostOptions = {},
): EffectHost {
  return createEffectHost(session, canvas, async () => effect, { id: effect.meta.id, ...options });
}

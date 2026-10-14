export type InitFailureReason = 'unsupported' | 'no-adapter' | 'device-failed';
export type GpuFailureReason = InitFailureReason | 'device-lost';

export interface GpuFailure {
  ok: false;
  reason: GpuFailureReason;
  message: string;
}

/** Plain, serialisable snapshot of what the device was given. Feeds the performance HUD. */
export interface GpuInfo {
  /** Enabled features, sorted. */
  features: string[];
  /** Limits of the device, i.e. what code may rely on. */
  limits: Record<string, number>;
  /** Often empty strings: browsers hide this to limit fingerprinting. */
  adapter: { vendor: string; architecture: string; device: string; description: string };
}

export interface Gpu {
  adapter: GPUAdapter;
  device: GPUDevice;
  format: GPUTextureFormat;
  info: GpuInfo;
}

export type GpuInitResult = { ok: true; gpu: Gpu } | (GpuFailure & { reason: InitFailureReason });
export type GpuState = { ok: true; gpu: Gpu } | GpuFailure;

export interface GpuOptions {
  powerPreference?: GPUPowerPreference;
  /** Requested only when the adapter has them; a missing one never fails init. Replaces the default list. */
  optionalFeatures?: readonly GPUFeatureName[];
}

export const DEFAULT_OPTIONAL_FEATURES: readonly GPUFeatureName[] = [
  'timestamp-query',
  'float32-filterable',
];

function failure(reason: InitFailureReason, error: unknown, what: string) {
  const detail = error instanceof Error ? error.message : String(error);
  return { ok: false, reason, message: `${what}: ${detail}` } as const;
}

function snapshotLimits(limits: GPUSupportedLimits): Record<string, number> {
  const out: Record<string, number> = {};
  // Limits are prototype getters, so Object.keys() would find nothing.
  for (const key in limits) {
    const value = (limits as unknown as Record<string, unknown>)[key];
    if (typeof value === 'number') out[key] = value;
  }
  return out;
}

function snapshotInfo(adapter: GPUAdapter, device: GPUDevice): GpuInfo {
  const info = adapter.info as GPUAdapterInfo | undefined;
  return {
    features: [...device.features].sort(),
    limits: snapshotLimits(device.limits),
    adapter: {
      vendor: info?.vendor ?? '',
      architecture: info?.architecture ?? '',
      device: info?.device ?? '',
      description: info?.description ?? '',
    },
  };
}

/** Never throws. */
export async function initGpu(options: GpuOptions = {}): Promise<GpuInitResult> {
  const gpu = typeof navigator === 'undefined' ? undefined : navigator.gpu;
  if (!gpu) {
    return {
      ok: false,
      reason: 'unsupported',
      message: 'navigator.gpu is not available in this browser.',
    };
  }

  let adapter: GPUAdapter | null;
  try {
    adapter = await gpu.requestAdapter(
      options.powerPreference ? { powerPreference: options.powerPreference } : undefined,
    );
  } catch (error) {
    return failure('no-adapter', error, 'requestAdapter failed');
  }
  if (!adapter) {
    return { ok: false, reason: 'no-adapter', message: 'requestAdapter returned no adapter.' };
  }

  const wanted = options.optionalFeatures ?? DEFAULT_OPTIONAL_FEATURES;
  const requiredFeatures = [...new Set(wanted)].filter((feature) => adapter.features.has(feature));

  let device: GPUDevice;
  try {
    device = await adapter.requestDevice({ label: 'watertemple', requiredFeatures });
  } catch (error) {
    return failure('device-failed', error, 'requestDevice failed');
  }

  return {
    ok: true,
    gpu: {
      adapter,
      device,
      format: gpu.getPreferredCanvasFormat(),
      info: snapshotInfo(adapter, device),
    },
  };
}

function createEmitter<T>() {
  const listeners = new Set<(value: T) => void>();
  return {
    subscribe(listener: (value: T) => void): () => void {
      listeners.add(listener);
      return () => listeners.delete(listener);
    },
    emit(value: T) {
      // Copy so a listener can unsubscribe itself, and isolate listeners from each other's errors.
      for (const listener of [...listeners]) {
        try {
          listener(value);
        } catch (error) {
          console.error('[webgpu] listener threw:', error);
        }
      }
    },
    clear() {
      listeners.clear();
    },
  };
}

export interface GpuErrorReport {
  kind: 'validation' | 'out-of-memory' | 'internal' | 'unknown';
  message: string;
  /** Occurrences of this exact error since it was first seen, including suppressed ones. */
  count: number;
}

export interface GpuErrorReporter {
  /** Starts listening to the device; returns a function that stops. */
  watch(device: GPUDevice): () => void;
  /** Called for each logged report, not for suppressed repeats. */
  subscribe(listener: (report: GpuErrorReport) => void): () => void;
}

export interface GpuErrorReporterOptions {
  /** An error seen again within this window is counted but not logged again. Default 5000. */
  repeatIntervalMs?: number;
  now?: () => number;
}

const MAX_TRACKED_ERRORS = 100;

function errorKind(error: GPUError): GpuErrorReport['kind'] {
  if (error instanceof GPUValidationError) return 'validation';
  if (error instanceof GPUOutOfMemoryError) return 'out-of-memory';
  if (error instanceof GPUInternalError) return 'internal';
  return 'unknown';
}

export function createGpuErrorReporter(options: GpuErrorReporterOptions = {}): GpuErrorReporter {
  const { repeatIntervalMs = 5000, now = () => performance.now() } = options;
  const emitter = createEmitter<GpuErrorReport>();
  const seen = new Map<string, { count: number; suppressed: number; loggedAt: number }>();

  function record(error: GPUError) {
    const kind = errorKind(error);
    const key = `${kind}\n${error.message}`;
    const time = now();
    let entry = seen.get(key);
    if (!entry) {
      // Bounded so a stream of ever-changing messages cannot grow the map without limit.
      if (seen.size >= MAX_TRACKED_ERRORS) seen.delete(seen.keys().next().value as string);
      entry = { count: 0, suppressed: 0, loggedAt: -Infinity };
      seen.set(key, entry);
    }
    entry.count++;
    if (time - entry.loggedAt < repeatIntervalMs) {
      entry.suppressed++;
      return;
    }
    const repeats = entry.suppressed
      ? ` (${entry.suppressed} identical error${entry.suppressed === 1 ? '' : 's'} suppressed)`
      : '';
    console.error(`[webgpu] uncaptured ${kind} error${repeats}: ${error.message}`);
    entry.loggedAt = time;
    entry.suppressed = 0;
    emitter.emit({ kind, message: error.message, count: entry.count });
  }

  return {
    watch(device) {
      const onError = (event: Event) => {
        // Without this the browser also logs every error itself, which is the flood we avoid.
        event.preventDefault();
        record((event as GPUUncapturedErrorEvent).error);
      };
      device.addEventListener('uncapturederror', onError);
      return () => device.removeEventListener('uncapturederror', onError);
    },
    subscribe: emitter.subscribe,
  };
}

export interface GpuSessionOptions extends GpuOptions {
  /** Give up after this many device losses within `lossWindowMs`. Default 3. */
  maxLosses?: number;
  /** Default 10000. */
  lossWindowMs?: number;
}

export interface GpuSession {
  /** Current device, or the reason there is none. Replaced after a recovered loss. */
  readonly state: GpuState;
  /**
   * Fires with the new state after a real device loss was recovered (remount effects on the new
   * device) or given up on (show the fallback). Not fired for the initial state or `destroy()`.
   */
  onChange(listener: (state: GpuState) => void): () => void;
  onUncapturedError(listener: (report: GpuErrorReport) => void): () => void;
  /** Intentional teardown: destroys the device and does not trigger recovery. */
  destroy(): void;
}

export async function createGpuSession(options: GpuSessionOptions = {}): Promise<GpuSession> {
  const { maxLosses = 3, lossWindowMs = 10_000, ...gpuOptions } = options;
  const changes = createEmitter<GpuState>();
  const reporter = createGpuErrorReporter();
  const lossTimes: number[] = [];
  let destroyed = false;
  let stopWatching: (() => void) | undefined;
  let state: GpuState;

  function adopt(result: GpuState) {
    state = result;
    stopWatching?.();
    stopWatching = undefined;
    if (!result.ok) return;
    const { device } = result.gpu;
    stopWatching = reporter.watch(device);
    void device.lost.then((lost) => onLost(device, lost));
  }

  async function onLost(device: GPUDevice, lost: GPUDeviceLostInfo) {
    // 'destroyed' means someone called device.destroy() on purpose, including destroy() below.
    if (lost.reason === 'destroyed' || destroyed) return;
    if (!state.ok || state.gpu.device !== device) return;

    const time = performance.now();
    lossTimes.push(time);
    while (lossTimes.length && time - (lossTimes[0] ?? time) > lossWindowMs) lossTimes.shift();

    const cause = `GPU device lost (${lost.reason}): ${lost.message}`;
    if (lossTimes.length >= maxLosses) {
      console.error(`[webgpu] ${cause}. Lost ${lossTimes.length} times in a row, giving up.`);
      settle({ ok: false, reason: 'device-lost', message: cause });
      return;
    }

    console.warn(`[webgpu] ${cause}. Reinitialising.`);
    const result = await initGpu(gpuOptions);
    if (destroyed) {
      if (result.ok) result.gpu.device.destroy();
      return;
    }
    settle(
      result.ok
        ? result
        : {
            ok: false,
            reason: 'device-lost',
            message: `${cause}. Recovery failed (${result.reason}): ${result.message}`,
          },
    );
  }

  function settle(next: GpuState) {
    adopt(next);
    changes.emit(next);
  }

  adopt(await initGpu(gpuOptions));

  return {
    get state() {
      return state;
    },
    onChange: changes.subscribe,
    onUncapturedError: reporter.subscribe,
    destroy() {
      if (destroyed) return;
      destroyed = true;
      stopWatching?.();
      changes.clear();
      if (state.ok) state.gpu.device.destroy();
    },
  };
}

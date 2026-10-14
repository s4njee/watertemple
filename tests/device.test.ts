import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import {
  createGpuErrorReporter,
  createGpuSession,
  DEFAULT_OPTIONAL_FEATURES,
  initGpu,
  type Gpu,
  type GpuErrorReport,
  type GpuInitResult,
  type GpuSession,
  type GpuSessionOptions,
  type GpuState,
} from '../src/gpu/device.ts';
import { fallbackCopy, showFallback } from '../src/gpu/fallback.ts';

const pause = (ms = 25) => new Promise<void>((resolve) => setTimeout(resolve, ms));

function expectOk(result: GpuInitResult | GpuState): Gpu {
  if (!result.ok) throw new Error(`expected a device, got ${result.reason}: ${result.message}`);
  return result.gpu;
}

/** Replaces navigator.gpu until the returned function is called. */
function installGpu(gpu: unknown): () => void {
  const original = Object.getOwnPropertyDescriptor(navigator, 'gpu');
  Object.defineProperty(navigator, 'gpu', { value: gpu, configurable: true });
  return () => {
    if (original) Object.defineProperty(navigator, 'gpu', original);
  };
}

async function withGpu<T>(gpu: unknown, run: () => Promise<T>): Promise<T> {
  const restore = installGpu(gpu);
  try {
    return await run();
  } finally {
    restore();
  }
}

function bound(target: object, prop: string | symbol): unknown {
  const value = Reflect.get(target, prop, target);
  return typeof value === 'function' ? value.bind(target) : value;
}

/**
 * Wraps a real device so a test can end its `lost` promise with reason 'unknown', which is how a
 * driver reset or GPU process crash looks to the page. Dawn offers no way to cause one for real.
 */
function losable(device: GPUDevice) {
  let lose!: (info: GPUDeviceLostInfo) => void;
  const simulated = new Promise<GPUDeviceLostInfo>((resolve) => (lose = resolve));
  const lost = Promise.race([device.lost, simulated]);
  const proxy = new Proxy(device, {
    get: (target, prop) => (prop === 'lost' ? lost : bound(target, prop)),
  });
  return {
    device,
    proxy,
    lose: (message = 'simulated device loss') =>
      lose({ reason: 'unknown', message } as unknown as GPUDeviceLostInfo),
  };
}

/** A navigator.gpu stand-in that hands out real Dawn devices which tests can make "lose" the GPU. */
function createLosableGpu() {
  const real = navigator.gpu;
  const devices: ReturnType<typeof losable>[] = [];
  const control = {
    adapterRequests: 0,
    nullAdapterOnRequest: new Set<number>(),
    hold: undefined as Promise<void> | undefined,
  };
  const gpu = {
    getPreferredCanvasFormat: () => real.getPreferredCanvasFormat(),
    async requestAdapter(options?: GPURequestAdapterOptions) {
      const request = ++control.adapterRequests;
      await control.hold;
      if (control.nullAdapterOnRequest.has(request)) return null;
      const adapter = await real.requestAdapter(options);
      if (!adapter) return null;
      return new Proxy(adapter, {
        get(target, prop) {
          if (prop !== 'requestDevice') return bound(target, prop);
          return async (descriptor?: GPUDeviceDescriptor) => {
            const entry = losable(await target.requestDevice(descriptor));
            devices.push(entry);
            return entry.proxy;
          };
        },
      });
    },
  };
  return { gpu, control, devices, latest: () => devices[devices.length - 1]! };
}

describe('initGpu', () => {
  const opened: GPUDevice[] = [];
  afterEach(() => {
    for (const device of opened.splice(0)) device.destroy();
  });

  async function init(options?: Parameters<typeof initGpu>[0]) {
    const gpu = expectOk(await initGpu(options));
    opened.push(gpu.device);
    return gpu;
  }

  it('creates a device with the canvas format and a serialisable info snapshot', async () => {
    const gpu = await init();
    expect(gpu.format).toBe(navigator.gpu.getPreferredCanvasFormat());
    expect(typeof gpu.device.createBuffer).toBe('function');

    const { info } = gpu;
    expect(info.features).toEqual([...gpu.device.features].sort());
    expect(Object.keys(info.limits).length).toBeGreaterThan(20);
    expect(info.limits.maxTextureDimension2D).toBe(gpu.device.limits.maxTextureDimension2D);
    expect(Object.values(info.limits).every((v) => typeof v === 'number')).toBe(true);
    for (const field of Object.values(info.adapter)) expect(typeof field).toBe('string');
    expect(JSON.parse(JSON.stringify(info))).toEqual(info);
  });

  it('requests the default optional features exactly where the adapter has them', async () => {
    const adapter = await navigator.gpu.requestAdapter();
    const gpu = await init();
    for (const feature of DEFAULT_OPTIONAL_FEATURES) {
      expect(gpu.device.features.has(feature)).toBe(adapter?.features.has(feature));
    }
  });

  it('requests only the intersection of optionalFeatures and the adapter features', async () => {
    const adapter = await navigator.gpu.requestAdapter();
    const gpu = await init({
      optionalFeatures: [
        'timestamp-query',
        'timestamp-query',
        'not-a-real-feature' as GPUFeatureName,
      ],
    });
    expect(gpu.device.features.has('timestamp-query')).toBe(
      adapter?.features.has('timestamp-query'),
    );
    expect(gpu.device.features.has('not-a-real-feature' as GPUFeatureName)).toBe(false);
    // The default list is replaced, not extended.
    expect(gpu.device.features.has('float32-filterable')).toBe(false);
  });

  it('requests no optional features when given an empty list', async () => {
    const gpu = await init({ optionalFeatures: [] });
    expect(gpu.device.features.has('timestamp-query')).toBe(false);
    expect(gpu.device.features.has('float32-filterable')).toBe(false);
  });

  it('passes powerPreference to requestAdapter only when set', async () => {
    const requestAdapter = vi.fn(async () => null);
    const stub = { getPreferredCanvasFormat: () => 'bgra8unorm', requestAdapter };
    await withGpu(stub, async () => {
      await initGpu();
      await initGpu({ powerPreference: 'low-power' });
    });
    expect(requestAdapter.mock.calls).toEqual([[undefined], [{ powerPreference: 'low-power' }]]);
  });

  it('copes with adapter info being unavailable', async () => {
    const real = navigator.gpu;
    const stub = {
      getPreferredCanvasFormat: () => real.getPreferredCanvasFormat(),
      requestAdapter: async () => {
        const adapter = (await real.requestAdapter())!;
        return new Proxy(adapter, {
          get: (target, prop) => (prop === 'info' ? undefined : bound(target, prop)),
        });
      },
    };
    const gpu = await withGpu(stub, init);
    expect(gpu.info.adapter).toEqual({
      vendor: '',
      architecture: '',
      device: '',
      description: '',
    });
  });

  describe('failures', () => {
    it("reports 'unsupported' when navigator.gpu is missing", async () => {
      const result = await withGpu(undefined, () => initGpu());
      expect(result).toMatchObject({ ok: false, reason: 'unsupported' });
      expect(result.ok === false && result.message).toMatch(/navigator\.gpu/);
    });

    it("reports 'no-adapter' when the browser returns no adapter", async () => {
      const stub = { requestAdapter: async () => null };
      const result = await withGpu(stub, () => initGpu());
      expect(result).toMatchObject({ ok: false, reason: 'no-adapter' });
    });

    it("reports 'no-adapter' when requestAdapter rejects", async () => {
      const stub = {
        requestAdapter: async () => {
          throw new Error('GPU process unavailable');
        },
      };
      const result = await withGpu(stub, () => initGpu());
      expect(result).toMatchObject({ ok: false, reason: 'no-adapter' });
      expect(result.ok === false && result.message).toMatch(/GPU process unavailable/);
    });

    it("reports 'device-failed' when requestDevice rejects", async () => {
      const stub = {
        getPreferredCanvasFormat: () => 'bgra8unorm',
        requestAdapter: async () => ({
          features: new Set<string>(),
          requestDevice: async () => {
            throw new DOMException('out of devices', 'OperationError');
          },
        }),
      };
      const result = await withGpu(stub, () => initGpu());
      expect(result).toMatchObject({ ok: false, reason: 'device-failed' });
      expect(result.ok === false && result.message).toMatch(/out of devices/);
    });
  });
});

describe('createGpuErrorReporter', () => {
  function setup(repeatIntervalMs = 1000) {
    let clock = 0;
    const reporter = createGpuErrorReporter({ repeatIntervalMs, now: () => clock });
    const device = new EventTarget() as unknown as GPUDevice;
    const reports: GpuErrorReport[] = [];
    reporter.subscribe((report) => reports.push(report));
    const error = vi.spyOn(console, 'error').mockImplementation(() => {});
    return {
      reporter,
      device,
      reports,
      error,
      at: (ms: number) => {
        clock = ms;
      },
      raise(gpuError: GPUError) {
        const event = new Event('uncapturederror', { cancelable: true });
        Object.defineProperty(event, 'error', { value: gpuError });
        device.dispatchEvent(event);
        return event;
      },
    };
  }

  afterEach(() => vi.restoreAllMocks());

  it('logs and reports the first error, and cancels the browser default so it is not logged twice', () => {
    const { reporter, device, reports, error, raise } = setup();
    reporter.watch(device);
    const event = raise(new GPUValidationError('bad bind group'));

    expect(event.defaultPrevented).toBe(true);
    expect(error).toHaveBeenCalledTimes(1);
    expect(String(error.mock.calls[0]?.[0])).toMatch(/validation.*bad bind group/);
    expect(reports).toEqual([{ kind: 'validation', message: 'bad bind group', count: 1 }]);
  });

  it('collapses an error repeated every frame into one log per interval', () => {
    const { reporter, device, reports, error, raise, at } = setup(1000);
    reporter.watch(device);
    for (let frame = 0; frame < 60; frame++) {
      at(frame * 16);
      raise(new GPUValidationError('bad bind group'));
    }
    expect(error).toHaveBeenCalledTimes(1);
    expect(reports).toHaveLength(1);

    at(1200);
    raise(new GPUValidationError('bad bind group'));
    expect(error).toHaveBeenCalledTimes(2);
    expect(String(error.mock.calls[1]?.[0])).toMatch(/59 identical errors suppressed/);
    expect(reports[1]).toEqual({ kind: 'validation', message: 'bad bind group', count: 61 });
  });

  it('treats different messages and different kinds as different errors', () => {
    const { reporter, device, reports, raise } = setup();
    reporter.watch(device);
    raise(new GPUValidationError('a'));
    raise(new GPUValidationError('b'));
    raise(new GPUOutOfMemoryError('a'));
    expect(reports.map((r) => `${r.kind}:${r.message}`)).toEqual([
      'validation:a',
      'validation:b',
      'out-of-memory:a',
    ]);
  });

  it('stops listening when the watch is cancelled or the subscriber leaves', () => {
    const { reporter, device, reports, error, raise } = setup();
    const unwatch = reporter.watch(device);
    unwatch();
    raise(new GPUValidationError('ignored'));
    expect(error).not.toHaveBeenCalled();

    const late: GpuErrorReport[] = [];
    reporter.watch(device);
    const unsubscribe = reporter.subscribe((report) => late.push(report));
    unsubscribe();
    raise(new GPUValidationError('heard'));
    expect(late).toEqual([]);
    expect(reports).toHaveLength(1);
  });

  it('keeps logging when a subscriber throws', () => {
    const { reporter, device, error, raise } = setup();
    reporter.subscribe(() => {
      throw new Error('bad listener');
    });
    reporter.watch(device);
    expect(() => raise(new GPUValidationError('x'))).not.toThrow();
    expect(error).toHaveBeenCalled();
  });
});

describe('createGpuSession', () => {
  let restoreGpu: (() => void) | undefined;
  const sessions: GpuSession[] = [];
  let fake: ReturnType<typeof createLosableGpu>;
  let warn: ReturnType<typeof vi.spyOn>;

  beforeEach(() => {
    fake = createLosableGpu();
    restoreGpu = installGpu(fake.gpu);
    warn = vi.spyOn(console, 'warn').mockImplementation(() => {});
    vi.spyOn(console, 'error').mockImplementation(() => {});
  });

  afterEach(() => {
    for (const session of sessions.splice(0)) session.destroy();
    for (const entry of fake.devices) entry.device.destroy();
    restoreGpu?.();
    vi.restoreAllMocks();
  });

  async function start(options?: GpuSessionOptions) {
    const session = await createGpuSession(options);
    sessions.push(session);
    return session;
  }

  function nextChange(session: GpuSession): Promise<GpuState> {
    return new Promise((resolve) => {
      const unsubscribe = session.onChange((state) => {
        unsubscribe();
        resolve(state);
      });
    });
  }

  /** Simulates a real loss of the current device and waits for the session to settle. */
  async function loseAndSettle(session: GpuSession, message?: string) {
    const settled = nextChange(session);
    fake.latest().lose(message);
    return settled;
  }

  it('starts with the initial device and does not notify for it', async () => {
    const session = await start();
    expectOk(session.state);
    expect(fake.control.adapterRequests).toBe(1);
  });

  it('exposes the failure when the first init fails', async () => {
    const restore = installGpu(undefined);
    try {
      const session = await start();
      expect(session.state).toMatchObject({ ok: false, reason: 'unsupported' });
      expect(() => session.destroy()).not.toThrow();
    } finally {
      restore();
    }
  });

  it('treats destroy() as intentional: no recovery, no notification', async () => {
    const session = await start();
    const onChange = vi.fn();
    session.onChange(onChange);

    session.destroy();
    await expect(fake.devices[0]!.device.lost).resolves.toMatchObject({ reason: 'destroyed' });
    await pause();

    expect(fake.control.adapterRequests).toBe(1);
    expect(onChange).not.toHaveBeenCalled();
    expect(warn).not.toHaveBeenCalled();
    expect(() => session.destroy()).not.toThrow();
  });

  it('also treats a direct device.destroy() as intentional', async () => {
    const session = await start();
    const onChange = vi.fn();
    session.onChange(onChange);

    expectOk(session.state).device.destroy();
    await pause();

    expect(fake.control.adapterRequests).toBe(1);
    expect(onChange).not.toHaveBeenCalled();
  });

  it('re-initialises after a real loss and hands subscribers the new device', async () => {
    const session = await start();
    const before = expectOk(session.state);
    const changes: GpuState[] = [];
    session.onChange((state) => changes.push(state));

    const after = expectOk(await loseAndSettle(session, 'driver reset'));

    expect(after.device).not.toBe(before.device);
    expect(session.state).toEqual({ ok: true, gpu: after });
    expect(changes).toHaveLength(1);
    expect(fake.control.adapterRequests).toBe(2);
    expect(String(warn.mock.calls[0]?.[0])).toMatch(/driver reset/);
    // The replacement is a working device.
    const buffer = after.device.createBuffer({ size: 16, usage: GPUBufferUsage.COPY_DST });
    expect(buffer.size).toBe(16);
  });

  it('reports a stale loss from an earlier device only once', async () => {
    const session = await start();
    const first = fake.latest();
    await loseAndSettle(session);
    first.lose('late duplicate');
    await pause();
    expect(fake.control.adapterRequests).toBe(2);
  });

  it('gives up with device-lost after repeated losses in a short window', async () => {
    const session = await start();
    const changes: GpuState[] = [];
    session.onChange((state) => changes.push(state));

    expectOk(await loseAndSettle(session));
    expectOk(await loseAndSettle(session));
    const final = await loseAndSettle(session, 'still broken');

    expect(final).toMatchObject({ ok: false, reason: 'device-lost' });
    expect(final.ok === false && final.message).toMatch(/still broken/);
    expect(session.state).toBe(final);
    expect(changes).toHaveLength(3);

    await pause();
    expect(fake.control.adapterRequests).toBe(3);
  });

  it('only counts losses inside the window', async () => {
    const session = await start({ lossWindowMs: 40 });
    for (let i = 0; i < 4; i++) {
      expectOk(await loseAndSettle(session));
      await pause(70);
    }
    expect(fake.control.adapterRequests).toBe(5);
  });

  it('honours maxLosses', async () => {
    const session = await start({ maxLosses: 1 });
    const state = await loseAndSettle(session);
    expect(state).toMatchObject({ ok: false, reason: 'device-lost' });
    expect(fake.control.adapterRequests).toBe(1);
  });

  it('surfaces a failed recovery as device-lost, keeping the cause', async () => {
    const session = await start();
    fake.control.nullAdapterOnRequest.add(2);
    const state = await loseAndSettle(session, 'gpu reset');
    expect(state).toMatchObject({ ok: false, reason: 'device-lost' });
    expect(state.ok === false && state.message).toMatch(/gpu reset.*no-adapter/);
  });

  it('discards a device that finishes initialising after destroy()', async () => {
    const session = await start();
    const onChange = vi.fn();
    session.onChange(onChange);

    let release!: () => void;
    fake.control.hold = new Promise<void>((resolve) => (release = resolve));
    fake.latest().lose();
    await pause();
    session.destroy();
    release();

    await vi.waitFor(() => expect(fake.devices).toHaveLength(2));
    await expect(fake.devices[1]!.device.lost).resolves.toMatchObject({ reason: 'destroyed' });
    expect(onChange).not.toHaveBeenCalled();
  });

  it('stops notifying a listener that unsubscribed', async () => {
    const session = await start();
    const onChange = vi.fn();
    const unsubscribe = session.onChange(onChange);
    unsubscribe();
    await loseAndSettle(session);
    expect(onChange).not.toHaveBeenCalled();
  });

  it('reports uncaptured errors once instead of on every occurrence', async () => {
    const session = await start();
    const reports: GpuErrorReport[] = [];
    session.onUncapturedError((report) => reports.push(report));
    const { device } = expectOk(session.state);

    for (let i = 0; i < 5; i++) device.createBuffer({ size: 4, usage: 0xffffff });
    await vi.waitFor(() => expect(reports).toHaveLength(1));
    await pause(50);

    expect(reports).toHaveLength(1);
    expect(reports[0]).toMatchObject({ kind: 'validation', count: 1 });
    expect(console.error).toHaveBeenCalledTimes(1);
  });
});

describe('fallback', () => {
  const reasons = ['unsupported', 'no-adapter', 'device-failed', 'device-lost'] as const;

  it('has distinct human copy for every failure reason', () => {
    const titles = new Set<string>();
    for (const reason of reasons) {
      const { title, body } = fallbackCopy(reason);
      expect(title.length).toBeGreaterThan(10);
      expect(body.length).toBeGreaterThan(30);
      titles.add(title);
    }
    expect(titles.size).toBe(reasons.length);
  });

  it('tells the user to reload only where reloading helps', () => {
    expect(fallbackCopy('device-lost').body).toMatch(/reload/i);
    expect(fallbackCopy('unsupported').body).not.toMatch(/reload/i);
  });

  /** Just enough DOM to run showFallback without a browser. */
  class FakeNode {
    children: FakeNode[] = [];
    attributes: Record<string, string> = {};
    parent: FakeNode | undefined;
    className = '';
    textContent = '';
    src = '';
    alt = 'unset';
    constructor(readonly tag: string) {}
    get ownerDocument() {
      return { createElement: (tag: string) => new FakeNode(tag) };
    }
    setAttribute(name: string, value: string) {
      this.attributes[name] = value;
    }
    append(...nodes: FakeNode[]) {
      for (const node of nodes) {
        node.parent = this;
        this.children.push(node);
      }
    }
    remove() {
      if (!this.parent) return;
      this.parent.children = this.parent.children.filter((child) => child !== this);
      this.parent = undefined;
    }
    find(className: string): FakeNode | undefined {
      if (this.className === className) return this;
      for (const child of this.children) {
        const found = child.find(className);
        if (found) return found;
      }
      return undefined;
    }
  }

  function mount(options: Parameters<typeof showFallback>[1]) {
    const container = new FakeNode('div');
    const remove = showFallback(container as unknown as HTMLElement, options);
    return { container, remove, root: container.children[0] };
  }

  it('renders an accessible status notice with the reason-specific copy', () => {
    const { container, root } = mount({ reason: 'no-adapter' });
    expect(container.children).toHaveLength(1);
    expect(root?.className).toBe('wt-fallback');
    expect(root?.attributes).toMatchObject({ role: 'status', 'data-reason': 'no-adapter' });
    expect(root?.find('wt-fallback__title')?.textContent).toBe(fallbackCopy('no-adapter').title);
    expect(root?.find('wt-fallback__body')?.textContent).toBe(fallbackCopy('no-adapter').body);
    expect(root?.find('wt-fallback__poster')).toBeUndefined();
    expect(root?.find('wt-fallback__details')).toBeUndefined();
  });

  it('shows a decorative poster and the technical message as plain text', () => {
    const message = 'requestDevice failed: <img src=x onerror=alert(1)>';
    const { root } = mount({
      reason: 'device-failed',
      posterUrl: '/posters/caustics.webp',
      message,
    });
    const poster = root?.find('wt-fallback__poster');
    expect(poster?.tag).toBe('img');
    expect(poster?.src).toBe('/posters/caustics.webp');
    expect(poster?.alt).toBe('');
    expect(root?.find('wt-fallback__detail')?.textContent).toBe(message);
  });

  it('removes itself, and removing twice is harmless', () => {
    const { container, remove } = mount({ reason: 'unsupported' });
    remove();
    expect(container.children).toHaveLength(0);
    expect(() => remove()).not.toThrow();
  });
});

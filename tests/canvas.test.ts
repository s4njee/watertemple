import { describe, expect, it, vi } from 'vitest';
import {
  computeCanvasSize,
  configureCanvas,
  observeCanvasSize,
  type CanvasSizeInput,
} from '../src/gpu/canvas.ts';

const base: CanvasSizeInput = {
  cssWidth: 800,
  cssHeight: 600,
  dpr: 1,
  maxDpr: 2,
  maxDimension: 8192,
};

describe('computeCanvasSize', () => {
  it('multiplies CSS size by the device pixel ratio', () => {
    expect(computeCanvasSize(base)).toEqual({ width: 800, height: 600, dpr: 1 });
    expect(computeCanvasSize({ ...base, dpr: 2 })).toEqual({ width: 1600, height: 1200, dpr: 2 });
  });

  it('rounds to the nearest pixel for fractional DPRs and sizes', () => {
    // 333 * 1.5 = 499.5 and 201 * 1.5 = 301.5
    expect(computeCanvasSize({ ...base, cssWidth: 333, cssHeight: 201, dpr: 1.5 })).toEqual({
      width: 500,
      height: 302,
      dpr: 1.5,
    });
    // 333.33 * 1.25 = 416.66
    expect(computeCanvasSize({ ...base, cssWidth: 333.33, cssHeight: 100, dpr: 1.25 })).toEqual({
      width: 417,
      height: 125,
      dpr: 1.25,
    });
  });

  it('uses the exact device-pixel size when given, ignoring CSS rounding', () => {
    const size = computeCanvasSize({
      ...base,
      cssWidth: 333.33,
      cssHeight: 200.5,
      dpr: 1.5,
      devicePixelWidth: 499,
      devicePixelHeight: 301,
    });
    expect(size).toEqual({ width: 499, height: 301, dpr: 1.5 });
  });

  it('clamps the DPR to maxDpr', () => {
    expect(computeCanvasSize({ ...base, dpr: 3 })).toEqual({ width: 1600, height: 1200, dpr: 2 });
    expect(computeCanvasSize({ ...base, dpr: 3, maxDpr: 1 })).toEqual({
      width: 800,
      height: 600,
      dpr: 1,
    });
  });

  it('scales an exact device-pixel size down when the DPR is clamped', () => {
    const size = computeCanvasSize({
      ...base,
      dpr: 3,
      devicePixelWidth: 2400,
      devicePixelHeight: 1801,
    });
    expect(size).toEqual({ width: 1600, height: 1201, dpr: 2 });
  });

  it('does not raise a low DPR up to maxDpr', () => {
    expect(computeCanvasSize({ ...base, dpr: 0.75, maxDpr: 2 })).toEqual({
      width: 600,
      height: 450,
      dpr: 0.75,
    });
  });

  it('clamps to the texture dimension limit, shrinking both axes to keep the aspect ratio', () => {
    const size = computeCanvasSize({ ...base, cssWidth: 6000, cssHeight: 3000, dpr: 2 });
    expect(size.width).toBe(8192);
    expect(size.height).toBe(4096);
    expect(size.dpr).toBeCloseTo((2 * 8192) / 12000);
  });

  it('never exceeds the limit on either axis', () => {
    const size = computeCanvasSize({
      ...base,
      cssWidth: 100,
      cssHeight: 20_000,
      dpr: 2,
      maxDimension: 4096,
    });
    expect(size.height).toBe(4096);
    expect(size.width).toBeLessThanOrEqual(4096);
    expect(size.width).toBeGreaterThanOrEqual(1);
  });

  it('is at least 1x1', () => {
    expect(computeCanvasSize({ ...base, cssWidth: 0, cssHeight: 0 })).toEqual({
      width: 1,
      height: 1,
      dpr: 1,
    });
    expect(computeCanvasSize({ ...base, cssWidth: 0.3, cssHeight: 0.2 })).toMatchObject({
      width: 1,
      height: 1,
    });
  });

  it('survives nonsense input', () => {
    const size = computeCanvasSize({
      cssWidth: Number.NaN,
      cssHeight: -5,
      dpr: 0,
      maxDpr: Number.NaN,
      maxDimension: 0,
    });
    expect(size).toEqual({ width: 1, height: 1, dpr: 1 });
  });
});

describe('configureCanvas', () => {
  function fakeCanvas(context: unknown) {
    return { getContext: vi.fn(() => context) } as unknown as HTMLCanvasElement;
  }
  const device = {} as GPUDevice;

  it('configures with premultiplied alpha by default and returns the context', () => {
    const context = { configure: vi.fn() };
    const canvas = fakeCanvas(context);
    expect(configureCanvas(canvas, device, 'bgra8unorm')).toBe(context);
    expect(canvas.getContext).toHaveBeenCalledWith('webgpu');
    expect(context.configure).toHaveBeenCalledWith({
      device,
      format: 'bgra8unorm',
      alphaMode: 'premultiplied',
    });
  });

  it('allows the alpha mode to be overridden', () => {
    const context = { configure: vi.fn() };
    configureCanvas(fakeCanvas(context), device, 'rgba8unorm', { alphaMode: 'opaque' });
    expect(context.configure).toHaveBeenCalledWith({
      device,
      format: 'rgba8unorm',
      alphaMode: 'opaque',
    });
  });

  it('throws a descriptive error when no WebGPU context is available', () => {
    expect(() => configureCanvas(fakeCanvas(null), device, 'bgra8unorm')).toThrow(
      /getContext\('webgpu'\) returned null/,
    );
  });
});

describe('observeCanvasSize', () => {
  const device = { limits: { maxTextureDimension2D: 4096 } };

  interface FakeEntry {
    contentRect: { width: number; height: number };
    devicePixelContentBoxSize?: { inlineSize: number; blockSize: number }[];
  }

  /** Records how the canvas was written to, because every write clears a real canvas. */
  function setup(
    options: {
      clientWidth?: number;
      clientHeight?: number;
      dpr?: number;
      rejectDevicePixelBox?: boolean;
      maxDpr?: number;
    } = {},
  ) {
    const writes: string[] = [];
    let width = 300;
    let height = 150;
    const canvas = {
      clientWidth: options.clientWidth ?? 400,
      clientHeight: options.clientHeight ?? 200,
      get width() {
        return width;
      },
      set width(value: number) {
        writes.push(`width=${value}`);
        width = value;
      },
      get height() {
        return height;
      },
      set height(value: number) {
        writes.push(`height=${value}`);
        height = value;
      },
    } as unknown as HTMLCanvasElement;

    let dpr = options.dpr ?? 1;
    let notify: (entries: FakeEntry[]) => void = () => {};
    const observeCalls: unknown[] = [];
    let disconnected = false;
    class FakeResizeObserver {
      constructor(callback: (entries: FakeEntry[]) => void) {
        notify = (entries) => {
          if (!disconnected) callback(entries);
        };
      }
      observe(_target: unknown, observeOptions?: { box?: string }) {
        observeCalls.push(observeOptions);
        if (options.rejectDevicePixelBox && observeOptions?.box === 'device-pixel-content-box') {
          throw new TypeError("'device-pixel-content-box' is not a valid ResizeObserverBoxOptions");
        }
      }
      disconnect() {
        disconnected = true;
      }
    }

    const onResize = vi.fn();
    const onPendingResize = vi.fn();
    const sizer = observeCanvasSize(canvas, device, {
      maxDpr: options.maxDpr,
      onResize,
      onPendingResize,
      deps: {
        ResizeObserver: FakeResizeObserver as unknown as typeof ResizeObserver,
        devicePixelRatio: () => dpr,
      },
    });
    return {
      canvas,
      sizer,
      onResize,
      onPendingResize,
      writes,
      observeCalls,
      observe: (entry: FakeEntry) => notify([entry]),
      setDpr: (value: number) => {
        dpr = value;
      },
    };
  }

  it('requests a redraw only for a changed pending size, without clearing the static canvas', () => {
    const { sizer, observe, writes, onPendingResize } = setup();
    sizer.apply();
    writes.length = 0;
    observe({ contentRect: { width: 500, height: 250 } });
    expect(onPendingResize).toHaveBeenCalledOnce();
    expect(writes).toEqual([]);
    observe({ contentRect: { width: 500, height: 250 } });
    expect(onPendingResize).toHaveBeenCalledOnce();
    sizer.setScale(0.5);
    expect(onPendingResize).toHaveBeenCalledTimes(2);
    expect(writes).toEqual([]);
    sizer.apply();
    expect(writes).toEqual(['width=250', 'height=125']);
    sizer.disconnect();
  });

  it('asks for the exact device-pixel box first', () => {
    const { observeCalls } = setup();
    expect(observeCalls).toEqual([{ box: 'device-pixel-content-box' }]);
  });

  it('applies a layout-derived size on the first apply, before the observer has reported', () => {
    const { sizer, onResize, canvas } = setup({ dpr: 2 });
    expect(sizer.apply()).toBe(true);
    expect(onResize).toHaveBeenCalledExactlyOnceWith(800, 400, 2);
    expect([canvas.width, canvas.height]).toEqual([800, 400]);
  });

  it('does not touch the canvas or call onResize when the observer fires', () => {
    const { sizer, observe, onResize, writes } = setup();
    sizer.apply();
    writes.length = 0;
    onResize.mockClear();

    observe({
      contentRect: { width: 500, height: 250 },
      devicePixelContentBoxSize: [{ inlineSize: 500, blockSize: 250 }],
    });
    expect(writes).toEqual([]);
    expect(onResize).not.toHaveBeenCalled();

    expect(sizer.apply()).toBe(true);
    expect(writes).toEqual(['width=500', 'height=250']);
    expect(onResize).toHaveBeenCalledExactlyOnceWith(500, 250, 1);
  });

  it('uses the device-pixel box size when the entry has one', () => {
    const { sizer, observe, onResize } = setup({ dpr: 1.5 });
    observe({
      contentRect: { width: 333.33, height: 200 },
      devicePixelContentBoxSize: [{ inlineSize: 499, blockSize: 300 }],
    });
    sizer.apply();
    expect(onResize).toHaveBeenCalledWith(499, 300, 1.5);
  });

  it('falls back to content box x dpr when the browser rejects the device-pixel box', () => {
    const { sizer, observe, observeCalls, onResize } = setup({
      dpr: 2,
      rejectDevicePixelBox: true,
    });
    expect(observeCalls).toEqual([{ box: 'device-pixel-content-box' }, { box: 'content-box' }]);

    observe({ contentRect: { width: 300.25, height: 100 } });
    sizer.apply();
    expect(onResize).toHaveBeenCalledWith(601, 200, 2);
  });

  it('falls back per entry when the box is accepted but the entry lacks the size', () => {
    const { sizer, observe, onResize } = setup({ dpr: 2 });
    observe({ contentRect: { width: 100, height: 50 }, devicePixelContentBoxSize: undefined });
    sizer.apply();
    expect(onResize).toHaveBeenCalledWith(200, 100, 2);
  });

  it('calls onResize only when the size actually changes', () => {
    const { sizer, observe, onResize, writes } = setup();
    sizer.apply();
    expect(sizer.apply()).toBe(false);

    observe({ contentRect: { width: 400, height: 200 } }); // same as the seeded size
    expect(sizer.apply()).toBe(false);
    expect(onResize).toHaveBeenCalledTimes(1);
    expect(writes).toEqual(['width=400', 'height=200']);
  });

  it('writes only the dimension that changed, since any write clears the canvas', () => {
    const { sizer, observe, writes } = setup();
    sizer.apply();
    writes.length = 0;
    observe({ contentRect: { width: 400, height: 300 } });
    sizer.apply();
    expect(writes).toEqual(['height=300']);
  });

  it('reports a DPR change even when the buffer size is identical', () => {
    const { sizer, observe, onResize, setDpr, writes } = setup({ maxDpr: 4 });
    sizer.apply();
    writes.length = 0;
    onResize.mockClear();

    // Moving to a denser monitor with a compensating CSS size: same device pixels, new ratio.
    setDpr(2);
    observe({
      contentRect: { width: 200, height: 100 },
      devicePixelContentBoxSize: [{ inlineSize: 400, blockSize: 200 }],
    });
    expect(sizer.apply()).toBe(true);
    expect(onResize).toHaveBeenCalledExactlyOnceWith(400, 200, 2);
    expect(writes).toEqual([]);
  });

  it('defaults maxDpr to 2 and honours an override', () => {
    const capped = setup({ dpr: 3 });
    capped.sizer.apply();
    expect(capped.onResize).toHaveBeenCalledWith(800, 400, 2);

    const wide = setup({ dpr: 3, maxDpr: 3 });
    wide.sizer.apply();
    expect(wide.onResize).toHaveBeenCalledWith(1200, 600, 3);
  });

  it('clamps to the device texture limit', () => {
    const { sizer, onResize } = setup({ clientWidth: 4000, clientHeight: 1000, dpr: 2 });
    sizer.apply();
    expect(onResize).toHaveBeenCalledTimes(1);
    const [width, height] = onResize.mock.calls[0] as [number, number, number];
    expect(width).toBe(4096);
    expect(height).toBe(1024);
  });

  it('stops observing on disconnect', () => {
    const { sizer, observe, onResize } = setup();
    sizer.apply();
    onResize.mockClear();
    sizer.disconnect();
    observe({ contentRect: { width: 999, height: 999 } });
    expect(sizer.apply()).toBe(false);
    expect(onResize).not.toHaveBeenCalled();
  });
});

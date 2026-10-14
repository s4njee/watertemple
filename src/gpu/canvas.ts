export interface CanvasConfigOptions {
  /** Default 'premultiplied' so canvases composite over the DOM. */
  alphaMode?: GPUCanvasAlphaMode;
}

export function configureCanvas(
  canvas: HTMLCanvasElement,
  device: GPUDevice,
  format: GPUTextureFormat,
  options: CanvasConfigOptions = {},
  toneMapping?: GPUCanvasToneMapping,
): GPUCanvasContext {
  const context = canvas.getContext('webgpu');
  if (!context) {
    throw new Error(
      "canvas.getContext('webgpu') returned null: the canvas already has a different kind of " +
        'context, or this browser has no WebGPU canvas support.',
    );
  }
  const alphaMode = options.alphaMode ?? 'premultiplied';
  // Unchanged call shape unless tone mapping is asked for, so SDR canvases configure exactly as before.
  if (toneMapping) context.configure({ device, format, alphaMode, toneMapping });
  else context.configure({ device, format, alphaMode });
  return context;
}

/** The result of asking for an HDR canvas; see `configureCanvasHdr`. */
export interface HdrCanvasConfig {
  context: GPUCanvasContext;
  /** The format of the textures `getCurrentTexture()` now returns: rgba16float when `hdr`, else the format asked for. */
  format: GPUTextureFormat;
  /** True when the canvas is rgba16float with `toneMapping: { mode: 'extended' }` and the browser confirmed it. */
  hdr: boolean;
  /** Why the SDR configuration was used instead; undefined when `hdr` is true. */
  reason?: string;
}

export interface HdrCanvasOptions extends CanvasConfigOptions {
  /** Overrides the display check, which exists so the fallback logic can run under test. */
  displayIsHdr?: () => boolean;
}

/**
 * `(dynamic-range: high)`: the display (as the browser sees it) can show more than SDR. It says
 * nothing about whether HDR is switched on in the OS, and some browsers answer true without being
 * able to present HDR canvases, so it is only the first of the checks in `configureCanvasHdr`.
 */
export function displaySupportsHdr(): boolean {
  try {
    return globalThis.matchMedia?.('(dynamic-range: high)').matches === true;
  } catch {
    return false;
  }
}

/**
 * Configures the canvas for HDR output if the display and browser can do it, otherwise exactly as
 * `configureCanvas(canvas, device, sdrFormat)` would. Never throws for lack of HDR.
 *
 * HDR means `rgba16float` + `toneMapping: { mode: 'extended' }`, colour space srgb. Values are
 * sRGB-encoded as for any other srgb canvas, 1.0 is SDR white, and values above 1 are brighter than
 * that, up to what the display allows (the spec projects the rest onto the display's range).
 * Support is confirmed by reading the configuration back: a browser that ignores the unknown
 * `toneMapping` member (the spec's own advice is to leave it out of `getConfiguration()`) reports no
 * `extended` mode, and is put back on the SDR path with the original format.
 */
export function configureCanvasHdr(
  canvas: HTMLCanvasElement,
  device: GPUDevice,
  sdrFormat: GPUTextureFormat,
  options: HdrCanvasOptions = {},
): HdrCanvasConfig {
  const sdr = (reason: string): HdrCanvasConfig => ({
    context: configureCanvas(canvas, device, sdrFormat, options),
    format: sdrFormat,
    hdr: false,
    reason,
  });
  if (!(options.displayIsHdr ?? displaySupportsHdr)()) return sdr('display is not HDR capable');

  let context: GPUCanvasContext;
  try {
    context = configureCanvas(canvas, device, 'rgba16float', options, { mode: 'extended' });
  } catch (error) {
    return sdr(`configure failed: ${error instanceof Error ? error.message : String(error)}`);
  }
  const applied = context.getConfiguration?.();
  if (applied?.toneMapping?.mode !== 'extended' || applied.format !== 'rgba16float') {
    return sdr('browser did not apply extended tone mapping');
  }
  return { context, format: 'rgba16float', hdr: true };
}

export interface CanvasSize {
  /** Drawing-buffer size in pixels. */
  width: number;
  height: number;
  /** Buffer pixels per CSS pixel actually in effect, after the maxDpr and dimension clamps. */
  dpr: number;
}

export interface CanvasSizeInput {
  /** Layout size in CSS pixels; may be fractional. */
  cssWidth: number;
  cssHeight: number;
  /** Exact size from a `device-pixel-content-box` observation, when the browser provides it. */
  devicePixelWidth?: number;
  devicePixelHeight?: number;
  /** The browser's devicePixelRatio. */
  dpr: number;
  maxDpr: number;
  /** Resolution scale, 0..1: renders fewer pixels and lets CSS stretch them. Default 1. */
  scale?: number;
  /** device.limits.maxTextureDimension2D */
  maxDimension: number;
}

function positive(value: number, fallback: number): number {
  return Number.isFinite(value) && value > 0 ? value : fallback;
}

export function computeCanvasSize(input: CanvasSizeInput): CanvasSize {
  const dpr = positive(input.dpr, 1);
  const resolutionScale = Math.min(1, positive(input.scale ?? 1, 1));
  const effectiveDpr = Math.min(dpr, positive(input.maxDpr, 1)) * resolutionScale;
  const maxDimension = Math.max(1, Math.floor(positive(input.maxDimension, 1)));
  const { devicePixelWidth, devicePixelHeight } = input;
  const exact = devicePixelWidth !== undefined && devicePixelHeight !== undefined;

  // Scaling the exact device-pixel size (rather than recomputing from CSS size) keeps the browser's
  // own pixel snapping when no clamp applies, since effectiveDpr / dpr is then exactly 1.
  const scale = effectiveDpr / dpr;
  const rawWidth = (exact ? devicePixelWidth : positive(input.cssWidth, 0) * dpr) * scale;
  const rawHeight = (exact ? devicePixelHeight : positive(input.cssHeight, 0) * dpr) * scale;

  // Shrink both axes together: clamping them independently would distort the aspect ratio, and
  // dpr stays truthful for effects that convert between CSS and buffer pixels.
  const fit = Math.min(1, maxDimension / Math.max(rawWidth, rawHeight, 1));
  const clamp = (value: number) => Math.min(maxDimension, Math.max(1, Math.round(value * fit)));

  return { width: clamp(rawWidth), height: clamp(rawHeight), dpr: effectiveDpr * fit };
}

export interface CanvasSizerDeps {
  ResizeObserver: typeof ResizeObserver;
  devicePixelRatio(): number;
}

export interface CanvasSizerOptions {
  /** Default 2: beyond that the extra pixels cost far more than they show. */
  maxDpr?: number;
  /** Initial resolution scale, 0..1. Default 1. */
  scale?: number;
  /** Called from `apply()`, only when width, height or dpr differ from the last call. */
  onResize(width: number, height: number, dpr: number): void;
  /** A changed size is pending: a paused host can request one static redraw. */
  onPendingResize?(): void;
  /** Overrides browser globals; exists so the observer logic can run under test. */
  deps?: Partial<CanvasSizerDeps>;
}

export interface CanvasSizer {
  /**
   * Applies the most recently observed size to the canvas and calls `onResize` if it changed.
   * Returns true when it did. Call it at the start of every frame, before encoding any commands.
   *
   * Assigning canvas.width or height clears the canvas. Doing that inside the observer callback
   * would show a blank canvas until the next frame is drawn; here the clear and the redraw happen in
   * the same task, so the browser never presents the cleared buffer. Until then the previous frame
   * is shown, stretched by CSS for at most one frame.
   */
  apply(): boolean;
  /** Changes the resolution scale (0..1); takes effect on the next `apply()`. */
  setScale(scale: number): void;
  disconnect(): void;
}

export function observeCanvasSize(
  canvas: HTMLCanvasElement,
  device: { limits: Pick<GPUSupportedLimits, 'maxTextureDimension2D'> },
  options: CanvasSizerOptions,
): CanvasSizer {
  const { maxDpr = 2, onResize } = options;
  const deps: CanvasSizerDeps = {
    ResizeObserver: globalThis.ResizeObserver,
    devicePixelRatio: () => globalThis.devicePixelRatio,
    ...options.deps,
  };
  const maxDimension = device.limits.maxTextureDimension2D;

  let scale = options.scale ?? 1;
  // The observer's first callback arrives after the first requestAnimationFrame, so seed from
  // layout to avoid drawing the first frame into a default 300x150 buffer.
  let observed: { cssWidth: number; cssHeight: number; devicePixels?: ResizeObserverSize } = {
    cssWidth: canvas.clientWidth,
    cssHeight: canvas.clientHeight,
  };

  const compute = (cssWidth: number, cssHeight: number, devicePixels?: ResizeObserverSize) => {
    observed = { cssWidth, cssHeight, devicePixels };
    return computeCanvasSize({
      cssWidth,
      cssHeight,
      devicePixelWidth: devicePixels?.inlineSize,
      devicePixelHeight: devicePixels?.blockSize,
      dpr: deps.devicePixelRatio(),
      maxDpr,
      scale,
      maxDimension,
    });
  };

  let pending = compute(observed.cssWidth, observed.cssHeight);
  let applied: CanvasSize | undefined;

  const queueSize = (next: CanvasSize) => {
    const changed =
      next.width !== pending.width || next.height !== pending.height || next.dpr !== pending.dpr;
    pending = next;
    if (changed) options.onPendingResize?.();
  };

  const observer = new deps.ResizeObserver((entries) => {
    const entry = entries[entries.length - 1];
    if (!entry) return;
    // Missing in Safari, which also may throw on the box option below; so check per entry too.
    const devicePixels = entry.devicePixelContentBoxSize?.[0];
    queueSize(compute(entry.contentRect.width, entry.contentRect.height, devicePixels));
  });
  try {
    observer.observe(canvas, { box: 'device-pixel-content-box' });
  } catch {
    // An unknown box value is a TypeError, not a silent no-op. Content-box x dpr can be a pixel
    // off at fractional zoom, which is the best that browsers without the exact box can do.
    observer.observe(canvas, { box: 'content-box' });
  }

  return {
    apply() {
      const next = pending;
      if (
        applied &&
        applied.width === next.width &&
        applied.height === next.height &&
        applied.dpr === next.dpr
      ) {
        return false;
      }
      // Even assigning an unchanged value clears the canvas, so only write what differs.
      if (canvas.width !== next.width) canvas.width = next.width;
      if (canvas.height !== next.height) canvas.height = next.height;
      applied = next;
      onResize(next.width, next.height, next.dpr);
      return true;
    },
    setScale(next) {
      scale = next;
      queueSize(compute(observed.cssWidth, observed.cssHeight, observed.devicePixels));
    },
    disconnect() {
      observer.disconnect();
    },
  };
}

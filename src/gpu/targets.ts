/** What a pass needs to render into: the view, plus the format its pipeline must be built for. */
export interface TargetView {
  readonly view: GPUTextureView;
  readonly format: GPUTextureFormat;
}

export interface RenderTargetOptions {
  label: string;
  format: GPUTextureFormat;
  width: number;
  height: number;
  /**
   * Replaces the default, `RENDER_ATTACHMENT | TEXTURE_BINDING`. Add `STORAGE_BINDING` for compute
   * writes and `COPY_SRC` for readback; drop `RENDER_ATTACHMENT` for formats that cannot have it.
   */
  usage?: GPUTextureUsageFlags;
}

/**
 * A 2D texture with a default view. `texture` and `view` change when `resize` reallocates, so
 * read them at encode time and rebuild bind groups that reference the old view.
 */
export interface RenderTarget extends TargetView {
  readonly label: string;
  readonly usage: GPUTextureUsageFlags;
  readonly texture: GPUTexture;
  readonly width: number;
  readonly height: number;
  /**
   * Reallocates only if the size changed, and returns whether it did. Sizes are rounded and
   * clamped to at least 1. **Contents are lost on reallocation**: the new texture is zeroed, so a
   * simulation has to re-seed itself.
   */
  resize(width: number, height: number): boolean;
  destroy(): void;
}

function toSize(value: number, what: string): number {
  if (!Number.isFinite(value)) throw new RangeError(`Render target ${what} must be finite`);
  return Math.max(1, Math.round(value));
}

export function createRenderTarget(device: GPUDevice, options: RenderTargetOptions): RenderTarget {
  const { label, format } = options;
  const usage =
    options.usage ?? GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING;
  let width = toSize(options.width, 'width');
  let height = toSize(options.height, 'height');
  let destroyed = false;

  const allocate = () => device.createTexture({ label, size: [width, height], format, usage });
  let texture = allocate();
  let view = texture.createView({ label: `${label} view` });

  return {
    label,
    format,
    usage,
    get texture() {
      return texture;
    },
    get view() {
      return view;
    },
    get width() {
      return width;
    },
    get height() {
      return height;
    },
    resize(newWidth, newHeight) {
      const w = toSize(newWidth, 'width');
      const h = toSize(newHeight, 'height');
      if (destroyed || (w === width && h === height)) return false;
      texture.destroy();
      width = w;
      height = h;
      texture = allocate();
      view = texture.createView({ label: `${label} view` });
      return true;
    },
    destroy() {
      destroyed = true;
      texture.destroy();
    },
  };
}

/**
 * Two identical targets for simulations that read last frame's state while writing this
 * frame's: render into `write`, sample `read`, then `swap()`.
 */
export interface PingPongTarget {
  readonly read: RenderTarget;
  readonly write: RenderTarget;
  swap(): void;
  /** Resizes both targets. Contents of both are lost if this returns true; see `RenderTarget`. */
  resize(width: number, height: number): boolean;
  destroy(): void;
}

export function createPingPongTarget(
  device: GPUDevice,
  options: RenderTargetOptions,
): PingPongTarget {
  const targets = [
    createRenderTarget(device, { ...options, label: `${options.label} A` }),
    createRenderTarget(device, { ...options, label: `${options.label} B` }),
  ] as const;
  let readIndex: 0 | 1 = 0;

  return {
    get read() {
      return targets[readIndex];
    },
    get write() {
      return targets[readIndex === 0 ? 1 : 0];
    },
    swap() {
      readIndex = readIndex === 0 ? 1 : 0;
    },
    resize(width, height) {
      const resized = targets[0].resize(width, height);
      targets[1].resize(width, height);
      return resized;
    },
    destroy() {
      targets[0].destroy();
      targets[1].destroy();
    },
  };
}

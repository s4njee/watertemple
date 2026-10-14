// The obstacle mask of the heightfield solver: which cells are solid ground and which are water.
//
// CONVENTION. One `r8unorm` texel per grid cell, the same size as the solver's grid, row 0 at the
// TOP (uv.y = 0), column 0 at the left. A value of 0.5 or more (stored byte 128..255) is SOLID: waves
// reflect off it, its height is held at 0, and shading passes should draw the obstacle themselves.
// Anything below 0.5 is water. Use 0 and 255.
//
// The texture has RENDER_ATTACHMENT, so a pass can draw the obstacles into `view` on the GPU (set
// `enabled = true` afterwards), and COPY_DST, so `set` / `setRects` can upload from the CPU.
// The mask is as big as the grid, so it is re-created whenever the grid is: rectangles given to
// `setRects` are remembered in uv and rasterised again on the new grid, but raw data from `set` and
// anything drawn on the GPU is lost; compare `Heightfield.generation` to know when to repaint.

export const OBSTACLE_FORMAT: GPUTextureFormat = 'r8unorm';

/** A rounded rectangle in canvas uv (0..1, top-left origin). */
export interface ObstacleRect {
  x: number;
  y: number;
  width: number;
  height: number;
  /** Corner radius in image heights (canvas height = 1). Default 0. */
  radius?: number;
}

export interface ObstacleMask {
  /** Current texture and view; both change when the grid is re-created. */
  readonly texture: GPUTexture;
  readonly view: GPUTextureView;
  readonly width: number;
  readonly height: number;
  /**
   * Whether the solver reads the mask at all. `set`, `setRects` and `clear` maintain it; set it to
   * true yourself after drawing into `view` on the GPU. When false the step shaders skip the mask
   * taps entirely.
   */
  enabled: boolean;
  /** Replaces the mask with `width * height` bytes (see the convention above). */
  set(data: Uint8Array): void;
  /** Replaces the mask with these rectangles, rasterised at cell centres. Remembered across re-creation. */
  setRects(rects: readonly ObstacleRect[]): void;
  /** Everything is water again. */
  clear(): void;
}

export interface InternalObstacleMask extends ObstacleMask {
  /** Re-allocates for a new grid size, re-rasterising remembered rectangles. */
  resize(width: number, height: number): void;
  destroy(): void;
}

/** Signed distance (image heights) from point p to a rounded rectangle given by centre, half-size and radius. */
function roundedBoxDistance(
  px: number,
  py: number,
  cx: number,
  cy: number,
  hx: number,
  hy: number,
  r: number,
): number {
  const qx = Math.abs(px - cx) - (hx - r);
  const qy = Math.abs(py - cy) - (hy - r);
  return Math.hypot(Math.max(qx, 0), Math.max(qy, 0)) + Math.min(Math.max(qx, qy), 0) - r;
}

/** Rasterises rectangles onto a `width * height` grid. Pure; exported for tests. */
export function rasterizeRects(
  rects: readonly ObstacleRect[],
  width: number,
  height: number,
): Uint8Array {
  const data = new Uint8Array(width * height);
  const aspect = width / height;
  for (const rect of rects) {
    if (![rect.x, rect.y, rect.width, rect.height].every(Number.isFinite)) continue;
    if (rect.width <= 0 || rect.height <= 0) continue;
    // Work in image heights so corner radii are round.
    const hx = (rect.width * aspect) / 2;
    const hy = rect.height / 2;
    const cx = (rect.x + rect.width / 2) * aspect;
    const cy = rect.y + rect.height / 2;
    const r = Math.min(Math.max(rect.radius ?? 0, 0), hx, hy);
    const x0 = Math.max(0, Math.floor(rect.x * width) - 1);
    const x1 = Math.min(width - 1, Math.ceil((rect.x + rect.width) * width) + 1);
    const y0 = Math.max(0, Math.floor(rect.y * height) - 1);
    const y1 = Math.min(height - 1, Math.ceil((rect.y + rect.height) * height) + 1);
    for (let j = y0; j <= y1; j++) {
      for (let i = x0; i <= x1; i++) {
        const px = ((i + 0.5) / width) * aspect;
        const py = (j + 0.5) / height;
        if (roundedBoxDistance(px, py, cx, cy, hx, hy, r) <= 0) data[j * width + i] = 255;
      }
    }
  }
  return data;
}

export function createObstacleMask(device: GPUDevice, label: string): InternalObstacleMask {
  let width = 1;
  let height = 1;
  let texture: GPUTexture | undefined;
  let view: GPUTextureView | undefined;
  let rects: readonly ObstacleRect[] = [];
  const state = { enabled: false };

  function upload(data: Uint8Array) {
    if (!texture) return;
    device.queue.writeTexture(
      { texture },
      data as Uint8Array<ArrayBuffer>,
      { bytesPerRow: width },
      [width, height],
    );
  }

  return {
    get texture() {
      return texture!;
    },
    get view() {
      return view!;
    },
    get width() {
      return width;
    },
    get height() {
      return height;
    },
    get enabled() {
      return state.enabled;
    },
    set enabled(value: boolean) {
      state.enabled = value;
    },
    set(data) {
      if (data.length !== width * height) {
        throw new RangeError(
          `Obstacle mask data must be ${width} x ${height} = ${width * height} bytes, got ${data.length}`,
        );
      }
      rects = [];
      upload(data);
      state.enabled = data.some((value) => value >= 128);
    },
    setRects(next) {
      rects = next.map((rect) => ({ ...rect }));
      const data = rasterizeRects(rects, width, height);
      upload(data);
      state.enabled = data.some((value) => value >= 128);
    },
    clear() {
      rects = [];
      upload(new Uint8Array(width * height));
      state.enabled = false;
    },
    resize(newWidth, newHeight) {
      texture?.destroy();
      width = newWidth;
      height = newHeight;
      texture = device.createTexture({
        label: `${label} obstacle mask`,
        size: [width, height],
        format: OBSTACLE_FORMAT,
        usage:
          GPUTextureUsage.TEXTURE_BINDING |
          GPUTextureUsage.RENDER_ATTACHMENT |
          GPUTextureUsage.COPY_DST |
          GPUTextureUsage.COPY_SRC,
      });
      view = texture.createView({ label: `${label} obstacle mask view` });
      if (rects.length > 0) {
        const data = rasterizeRects(rects, width, height);
        upload(data);
        state.enabled = data.some((value) => value >= 128);
      } else {
        state.enabled = false;
      }
    },
    destroy() {
      texture?.destroy();
      texture = undefined;
    },
  };
}

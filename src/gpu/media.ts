import { createTextureFromSource, updateTextureFromSource } from './texture.ts';

/*
 * Getting an image or a video into a texture and keeping it current: the first step of every effect
 * that distorts, reflects or refracts something the site already has (src/effects/media-distortion).
 *
 *   createMediaTexture(device, source)        <img>, <canvas>, ImageBitmap, ImageData or <video>
 *   createMediaFromFile(device, file)         an image or video File (drag and drop, <input type=file>)
 *   mediaFromTexture(texture)                 wrap a texture you rendered yourself in the same interface
 *   media.update(encoder)                     once per frame: uploads a new video frame if one is due
 *   fitScale(canvasAspect, mediaAspect, fit)  cover / contain, matching `media_fit_scale` in media.wgsl
 *
 * Colour. The texture is `rgba8unorm-srgb`: the file's sRGB bytes are stored untouched and the sampler
 * decodes them, so the shader sees LINEAR light, filters and mips blend in linear space, and the shader's
 * last step is `linear_to_srgb`. Wide-gamut (Display P3) sources are converted to sRGB by the browser.
 * Orientation. Row 0 of the texture is the top row of the picture, so `uv` from fullscreen.wgsl (origin
 * top-left) shows it upright; EXIF rotation is applied by the browser's decoder. Alpha is stored
 * straight (not premultiplied); photos and video are opaque anyway.
 *
 * Video, and why it is copied. Two ways to read a <video> in WebGPU:
 *   1. copyExternalImageToTexture into a normal texture, once per new frame (this file).
 *   2. importExternalTexture, a zero-copy `texture_external` that is valid only for the current task.
 * (2) saves one GPU copy of the frame, which for a 720p to 1080p frame is a fraction of a millisecond
 * on a desktop GPU. It costs a lot of what an effect wants: `texture_external` can only be read with
 * textureSampleBaseClampToEdge (no mip levels, so no blur or minification filtering, no custom wrap
 * mode, a fixed sampler), it has to be re-imported and its bind group rebuilt every frame, the shader
 * needs a second code path for stills, and it returns non-linear sRGB that the shader must decode
 * itself. Every refraction helper in src/wgsl takes a plain `texture_2d`. Support is also thinner:
 * importExternalTexture is in Chrome and Safari 26 but, at the time of writing, not in Firefox, which
 * has no `texture_external` at all. So: copy path by default, no external-texture path. Revisit it for
 * a full-screen 4K video with a trivial shader, where the copy and the mip chain are the main cost.
 *
 * "Only when a new frame exists". A 24 or 30 fps video on a 60 or 120 Hz display has a new frame every
 * second to fourth rAF. Uploading on every rAF would copy the same picture again and regenerate the
 * mips for nothing. `requestVideoFrameCallback` fires once per presented frame (Chrome, Edge, Safari;
 * Firefox from 132); without it the fallback compares `currentTime`, which is coarser but correct.
 * `videoFrameDue` is that decision as a pure function.
 */

export type MediaFit = 'cover' | 'contain';

export type MediaSource = GPUCopyExternalImageSource;

// --- Pure logic (no DOM, no GPU) -----------------------------------------------------------------

/**
 * Scale about the centre that turns canvas uv into media uv (`media_uv = (uv - 0.5) * scale + 0.5`),
 * the CSS `object-fit` rules. Same numbers as `media_fit_scale` in src/wgsl/media.wgsl.
 * Cover never leaves [0, 1]; contain leaves it over the bars. With r = canvasAspect / mediaAspect:
 * r > 1 gives cover (1, 1/r) and contain (r, 1); r < 1 gives cover (r, 1) and contain (1, 1/r).
 */
export function fitScale(
  canvasAspect: number,
  mediaAspect: number,
  fit: MediaFit,
): [number, number] {
  const r = Math.max(canvasAspect, 1e-4) / Math.max(mediaAspect, 1e-4);
  const cover: [number, number] = r > 1 ? [1, 1 / r] : [r, 1];
  const shortest = Math.min(cover[0], cover[1]);
  return fit === 'contain' ? [cover[0] / shortest, cover[1] / shortest] : cover;
}

/** Media uv for a canvas uv under a fit scale. */
export function fitUv(uv: readonly [number, number], scale: readonly [number, number]) {
  return [(uv[0] - 0.5) * scale[0] + 0.5, (uv[1] - 0.5) * scale[1] + 0.5] as [number, number];
}

/** Shrinks `width` x `height` to fit inside `max` on the long side, keeping the aspect. Never upscales. */
export function limitSize(
  width: number,
  height: number,
  max: number,
): { width: number; height: number; scaled: boolean } {
  const longest = Math.max(width, height);
  if (longest <= max) return { width, height, scaled: false };
  const k = max / longest;
  return {
    width: Math.max(1, Math.round(width * k)),
    height: Math.max(1, Math.round(height * k)),
    scaled: true,
  };
}

/** What a dropped or chosen file is, from its MIME type or, failing that, its extension. */
export function classifyFile(file: {
  name?: string;
  type?: string;
}): 'image' | 'video' | undefined {
  const type = (file.type ?? '').toLowerCase();
  if (type.startsWith('image/')) return 'image';
  if (type.startsWith('video/')) return 'video';
  const extension = /\.([a-z0-9]+)$/i.exec(file.name ?? '')?.[1]?.toLowerCase() ?? '';
  if (['png', 'jpg', 'jpeg', 'webp', 'gif', 'avif', 'bmp', 'svg'].includes(extension)) {
    return 'image';
  }
  if (['mp4', 'webm', 'mov', 'm4v', 'ogv', 'ogg'].includes(extension)) return 'video';
  return undefined;
}

/** `HTMLMediaElement.readyState` value meaning the current frame can be drawn. */
export const HAVE_CURRENT_DATA = 2;

export interface VideoUploadState {
  readyState: number;
  /** `videoWidth` / `videoHeight`: 0 until the metadata has loaded. */
  width: number;
  height: number;
  currentTime: number;
  /** `currentTime` at the last upload, or undefined if nothing has been uploaded yet. */
  lastUploadedTime: number | undefined;
  /** The browser has `requestVideoFrameCallback`. */
  frameCallbackSupported: boolean;
  /** The frame callback fired since the last upload. */
  frameSignalled: boolean;
}

/**
 * Should this frame's render upload the video again? Never before the first frame has decoded, always
 * for the first one, and after that only when the browser says a new frame was presented (or, without
 * `requestVideoFrameCallback`, when the playback position moved).
 */
export function videoFrameDue(state: VideoUploadState): boolean {
  if (state.readyState < HAVE_CURRENT_DATA || state.width <= 0 || state.height <= 0) return false;
  if (state.lastUploadedTime === undefined) return true;
  return state.frameCallbackSupported
    ? state.frameSignalled
    : state.currentTime !== state.lastUploadedTime;
}

// --- The texture handle ---------------------------------------------------------------------------

export interface MediaTexture {
  readonly kind: 'still' | 'video';
  /** All mip levels, `rgba8unorm-srgb` (linear on sampling). Replaced when the media changes size. */
  readonly view: GPUTextureView;
  readonly texture: GPUTexture;
  readonly width: number;
  readonly height: number;
  /** width / height of the media itself. */
  readonly aspect: number;
  readonly mipLevelCount: number;
  /** Bumps whenever `view` is replaced (a video changed resolution): rebuild bind groups then. */
  readonly version: number;
  /**
   * Call once per frame before drawing. For a video, uploads the current frame if a new one is due
   * and records its mip passes into `encoder`; returns whether it did. A still returns false.
   */
  update(encoder: GPUCommandEncoder): boolean;
  destroy(): void;
}

export interface MediaTextureOptions {
  label?: string;
  /** Mip chain for minification and blur. Default true. */
  mipmaps?: boolean;
  /**
   * Stills larger than this on the long side are downscaled while decoding (default 4096; a 4096 square
   * with mips is about 90 MB). Never larger than the device allows. Videos are not resized.
   */
  maxSize?: number;
  /** Extra cleanup run by `destroy()`, e.g. revoking an object URL. */
  onDestroy?: () => void;
}

function isVideo(source: MediaSource): source is HTMLVideoElement {
  return 'videoWidth' in source;
}

function isImageElement(source: MediaSource): source is HTMLImageElement {
  return 'naturalWidth' in source;
}

function sizeOf(source: MediaSource): { width: number; height: number } {
  if (isVideo(source)) return { width: source.videoWidth, height: source.videoHeight };
  if (isImageElement(source)) return { width: source.naturalWidth, height: source.naturalHeight };
  const sized = source as { displayWidth?: number; displayHeight?: number } & {
    width: number;
    height: number;
  };
  return {
    width: sized.displayWidth ?? sized.width,
    height: sized.displayHeight ?? sized.height,
  };
}

/** Resolves once a video has a frame to show. Rejects on a media error or after `timeoutMs`. */
export function whenVideoReady(video: HTMLVideoElement, timeoutMs = 10_000): Promise<void> {
  if (video.readyState >= HAVE_CURRENT_DATA && video.videoWidth > 0) return Promise.resolve();
  return new Promise((resolve, reject) => {
    const finish = (error?: Error) => {
      clearTimeout(timer);
      for (const type of ['loadeddata', 'canplay', 'playing'])
        video.removeEventListener(type, ready);
      video.removeEventListener('error', failed);
      if (error) reject(error);
      else resolve();
    };
    const ready = () => {
      if (video.readyState >= HAVE_CURRENT_DATA && video.videoWidth > 0) finish();
    };
    const failed = () =>
      finish(new Error(`the video failed to load (${video.error?.message ?? ''})`));
    const timer = setTimeout(
      () => finish(new Error('timed out waiting for the first video frame')),
      timeoutMs,
    );
    for (const type of ['loadeddata', 'canplay', 'playing']) video.addEventListener(type, ready);
    video.addEventListener('error', failed);
  });
}

async function decodeToBitmap(
  source: ImageBitmapSource,
  size?: { width: number; height: number },
): Promise<ImageBitmap> {
  return createImageBitmap(source, {
    // Colour photos: let the browser map wide-gamut files into sRGB. Alpha stays straight.
    colorSpaceConversion: 'default',
    premultiplyAlpha: 'none',
    ...(size ? { resizeWidth: size.width, resizeHeight: size.height, resizeQuality: 'high' } : {}),
  });
}

/**
 * Makes a texture from an image, canvas, bitmap or video and, for a video, a handle that re-uploads
 * it when a new frame is due. A video must already have its metadata (`whenVideoReady`).
 * `HTMLImageElement`s are decoded and passed through `createImageBitmap` first: that is the one path
 * every browser accepts (`copyExternalImageToTexture` with an <img> is Chrome-only), and it is where
 * oversized pictures are downscaled.
 */
export async function createMediaTexture(
  device: GPUDevice,
  source: MediaSource,
  options: MediaTextureOptions = {},
): Promise<MediaTexture> {
  const label = options.label ?? 'media';
  const mipmaps = options.mipmaps ?? true;
  const limit = Math.min(options.maxSize ?? 4096, device.limits.maxTextureDimension2D);

  if (isVideo(source)) return createVideoMedia(device, source, label, mipmaps, options.onDestroy);

  if (isImageElement(source) && !source.complete) await source.decode();
  const size = sizeOf(source);
  if (size.width === 0 || size.height === 0) {
    throw new Error(`createMediaTexture: ${label} has no pixels (an <img> must have loaded)`);
  }
  const fitted = limitSize(size.width, size.height, limit);
  let bitmap: ImageBitmap | undefined;
  let upload: MediaSource = source;
  if (isImageElement(source) || fitted.scaled) {
    bitmap = await decodeToBitmap(source as ImageBitmapSource, fitted.scaled ? fitted : undefined);
    upload = bitmap;
  }
  try {
    const texture = createTextureFromSource(device, upload, { label, mipmaps, srgb: true });
    return stillMedia(texture, options.onDestroy);
  } finally {
    // The copy has captured the pixels already.
    bitmap?.close();
  }
}

/** Wraps a texture you made yourself (a procedural scene rendered once) as a still `MediaTexture`. */
export function mediaFromTexture(texture: GPUTexture, onDestroy?: () => void): MediaTexture {
  return stillMedia(texture, onDestroy);
}

function stillMedia(texture: GPUTexture, onDestroy?: () => void): MediaTexture {
  const view = texture.createView({ label: `${texture.label} view` });
  let destroyed = false;
  return {
    kind: 'still',
    texture,
    view,
    width: texture.width,
    height: texture.height,
    aspect: texture.width / texture.height,
    mipLevelCount: texture.mipLevelCount,
    version: 0,
    update: () => false,
    destroy() {
      if (destroyed) return;
      destroyed = true;
      texture.destroy();
      onDestroy?.();
    },
  };
}

function createVideoMedia(
  device: GPUDevice,
  video: HTMLVideoElement,
  label: string,
  mipmaps: boolean,
  onDestroy?: () => void,
): MediaTexture {
  if (video.videoWidth === 0 || video.videoHeight === 0) {
    throw new Error(
      'createMediaTexture: the video has no size yet; await whenVideoReady(video) first',
    );
  }
  const frameCallbackSupported = typeof video.requestVideoFrameCallback === 'function';
  let signalled = false;
  let callbackId: number | undefined;
  const arm = () => {
    callbackId = video.requestVideoFrameCallback(() => {
      signalled = true;
      arm();
    });
  };
  if (frameCallbackSupported) arm();

  let texture = createTextureFromSource(device, video, { label, mipmaps, srgb: true });
  let view = texture.createView({ label: `${label} view` });
  let version = 0;
  let lastUploadedTime: number | undefined = video.currentTime;
  let stage: HTMLCanvasElement | undefined;
  let destroyed = false;

  /** copyExternalImageToTexture straight from the video; if a browser refuses that, via a 2D canvas. */
  function upload(encoder: GPUCommandEncoder) {
    if (!stage) {
      try {
        updateTextureFromSource(device, texture, video, { encoder });
        return;
      } catch (error) {
        if (typeof document === 'undefined') throw error;
        stage = document.createElement('canvas');
      }
    }
    stage.width = video.videoWidth;
    stage.height = video.videoHeight;
    stage.getContext('2d')?.drawImage(video, 0, 0);
    updateTextureFromSource(device, texture, stage, { encoder });
  }

  const handle: MediaTexture = {
    kind: 'video',
    get texture() {
      return texture;
    },
    get view() {
      return view;
    },
    get width() {
      return texture.width;
    },
    get height() {
      return texture.height;
    },
    get aspect() {
      return texture.width / texture.height;
    },
    get mipLevelCount() {
      return texture.mipLevelCount;
    },
    get version() {
      return version;
    },
    update(encoder) {
      if (destroyed) return false;
      const due = videoFrameDue({
        readyState: video.readyState,
        width: video.videoWidth,
        height: video.videoHeight,
        currentTime: video.currentTime,
        lastUploadedTime,
        frameCallbackSupported,
        frameSignalled: signalled,
      });
      if (!due) return false;
      signalled = false;
      lastUploadedTime = video.currentTime;
      if (video.videoWidth !== texture.width || video.videoHeight !== texture.height) {
        // Adaptive streams change resolution mid-play: allocate again and tell the caller.
        texture.destroy();
        texture = createTextureFromSource(device, video, { label, mipmaps, srgb: true });
        view = texture.createView({ label: `${label} view` });
        version++;
        return true;
      }
      upload(encoder);
      return true;
    },
    destroy() {
      if (destroyed) return;
      destroyed = true;
      if (callbackId !== undefined) video.cancelVideoFrameCallback(callbackId);
      texture.destroy();
      onDestroy?.();
    },
  };
  return handle;
}

/**
 * An image or video chosen by the visitor (a dropped or picked `File`). Everything stays in the page:
 * the file is read through an object URL or `createImageBitmap`, nothing is uploaded anywhere. A video
 * is muted, looped and played inline; `destroy()` pauses it and revokes the URL.
 */
export async function createMediaFromFile(
  device: GPUDevice,
  file: File,
  options: MediaTextureOptions = {},
): Promise<MediaTexture> {
  const kind = classifyFile(file);
  if (!kind) throw new Error(`${file.name || 'that file'} is not an image or a video`);
  const label = options.label ?? file.name ?? 'dropped file';

  if (kind === 'image') {
    if (typeof createImageBitmap !== 'function') {
      throw new Error('createMediaFromFile needs a browser (createImageBitmap)');
    }
    const bitmap = await decodeToBitmap(file);
    try {
      return await createMediaTexture(device, bitmap, { ...options, label });
    } finally {
      bitmap.close();
    }
  }

  if (typeof document === 'undefined' || typeof URL.createObjectURL !== 'function') {
    throw new Error('createMediaFromFile needs a browser to play a video');
  }
  const url = URL.createObjectURL(file);
  const video = document.createElement('video');
  video.muted = true;
  video.loop = true;
  video.playsInline = true;
  video.preload = 'auto';
  video.src = url;
  const release = () => {
    video.pause();
    video.removeAttribute('src');
    video.load();
    URL.revokeObjectURL(url);
  };
  try {
    await whenVideoReady(video);
    await video.play();
    return await createMediaTexture(device, video, {
      ...options,
      label,
      onDestroy: () => {
        release();
        options.onDestroy?.();
      },
    });
  } catch (error) {
    release();
    throw error;
  }
}

import { createTextureFromSource } from '../texture.ts';
/** Stable path: only image/video/canvas pixels accepted by copyExternalImageToTexture. */
export function uploadBackdrop(device: GPUDevice, source: GPUCopyExternalImageSource): GPUTexture {
  return createTextureFromSource(device, source, { label: 'Uploaded UI backdrop', srgb: true });
}
/** A capability probe only; experimental method presence is not permission to capture any DOM. */
export function htmlCanvasCaptureSupport(queue: object): 'draw' | 'legacy-copy' | 'unavailable' {
  const candidate = queue as {
    drawElementImageToTexture?: unknown;
    copyElementImageToTexture?: unknown;
  };
  if (typeof candidate.drawElementImageToTexture === 'function') return 'draw';
  if (typeof candidate.copyElementImageToTexture === 'function') return 'legacy-copy';
  return 'unavailable';
}
/**
 * Rasterise a self-contained XHTML fragment. Callers supply inline CSS/fonts/images; this does
 * not clone computed styles or capture a live page. Browser security may reject the decode/upload.
 */
export async function rasteriseForeignObject(
  xhtml: string,
  width: number,
  height: number,
): Promise<ImageBitmap> {
  const w = Math.max(1, Math.round(width)),
    h = Math.max(1, Math.round(height));
  const svg = `<svg xmlns="http://www.w3.org/2000/svg" width="${w}" height="${h}"><foreignObject width="100%" height="100%"><div xmlns="http://www.w3.org/1999/xhtml">${xhtml}</div></foreignObject></svg>`;
  return createImageBitmap(new Blob([svg], { type: 'image/svg+xml' }));
}

import { createTextureFromData } from '../texture.ts';
import { ATLAS_GZIP } from './atlas-data.ts';
export const ATLAS_WIDTH = 384,
  ATLAS_HEIGHT = 144;
let decoded: Promise<Uint8Array<ArrayBuffer>> | undefined;
export function decodeAtlas(): Promise<Uint8Array<ArrayBuffer>> {
  decoded ??= (async () => {
    const encoded = Uint8Array.from(atob(ATLAS_GZIP), (c) => c.charCodeAt(0));
    const stream = new Blob([encoded]).stream().pipeThrough(new DecompressionStream('gzip'));
    const data = new Uint8Array(await new Response(stream).arrayBuffer());
    if (data.length !== ATLAS_WIDTH * ATLAS_HEIGHT * 4) throw new Error('Corrupt UI MSDF atlas');
    return data;
  })();
  return decoded;
}
export async function createTextAtlas(device: GPUDevice) {
  return createTextureFromData(device, {
    label: 'Anton MSDF (linear data)',
    width: ATLAS_WIDTH,
    height: ATLAS_HEIGHT,
    data: await decodeAtlas(),
  });
}

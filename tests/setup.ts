import { create, globals } from 'webgpu';
// Some tests build shaders from raw text with `#include`, so register Water Temple's shared WGSL
// library with the extracted temple core (production shaders carry their own chunks).
import { setShaderLibrary } from '@mk7s/hologpu-temple-core/gpu/shader';
import { getChunk } from '../src/wgsl/index.ts';
setShaderLibrary(getChunk);

// Dawn's Node bindings stand in for the browser's WebGPU implementation, so GPU code is
// validated by a real implementation without launching a browser.
Object.assign(globalThis, globals);
Object.defineProperty(globalThis.navigator, 'gpu', { value: create([]), configurable: true });

import { create, globals } from 'webgpu';
// Register Water Temple's shared WGSL library with the extracted temple core, so shaders that
// use `#include` resolve (the shared package has Fire Temple's chunks, not Water's).
import '../src/gpu/shader.ts';

// Dawn's Node bindings stand in for the browser's WebGPU implementation, so GPU code is
// validated by a real implementation without launching a browser.
Object.assign(globalThis, globals);
Object.defineProperty(globalThis.navigator, 'gpu', { value: create([]), configurable: true });

import { create, globals } from 'webgpu';

// Dawn's Node bindings stand in for the browser's WebGPU implementation, so GPU code is
// validated by a real implementation without launching a browser.
Object.assign(globalThis, globals);
Object.defineProperty(globalThis.navigator, 'gpu', { value: create([]), configurable: true });

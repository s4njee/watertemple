import { createGpuSession } from './gpu/device.ts';
import { setShaderLibrary } from '@mk7s/hologpu-temple-core/gpu/shader';
import { getChunk } from './wgsl/index.ts';
import './gpu/shader-overlay.css';
import { mountApp } from './shell/app.ts';

// Shaders assembled from TypeScript text still resolve `#include` at run time from the shared
// library; effect shaders carry their own chunks (see vite-plugin-wgsl.ts).
setShaderLibrary(getChunk);

if (import.meta.env.DEV) {
  const { installShaderOverlay } = await import('./gpu/shader-overlay.ts');
  installShaderOverlay();
}

// A single device is shared by every canvas on the page.
const session = await createGpuSession();
mountApp(document.querySelector<HTMLElement>('#app')!, { session });

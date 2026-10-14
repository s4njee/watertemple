import { createGpuSession } from './gpu/device.ts';
// Registers Water Temple's WGSL library with the extracted temple core (see src/gpu/shader.ts).
import './gpu/shader.ts';
import './gpu/shader-overlay.css';
import { mountApp } from './shell/app.ts';

if (import.meta.env.DEV) {
  const { installShaderOverlay } = await import('./gpu/shader-overlay.ts');
  installShaderOverlay();
}

// A single device is shared by every canvas on the page.
const session = await createGpuSession();
mountApp(document.querySelector<HTMLElement>('#app')!, { session });

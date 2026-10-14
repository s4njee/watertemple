import { createGpuSession } from './gpu/device.ts';
import { mountApp } from './shell/app.ts';

if (import.meta.env.DEV) {
  const { installShaderOverlay } = await import('./gpu/shader-overlay.ts');
  installShaderOverlay();
}

// A single device is shared by every canvas on the page.
const session = await createGpuSession();
mountApp(document.querySelector<HTMLElement>('#app')!, { session });

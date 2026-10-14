import type { GpuSession } from '../gpu/device.ts';

/** What every shell component is handed. */
export interface ShellContext {
  /** The one device shared by every canvas on the page. */
  session: GpuSession;
}

/** Removes a mounted component: its DOM, listeners, and GPU resources. */
export type Unmount = () => void;

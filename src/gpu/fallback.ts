import type { GpuFailureReason } from './device.ts';
import './fallback.css';

export interface FallbackOptions {
  reason: GpuFailureReason;
  /** Technical detail, e.g. GpuFailure.message. Shown folded away, as text. */
  message?: string;
  /** Still image of the effect, shown behind the notice. */
  posterUrl?: string;
}

export interface FallbackCopy {
  title: string;
  body: string;
}

const COPY: Record<GpuFailureReason, FallbackCopy> = {
  unsupported: {
    title: 'WebGPU is not available in this browser',
    body: 'These effects run on WebGPU. Try an up-to-date Chrome, Edge, Safari or Firefox, and check that hardware acceleration is turned on.',
  },
  'no-adapter': {
    title: 'No compatible graphics adapter was found',
    body: 'Your browser supports WebGPU but could not get a GPU to use. This happens when hardware acceleration is off, the GPU is blocklisted, or its drivers are out of date.',
  },
  'device-failed': {
    title: 'The graphics device could not be started',
    body: 'A GPU was found but WebGPU could not set it up. Closing other graphics-heavy tabs and reloading the page often helps.',
  },
  'device-lost': {
    title: 'The graphics device was lost',
    body: 'The GPU stopped responding or was reset, which can follow a driver update, sleep, or running out of memory. Reload the page to try again.',
  },
};

export function fallbackCopy(reason: GpuFailureReason): FallbackCopy {
  return COPY[reason];
}

/** Returns a function that removes the notice again. */
export function showFallback(container: HTMLElement, options: FallbackOptions): () => void {
  const doc = container.ownerDocument;
  const copy = fallbackCopy(options.reason);

  function el(tag: string, className: string, text?: string): HTMLElement {
    const node = doc.createElement(tag);
    node.className = className;
    if (text !== undefined) node.textContent = text;
    return node;
  }

  const root = el('div', 'wt-fallback');
  root.setAttribute('role', 'status');
  root.setAttribute('data-reason', options.reason);

  if (options.posterUrl) {
    const poster = doc.createElement('img');
    poster.className = 'wt-fallback__poster';
    poster.src = options.posterUrl;
    // The notice carries the meaning; the poster is decoration.
    poster.alt = '';
    root.append(poster);
  }

  const panel = el('div', 'wt-fallback__panel');
  panel.append(el('p', 'wt-fallback__title', copy.title), el('p', 'wt-fallback__body', copy.body));

  if (options.message) {
    const details = el('details', 'wt-fallback__details');
    details.append(
      el('summary', 'wt-fallback__summary', 'Technical details'),
      el('p', 'wt-fallback__detail', options.message),
    );
    panel.append(details);
  }

  root.append(panel);
  container.append(root);
  return () => root.remove();
}

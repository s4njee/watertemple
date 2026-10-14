import type { Unmount } from './types.ts';

/**
 * Mounts a component from a dynamically imported module. Heavy panels (controls, highlighter,
 * markdown) stay out of the first load, and one that fails to load shows a note in its own slot
 * instead of taking the page down with it.
 */
export function lazyMount(
  container: HTMLElement,
  name: string,
  load: () => Promise<Unmount>,
): Unmount {
  let unmount: Unmount | undefined;
  let cancelled = false;

  load().then(
    (mounted) => {
      if (cancelled) mounted();
      else unmount = mounted;
    },
    (error: unknown) => {
      console.error(`[watertemple] could not load ${name}:`, error);
      if (cancelled) return;
      const note = document.createElement('p');
      note.className = 'wt-muted';
      note.textContent = `The ${name} panel failed to load.`;
      container.append(note);
      unmount = () => note.remove();
    },
  );

  return () => {
    cancelled = true;
    unmount?.();
  };
}

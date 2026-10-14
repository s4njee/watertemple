/** Observe whether a host should animate. A first static frame is still drawn by the host. */
export function observeMotionEligibility(
  element: HTMLElement,
  changed: (eligible: boolean) => void,
): () => void {
  const doc = element.ownerDocument;
  const owner = doc.defaultView;
  const reduced = owner?.matchMedia?.('(prefers-reduced-motion: reduce)');
  let inView = true;
  let previous: boolean | undefined;
  const update = () => {
    const eligible = !doc.hidden && inView && !reduced?.matches;
    if (eligible !== previous) {
      previous = eligible;
      changed(eligible);
    }
  };
  const observer = owner?.IntersectionObserver
    ? new owner.IntersectionObserver((entries) => {
        for (const entry of entries) if (entry.target === element) inView = entry.isIntersecting;
        update();
      })
    : undefined;
  observer?.observe(element);
  doc.addEventListener('visibilitychange', update);
  reduced?.addEventListener('change', update);
  update();
  return () => {
    observer?.disconnect();
    doc.removeEventListener('visibilitychange', update);
    reduced?.removeEventListener('change', update);
  };
}

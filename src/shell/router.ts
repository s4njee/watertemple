// The whole app state lives in the query string, on one HTML file, so every view is a
// shareable link and the site works on any static host without rewrite rules.
//
// Reserved keys: effect, tab, layout, hud, hdr, tier, tag. Effect parameters use the `p.` prefix.

export type QueryChanges = Readonly<Record<string, string | null | undefined>>;

const listeners = new Set<(query: URLSearchParams) => void>();

/** Pure: `null` or `undefined` removes a key, anything else sets it. */
export function applyQuery(search: string, changes: QueryChanges): string {
  const query = new URLSearchParams(search);
  for (const [key, value] of Object.entries(changes)) {
    if (value === null || value === undefined) query.delete(key);
    else query.set(key, value);
  }
  const text = query.toString();
  return text ? `?${text}` : '';
}

export function readQuery(): URLSearchParams {
  return new URLSearchParams(location.search);
}

/** A real URL for `<a href>`, so links can be opened in a new tab. */
export function hrefFor(changes: QueryChanges, from: string = location.search): string {
  return location.pathname + applyQuery(from, changes);
}

function emit() {
  const query = readQuery();
  for (const listener of [...listeners]) listener(query);
}

export interface NavigateOptions {
  /** Replace the history entry instead of adding one. Use for continuous changes such as sliders. */
  replace?: boolean;
  /** Skip notifying listeners: the caller already reflects the change. */
  silent?: boolean;
}

export function updateQuery(changes: QueryChanges, options: NavigateOptions = {}): void {
  const url = hrefFor(changes);
  if (url === location.pathname + location.search) return;
  if (options.replace) history.replaceState(null, '', url);
  else history.pushState(null, '', url);
  if (!options.silent) emit();
}

/** Goes to a link produced by `hrefFor`, dropping every key the link does not carry. */
export function navigate(href: string): void {
  if (href === location.pathname + location.search) return;
  history.pushState(null, '', href);
  emit();
}

export function onQueryChange(listener: (query: URLSearchParams) => void): () => void {
  listeners.add(listener);
  return () => listeners.delete(listener);
}

/**
 * Makes same-origin `<a data-link>` clicks navigate in place. Modified clicks (new tab, download)
 * are left to the browser.
 */
export function startRouter(): void {
  addEventListener('popstate', emit);
  addEventListener('click', (event) => {
    if (event.defaultPrevented || event.button !== 0) return;
    if (event.metaKey || event.ctrlKey || event.shiftKey || event.altKey) return;
    const link = (event.target as Element | null)?.closest?.('a[data-link]');
    if (!(link instanceof HTMLAnchorElement) || link.origin !== location.origin) return;
    event.preventDefault();
    navigate(link.pathname + link.search);
  });
}

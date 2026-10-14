import { Marked } from 'marked';
import { highlightCode, languageOf, plainCode } from './highlight.ts';
import type { Unmount } from './types.ts';
import './notes.css';

const globbed = import.meta.glob<string>('../effects/*/notes.md', {
  query: '?raw',
  import: 'default',
});

interface NotesStore {
  notes: typeof globbed;
  listeners: Set<() => void>;
}

// Editing a write-up refreshes the open one instead of reloading the page. See source-viewer.ts
// for why this module accepts its own updates and keeps the loaders in `hot.data`.
const previous = import.meta.hot?.data.notesStore as NotesStore | undefined;
const store: NotesStore = previous ?? { notes: globbed, listeners: new Set() };
store.notes = globbed;
if (import.meta.hot) {
  import.meta.hot.data.notesStore = store;
  import.meta.hot.accept();
  if (previous) for (const listener of [...store.listeners]) listener();
}

// The markdown is written in this repository, so its HTML is trusted and inserted as is.
const markdown = new Marked({ gfm: true });

export function isExternalHref(href: string): boolean {
  return /^(?:[a-z][a-z0-9+.-]*:)?\/\//i.test(href);
}

function enhance(root: HTMLElement): void {
  for (const link of root.querySelectorAll<HTMLAnchorElement>('a[href]')) {
    if (!isExternalHref(link.getAttribute('href') ?? '')) continue;
    link.target = '_blank';
    link.rel = 'noopener noreferrer';
  }

  for (const block of root.querySelectorAll<HTMLElement>('pre > code')) {
    const pre = block.parentElement!;
    const code = block.textContent ?? '';
    const language = languageOf(/language-(\S+)/.exec(block.className)?.[1] ?? '');
    pre.className = 'wt-source__code';
    pre.tabIndex = 0;
    block.className = '';
    block.replaceChildren(plainCode(code));
    if (!language) continue;
    highlightCode(code, language).then(
      (lines) => block.isConnected && block.replaceChildren(lines),
      (error: unknown) => console.warn('[watertemple] syntax highlighting unavailable:', error),
    );
  }
}

export function mountNotes(container: HTMLElement, effectId: string): Unmount {
  const article = document.createElement('article');
  article.className = 'wt-notes wt-prose';
  article.setAttribute('aria-label', 'Write-up');
  container.append(article);

  let disposed = false;
  function render() {
    const load = store.notes[`../effects/${effectId}/notes.md`];
    if (!load) {
      const note = document.createElement('p');
      note.className = 'wt-muted';
      note.textContent = `No write-up yet. Add src/effects/${effectId}/notes.md following docs/effect-notes-template.md.`;
      article.replaceChildren(note);
      return;
    }
    load().then(
      (text) => {
        if (disposed) return;
        article.innerHTML = markdown.parse(text, { async: false });
        enhance(article);
      },
      (error: unknown) => {
        console.error('[watertemple] could not load the write-up:', error);
        if (!disposed) article.textContent = 'The write-up could not be loaded.';
      },
    );
  }

  render();
  store.listeners.add(render);
  return () => {
    disposed = true;
    store.listeners.delete(render);
    article.remove();
  };
}

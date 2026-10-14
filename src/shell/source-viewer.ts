import { getCompiledShader, onCompiledShadersChanged } from '../gpu/shader.ts';
import { getChunk } from '../wgsl/index.ts';
import { weatherSourceFiles } from '../gpu/weather/sources.ts';
import { uiSourceFiles } from '../gpu/ui/sources.ts';
import { oceanSourceFiles } from '../gpu/ocean/sources.ts';
import { preprocess, PreprocessError } from '../wgsl/preprocess.ts';
import { highlightCode, languageOf, plainCode, splitLines } from './highlight.ts';
import type { Unmount } from './types.ts';
import './source-viewer.css';

const globbed = import.meta.glob<string>(
  [
    '../effects/*/*.{ts,wgsl}',
    '../gpu/ocean/*.{ts,wgsl}',
    '../gpu/ui/*.{ts,wgsl}',
    '../gpu/weather/*.{ts,wgsl}',
  ],
  {
    query: '?raw',
    import: 'default',
  },
);

interface SourceStore {
  sources: typeof globbed;
  listeners: Set<() => void>;
}

// An edit to an effect's file reaches this module too (it imports the text), so it accepts the
// update itself; otherwise Vite would reload the page and shader hot reload would be lost. Other
// modules keep functions from the first copy, so the live loaders sit in `hot.data`, which every
// copy shares, and mounted viewers refresh through its listeners.
const previous = import.meta.hot?.data.sourceStore as SourceStore | undefined;
const store: SourceStore = previous ?? { sources: globbed, listeners: new Set() };
store.sources = globbed;
if (import.meta.hot) {
  import.meta.hot.data.sourceStore = store;
  import.meta.hot.accept();
  if (previous) for (const listener of [...store.listeners]) listener();
}

export interface EffectFile {
  /** Base name, e.g. "hello.wgsl". */
  name: string;
  /** Key into the glob of lazy loaders. */
  path: string;
}

/** Shaders first (they are the point), then the effect, other TypeScript, and metadata last. */
export function fileRank(name: string): number {
  if (name.endsWith('.wgsl')) return 0;
  if (name === 'index.ts') return 1;
  return name === 'meta.ts' ? 3 : 2;
}

/** The source files of effect `id` among glob keys like "../effects/hello/index.ts", in display order. */
export function effectFiles(paths: readonly string[], id: string): EffectFile[] {
  const prefix = `../effects/${id}/`;
  return paths
    .filter((path) => path.startsWith(prefix) && !path.slice(prefix.length).includes('/'))
    .map((path) => ({ name: path.slice(prefix.length), path }))
    .sort((a, b) => fileRank(a.name) - fileRank(b.name) || a.name.localeCompare(b.name));
}

/**
 * Every file `source` pulls in through `#include`, transitively, in order of first inclusion.
 * `lookup` supplies chunk text; a name it cannot resolve is still listed, but nothing below it is.
 */
export function extractIncludes(
  source: string,
  lookup: (name: string) => string | undefined = getChunk,
): string[] {
  const names: string[] = [];
  try {
    preprocess(source, '', (name) => {
      names.push(name);
      return lookup(name) ?? '';
    });
  } catch (error) {
    // A malformed directive ends the list; the compile overlay reports it properly.
    if (!(error instanceof PreprocessError)) throw error;
  }
  return names;
}

export function lineCountText(text: string): string {
  const count = splitLines(text).length;
  return `${count} ${count === 1 ? 'line' : 'lines'}`;
}

type Detail = { kind: 'chunk'; name: string } | { kind: 'expanded'; label: string };

interface View {
  title: string;
  note?: string;
  text: string;
  language: ReturnType<typeof languageOf>;
}

function el<K extends keyof HTMLElementTagNameMap>(
  tag: K,
  className: string,
  text?: string,
): HTMLElementTagNameMap[K] {
  const node = document.createElement(tag);
  node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
}

function groupButton(label: string, key: string): HTMLButtonElement {
  const node = el('button', 'wt-button wt-source__button', label);
  node.type = 'button';
  node.dataset.key = key;
  return node;
}

async function copyText(text: string): Promise<boolean> {
  try {
    await navigator.clipboard.writeText(text);
    return true;
  } catch {
    // Plain-http pages (a dev server on a LAN address) have no async clipboard.
    const area = document.createElement('textarea');
    area.value = text;
    area.setAttribute('readonly', '');
    area.style.position = 'fixed';
    area.style.opacity = '0';
    document.body.append(area);
    area.select();
    try {
      return document.execCommand('copy');
    } catch {
      return false;
    } finally {
      area.remove();
    }
  }
}

export function mountSourceViewer(container: HTMLElement, effectId: string): Unmount {
  const root = el('section', 'wt-source');
  root.setAttribute('aria-label', 'Source code');
  container.append(root);

  const files = [
    ...oceanSourceFiles(Object.keys(store.sources), effectId),
    ...uiSourceFiles(Object.keys(store.sources), effectId),
    ...weatherSourceFiles(Object.keys(store.sources), effectId),
    ...effectFiles(Object.keys(store.sources), effectId),
  ];
  if (files.length === 0) {
    root.append(el('p', 'wt-muted', 'This effect has no source files to show.'));
    return () => root.remove();
  }

  let file = files[0]!;
  let detail: Detail | undefined;
  let shown: { title: string; text: string } | undefined;
  let fileText = '';
  let renderId = 0;
  let copyTimer: ReturnType<typeof setTimeout> | undefined;
  let disposed = false;
  const fileTexts = new Map<string, Promise<string>>();

  const bar = el('div', 'wt-source__bar');
  const fileGroup = el('div', 'wt-source__group');
  fileGroup.setAttribute('role', 'group');
  fileGroup.setAttribute('aria-label', 'Effect files');
  const copyButton = el('button', 'wt-button', 'Copy');
  copyButton.type = 'button';
  bar.append(fileGroup, copyButton);

  const related = el('div', 'wt-source__related');
  related.setAttribute('role', 'group');
  related.setAttribute('aria-label', 'Shader includes and expanded source');

  const caption = el('div', 'wt-source__caption');
  const captionText = el('p', 'wt-data wt-muted');
  const status = el('span', 'wt-source__status');
  status.setAttribute('role', 'status');
  caption.append(captionText, status);

  const pre = el('pre', 'wt-source__code');
  pre.tabIndex = 0;
  const code = el('code', '');
  pre.append(code);

  root.append(bar, related, caption, pre);

  const compiledFor = (name: string) =>
    getCompiledShader(
      name.startsWith('ocean/') || name.startsWith('ui/') || name.startsWith('weather/')
        ? name
        : `${effectId}/${name}`,
    );

  const fileButtons = files.map((candidate) => {
    const node = groupButton(candidate.name, candidate.name);
    node.addEventListener('click', () => {
      file = candidate;
      detail = undefined;
      void update();
    });
    fileGroup.append(node);
    return node;
  });

  function loadFile(target: EffectFile): Promise<string> {
    let text = fileTexts.get(target.path);
    if (!text) {
      text = store.sources[target.path]!();
      fileTexts.set(target.path, text);
    }
    return text;
  }

  async function currentView(): Promise<View> {
    if (detail?.kind === 'expanded') {
      const shader = getCompiledShader(detail.label);
      return {
        title: `Expanded ${detail.label}`,
        note: 'Includes inlined, exactly as sent to the GPU. Self-contained.',
        text: shader?.code ?? '',
        language: 'wgsl',
      };
    }
    if (detail?.kind === 'chunk') {
      const { name } = detail;
      const generated = compiledFor(file.name)?.chunks[name];
      if (generated !== undefined) {
        return {
          title: name,
          note: 'Generated at runtime by the effect; not a file in the repository.',
          text: generated,
          language: 'wgsl',
        };
      }
      return {
        title: `src/wgsl/${name}`,
        note: 'Shared library chunk.',
        text: getChunk(name) ?? '',
        language: 'wgsl',
      };
    }
    return {
      title: file.path.replace(/^\.\.\//, 'src/'),
      text: await loadFile(file),
      language: languageOf(file.name),
    };
  }

  function updateRelated() {
    const focused = related.contains(document.activeElement)
      ? (document.activeElement as HTMLElement).dataset.key
      : undefined;
    related.replaceChildren();
    related.hidden = languageOf(file.name) !== 'wgsl';
    if (related.hidden) return;

    const compiled = compiledFor(file.name);
    const lookup = (name: string) => compiled?.chunks[name] ?? getChunk(name);
    // What was actually compiled wins; before that, read the includes out of the file itself.
    const includes = compiled?.includes ?? extractIncludes(fileText, lookup);

    const add = (
      group: HTMLElement,
      label: string,
      key: string,
      next: Detail | undefined,
      hint: string,
    ) => {
      const button = groupButton(label, key);
      if (next) {
        button.addEventListener('click', () => {
          detail = JSON.stringify(detail) === JSON.stringify(next) ? undefined : next;
          void update();
        });
      } else {
        button.disabled = true;
        button.title = hint;
      }
      group.append(button);
    };

    const rowOf = (label: string) => {
      const group = el('div', 'wt-source__group');
      group.append(el('span', 'wt-label', label));
      related.append(group);
      return group;
    };
    let group = rowOf('Includes');
    for (const name of includes) {
      const viewable = lookup(name) !== undefined;
      add(
        group,
        name,
        `chunk:${name}`,
        viewable ? { kind: 'chunk', name } : undefined,
        'Provided by the effect at runtime; it shows once the shader has compiled.',
      );
    }
    if (includes.length === 0) group.append(el('span', 'wt-muted wt-data', 'none'));
    group = rowOf('Output');
    add(
      group,
      'Expanded shader',
      'expanded',
      compiled && { kind: 'expanded', label: compiled.label },
      'Available once the effect has compiled this shader.',
    );

    if (focused) related.querySelector<HTMLElement>(`[data-key="${CSS.escape(focused)}"]`)?.focus();
  }

  function pressed() {
    for (const node of fileButtons) {
      node.setAttribute('aria-pressed', String(node.dataset.key === file.name && !detail));
    }
    for (const node of related.querySelectorAll<HTMLButtonElement>('button')) {
      const key = node.dataset.key ?? '';
      const on =
        (detail?.kind === 'chunk' && key === `chunk:${detail.name}`) ||
        (detail?.kind === 'expanded' && key === 'expanded');
      node.setAttribute('aria-pressed', String(on));
    }
  }

  function paint(view: View, id: number) {
    captionText.textContent = `${view.title} · ${lineCountText(view.text)}${view.note ? ` · ${view.note}` : ''}`;
    copyButton.disabled = view.text === '';
    if (shown?.title === view.title && shown.text === view.text) return;
    if (shown?.title !== view.title) pre.scrollLeft = 0;
    shown = { title: view.title, text: view.text };
    pre.setAttribute('aria-label', view.title);
    pre.style.setProperty(
      '--wt-source-digits',
      String(String(splitLines(view.text).length).length),
    );
    code.replaceChildren(plainCode(view.text));
    if (!view.language) return;
    highlightCode(view.text, view.language).then(
      (lines) => {
        if (id === renderId && !disposed) code.replaceChildren(lines);
      },
      (error: unknown) => console.warn('[watertemple] syntax highlighting unavailable:', error),
    );
  }

  async function update() {
    const id = ++renderId;
    pressed();
    const view = await currentView();
    if (id !== renderId || disposed) return;
    if (!detail) fileText = view.text;
    updateRelated();
    pressed();
    paint(view, id);
  }

  copyButton.addEventListener('click', async () => {
    const view = await currentView();
    const ok = await copyText(view.text);
    if (disposed) return;
    clearTimeout(copyTimer);
    const name =
      detail?.kind === 'expanded'
        ? 'expanded shader'
        : detail?.kind === 'chunk'
          ? detail.name
          : file.name;
    status.dataset.state = ok ? 'ok' : 'error';
    status.textContent = ok
      ? `Copied ${name} (${lineCountText(view.text)}) to the clipboard.`
      : 'Could not copy. Select the code and copy it by hand.';
    copyButton.textContent = ok ? 'Copied' : 'Copy';
    copyTimer = setTimeout(() => {
      status.textContent = '';
      copyButton.textContent = 'Copy';
    }, 3000);
  });

  const stopCompiled = onCompiledShadersChanged(() => void update());
  const refresh = () => {
    fileTexts.clear();
    void update();
  };
  store.listeners.add(refresh);

  void update();

  return () => {
    disposed = true;
    clearTimeout(copyTimer);
    stopCompiled();
    store.listeners.delete(refresh);
    root.remove();
  };
}

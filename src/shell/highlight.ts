import type { HighlighterCore } from 'shiki/core';
import './source-viewer.css';

export type CodeLanguage = 'wgsl' | 'typescript';

const LANGUAGES: Readonly<Record<string, CodeLanguage>> = {
  wgsl: 'wgsl',
  ts: 'typescript',
  typescript: 'typescript',
};

/** Language of a file name ("index.ts") or a markdown fence tag ("wgsl"); undefined if unsupported. */
export function languageOf(nameOrTag: string): CodeLanguage | undefined {
  return LANGUAGES[nameOrTag.trim().toLowerCase().split('.').pop() ?? ''];
}

/** Lines of `code`; a final line break does not start another line. */
export function splitLines(code: string): string[] {
  const lines = code.split(/\r\n|\n|\r/);
  if (lines.length > 1 && lines[lines.length - 1] === '') lines.pop();
  return lines;
}

interface Token {
  content: string;
  style?: Readonly<Record<string, string>>;
}

function lineElement(tokens: readonly Token[]): HTMLElement {
  const line = document.createElement('span');
  line.className = 'wt-source__line';
  for (const token of tokens) {
    if (!token.style) {
      line.append(token.content);
      continue;
    }
    const span = document.createElement('span');
    span.textContent = token.content;
    for (const [name, value] of Object.entries(token.style)) span.style.setProperty(name, value);
    line.append(span);
  }
  return line;
}

/**
 * The code as one block-level span per line, readable immediately. Highlighted output has the
 * same structure, so swapping it in does not move anything.
 */
export function plainCode(code: string): DocumentFragment {
  const fragment = document.createDocumentFragment();
  for (const line of splitLines(code)) fragment.append(lineElement([{ content: line }]));
  return fragment;
}

// Both palettes are emitted as CSS variables and picked in CSS by `light-dark()`, so the site's
// theme toggle recolours code without re-highlighting.
const THEMES = { light: 'github-light-default', dark: 'github-dark-default' } as const;

let highlighter: Promise<HighlighterCore> | undefined;

function loadHighlighter(): Promise<HighlighterCore> {
  highlighter ??= (async () => {
    const [core, engine, light, dark, wgsl, typescript] = await Promise.all([
      import('shiki/core'),
      import('shiki/engine/javascript'),
      import('shiki/themes/github-light-default.mjs'),
      import('shiki/themes/github-dark-default.mjs'),
      import('shiki/langs/wgsl.mjs'),
      import('shiki/langs/typescript.mjs'),
    ]);
    return core.createHighlighterCore({
      themes: [light.default, dark.default],
      langs: [wgsl.default, typescript.default],
      engine: engine.createJavaScriptRegexEngine({ forgiving: true }),
    });
  })();
  highlighter.catch(() => (highlighter = undefined));
  return highlighter;
}

/** Highlighted lines for `code`. Rejects if the highlighter cannot load; callers keep `plainCode`. */
export async function highlightCode(
  code: string,
  language: CodeLanguage,
): Promise<DocumentFragment> {
  const { tokens } = (await loadHighlighter()).codeToTokens(splitLines(code).join('\n'), {
    lang: language,
    themes: THEMES,
    defaultColor: false,
  });
  const fragment = document.createDocumentFragment();
  for (const line of tokens) {
    fragment.append(
      lineElement(line.map((token) => ({ content: token.content, style: token.htmlStyle }))),
    );
  }
  return fragment;
}

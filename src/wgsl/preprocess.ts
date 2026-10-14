/** Where one line of preprocessed output came from. `line` is 1-based. */
export interface SourceLine {
  file: string;
  line: number;
}

export interface Preprocessed {
  code: string;
  /** `lines[i]` is the origin of output line `i + 1`. Output lines are copied verbatim, so columns carry over unchanged. */
  lines: SourceLine[];
}

/** A malformed or unresolvable `#include`; points at the directive in the file that contains it. */
export class PreprocessError extends Error {
  readonly reason: string;
  readonly file: string;
  /** 1-based. */
  readonly line: number;
  /** 1-based column of the `#include`. */
  readonly column: number;
  readonly sourceLine: string;

  constructor(reason: string, file: string, line: number, sourceLine: string) {
    super(`${file}:${line}: ${reason}`);
    this.name = 'PreprocessError';
    this.reason = reason;
    this.file = file;
    this.line = line;
    this.column = Math.max(0, sourceLine.search(/\S/)) + 1;
    this.sourceLine = sourceLine;
  }
}

// These are exactly the line breaks the WGSL spec defines (CRLF counts once, LF+CR twice). Dawn
// numbers its diagnostics with them, so splitting on anything else would skew every mapped line.
const LINE_BREAK = /\r\n|[\n\v\f\r\u0085\u2028\u2029]/;
const INCLUDE_START = /^\s*#include\b/;
const INCLUDE = /^\s*#include\s+"([^"]+)"\s*(?:\/\/.*)?$/;

/** Block-comment nesting depth after `text`. WGSL block comments nest, and a line comment hides the rest of its line. */
function blockDepthAfter(text: string, depth: number): number {
  for (let i = 0; i < text.length; i++) {
    const pair = text.slice(i, i + 2);
    if (pair === '/*') {
      depth++;
      i++;
    } else if (depth > 0 && pair === '*/') {
      depth--;
      i++;
    } else if (depth === 0 && pair === '//') {
      break;
    }
  }
  return depth;
}

/**
 * Expands `#include "name.wgsl"` directives, one per line, flat namespace.
 *
 * - Each file is included at most once, at its first occurrence, because WGSL rejects duplicate
 *   declarations. The root counts as already included, so cycles (and diamonds) are harmless;
 *   WGSL declarations are order-independent, so nothing is lost by it.
 * - A directive is only recognised at the start of a line outside any block comment. `// #include`
 *   and an `#include` inside a block comment (which nests, as in WGSL) are left alone.
 * - A line that starts with `#include` but is not `#include "name"` (optionally followed by a
 *   `//` comment) is an error rather than a confusing "invalid character" from the WGSL compiler.
 * - Lines are split on WGSL's line breaks and rejoined with `\n`; a leading BOM is dropped
 *   because WGSL rejects it.
 */
export function preprocess(
  source: string,
  file: string,
  resolve: (name: string) => string | undefined,
): Preprocessed {
  const out: string[] = [];
  const lines: SourceLine[] = [];
  const included = new Set([file]);

  const visit = (text: string, name: string) => {
    // Comment state is per file: an unterminated comment in a chunk must not hide its parent.
    let depth = 0;
    text
      .replace(/^\uFEFF/, '')
      .split(LINE_BREAK)
      .forEach((lineText, index) => {
        const line = index + 1;
        if (depth === 0 && INCLUDE_START.test(lineText)) {
          const target = INCLUDE.exec(lineText)?.[1];
          if (target === undefined) {
            throw new PreprocessError(
              'malformed #include, expected #include "name.wgsl"',
              name,
              line,
              lineText,
            );
          }
          if (included.has(target)) return;
          const chunk = resolve(target);
          if (chunk === undefined) {
            throw new PreprocessError(`#include "${target}" not found`, name, line, lineText);
          }
          included.add(target);
          visit(chunk, target);
          return;
        }
        depth = blockDepthAfter(lineText, depth);
        out.push(lineText);
        lines.push({ file: name, line });
      });
  };

  visit(source, file);
  return { code: out.join('\n'), lines };
}

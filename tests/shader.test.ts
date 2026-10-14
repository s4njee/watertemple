import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import {
  checkShader,
  clearDiagnostics,
  createShader,
  formatDiagnostic,
  getDiagnostics,
  groupDiagnosticsByFile,
  mapCompilationMessages,
  onDiagnosticsChanged,
  resolveShader,
  type CheckedShader,
  type ShaderDiagnostic,
  type ShaderSource,
} from '../src/gpu/shader.ts';
import { chunkNames, getChunk, onChunksChanged } from '../src/wgsl/index.ts';
import { PreprocessError } from '../src/wgsl/preprocess.ts';
import { captureValidationError, getTestDevice } from './gpu.ts';

const lines = (...text: string[]) => text.join('\n');

const LIB = lines(
  'fn helper() -> f32 {',
  '  return 1.0;',
  '}',
  'fn broken() -> f32 {',
  '  return missing;',
  '}',
);

/** Compiles through the real WebGPU implementation; the validation error is Dawn's own verdict. */
async function check(source: ShaderSource) {
  const device = await getTestDevice();
  let checked!: CheckedShader;
  const validationError = await captureValidationError(device, async () => {
    checked = await checkShader(device, source);
  });
  return { ...checked, validationError };
}

const errorsOf = (diagnostics: ShaderDiagnostic[]) =>
  diagnostics.filter((d) => d.severity === 'error');

function diagnostic(overrides: Partial<ShaderDiagnostic> = {}): ShaderDiagnostic {
  return {
    severity: 'error',
    message: 'unresolved value',
    shader: 'main.wgsl',
    file: 'main.wgsl',
    line: 2,
    column: 10,
    length: 4,
    sourceLine: '  return nope;',
    ...overrides,
  };
}

let errorLog: ReturnType<typeof vi.spyOn>;
let warnLog: ReturnType<typeof vi.spyOn>;

beforeEach(() => {
  clearDiagnostics();
  errorLog = vi.spyOn(console, 'error').mockImplementation(() => {});
  warnLog = vi.spyOn(console, 'warn').mockImplementation(() => {});
});

afterEach(() => {
  vi.restoreAllMocks();
  clearDiagnostics();
});

describe('formatDiagnostic', () => {
  it('prints location, source line and carets under the span', () => {
    expect(formatDiagnostic(diagnostic({ message: "unresolved value 'nope'" }))).toBe(
      lines(
        "main.wgsl:2:10: error: unresolved value 'nope'",
        '2 |   return nope;',
        '  |          ^^^^',
      ),
    );
  });

  it('names the shader when the file is an included chunk', () => {
    const text = formatDiagnostic(diagnostic({ file: 'lib.wgsl', line: 12, column: 3, length: 1 }));
    expect(text).toBe(
      lines(
        'lib.wgsl:12:3: error: unresolved value (while compiling main.wgsl)',
        '12 |   return nope;',
        '   |   ^',
      ),
    );
  });

  it('keeps tabs in the caret indent so it lines up', () => {
    const text = formatDiagnostic(
      diagnostic({ sourceLine: '\t\tx = nope;', column: 7, length: 4 }),
    );
    expect(text.split('\n')[2]).toBe('  | \t\t    ^^^^');
  });

  it('draws one caret for an empty span and clamps a span running past the line', () => {
    expect(formatDiagnostic(diagnostic({ length: 0 })).split('\n')[2]).toBe('  |          ^');
    expect(formatDiagnostic(diagnostic({ length: 99 })).split('\n')[2]).toBe('  |          ^^^^^');
  });

  it('puts a caret just past the end of the line for an end-of-line position', () => {
    const text = formatDiagnostic(diagnostic({ sourceLine: 'fn f() {', column: 9, length: 0 }));
    expect(text.split('\n')[2]).toBe('  |         ^');
  });

  it('prints only the header when the compiler gave no position', () => {
    expect(formatDiagnostic(diagnostic({ line: 0, column: 0, length: 0, sourceLine: '' }))).toBe(
      'main.wgsl: error: unresolved value',
    );
  });
});

describe('mapCompilationMessages', () => {
  const preprocessed = {
    code: lines('// root', 'fn a() {', '}', 'fn main() {}'),
    lines: [
      { file: 'root.wgsl', line: 1 },
      { file: 'a.wgsl', line: 7 },
      { file: 'a.wgsl', line: 8 },
      { file: 'root.wgsl', line: 3 },
    ],
  };
  const message = (lineNum: number, linePos: number, length = 2) => ({
    type: 'error' as const,
    message: 'm',
    lineNum,
    linePos,
    length,
  });

  it('points each message at its original file, line and text', () => {
    const [chunk, root] = mapCompilationMessages('root.wgsl', preprocessed, [
      message(2, 4),
      message(4, 1),
    ]);
    expect(chunk).toMatchObject({
      file: 'a.wgsl',
      line: 7,
      column: 4,
      length: 2,
      sourceLine: 'fn a() {',
      shader: 'root.wgsl',
    });
    expect(root).toMatchObject({
      file: 'root.wgsl',
      line: 3,
      column: 1,
      sourceLine: 'fn main() {}',
    });
  });

  it('keeps messages without a position, or beyond the source, on the shader itself', () => {
    const mapped = mapCompilationMessages('root.wgsl', preprocessed, [
      message(0, 0, 0),
      message(99, 1),
    ]);
    for (const d of mapped) {
      expect(d).toMatchObject({ file: 'root.wgsl', line: 0, column: 0, sourceLine: '' });
    }
  });
});

describe('groupDiagnosticsByFile', () => {
  it('groups in order of first appearance and keeps order within a file', () => {
    const a1 = diagnostic({ file: 'a.wgsl', line: 1 });
    const b = diagnostic({ file: 'b.wgsl', line: 1 });
    const a2 = diagnostic({ file: 'a.wgsl', line: 2 });
    expect(groupDiagnosticsByFile([a1, b, a2])).toEqual([
      { file: 'a.wgsl', diagnostics: [a1, a2] },
      { file: 'b.wgsl', diagnostics: [b] },
    ]);
  });
});

describe('compile diagnostics (real compiler)', () => {
  it('reports nothing for a clean shader', async () => {
    const { diagnostics, validationError } = await check({
      label: 'clean.wgsl',
      code: 'fn f() -> f32 { return 1.0; }',
    });
    expect(diagnostics).toEqual([]);
    expect(validationError).toBeNull();
    expect(getDiagnostics()).toEqual([]);
    expect(errorLog).not.toHaveBeenCalled();
  });

  it('locates an error in the top-level file', async () => {
    const { diagnostics, validationError } = await check({
      label: 'main.wgsl',
      code: lines('fn f() -> f32 {', '  return nope;', '}'),
    });
    expect(validationError).not.toBeNull();
    expect(diagnostics).toHaveLength(1);
    expect(diagnostics[0]).toMatchObject({
      severity: 'error',
      shader: 'main.wgsl',
      file: 'main.wgsl',
      line: 2,
      column: 10,
      length: 4,
      sourceLine: '  return nope;',
    });
    expect(diagnostics[0]!.message).toContain('nope');
  });

  it('points an error inside an included chunk at the chunk and its own line', async () => {
    const source: ShaderSource = {
      label: 'main.wgsl',
      code: lines('// root', '#include "lib.wgsl"', 'fn main() {}'),
      chunks: { 'lib.wgsl': LIB },
    };
    // The bug is on line 5 of lib.wgsl but line 6 of what the compiler sees.
    expect(resolveShader(source).code.split('\n')[5]).toBe('  return missing;');

    const { diagnostics } = await check(source);
    expect(diagnostics).toHaveLength(1);
    expect(diagnostics[0]).toMatchObject({
      file: 'lib.wgsl',
      line: 5,
      column: 10,
      shader: 'main.wgsl',
      sourceLine: '  return missing;',
    });
    expect(formatDiagnostic(diagnostics[0]!)).toContain('lib.wgsl:5:10: error:');
  });

  it('maps an error after an include back past the lines the include added', async () => {
    const { diagnostics } = await check({
      label: 'main.wgsl',
      code: lines('#include "lib.wgsl"', 'fn main() -> f32 {', '  return nope;', '}'),
      chunks: { 'lib.wgsl': 'fn helper() -> f32 {\n  return 1.0;\n}' },
    });
    expect(diagnostics).toHaveLength(1);
    // Compiler line 5, original line 3.
    expect(diagnostics[0]).toMatchObject({ file: 'main.wgsl', line: 3, column: 10 });
  });

  it('maps correctly through CRLF files', async () => {
    // The compiler stops at the first error, so each file gets its own compile.
    const chunks = { 'lib.wgsl': 'fn helper() -> f32 {\r\n  return missing;\r\n}\r\n' };
    const inChunk = await check({
      label: 'main.wgsl',
      code: 'const a = 1.0;\r\n#include "lib.wgsl"\r\nfn main() {}\r\n',
      chunks,
    });
    expect(inChunk.diagnostics).toHaveLength(1);
    expect(inChunk.diagnostics[0]).toMatchObject({
      file: 'lib.wgsl',
      line: 2,
      column: 10,
      sourceLine: '  return missing;',
    });

    const inRoot = await check({
      label: 'main.wgsl',
      code: 'const a = 1.0;\r\n#include "lib.wgsl"\r\nfn main() -> f32 {\r\n  return nope;\r\n}\r\n',
      chunks: { 'lib.wgsl': 'fn helper() {}\r\n' },
    });
    expect(inRoot.diagnostics).toHaveLength(1);
    expect(inRoot.diagnostics[0]).toMatchObject({
      file: 'main.wgsl',
      line: 4,
      column: 10,
      sourceLine: '  return nope;',
    });
  });

  it('counts columns in UTF-16 units, like string indices', async () => {
    const code = lines('// 🌊', 'fn f() -> f32 { /* 🌊é */ return nope; }');
    const { diagnostics } = await check({ label: 'main.wgsl', code });
    const [d] = diagnostics;
    expect(d).toMatchObject({ line: 2, column: 34, length: 4 });
    expect(d!.sourceLine.slice(d!.column - 1, d!.column - 1 + d!.length)).toBe('nope');
  });

  it('lines up with the compiler when the source uses other WGSL line breaks', async () => {
    const code = ['fn f() -> f32 {', '  let a = 1.0;', '  return nope;', '}'].join('\u2028');
    const { diagnostics } = await check({ label: 'main.wgsl', code });
    expect(diagnostics[0]).toMatchObject({ line: 3, column: 10, sourceLine: '  return nope;' });
  });

  it('reports a warning without failing validation', async () => {
    const { diagnostics, validationError } = await check({
      label: 'warn.wgsl',
      code: lines('fn f() -> f32 {', '  return 1.0;', '  return 2.0;', '}'),
    });
    expect(validationError).toBeNull();
    expect(errorsOf(diagnostics)).toEqual([]);
    expect(diagnostics).toHaveLength(1);
    expect(diagnostics[0]).toMatchObject({
      severity: 'warning',
      line: 3,
      column: 3,
      length: 6,
      sourceLine: '  return 2.0;',
    });
    expect(getDiagnostics()).toEqual(diagnostics);
    expect(errorLog).not.toHaveBeenCalled();
    expect(warnLog).toHaveBeenCalledOnce();
  });

  it('maps a note in another file onto that file', async () => {
    const { diagnostics } = await check({
      label: 'main.wgsl',
      code: lines('#include "lib.wgsl"', 'fn dup() {}'),
      chunks: { 'lib.wgsl': 'fn dup() {}' },
    });
    // Dawn reports the redeclaration as an error and the first declaration as a note.
    expect(diagnostics.map((d) => [d.severity, d.file, d.line])).toEqual([
      ['error', 'main.wgsl', 2],
      ['info', 'lib.wgsl', 1],
    ]);
  });

  it('keeps a message with no position on the shader, at line 0', async () => {
    const { diagnostics } = await check({
      label: 'main.wgsl',
      code: `fn f() {}\n${String.fromCharCode(0)}`,
    });
    expect(diagnostics).toHaveLength(1);
    expect(diagnostics[0]).toMatchObject({
      severity: 'error',
      file: 'main.wgsl',
      line: 0,
      column: 0,
      sourceLine: '',
    });
    expect(formatDiagnostic(diagnostics[0]!)).toMatch(/^main\.wgsl: error: /);
  });
});

describe('diagnostics store', () => {
  const broken: ShaderSource = {
    label: 'fx/main.wgsl',
    code: lines('#include "lib.wgsl"', 'fn main() {}'),
    chunks: { 'lib.wgsl': LIB },
  };
  const fixed: ShaderSource = {
    ...broken,
    chunks: { 'lib.wgsl': LIB.replace('missing', '2.0') },
  };

  it('records errors from createShader, logs them, and clears them on a clean compile', async () => {
    const device = await getTestDevice();
    const listener = vi.fn();
    const unsubscribe = onDiagnosticsChanged(listener);

    await captureValidationError(device, async () => {
      createShader(device, broken);
      await vi.waitFor(() => expect(getDiagnostics()).toHaveLength(1));
    });
    expect(getDiagnostics()[0]).toMatchObject({
      file: 'lib.wgsl',
      line: 5,
      shader: 'fx/main.wgsl',
    });
    expect(listener).toHaveBeenCalledTimes(1);
    expect(errorLog).toHaveBeenCalledOnce();
    const logged = String(errorLog.mock.calls[0]![0]);
    expect(logged).toContain('fx/main.wgsl');
    expect(logged).toContain('lib.wgsl:5:10: error:');
    expect(logged).toContain('5 |   return missing;');

    const module = createShader(device, fixed);
    expect(module).toBeDefined();
    await vi.waitFor(() => expect(getDiagnostics()).toEqual([]));
    expect(listener).toHaveBeenCalledTimes(2);

    // A clean compile with nothing to clear is not a change.
    await checkShader(device, fixed);
    expect(listener).toHaveBeenCalledTimes(2);

    unsubscribe();
    await captureValidationError(device, () => checkShader(device, broken));
    expect(listener).toHaveBeenCalledTimes(2);
  });

  it('keeps shaders independent: fixing one leaves the other', async () => {
    const other: ShaderSource = { label: 'other.wgsl', code: 'fn f() -> f32 { return nope; }' };
    await check(broken);
    await check(other);
    expect(new Set(getDiagnostics().map((d) => d.shader))).toEqual(
      new Set(['fx/main.wgsl', 'other.wgsl']),
    );

    await check(fixed);
    expect(getDiagnostics().map((d) => d.shader)).toEqual(['other.wgsl']);
  });

  it('hands out a stable snapshot until something changes', async () => {
    await check(broken);
    const first = getDiagnostics();
    expect(getDiagnostics()).toBe(first);
    await check(fixed);
    expect(getDiagnostics()).not.toBe(first);
    expect(first).toHaveLength(1);
  });

  it('ignores the result of a compile that a newer one of the same label superseded', async () => {
    const [stale, latest] = await Promise.all([check(broken), check(fixed)]);
    expect(stale.diagnostics).toHaveLength(1);
    expect(latest.diagnostics).toEqual([]);
    expect(getDiagnostics()).toEqual([]);
    expect(errorLog).not.toHaveBeenCalled();
  });

  it('records a failed #include, throws it, and clears once it is fixed', async () => {
    const device = await getTestDevice();
    const source: ShaderSource = {
      label: 'fx/main.wgsl',
      code: lines('// root', '  #include "nope.wgsl"'),
    };
    expect(() => createShader(device, source)).toThrowError(PreprocessError);
    await expect(checkShader(device, source)).rejects.toBeInstanceOf(PreprocessError);

    expect(getDiagnostics()).toEqual([
      {
        severity: 'error',
        message: '#include "nope.wgsl" not found',
        shader: 'fx/main.wgsl',
        file: 'fx/main.wgsl',
        line: 2,
        column: 3,
        length: '#include "nope.wgsl"'.length,
        sourceLine: '  #include "nope.wgsl"',
      },
    ]);
    expect(errorLog).toHaveBeenCalled();

    await check({ ...source, code: '// root', chunks: {} });
    expect(getDiagnostics()).toEqual([]);
  });
});

describe('shared chunks', () => {
  it('lists the shared library by file name', () => {
    const names = chunkNames();
    expect(names).toEqual([...names].sort());
    for (const name of names) {
      expect(name).toMatch(/^[^/]+\.wgsl$/);
      expect(typeof getChunk(name)).toBe('string');
    }
    expect(getChunk('definitely-not-a-chunk.wgsl')).toBeUndefined();
  });

  it('includes every shared chunk by name, with its own includes expanded', () => {
    for (const name of chunkNames()) {
      const { code, lines: map } = resolveShader({
        label: 'main.wgsl',
        code: `#include "${name}"`,
      });
      const outLines = code.split('\n');
      expect(map).toHaveLength(outLines.length);
      expect(map.some((l) => l.file === name)).toBe(true);
      expect(map.every((l) => l.file !== 'main.wgsl')).toBe(true);
      for (const line of getChunk(name)!.split(/\r?\n/)) {
        if (!/^\s*#include\b/.test(line)) expect(outLines).toContain(line);
      }
      expect(outLines.some((line) => /^\s*#include\b/.test(line))).toBe(false);
    }
  });

  it('prefers an effect-local chunk over a shared one of the same name', () => {
    const { code } = resolveShader({
      label: 'main.wgsl',
      code: '#include "noise.wgsl"',
      chunks: { 'noise.wgsl': '// local noise' },
    });
    expect(code).toBe('// local noise');
  });

  it('fails clearly when neither the effect nor the library has the chunk', () => {
    expect(() =>
      resolveShader({ label: 'main.wgsl', code: '#include "definitely-not-a-chunk.wgsl"' }),
    ).toThrowError('main.wgsl:1: #include "definitely-not-a-chunk.wgsl" not found');
  });

  it('lets listeners unsubscribe', () => {
    const listener = vi.fn();
    const unsubscribe = onChunksChanged(listener);
    unsubscribe();
    expect(listener).not.toHaveBeenCalled();
  });
});

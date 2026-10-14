import { describe, expect, it } from 'vitest';
import { preprocess, PreprocessError } from '../src/wgsl/preprocess.ts';

const run = (source: string, files: Record<string, string> = {}, root = 'root.wgsl') =>
  preprocess(source, root, (name) => files[name]);

const lines = (...text: string[]) => text.join('\n');
const char = (code: number) => String.fromCharCode(code);

/** Every output line must be byte-for-byte the line its `lines` entry points at. */
function expectExactMap(
  result: ReturnType<typeof run>,
  files: Record<string, string>,
  root = 'root.wgsl',
) {
  const outLines = result.code.split('\n');
  expect(result.lines).toHaveLength(outLines.length);
  outLines.forEach((text, i) => {
    const origin = result.lines[i]!;
    const body = origin.file === root ? files[root]! : files[origin.file]!;
    const originLines = body.split(/\r\n|[\n\v\f\r\u0085\u2028\u2029]/);
    expect(originLines[origin.line - 1], `output line ${i + 1}`).toBe(text);
  });
}

describe('preprocess', () => {
  it('passes a file without includes through unchanged', () => {
    const source = lines('fn a() {}', '', 'fn b() {}', '');
    const result = run(source);
    expect(result.code).toBe(source);
    expect(result.lines).toEqual([
      { file: 'root.wgsl', line: 1 },
      { file: 'root.wgsl', line: 2 },
      { file: 'root.wgsl', line: 3 },
      { file: 'root.wgsl', line: 4 },
    ]);
  });

  it('replaces an include with the chunk and maps every line', () => {
    const files = { 'a.wgsl': lines('fn a() {', '  // body', '}') };
    const source = lines('// head', '#include "a.wgsl"', 'fn main() {}');
    const result = run(source, files);
    expect(result.code).toBe(lines('// head', 'fn a() {', '  // body', '}', 'fn main() {}'));
    expect(result.lines).toEqual([
      { file: 'root.wgsl', line: 1 },
      { file: 'a.wgsl', line: 1 },
      { file: 'a.wgsl', line: 2 },
      { file: 'a.wgsl', line: 3 },
      { file: 'root.wgsl', line: 3 },
    ]);
  });

  it('expands nested includes in place', () => {
    const files = {
      'outer.wgsl': lines('// outer 1', '#include "inner.wgsl"', '// outer 3'),
      'inner.wgsl': lines('// inner 1', '// inner 2'),
    };
    const source = lines('#include "outer.wgsl"', '// root 2');
    const result = run(source, files);
    expect(result.code).toBe(
      lines('// outer 1', '// inner 1', '// inner 2', '// outer 3', '// root 2'),
    );
    expect(result.lines.map((l) => `${l.file}:${l.line}`)).toEqual([
      'outer.wgsl:1',
      'inner.wgsl:1',
      'inner.wgsl:2',
      'outer.wgsl:3',
      'root.wgsl:2',
    ]);
    expectExactMap(result, { ...files, 'root.wgsl': source });
  });

  it('includes a diamond dependency once, at its first occurrence', () => {
    const files = {
      'left.wgsl': lines('#include "base.wgsl"', '// left'),
      'right.wgsl': lines('#include "base.wgsl"', '// right'),
      'base.wgsl': '// base',
    };
    const source = lines('#include "left.wgsl"', '#include "right.wgsl"', '#include "base.wgsl"');
    const result = run(source, files);
    expect(result.code).toBe(lines('// base', '// left', '// right'));
    expect(result.lines.map((l) => `${l.file}:${l.line}`)).toEqual([
      'base.wgsl:1',
      'left.wgsl:2',
      'right.wgsl:2',
    ]);
  });

  it('includes a file only once when the same include is repeated', () => {
    const result = run(lines('#include "a.wgsl"', '#include "a.wgsl"', 'x'), { 'a.wgsl': 'a' });
    expect(result.code).toBe(lines('a', 'x'));
  });

  it('survives include cycles, between chunks and back to the root', () => {
    const files = {
      'a.wgsl': lines('#include "b.wgsl"', '// a'),
      'b.wgsl': lines('#include "a.wgsl"', '#include "root.wgsl"', '// b'),
    };
    const source = lines('#include "a.wgsl"', '// root');
    const result = run(source, files);
    expect(result.code).toBe(lines('// b', '// a', '// root'));
    expect(run('#include "self.wgsl"', { 'self.wgsl': '#include "self.wgsl"\n// s' }).code).toBe(
      '// s',
    );
  });

  describe('missing includes', () => {
    it('names the including file and line', () => {
      expect(() => run(lines('// ok', '', '  #include "nope.wgsl"'))).toThrowError(
        'root.wgsl:3: #include "nope.wgsl" not found',
      );
    });

    it('names the chunk when the include is nested', () => {
      const files = { 'a.wgsl': lines('// 1', '// 2', '#include "nope.wgsl"') };
      let error: unknown;
      try {
        run('#include "a.wgsl"', files);
      } catch (caught) {
        error = caught;
      }
      expect(error).toBeInstanceOf(PreprocessError);
      expect(error).toMatchObject({
        file: 'a.wgsl',
        line: 3,
        column: 1,
        sourceLine: '#include "nope.wgsl"',
        reason: '#include "nope.wgsl" not found',
      });
    });

    it('reports the column of an indented directive', () => {
      let error: unknown;
      try {
        run('\t  #include "nope.wgsl"');
      } catch (caught) {
        error = caught;
      }
      expect(error).toMatchObject({ column: 4, line: 1 });
    });

    it('accepts an empty chunk', () => {
      expect(run(lines('a', '#include "empty.wgsl"', 'b'), { 'empty.wgsl': '' }).code).toBe(
        lines('a', '', 'b'),
      );
    });

    it.each([
      '#include noquotes.wgsl',
      '#include <a.wgsl>',
      "#include 'a.wgsl'",
      '#include ""',
      '#include',
    ])('rejects the malformed directive %s', (directive) => {
      expect(() => run(lines('// x', directive), { 'a.wgsl': '' })).toThrowError(
        /^root\.wgsl:2: malformed #include/,
      );
    });

    it('does not mistake a longer word for the directive', () => {
      expect(run('#includes are fine here').code).toBe('#includes are fine here');
    });
  });

  describe('line endings', () => {
    it('handles CRLF in the root and in chunks', () => {
      const files = { 'a.wgsl': 'fn a() {\r\n}\r\n' };
      const source = 'let x = 1;\r\n#include "a.wgsl"\r\nlet y = 2;\r\n';
      const result = run(source, files);
      expect(result.code).toBe('let x = 1;\nfn a() {\n}\n\nlet y = 2;\n');
      expect(result.code).not.toContain('\r');
      expect(result.lines.map((l) => `${l.file}:${l.line}`)).toEqual([
        'root.wgsl:1',
        'a.wgsl:1',
        'a.wgsl:2',
        'a.wgsl:3',
        'root.wgsl:3',
        'root.wgsl:4',
      ]);
      expectExactMap(result, { ...files, 'root.wgsl': source });
    });

    it('splits on every WGSL line break, so line numbers agree with the compiler', () => {
      // Dawn counts each of these as one line break.
      const breaks = ['\n', '\r', '\r\n', '\v', '\f', char(0x85), char(0x2028), char(0x2029)];
      for (const br of breaks) {
        const result = run(`a${br}#include "c.wgsl"${br}b`, { 'c.wgsl': 'c' });
        expect(result.code, JSON.stringify(br)).toBe('a\nc\nb');
        expect(result.lines.at(-1)).toEqual({ file: 'root.wgsl', line: 3 });
      }
    });

    it('counts LF followed by CR as two line breaks, as the compiler does', () => {
      const result = run('a\n\r#include "c.wgsl"\n\rb', { 'c.wgsl': 'c' });
      expect(result.code).toBe('a\n\nc\n\nb');
      expect(result.lines.at(-1)).toEqual({ file: 'root.wgsl', line: 5 });
    });

    it('drops a leading byte order mark from every file', () => {
      const files = { 'a.wgsl': '\uFEFF// a' };
      expect(run('\uFEFF// root\n#include "a.wgsl"', files).code).toBe('// root\n// a');
    });
  });

  describe('comments', () => {
    const files = { 'a.wgsl': '// a' };

    it('does not expand a directive in a line comment', () => {
      for (const text of [
        '// #include "a.wgsl"',
        '  //#include "a.wgsl"',
        'let x = 1; // #include "a.wgsl"',
      ]) {
        expect(run(text, files).code).toBe(text);
      }
    });

    it('does not expand a directive inside a block comment', () => {
      const source = lines('/*', '#include "a.wgsl"', '*/', '#include "a.wgsl"');
      expect(run(source, files).code).toBe(lines('/*', '#include "a.wgsl"', '*/', '// a'));
    });

    it('tracks nested block comments', () => {
      const source = lines(
        '/* outer /* inner */',
        '#include "a.wgsl"',
        'still outer */',
        '#include "a.wgsl"',
      );
      expect(run(source, files).code).toBe(
        lines('/* outer /* inner */', '#include "a.wgsl"', 'still outer */', '// a'),
      );
    });

    it('resumes expanding after the comment closes on the same line as code', () => {
      const source = lines('let a = 1; /* start', 'end */ let b = 2;', '#include "a.wgsl"');
      expect(run(source, files).code).toBe(
        lines('let a = 1; /* start', 'end */ let b = 2;', '// a'),
      );
    });

    it('lets a line comment hide a comment opener', () => {
      const source = lines('// not a block /*', '#include "a.wgsl"');
      expect(run(source, files).code).toBe(lines('// not a block /*', '// a'));
    });

    it('ignores a block opener hiding behind a comment opener inside a block comment', () => {
      // `//` inside a block comment is plain text, so `*/` still closes it.
      const source = lines('/* // */', '#include "a.wgsl"');
      expect(run(source, files).code).toBe(lines('/* // */', '// a'));
    });

    it('does not let `*/` outside a comment swallow a following `/*`', () => {
      const source = lines('let x = a*/*c*/b;', '#include "a.wgsl"');
      expect(run(source, files).code).toBe(lines('let x = a*/*c*/b;', '// a'));
    });

    it('keeps an unterminated comment in a chunk from hiding the includer', () => {
      const result = run(lines('#include "open.wgsl"', '#include "a.wgsl"'), {
        ...files,
        'open.wgsl': '/* never closed',
      });
      expect(result.code).toBe(lines('/* never closed', '// a'));
    });

    it('allows a trailing comment after a directive', () => {
      expect(run('#include "a.wgsl" // shared helper', files).code).toBe('// a');
    });
  });

  it('keeps the map exact through a mixed tree', () => {
    const files: Record<string, string> = {
      'root.wgsl': lines(
        '// root',
        '#include "hash.wgsl"',
        '/* block',
        '#include "skipped.wgsl"',
        '*/',
        '#include "noise.wgsl"',
        '@fragment fn fs() {}',
        '',
      ),
      'hash.wgsl': 'fn hash() {}\r\n// tail\r\n',
      'noise.wgsl': lines('#include "hash.wgsl"', 'fn noise() {', '  hash();', '}'),
      'skipped.wgsl': 'fn skipped() {}',
    };
    const result = run(files['root.wgsl']!, files);
    expect(result.code).not.toContain('fn skipped');
    expectExactMap(result, files);
    expect(result.code.split('\n')).toHaveLength(result.lines.length);
  });
});

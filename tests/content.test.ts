import { afterEach, describe, expect, it } from 'vitest';
import helloWgsl from '../src/effects/hello/hello.wgsl';
import { listEffects, loadEffect, effectPoster } from '../src/effects/registry.ts';
import {
  clearCompiledShaders,
  createShader,
  getCompiledShader,
  getCompiledShaders,
  onCompiledShadersChanged,
  resolveShader,
} from '../src/gpu/shader.ts';
import { languageOf, splitLines } from '../src/shell/highlight.ts';
import { isExternalHref } from '../src/shell/notes.ts';
import {
  budgetText,
  computeText,
  costRows,
  featureStatus,
  formatKb,
  formatMs,
  mobileText,
  NOT_MEASURED,
  resolutionText,
  smooth,
  sumPasses,
  tierLabel,
} from '../src/shell/scorecard.ts';
import {
  effectFiles,
  extractIncludes,
  fileRank,
  lineCountText,
} from '../src/shell/source-viewer.ts';
import { getChunk } from '../src/wgsl/index.ts';
import { getTestDevice } from './gpu.ts';

const lines = (...text: string[]) => text.join('\n');

describe('extractIncludes', () => {
  const chunks: Record<string, string> = {
    'a.wgsl': lines('#include "b.wgsl"', 'fn a() {}'),
    'b.wgsl': 'fn b() {}',
  };
  const lookup = (name: string) => chunks[name];

  it('lists direct includes in order', () => {
    const source = lines('#include "b.wgsl"', '#include "c.wgsl"', 'fn main() {}');
    expect(extractIncludes(source, lookup)).toEqual(['b.wgsl', 'c.wgsl']);
  });

  it('follows includes into chunks and lists each file once', () => {
    const source = lines('#include "a.wgsl"', '#include "b.wgsl"');
    expect(extractIncludes(source, lookup)).toEqual(['a.wgsl', 'b.wgsl']);
  });

  it('ignores directives inside comments', () => {
    const source = lines(
      '// #include "a.wgsl"',
      '/*',
      '#include "b.wgsl"',
      '*/',
      '#include "c.wgsl"',
    );
    expect(extractIncludes(source, lookup)).toEqual(['c.wgsl']);
  });

  it('keeps names it cannot resolve and stops at a malformed directive', () => {
    expect(extractIncludes('#include "gone.wgsl"', lookup)).toEqual(['gone.wgsl']);
    expect(extractIncludes(lines('#include "b.wgsl"', '#include nonsense'), lookup)).toEqual([
      'b.wgsl',
    ]);
  });

  it('finds what the real hello shader pulls in from the shared library', () => {
    const includes = extractIncludes(
      helloWgsl.code,
      (name) => helloWgsl.chunks[name] ?? getChunk(name),
    );
    expect(includes.slice(0, 2)).toEqual(['fullscreen.wgsl', 'fbm.wgsl']);
    expect(includes).toContain('noise.wgsl');
    expect(includes).toContain('params.wgsl');
  });
});

describe('effectFiles', () => {
  const paths = [
    '../effects/hello/meta.ts',
    '../effects/hello/index.ts',
    '../effects/hello/hello.wgsl',
    '../effects/hello/blur.wgsl',
    '../effects/hello/helpers.ts',
    '../effects/hello/deep/nested.ts',
    '../effects/other/index.ts',
    '../effects/hello-two/index.ts',
  ];

  it('keeps only the effect’s own files: shaders, then index, other TypeScript, metadata', () => {
    expect(effectFiles(paths, 'hello').map((file) => file.name)).toEqual([
      'blur.wgsl',
      'hello.wgsl',
      'index.ts',
      'helpers.ts',
      'meta.ts',
    ]);
  });

  it('returns the glob key so the file can be loaded', () => {
    expect(effectFiles(paths, 'other')).toEqual([
      { name: 'index.ts', path: '../effects/other/index.ts' },
    ]);
    expect(effectFiles(paths, 'missing')).toEqual([]);
  });

  it('ranks files', () => {
    expect(
      ['meta.ts', 'x.ts', 'index.ts', 'a.wgsl'].sort((a, b) => fileRank(a) - fileRank(b)),
    ).toEqual(['a.wgsl', 'index.ts', 'x.ts', 'meta.ts']);
  });
});

describe('code helpers', () => {
  it('maps file names and fence tags to a language', () => {
    expect(languageOf('hello.wgsl')).toBe('wgsl');
    expect(languageOf('index.ts')).toBe('typescript');
    expect(languageOf('WGSL')).toBe('wgsl');
    expect(languageOf('ts')).toBe('typescript');
    expect(languageOf('notes.md')).toBeUndefined();
    expect(languageOf('')).toBeUndefined();
  });

  it('splits lines without inventing one after a final line break', () => {
    expect(splitLines('a\nb\n')).toEqual(['a', 'b']);
    expect(splitLines('a\r\nb')).toEqual(['a', 'b']);
    expect(splitLines('')).toEqual(['']);
    expect(splitLines('a\n\n')).toEqual(['a', '']);
    expect(lineCountText('a\nb\n')).toBe('2 lines');
    expect(lineCountText('a')).toBe('1 line');
  });

  it('opens only absolute links in a new tab', () => {
    expect(isExternalHref('https://example.com/x')).toBe(true);
    expect(isExternalHref('//example.com')).toBe(true);
    expect(isExternalHref('?effect=hello')).toBe(false);
    expect(isExternalHref('#pitfalls')).toBe(false);
    expect(isExternalHref('../docs/a.md')).toBe(false);
  });
});

describe('compiled shader registry', () => {
  afterEach(() => clearCompiledShaders());

  const source = {
    label: 'fx/main.wgsl',
    code: lines(
      '#include "gen.wgsl"',
      '@fragment fn fs() -> @location(0) vec4f { return vec4f(K); }',
    ),
    chunks: { 'gen.wgsl': 'const K: f32 = 1.0;' },
  };

  it('remembers exactly what was compiled, includes inlined', async () => {
    const device = await getTestDevice();
    createShader(device, source);
    const entry = getCompiledShader('fx/main.wgsl');
    expect(entry?.code).toBe(resolveShader(source).code);
    expect(entry?.code).toContain('const K: f32 = 1.0;');
    expect(entry?.code).not.toContain('#include');
    expect(entry?.includes).toEqual(['gen.wgsl']);
    expect(entry?.chunks).toEqual({ 'gen.wgsl': 'const K: f32 = 1.0;' });
  });

  it('lists shared library includes transitively, without marking them effect-local', async () => {
    const device = await getTestDevice();
    createShader(device, { label: 'fx/noise.wgsl', code: '#include "fbm.wgsl"' });
    const entry = getCompiledShader('fx/noise.wgsl');
    expect(entry?.includes).toEqual(expect.arrayContaining(['fbm.wgsl', 'noise.wgsl']));
    expect(entry?.chunks).toEqual({});
  });

  it('keeps one entry per label, replaced by the newest compile', async () => {
    const device = await getTestDevice();
    createShader(device, source);
    createShader(device, { ...source, chunks: { 'gen.wgsl': 'const K: f32 = 2.0;' } });
    expect(getCompiledShaders()).toHaveLength(1);
    expect(getCompiledShader('fx/main.wgsl')?.code).toContain('K: f32 = 2.0');
  });

  it('notifies on a change but not when the same source is compiled again', async () => {
    const device = await getTestDevice();
    let calls = 0;
    const stop = onCompiledShadersChanged(() => calls++);
    createShader(device, source);
    expect(calls).toBe(1);
    createShader(device, source);
    expect(calls).toBe(1);
    createShader(device, { ...source, chunks: { 'gen.wgsl': 'const K: f32 = 3.0;' } });
    expect(calls).toBe(2);
    stop();
    clearCompiledShaders();
    expect(calls).toBe(2);
  });

  it('is bounded, dropping the oldest labels first', async () => {
    const device = await getTestDevice();
    for (let i = 0; i < 80; i++) {
      createShader(device, { label: `many/${i}.wgsl`, code: `const N${i}: f32 = 1.0;` });
    }
    const all = getCompiledShaders();
    expect(all.length).toBeLessThanOrEqual(64);
    expect(all.at(-1)?.label).toBe('many/79.wgsl');
    expect(getCompiledShader('many/0.wgsl')).toBeUndefined();
  });

  it('records nothing for a shader whose include cannot be resolved', async () => {
    const device = await getTestDevice();
    expect(() =>
      createShader(device, { label: 'fx/bad.wgsl', code: '#include "nope.wgsl"' }),
    ).toThrow();
    expect(getCompiledShader('fx/bad.wgsl')).toBeUndefined();
  });

  it('replaces the snapshot array instead of mutating it', async () => {
    const device = await getTestDevice();
    const before = getCompiledShaders();
    createShader(device, source);
    expect(before).toHaveLength(0);
    expect(getCompiledShaders()).toHaveLength(1);
  });
});

describe('scorecard formatting', () => {
  it('states the tier budgets from the plan', () => {
    expect(tierLabel('A')).toBe('Tier A · Drop-in');
    expect(budgetText('A')).toBe('≤ 2 ms per frame at 1080p on an integrated GPU');
    expect(budgetText('B')).toContain('6 ms');
    expect(budgetText('C')).toContain('12 ms');
    expect(budgetText('C')).toContain('Apple Silicon');
  });

  it('formats times and sizes, and never invents a number for a missing one', () => {
    expect(formatMs(0.8412)).toBe('0.84 ms');
    expect(formatMs(12.34)).toBe('12.3 ms');
    expect(formatKb(undefined)).toBe(NOT_MEASURED);
    expect(formatKb(4.26)).toBe('4.3 kB gzipped');
    expect(formatKb(42.6)).toBe('43 kB gzipped');
  });

  it('describes compute and mobile status', () => {
    expect(computeText(false)).toBe('No');
    expect(computeText(true)).toMatch(/no WebGL2/);
    expect(mobileText('untested')).toBe('Untested');
    expect(mobileText('no')).toBe('No');
    expect(mobileText('yes')).toMatch(/^Yes/);
  });

  it('judges measured devices against the tier budget, and shows no rows when nothing is measured', () => {
    expect(costRows('A', undefined)).toEqual([]);
    expect(costRows('A', [])).toEqual([]);
    const rows = costRows('A', [
      { device: 'M2', ms: 0.9 },
      { device: 'UHD 620', ms: 2.6 },
    ]);
    expect(rows.map((row) => row.device)).toEqual(['M2', 'UHD 620']);
    expect(rows[0]?.value).toBe('0.90 ms');
    expect(rows[0]?.verdict).toBe('within the 2 ms budget');
    expect(rows[1]?.verdict).toBe('over the 2 ms budget');
  });

  it('reports whether this device has a feature, or that it is still unknown', () => {
    const set = new Set(['timestamp-query']);
    expect(featureStatus('timestamp-query', set)).toBe('available');
    expect(featureStatus('subgroups', set)).toBe('missing');
    expect(featureStatus('subgroups', undefined)).toBe('unknown');
  });

  it('sums pass timings, smooths readings, and formats the resolution', () => {
    expect(sumPasses([])).toBe(0);
    expect(
      sumPasses([
        { label: 'a', ms: 0.25 },
        { label: 'b', ms: 0.5 },
      ]),
    ).toBeCloseTo(0.75);
    expect(smooth(undefined, 4)).toBe(4);
    expect(smooth(4, 8, 0.5)).toBe(6);
    expect(resolutionText(1920, 1080)).toBe('1920 × 1080 (2.07 MP)');
  });
});

describe('effect write-ups', () => {
  const files = import.meta.glob<string>('../src/effects/*/notes.md', {
    query: '?raw',
    import: 'default',
    eager: true,
  });
  const template = Object.values(
    import.meta.glob<string>('../docs/effect-notes-template.md', {
      query: '?raw',
      import: 'default',
      eager: true,
    }),
  )[0]!;
  const headings = (markdown: string) =>
    [...markdown.replace(/<!--[\s\S]*?-->/g, '').matchAll(/^## (.+)$/gm)].map((match) => match[1]);

  it('has a template with the five agreed headings', () => {
    expect(headings(template)).toEqual([
      'How it works',
      'Cost',
      'Pitfalls',
      'When to use it',
      'References',
    ]);
  });

  it('exists for hello', () => {
    expect(Object.keys(files)).toContain('../src/effects/hello/notes.md');
    expect(listEffects().some((meta) => meta.id === 'hello')).toBe(true);
  });

  it.each(listEffects().filter((meta) => meta.epic === '7'))(
    '$id has a complete research package',
    async (meta) => {
      const notes = files[`../src/effects/${meta.id}/notes.md`];
      expect(notes).toBeDefined();
      expect(effectPoster(meta.id)).toBeDefined();
      expect(meta.scorecard?.fallback).toBeTruthy();
      expect(meta.scorecard?.compute).toBe(true);
      expect(
        meta.scorecard?.gpuMs?.every((sample) => Number.isFinite(sample.ms) && sample.ms > 0),
      ).toBe(true);
      expect(meta.scorecard?.bundleKb).toBeGreaterThan(0);
      const effect = await loadEffect(meta.id);
      expect(Object.keys(effect.presets ?? {}).length).toBeGreaterThanOrEqual(2);
    },
  );

  it.each(listEffects().filter((meta) => meta.epic === '8'))(
    '$id has ocean notes, poster, presets and measured costs',
    async (meta) => {
      expect(files[`../src/effects/${meta.id}/notes.md`]).toBeDefined();
      expect(effectPoster(meta.id)).toBeDefined();
      expect(meta.scorecard?.fallback).toBeTruthy();
      expect(meta.scorecard?.bundleKb).toBeGreaterThan(0);
      expect(meta.scorecard?.gpuMs?.length).toBeGreaterThan(0);
      expect(meta.scorecard?.gpuMs?.every((s) => Number.isFinite(s.ms) && s.ms > 0)).toBe(true);
      expect(Object.keys((await loadEffect(meta.id)).presets ?? {}).length).toBeGreaterThanOrEqual(
        2,
      );
    },
  );

  it.each(listEffects().filter((meta) => meta.epic === '9'))(
    '$id has a complete liquid UI research package',
    async (meta) => {
      expect(files[`../src/effects/${meta.id}/notes.md`]).toBeDefined();
      expect(effectPoster(meta.id)).toBeDefined();
      expect(meta.scorecard?.compute).toBe(false);
      expect(meta.scorecard?.fallback).toBeTruthy();
      expect(meta.scorecard?.bundleKb).toBeGreaterThan(0);
      expect(meta.scorecard?.gpuMs?.length).toBeGreaterThan(0);
      expect(meta.scorecard?.gpuMs?.every((s) => Number.isFinite(s.ms) && s.ms > 0)).toBe(true);
      const effect = await loadEffect(meta.id);
      expect(Object.keys(effect.presets ?? {}).length).toBeGreaterThanOrEqual(2);
      for (const preset of Object.values(effect.presets ?? {}))
        expect(Object.keys(preset).every((key) => key in effect.params)).toBe(true);
    },
  );

  it.each(listEffects().filter((meta) => meta.epic === '10'))(
    '$id has a complete weather research package',
    async (meta) => {
      expect(files[`../src/effects/${meta.id}/notes.md`]).toBeDefined();
      expect(effectPoster(meta.id)).toBeDefined();
      expect(meta.scorecard?.compute).toBe(false);
      expect(meta.scorecard?.fallback).toBeTruthy();
      expect(meta.scorecard?.bundleKb).toBeGreaterThan(0);
      expect(meta.scorecard?.gpuMs?.length).toBeGreaterThan(0);
      const effect = await loadEffect(meta.id);
      expect(Object.keys(effect.presets ?? {}).length).toBeGreaterThanOrEqual(2);
      for (const preset of Object.values(effect.presets ?? {}))
        expect(Object.keys(preset).every((key) => key in effect.params)).toBe(true);
    },
  );

  it.each(Object.entries(files))('%s follows the template headings', (_path, markdown) => {
    expect(headings(markdown)).toEqual(headings(template));
  });
});

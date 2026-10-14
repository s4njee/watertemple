import { getChunk } from '../wgsl/index.ts';
import { preprocess, PreprocessError, type Preprocessed } from '../wgsl/preprocess.ts';

export interface ShaderSource {
  /** GPU object label and the file name shown in diagnostics, e.g. "caustics/caustics.wgsl". */
  label: string;
  code: string;
  /** Effect-local include targets, looked up before the shared library in src/wgsl. */
  chunks?: Readonly<Record<string, string>>;
}

export type DiagnosticSeverity = 'error' | 'warning' | 'info';

export interface ShaderDiagnostic {
  severity: DiagnosticSeverity;
  message: string;
  /** Label of the shader that was being compiled. */
  shader: string;
  /** The original file the message points into: the shader itself or an included chunk. */
  file: string;
  /** 1-based; 0 when the compiler gave no position (`file` is then the shader). */
  line: number;
  /** 1-based, in UTF-16 code units like JavaScript string indices; 0 when there is no position. */
  column: number;
  /** Length of the offending span in UTF-16 code units; may run past the end of the line. */
  length: number;
  /** Text of the offending line in the original file. */
  sourceLine: string;
}

export function resolveShader(source: ShaderSource): Preprocessed {
  return preprocess(source.code, source.label, (name) => source.chunks?.[name] ?? getChunk(name));
}

type CompilationMessage = Pick<
  GPUCompilationMessage,
  'type' | 'message' | 'lineNum' | 'linePos' | 'length'
>;

/**
 * Maps compiler messages, positioned in the preprocessed code, back to the original files.
 * Dawn reports 1-based `lineNum` and `linePos` (in UTF-16 units) and `lineNum` 0 when a message
 * has no position, e.g. an illegal character in the source.
 */
export function mapCompilationMessages(
  shader: string,
  { code, lines }: Preprocessed,
  messages: readonly CompilationMessage[],
): ShaderDiagnostic[] {
  const codeLines = code.split('\n');
  return messages.map((message) => {
    const origin = message.lineNum > 0 ? lines[message.lineNum - 1] : undefined;
    return {
      severity: message.type,
      message: message.message,
      shader,
      file: origin?.file ?? shader,
      line: origin?.line ?? 0,
      column: origin ? message.linePos : 0,
      length: message.length,
      sourceLine: origin ? (codeLines[message.lineNum - 1] ?? '') : '',
    };
  });
}

/** The numbered source line with the offending span underlined; empty without a position. */
export function diagnosticExcerpt(diagnostic: ShaderDiagnostic): string {
  const { line, column, sourceLine } = diagnostic;
  if (line <= 0) return '';
  const gutter = String(line);
  // Keep tabs so the carets line up under the source however wide the viewer renders a tab.
  const indent = sourceLine.slice(0, column - 1).replace(/[^\t]/g, ' ');
  const width = Math.max(1, Math.min(diagnostic.length, sourceLine.length - (column - 1)));
  return `${gutter} | ${sourceLine}\n${' '.repeat(gutter.length)} | ${indent}${'^'.repeat(width)}`;
}

/** A readable block: `file:line:col: severity: message`, the source line, and carets under it. */
export function formatDiagnostic(diagnostic: ShaderDiagnostic): string {
  const { file, line, column, severity, message, shader } = diagnostic;
  const where = line > 0 ? `${file}:${line}:${column}` : file;
  const via = file === shader ? '' : ` (while compiling ${shader})`;
  const head = `${where}: ${severity}: ${message}${via}`;
  return line > 0 ? `${head}\n${diagnosticExcerpt(diagnostic)}` : head;
}

/** Groups by original file, in order of first appearance, keeping each file's order. */
export function groupDiagnosticsByFile(
  diagnostics: readonly ShaderDiagnostic[],
): { file: string; diagnostics: ShaderDiagnostic[] }[] {
  const groups = new Map<string, ShaderDiagnostic[]>();
  for (const diagnostic of diagnostics) {
    const group = groups.get(diagnostic.file);
    if (group) group.push(diagnostic);
    else groups.set(diagnostic.file, [diagnostic]);
  }
  return [...groups].map(([file, grouped]) => ({ file, diagnostics: grouped }));
}

// --- Diagnostics store -------------------------------------------------------------------------

const byShader = new Map<string, readonly ShaderDiagnostic[]>();
const storeListeners = new Set<() => void>();
let snapshot: readonly ShaderDiagnostic[] | undefined;

function setDiagnostics(shader: string, diagnostics: readonly ShaderDiagnostic[]): void {
  if (diagnostics.length === 0 && !byShader.has(shader)) return;
  if (diagnostics.length === 0) byShader.delete(shader);
  else byShader.set(shader, diagnostics);
  snapshot = undefined;
  for (const listener of [...storeListeners]) listener();
}

/** Current diagnostics of every shader whose latest compile had any. The array is replaced, never mutated. */
export function getDiagnostics(): readonly ShaderDiagnostic[] {
  return (snapshot ??= Object.freeze([...byShader.values()].flat()));
}

export function onDiagnosticsChanged(listener: () => void): () => void {
  storeListeners.add(listener);
  return () => storeListeners.delete(listener);
}

/** Forgets every recorded diagnostic. For tests and tools; normal operation clears per label. */
export function clearDiagnostics(): void {
  if (byShader.size === 0) return;
  byShader.clear();
  snapshot = undefined;
  for (const listener of [...storeListeners]) listener();
}

// --- Compiled source registry ------------------------------------------------------------------

/** What `createShader` last sent to the compiler under one label. */
export interface CompiledShader {
  label: string;
  /** The preprocessed WGSL exactly as given to `createShaderModule`: includes inlined. */
  code: string;
  /** Every file pulled in by `#include`, transitively, in order of first inclusion. */
  includes: readonly string[];
  /** The effect-local include targets that were used, e.g. a chunk generated at runtime. */
  chunks: Readonly<Record<string, string>>;
}

// Bounded, and replaced per label, so shaders edited over a long dev session do not pile up.
const MAX_COMPILED_SHADERS = 64;
const compiled = new Map<string, CompiledShader>();
const compiledListeners = new Set<() => void>();
let compiledSnapshot: readonly CompiledShader[] | undefined;

function sameCompiled(a: CompiledShader | undefined, b: CompiledShader): boolean {
  if (!a || a.code !== b.code) return false;
  const names = Object.keys(b.chunks);
  return (
    names.length === Object.keys(a.chunks).length && names.every((n) => a.chunks[n] === b.chunks[n])
  );
}

function recordCompiled(source: ShaderSource, { code, lines }: Preprocessed): void {
  const { label } = source;
  const includes = [...new Set(lines.map((line) => line.file))].filter((file) => file !== label);
  const chunks: Record<string, string> = {};
  for (const name of includes) {
    const chunk = source.chunks?.[name];
    if (chunk !== undefined) chunks[name] = chunk;
  }
  const entry: CompiledShader = { label, code, includes, chunks };

  const unchanged = sameCompiled(compiled.get(label), entry);
  compiled.delete(label);
  compiled.set(label, entry);
  if (compiled.size > MAX_COMPILED_SHADERS) compiled.delete(compiled.keys().next().value as string);
  if (unchanged) return;
  compiledSnapshot = undefined;
  for (const listener of [...compiledListeners]) listener();
}

/** The latest compiled source of every shader, oldest first. The array is replaced, never mutated. */
export function getCompiledShaders(): readonly CompiledShader[] {
  return (compiledSnapshot ??= Object.freeze([...compiled.values()]));
}

export function getCompiledShader(label: string): CompiledShader | undefined {
  return compiled.get(label);
}

export function onCompiledShadersChanged(listener: () => void): () => void {
  compiledListeners.add(listener);
  return () => compiledListeners.delete(listener);
}

/** Forgets every recorded source. For tests and tools. */
export function clearCompiledShaders(): void {
  if (compiled.size === 0) return;
  compiled.clear();
  compiledSnapshot = undefined;
  for (const listener of [...compiledListeners]) listener();
}

// --- Compiling ---------------------------------------------------------------------------------

export interface CheckedShader {
  module: GPUShaderModule;
  /** Every compiler message (errors, warnings, and the notes attached to them), original positions. */
  diagnostics: ShaderDiagnostic[];
}

// A newer compile of the same label supersedes an older one whose result is still in flight.
const latestAttempt = new Map<string, number>();

function report(label: string, attempt: number, diagnostics: ShaderDiagnostic[]): void {
  if (latestAttempt.get(label) !== attempt) return;
  setDiagnostics(label, diagnostics);
  if (diagnostics.length === 0) return;
  const hasErrors = diagnostics.some((d) => d.severity === 'error');
  const text = diagnostics.map(formatDiagnostic).join('\n\n');
  if (hasErrors) console.error(`[wgsl] ${label} failed to compile\n\n${text}`);
  else console.warn(`[wgsl] ${label} compiled with warnings\n\n${text}`);
}

function startCompile(
  device: GPUDevice,
  source: ShaderSource,
): { module: GPUShaderModule; checked: Promise<CheckedShader> } {
  const { label } = source;
  const attempt = (latestAttempt.get(label) ?? 0) + 1;
  latestAttempt.set(label, attempt);

  let preprocessed: Preprocessed;
  try {
    preprocessed = resolveShader(source);
  } catch (error) {
    if (error instanceof PreprocessError) {
      report(label, attempt, [
        {
          severity: 'error',
          message: error.reason,
          shader: label,
          file: error.file,
          line: error.line,
          column: error.column,
          length: error.sourceLine.trim().length,
          sourceLine: error.sourceLine,
        },
      ]);
    }
    throw error;
  }

  recordCompiled(source, preprocessed);
  const module = device.createShaderModule({ label, code: preprocessed.code });
  const checked = module.getCompilationInfo().then(({ messages }) => {
    const diagnostics = mapCompilationMessages(label, preprocessed, messages);
    report(label, attempt, diagnostics);
    return { module, diagnostics };
  });
  return { module, checked };
}

/**
 * Preprocesses and compiles a shader. Returns the module immediately; the compiler's verdict is
 * checked in the background. Errors are logged and recorded under `source.label` (see
 * `getDiagnostics`), and the next clean compile of that label clears them. A failed `#include`
 * is recorded the same way and then thrown, because there is no source to compile.
 */
export function createShader(device: GPUDevice, source: ShaderSource): GPUShaderModule {
  const { module, checked } = startCompile(device, source);
  checked.catch((error: unknown) =>
    console.warn(`[wgsl] ${source.label}: no compile report`, error),
  );
  return module;
}

/** Like `createShader`, but resolves once the compiler has reported. Rejects on a failed `#include`. */
export function checkShader(device: GPUDevice, source: ShaderSource): Promise<CheckedShader> {
  try {
    return startCompile(device, source).checked;
  } catch (error) {
    return Promise.reject(error);
  }
}

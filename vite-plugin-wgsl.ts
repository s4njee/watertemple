import { existsSync, readFileSync } from 'node:fs';
import { basename, dirname, relative, resolve, sep } from 'node:path';
import type { Plugin } from 'vite';
import { listIncludes } from './src/wgsl/preprocess.ts';

/*
 * `import shader from './flame.wgsl'` yields a ShaderSource: `{ label, code, chunks }`, where
 * `chunks` holds every file the shader pulls in through `#include`, transitively. Each of those is
 * imported as its own `?raw` module, so the bundler places a shared chunk next to the effects that
 * use it and nothing lands in the eager bundle. (`import text from './flame.wgsl?raw'` still gives
 * the plain text.)
 *
 * An include is looked up beside the including file first, then in the shared library
 * (src/wgsl). A name found in neither is left for run time: the effect supplies it through
 * `chunks`, as it does for uniform structs generated from TypeScript.
 */

export interface WgslPluginOptions {
  /** Directory of the shared include library, relative to the project root. Default "src/wgsl". */
  library?: string;
}

const WGSL = /\.wgsl$/;

/** "src/effects/hello/hello.wgsl" -> "hello/hello.wgsl"; library files are known by base name. */
export function shaderLabel(root: string, library: string, file: string): string {
  if (dirname(file) === library) return basename(file);
  const path = relative(root, file).split(sep).join('/');
  return path.replace(/^src\/(?:effects|gpu)\//, '').replace(/^src\//, '');
}

export interface IncludeClosure {
  /** Include name -> absolute file, in order of first inclusion. */
  files: Map<string, string>;
  /** Names no file was found for; the effect must supply them at run time. */
  runtime: string[];
}

/** Every file `file` includes, transitively, resolved beside the includer and then in `library`. */
export function includeClosure(
  file: string,
  library: string,
  read: (path: string) => string = (path) => readFileSync(path, 'utf8'),
  exists: (path: string) => boolean = existsSync,
): IncludeClosure {
  const files = new Map<string, string>();
  const runtime = new Set<string>();

  const visit = (path: string) => {
    for (const name of listIncludes(read(path))) {
      if (files.has(name) || runtime.has(name)) continue;
      const local = resolve(dirname(path), name);
      const shared = resolve(library, name);
      const found = exists(local) ? local : exists(shared) ? shared : undefined;
      if (found === undefined) {
        runtime.add(name);
        continue;
      }
      if (found === file) continue;
      files.set(name, found);
      visit(found);
    }
  };

  visit(file);
  return { files, runtime: [...runtime] };
}

export function wgsl(options: WgslPluginOptions = {}): Plugin {
  let root = process.cwd();
  let library = resolve(root, options.library ?? 'src/wgsl');

  return {
    name: 'firetemple:wgsl',
    enforce: 'pre',

    configResolved(config) {
      root = config.root;
      library = resolve(root, options.library ?? 'src/wgsl');
    },

    load(id) {
      if (!WGSL.test(id)) return null;
      const closure = includeClosure(id, library);
      for (const file of closure.files.values()) this.addWatchFile(file);

      const names = [...closure.files.keys()];
      const imports = names.map(
        (name, index) =>
          `import c${index} from ${JSON.stringify(`${closure.files.get(name)}?raw`)};`,
      );
      const entries = names.map((name, index) => `${JSON.stringify(name)}: c${index}`);
      return [
        `import code from ${JSON.stringify(`${id}?raw`)};`,
        ...imports,
        `export default {`,
        `  label: ${JSON.stringify(shaderLabel(root, library, id))},`,
        `  code,`,
        `  chunks: { ${entries.join(', ')} },`,
        `};`,
      ].join('\n');
    },

    // The list of includes is computed in `load`, so a module must be loaded again, not patched,
    // when any shader file changes: an edit may have added or removed an `#include`.
    hotUpdate({ file }) {
      if (!WGSL.test(file)) return;
      const graph = this.environment.moduleGraph;
      for (const module of graph.idToModuleMap.values()) {
        if (module.id && WGSL.test(module.id)) graph.invalidateModule(module);
      }
    },
  };
}

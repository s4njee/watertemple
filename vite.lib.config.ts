// Library build (M3 story 3.4): emit plain ESM with WGSL inlined as strings, so an effect can
// be consumed without Vite. Run with `npm run build:lib`.
import { existsSync, readdirSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { defineConfig } from 'vite';
import { wgsl } from './vite-plugin-wgsl.ts';

const src = fileURLToPath(new URL('./src', import.meta.url));

/** One entry per source module, named by its path, so consumers import `.../effects/<id>/index`. */
function entries(dir: string, prefix = ''): Record<string, string> {
  const out: Record<string, string> = {};
  for (const entry of readdirSync(dir, { withFileTypes: true })) {
    const path = `${dir}/${entry.name}`;
    if (entry.isDirectory()) Object.assign(out, entries(path, `${prefix}${entry.name}/`));
    else if (
      entry.name.endsWith('.ts') &&
      !entry.name.endsWith('.d.ts') &&
      !entry.name.endsWith('.test.ts')
    )
      out[`${prefix}${entry.name.slice(0, -3)}`] = path;
  }
  return out;
}

export default defineConfig({
  plugins: [wgsl()],
  build: {
    outDir: 'lib',
    emptyOutDir: true,
    target: 'es2022',
    sourcemap: true,
    minify: false,
    lib: { entry: entries(src), formats: ['es'] },
    rollupOptions: {
      // Keep dependencies external; consumers install them.
      external: (id) => !id.startsWith('.') && !id.startsWith('/') && !id.startsWith('\0'),
    },
  },
});

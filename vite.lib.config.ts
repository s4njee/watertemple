// Library build (M3 story 3.4, packaged in M4 story 4.4): emit plain ESM with WGSL inlined as
// strings, so an effect can be consumed without Vite. Run with `npm run build:lib`.
import { existsSync, readdirSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { defineConfig } from 'vite';
import { wgsl } from './vite-plugin-wgsl.ts';

const root = fileURLToPath(new URL('.', import.meta.url));
const effectsDir = fileURLToPath(new URL('./src/effects', import.meta.url));

const effectEntries = Object.fromEntries(
  readdirSync(effectsDir)
    .filter(
      (name) =>
        !name.startsWith('_') &&
        existsSync(fileURLToPath(new URL(`./src/effects/${name}/index.ts`, import.meta.url))),
    )
    .map((name) => [
      `effects/${name}/index`,
      fileURLToPath(new URL(`./src/effects/${name}/index.ts`, import.meta.url)),
    ]),
);

export default defineConfig({
  // The WGSL plugin turns every `import shader from './x.wgsl'` into JS data with the shader
  // text inlined, so the built output needs no bundler-specific shader import.
  plugins: [wgsl()],
  build: {
    outDir: 'lib',
    emptyOutDir: true,
    target: 'es2022',
    sourcemap: true,
    minify: false,
    lib: {
      entry: {
        index: fileURLToPath(new URL('./src/lib/index.ts', import.meta.url)),
        ...effectEntries,
      },
      formats: ['es'],
    },
    rollupOptions: {
      // Keep dependencies external; consumers install them.
      external: (id) => !id.startsWith('.') && !id.startsWith('/') && !id.startsWith('\0'),
    },
  },
  root,
});

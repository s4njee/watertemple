import { readFileSync } from 'node:fs';
import { defineConfig } from 'vitest/config';

export default defineConfig({
  // Static hosts that serve from a sub-path (e.g. GitHub Pages) set BASE_PATH at build time.
  base: process.env.BASE_PATH ?? '/',
  build: {
    manifest: true,
    target: 'es2022',
    rolldownOptions: {
      // The gallery and standalone pages that place effects behind ordinary DOM content.
      input: [
        'index.html',
        'ink-orbit.html',
        'aurora.html',
        'neon-calligraphy.html',
        'dichroic-glass.html',
        'bioluminescent-currents.html',
        'liquid-chrome.html',
        'silk-embers.html',
        'eclipse-halos.html',
        'neon-mycelium.html',
        'game-of-life.html',
        'lantern.html',
        'capture-spike.html',
      ],
    },
  },
  plugins: [
    {
      name: 'ui-research-artifacts',
      generateBundle() {
        for (const fileName of [
          'docs/backdrop-capture-study.md',
          'docs/ui-cost-study.md',
          'docs/weather-cost-study.md',
          'docs/weather-cost-study.json',
          'docs/weather-bundle-sizes.json',
          'docs/ui-cost-study.json',
          'docs/ui-bundle-sizes.json',
          'assets/epic-9/capture-spike.jpg',
          'scripts/ui/README.md',
          'scripts/ui/generate-msdf.cpp',
          'scripts/ui/pack-atlas.py',
          'scripts/ui/MSDFGEN-LICENSE.txt',
        ]) {
          this.emitFile({ type: 'asset', fileName, source: readFileSync(fileName) });
        }
      },
    },
  ],
  test: {
    environment: 'node',
    include: ['tests/**/*.test.ts'],
    setupFiles: ['tests/setup.ts'],
  },
});

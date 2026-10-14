import { log } from 'node:console';
import { readFileSync, writeFileSync } from 'node:fs';
import { gzipSync } from 'node:zlib';

// Build first. Report the effect entry separately from the transitive shared runtime/shaders.
const manifest = JSON.parse(readFileSync('dist/.vite/manifest.json', 'utf8'));
const ids = ['particles', 'sph-2d', 'pbf-2d', 'pbf-3d', 'mpm-3d', 'fluid-surface'];
const size = (file) => gzipSync(readFileSync(`dist/${file}`)).length;
const entries = ids.map((id) => {
  const key = `src/effects/${id}/index.ts`;
  const entry = manifest[key];
  if (!entry) throw new Error(`Missing effect build entry: ${key}`);
  const files = new Set();
  const visit = (key) => {
    const item = manifest[key];
    if (!item || files.has(item.file)) return;
    files.add(item.file);
    for (const dependency of item.imports ?? []) visit(dependency);
  };
  visit(key);
  const ownFiles = new Set([entry.file]);
  // A second entry point (the lantern page) can move an effect into a shared chunk,
  // leaving a tiny re-export wrapper. Include that module, not just the wrapper.
  if (entry.imports?.length === 1) {
    const shared = manifest[entry.imports[0]];
    if (shared?.name === id) ownFiles.add(shared.file);
  }
  return {
    id,
    entryGzipKb: Math.round([...ownFiles].reduce((n, file) => n + size(file), 0) / 10) / 100,
    withSharedGzipKb: Math.round([...files].reduce((n, file) => n + size(file), 0) / 10) / 100,
    files: [...files],
  };
});
writeFileSync('docs/particle-bundle-sizes.json', JSON.stringify(entries, null, 2) + '\n');
for (const entry of entries)
  log(
    `${entry.id}: ${entry.entryGzipKb} kB own / ${entry.withSharedGzipKb} kB with shared imports`,
  );

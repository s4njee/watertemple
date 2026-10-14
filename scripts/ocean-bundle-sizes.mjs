import { log } from 'node:console';
import { readFileSync, writeFileSync } from 'node:fs';
import { gzipSync } from 'node:zlib';

// Build first. Report the effect entry separately from the transitive shared runtime/shaders.
const manifest = JSON.parse(readFileSync('dist/.vite/manifest.json', 'utf8'));
const ids = [
  'gerstner-ocean',
  'fft-ocean',
  'shoreline',
  'river',
  'waterfall',
  'waterline',
  'calm-lake',
];
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
  // The scene wrappers share their implementation. Count the renderer and its two local
  // shader chunks as effect code, rather than claiming the wrapper is a standalone effect.
  const oceanKey = (entry.imports ?? []).find((k) => manifest[k]?.name === 'effect');
  const effectCodeFiles = new Set(ownFiles);
  if (oceanKey) {
    effectCodeFiles.add(manifest[oceanKey].file);
    for (const dependency of manifest[oceanKey].imports ?? []) {
      if (['fft', 'scene'].includes(manifest[dependency]?.name))
        effectCodeFiles.add(manifest[dependency].file);
    }
  }
  for (const dependency of entry.imports ?? []) {
    if (manifest[dependency]?.name === 'meta') effectCodeFiles.add(manifest[dependency].file);
  }
  return {
    id,
    entryGzipKb: Math.round([...ownFiles].reduce((n, file) => n + size(file), 0) / 10) / 100,
    effectCodeGzipKb:
      Math.round([...effectCodeFiles].reduce((n, file) => n + size(file), 0) / 10) / 100,
    effectCodeFiles: [...effectCodeFiles],
    withSharedGzipKb: Math.round([...files].reduce((n, file) => n + size(file), 0) / 10) / 100,
    sharedOceanGzipKb:
      Math.round(
        (entry.imports ?? [])
          .filter((k) => manifest[k]?.name === 'effect')
          .reduce((n, k) => n + size(manifest[k].file), 0) / 10,
      ) / 100,
    files: [...files],
  };
});
writeFileSync('docs/ocean-bundle-sizes.json', JSON.stringify(entries, null, 2) + '\n');
for (const entry of entries)
  log(
    `${entry.id}: ${entry.effectCodeGzipKb} kB effect code / ${entry.withSharedGzipKb} kB with shared imports`,
  );

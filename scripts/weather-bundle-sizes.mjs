import { log } from 'node:console';
import { readFileSync, writeFileSync } from 'node:fs';
import { gzipSync } from 'node:zlib';
const manifest = JSON.parse(readFileSync('dist/.vite/manifest.json', 'utf8'));
const ids = ['rain-streaks', 'rain-rings', 'bubbles', 'splash-burst', 'fogged-glass', 'frost-melt'];
const size = (file) => gzipSync(readFileSync(`dist/${file}`)).length;
const entries = ids.map((id) => {
  const key = `src/effects/${id}/index.ts`,
    entry = manifest[key];
  if (!entry) throw new Error(`Missing ${key}`);
  const files = new Set(),
    owned = new Set();
  const visit = (key) => {
    const item = manifest[key];
    if (!item || files.has(item.file)) return;
    files.add(item.file);
    if (
      key.startsWith('src/gpu/weather/') ||
      key.startsWith(`src/effects/${id}/`) ||
      /weather/.test(item.name ?? '')
    )
      owned.add(item.file);
    for (const dependency of item.imports ?? []) visit(dependency);
  };
  visit(key);
  owned.add(entry.file);
  for (const dependency of entry.imports ?? []) {
    if (manifest[dependency]?.name === 'meta') owned.add(manifest[dependency].file);
  }
  // Rolldown may key shared weather chunks by a generated _effect-* entry. Follow the direct factory.
  const factory = (entry.imports ?? []).find((key) => manifest[key]?.name === 'effect');
  if (factory) {
    const local = manifest[factory];
    owned.add(local.file);
    for (const key of local.imports ?? []) {
      const dep = manifest[key];
      if (
        ['display', 'particles', 'memory', 'params', 'backdrop', 'scene', 'dynamics'].includes(
          dep?.name,
        )
      )
        owned.add(dep.file);
    }
  }
  return {
    id,
    entryGzipKb: Math.round(size(entry.file) / 10) / 100,
    effectCodeGzipKb: Math.round([...owned].reduce((n, f) => n + size(f), 0) / 10) / 100,
    withSharedGzipKb: Math.round([...files].reduce((n, f) => n + size(f), 0) / 10) / 100,
    effectCodeFiles: [...owned],
    files: [...files],
  };
});
writeFileSync('docs/weather-bundle-sizes.json', JSON.stringify(entries, null, 2) + '\n');
for (const row of entries)
  log(`${row.id}: ${row.effectCodeGzipKb} kB owned / ${row.withSharedGzipKb} kB transitive`);

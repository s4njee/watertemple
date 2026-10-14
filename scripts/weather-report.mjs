import { readFileSync, writeFileSync, existsSync, copyFileSync } from 'node:fs';
import { log } from 'node:console';
import { format, resolveConfig } from 'prettier';
const study = JSON.parse(readFileSync('docs/weather-cost-study.json', 'utf8'));
if (!study.metric.startsWith('GPU timestamp')) {
  throw new Error(
    'CPU wall-time samples cannot populate GPU scorecards; keep the labelled raw study instead.',
  );
}
const sizes = JSON.parse(readFileSync('docs/weather-bundle-sizes.json', 'utf8'));
const table = [
  '| Effect / configuration | Median frame span (ms) | Workload slope (ms) | Target tier |',
  '|---|---:|---:|---|',
];
for (const row of study.rows) {
  const p = row.params,
    detail = [];
  if (row.inputSegments) detail.push(`${row.inputSegments} input segments`);
  if (row.id === 'rain-streaks') detail.push(`${p.count} streaks`);
  if (row.id === 'bubbles') detail.push(`${p.count} ${p.style}`);
  if (row.id === 'frost-melt') detail.push(p.cycle);
  if (row.id === 'splash-burst') detail.push(`${p.count} droplets per burst`);
  table.push(
    `| ${row.id} / ${detail.join(', ') || 'default'} | ${row.spanMedianMs.toFixed(3)} | ${row.slopeMedianMs.toFixed(3)} | ${['fogged-glass', 'frost-melt'].includes(row.id) ? 'B (6 ms)' : 'A (2 ms)'} |`,
  );
}
const doc = 'docs/weather-cost-study.md';
let markdown = readFileSync(doc, 'utf8');
const start = markdown.indexOf('| Effect / configuration'),
  end = markdown.indexOf('\n\nAll configurations', start);
if (start < 0 || end < 0) throw new Error('Weather report table anchors are missing');
markdown = markdown.slice(0, start) + table.join('\n') + markdown.slice(end);
writeFileSync(doc, await format(markdown, { ...(await resolveConfig(doc)), parser: 'markdown' }));
for (const size of sizes) {
  const rows = study.rows.filter((row) => row.id === size.id);
  if (!rows.length) throw new Error(`Missing timing rows for ${size.id}`);
  const ms =
    Math.round(
      Math.max(...rows.map((row) => Math.max(row.spanMedianMs, row.slopeMedianMs))) * 1000,
    ) / 1000;
  if (!Number.isFinite(ms) || ms <= 0) throw new Error(`Invalid timing for ${size.id}`);
  const file = `src/effects/${size.id}/meta.ts`;
  let meta = readFileSync(file, 'utf8')
    .replace(/bundleKb: [\d.]+/, `bundleKb: ${size.effectCodeGzipKb}`)
    .replace(/ms: [\d.]+/, `ms: ${ms}`)
    .replace(
      /device: '[^']*'/,
      `device: ${JSON.stringify(study.deviceLabel ?? study.adapter.device ?? study.adapter.description)}`,
    );
  writeFileSync(file, await format(meta, { ...(await resolveConfig(file)), parser: 'typescript' }));
  const notes = `src/effects/${size.id}/notes.md`;
  let text = readFileSync(notes, 'utf8')
    .replace(
      /conservative complete-frame cost \*\*[\d.]+ ms/,
      `conservative complete-frame cost **${ms.toFixed(3)} ms`,
    )
    .replace(
      /Owned effect code is \*\*[\d.]+ kB/,
      `Owned effect code is **${size.effectCodeGzipKb.toFixed(2)} kB`,
    );
  writeFileSync(notes, text);
  log(`${size.id}: ${ms} ms / ${size.effectCodeGzipKb} kB owned`);
}
for (const name of [
  'weather-cost-study.md',
  'weather-cost-study.json',
  'weather-bundle-sizes.json',
]) {
  if (existsSync('dist/docs')) copyFileSync(`docs/${name}`, `dist/docs/${name}`);
}

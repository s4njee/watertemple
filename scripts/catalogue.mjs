// Emits catalogue.json: one record per effect, with a `status` derived from the existing
// evidence (tier, compute, flash rating, mobile). M3 stories 3.9 and 3.10.
import fs from 'node:fs';
import path from 'node:path';
import { createRequire } from 'node:module';
import { fileURLToPath } from 'node:url';

const require = createRequire(import.meta.url);
const ts = require('typescript');
const root = path.dirname(path.dirname(fileURLToPath(import.meta.url)));
const dir = path.join(root, 'src/effects');

async function readMeta(file) {
  const source = ts.transpileModule(fs.readFileSync(file, 'utf8'), {
    compilerOptions: { module: ts.ModuleKind.ESNext, target: ts.ScriptTarget.ES2022 },
  }).outputText;
  const data = 'data:text/javascript;base64,' + Buffer.from(source).toString('base64');
  return (await import(data)).meta;
}

/** Only `ready` effects get a stable export, a docs-page promise and semver protection (D7). */
function statusOf(meta) {
  const s = meta.scorecard;
  if (!s) return 'research';
  const flash = s.flash ?? 'none';
  // Ready: a shader-only tier-A effect that is flash-safe and reads over a light page.
  const ready = meta.tier === 'A' && !s.compute && flash !== 'high' && s.lightPage !== 'no';
  return ready ? 'ready' : 'demo';
}

const effects = [];
for (const id of fs.readdirSync(dir)) {
  const metaFile = path.join(dir, id, 'meta.ts');
  if (!fs.existsSync(metaFile)) continue;
  const meta = await readMeta(metaFile);
  const poster =
    ['poster.webp', 'poster.png']
      .map((file) => path.join('src/effects', id, file))
      .find((file) => fs.existsSync(path.join(root, file))) ?? null;
  effects.push({
    id: meta.id,
    title: meta.title,
    summary: meta.summary ?? '',
    epic: meta.epic,
    tier: meta.tier,
    tags: [...(meta.tags ?? [])],
    status: statusOf(meta),
    poster,
    scorecard: meta.scorecard ?? null,
  });
}
effects.sort((a, b) => Number(a.epic) - Number(b.epic) || a.id.localeCompare(b.id));

const catalogue = { project: path.basename(root), effects };
fs.writeFileSync(path.join(root, 'catalogue.json'), JSON.stringify(catalogue, null, 2) + '\n');
console.log(`${effects.length} effects, ${effects.filter((e) => e.status === 'ready').length} ready`);

// Rewrite relative `.ts` import specifiers in the emitted declarations to `.js`, so a consumer's
// resolver (which looks for `*.d.ts` beside `*.js`, not beside `*.ts`) finds them at the same path
// the library build emits.
import { readdirSync, readFileSync, statSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

const root = 'types';
let changed = 0;
const walk = (dir) => {
  for (const entry of readdirSync(dir)) {
    const path = join(dir, entry);
    if (statSync(path).isDirectory()) walk(path);
    else if (entry.endsWith('.d.ts')) {
      const source = readFileSync(path, 'utf8');
      const rewritten = source.replace(/(['"])(\.\.?\/[^'"]+?)\.ts\1/g, '$1$2.js$1');
      if (rewritten !== source) {
        writeFileSync(path, rewritten);
        changed += 1;
      }
    }
  }
};
walk(root);
console.log(`rewrote .ts specifiers in ${changed} declaration files`);

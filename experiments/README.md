# Particle development experiments

The former scratch tests are preserved here. They include long parameter sweeps,
manual screenshots and timing probes; they are not regression tests and are excluded
from the normal test, typecheck and lint commands. Their shared GPU helpers remain
in `tests/`. Use `npx vitest run --config experiments/vite.config.ts <file>` to run one.

The maintained scale study is `npm run study:particles`; its JSON and methodology
are in `docs/particle-scale-study.json` and `docs/particle-scale-study.md`.

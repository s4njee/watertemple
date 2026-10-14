# Water Temple

A research gallery of liquid-looking **WebGPU** effects, judged by one question:
**would you actually ship this on a website?** Every effect gets a live demo,
readable source, a write-up and a practicality scorecard.

![Water Temple effects](docs/media/hero.webp)

Part of the [holofx collection](https://github.com/s4njee/holofx) alongside
[hologpu](https://github.com/s4njee/hologpu),
[firetemple](https://github.com/s4njee/firetemple) and
[lighttemple](https://github.com/s4njee/lighttemple).

## Run

```sh
npm install
npm run dev        # http://localhost:5173
npm run build      # typecheck + static build to dist/
npm test           # vitest, including WGSL/Dawn tests
```

Requires a WebGPU-capable browser (Chrome/Edge 113+, Safari 26+, Firefox 141+ on
Windows). Without WebGPU each demo shows its fallback poster and the page stays
readable.

## Use one effect

The package is published as `@mk7s/watertemple`:

```sh
npm install @mk7s/watertemple @mk7s/hologpu-temple-core
```

Every effect is a self-contained ES module with a declared contract: metadata, a
parameter schema, presets, and a `create` that returns `{ resize, frame, destroy }`.
Import it from its subpath and drive it from a `GPUDevice` you own:

```ts
import calmLake from '@mk7s/watertemple/effects/calm-lake';

const effect = await calmLake.create({ device, format, canvas, params, profiler });
effect.resize(width, height, dpr);
// each frame:
effect.frame({ time, delta }, encoder, target);
// on teardown:
effect.destroy();
```

Or let the host own the loop; it creates no DOM and imports no gallery code:

```ts
import { createGpuSession, mount } from '@mk7s/watertemple';
import calmLake from '@mk7s/watertemple/effects/calm-lake';

const session = await createGpuSession();
const host = mount(session, canvas, calmLake);
// later: host.destroy();
```

Both entries ship with types. See the per-effect `notes.md` for the current shape and
cost.

## Layout

```
src/effects/     one folder per effect: index.ts, shader(s), meta, notes, poster
src/gpu/         device/context, targets, passes, simulations, shared WGSL
src/shell/       demo host, controls, scorecard, fallbacks
docs/            studies, effect-authoring guide, support matrix
experiments/     scratch tests and spikes
```

See [plan.md](plan.md) for the roadmap and the
[holofx backlog](https://github.com/s4njee/holofx/blob/main/backlog.md) for the
cross-repo (superproject) workflow.

## Licence

MIT. See [LICENSE](LICENSE) and [CREDITS.md](CREDITS.md).

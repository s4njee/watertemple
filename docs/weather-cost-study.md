# Weather study — Epic 10

Six registered effects implement stories 10.1–10.6, including the frost-and-melt stretch story. No effect needs compute or optional WebGPU features. The measurements below are local M1 Max evidence, not mobile or cross-device certification.

## Method

Run `npm run study:weather`. Raw adapter identifiers, parameters and one/four-repeat timestamp samples live in [weather-cost-study.json](weather-cost-study.json). Each configuration warms eight frames and alternates one and four repeats for nine rounds. GPU cost uses the span from the first pass begin to the last pass end, never a sum of overlapping pass times. The workload slope is `(span(4)−span(1))/3`. The scorecards conservatively take the larger median span/slope over each effect’s tested configurations. Timestamp quantization and background GPU load limit precision; the numbers are reported as local observations. If timestamps are absent, the harness labels wall time instead of claiming GPU time.

Resolution: 1920×1080. Adapter: Apple M1 Max, Dawn/Metal. Initialization and one-time scene/mipmap bakes occur before the warm steady-frame samples. CPU encoding, particle sorting and display compositor cost are not included in the GPU timestamp metric.

## Results

| Effect / configuration                                  | Median frame span (ms) | Workload slope (ms) | Target tier |
| ------------------------------------------------------- | ---------------------: | ------------------: | ----------- |
| rain-streaks / 320 streaks                              |                  0.197 |               0.240 | A (2 ms)    |
| rain-rings / default                                    |                  0.786 |               0.765 | A (2 ms)    |
| bubbles / 48 underwater                                 |                  0.197 |               0.109 | A (2 ms)    |
| splash-burst / 48 droplets per burst                    |                  0.197 |               0.131 | A (2 ms)    |
| fogged-glass / default                                  |                  0.524 |               0.306 | B (6 ms)    |
| frost-melt / freeze and thaw                            |                  0.524 |               0.481 | B (6 ms)    |
| rain-streaks / 1000 streaks                             |                  0.328 |               0.240 | A (2 ms)    |
| bubbles / 140 soap                                      |                  0.328 |               0.262 | A (2 ms)    |
| splash-burst / 1 input segments, 128 droplets per burst |                  0.197 |               0.109 | A (2 ms)    |
| fogged-glass / 16 input segments                        |                  1.442 |               0.066 | B (6 ms)    |
| frost-melt / freeze                                     |                  0.524 |               0.393 | B (6 ms)    |
| frost-melt / melt                                       |                  0.590 |               0.415 | B (6 ms)    |

All configurations fit the stated budgets on this local adapter. Other integrated GPUs, Android and iPhone remain unmeasured under Epic 13.2. The `mobile` scorecards therefore say untested.

## Resource and behavior bounds

- Rain: 16–1,000 active streaks, with ground splashes and rings sharing a hard 2,048-particle cap. Size, fall speed and opacity depend on depth; instances blend far to near.
- Bubbles: 4–140 live sphere impostors; bounded analytic rising/wobbling motion. Soap-film thickness changes wavelength-dependent phase, with an explicit three-wavelength approximation.
- Splash bursts: 8–128 particles per press, finite lifetimes and analytic gravity. Ambient bursts can be disabled.
- Ripple rings: nine procedural event candidates per pixel, finite ring extent and fading lifetimes.
- Fog/frost: two `rgba16float` history textures, at most 1,024×256 cells; sharp-to-blurred scene mixing and a persistent wipe field. Frost ice loss under heat feeds a water channel, which advects downward and makes bead lenses.
- All effects: mipmapped authored scene backdrop, at most 768 pixels on its long side and at most half the canvas resolution; 96 KiB particle buffer. No arbitrary DOM capture is claimed. Resize rebuilds references and the shared host handles device loss, hidden/offscreen pause and reduced motion.

## Packaging

Build, then run `node scripts/weather-bundle-sizes.mjs` and `npm run report:weather` to synchronize measured scorecards and notes. [weather-bundle-sizes.json](weather-bundle-sizes.json) distinguishes tiny entry wrappers, the shared owned weather renderer/shaders, and the larger transitive runtime/WGSL load. These are not standalone embed sizes; Epic 12.1 owns that packaging work.

## Verification and captures

`tests/weather.test.ts` verifies wind/depth/ground impacts, ballistic and wobble integration across frame partitions, bounded particle storage, press-only feedback, persistent connected wipes and refogging, ice-to-water transition and downward runoff, every preset, float output, portrait/DPR changes and source listing. `tests/effects.test.ts` exercises the common lifecycle for all six. Captured GPU pixels produce the fallback WebP posters; the reproducible capture command is `WEATHER_CAPTURE=output/epic-10 npx vitest run tests/weather-capture.test.ts`. Native browser checks verify every effect in Hero, Cards, Band and Backdrop contexts, preset controls, expanded source tabs and pointer interaction. `output/epic-10/layout-validation.json` records all 24 desktop context mounts and six phone-width mounts with positive canvas sizes and no horizontal overflow. These are responsive-layout checks on the desktop browser, not physical-phone performance measurements.

## Approximation limits

Particles are 2D presentation layers rather than a 3D fluid solver. Independent ring waves do not reflect or interfere physically. Bubble optics omit complete spectral multiple reflections. Frost uses designed branching geometry rather than diffusion-limited growth; runoff is an advected coverage field with evaporation, not mass-conserving discrete water. These choices are exposed in each effect’s notes.

# Water Temple — WebGPU liquid effects research site

A research site that implements, measures, and documents liquid-looking WebGPU effects, judged by one question: **would you actually ship this on a website?** Every effect gets a live demo, readable source, a write-up, and a practicality scorecard.

## Goals

- Cover a wide range of liquid techniques, from one-pass fragment shaders to compute-driven simulations.
- Show each effect in a realistic web context (hero, card, button, background), not only on a bare canvas.
- Record what each effect costs: GPU time, bundle size, device coverage, fallback story.
- Keep every effect a self-contained module that can be lifted into another project.

## Non-goals

- A general-purpose rendering engine. A published npm library and a shared temple core are now planned; see stories 3.3–3.5 and M4 in the [holofx backlog](../holofx/backlog.md).
- Offline-quality or physically exact simulation. Plausible and cheap beats correct and slow.
- Full WebGL2 parity. WebGL2 ports are done only where the comparison is the research.

## Decisions

Defaults chosen so work can start. Each is cheap to revisit before Epic 1 is finished.

| Decision            | Default                                                  | Why                                                                                                                                       |
| ------------------- | -------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------- |
| Build tooling       | Vite + TypeScript (strict)                               | Requested. Fast HMR suits shader iteration.                                                                                               |
| Rendering API       | Raw WebGPU + WGSL, thin in-house helpers                 | The research subject is WebGPU itself. Smallest bundles, nothing hidden. Three.js/TSL ports are a comparison story (12.10), not the base. |
| UI framework        | None: vanilla TS, lazy-loaded effect modules             | Keeps effects framework-agnostic and copyable.                                                                                            |
| Shader loading      | `.wgsl?raw` imports plus a small `#include` preprocessor | No plugin dependency; shared chunks stay DRY.                                                                                             |
| Controls            | Tweakpane, generated from each effect's param schema     | One schema drives the UI, docs table, and URL state.                                                                                      |
| Math                | `wgpu-matrix`                                            | Small, WebGPU-conventions (0..1 depth).                                                                                                   |
| Hosting             | Static build, any static host                            | No server needed.                                                                                                                         |
| No-WebGPU behaviour | Poster image or captured video per effect                | Honest fallback; mirrors what a production site would do.                                                                                 |

## Practicality tiers

Every effect is tagged with a tier. Budgets are starting targets to be validated in Epic 13.

| Tier            | Meaning                                             | Shape                                  | GPU budget @1080p                   |
| --------------- | --------------------------------------------------- | -------------------------------------- | ----------------------------------- |
| **A** Drop-in   | Safe as ambient decoration, including mobile        | One or two fragment passes, no compute | ≤ 2 ms on integrated GPU            |
| **B** Feature   | Hero sections and interactive set pieces            | Multi-pass or a small compute sim      | ≤ 6 ms on integrated GPU            |
| **C** Showpiece | Dedicated page or landing experience, desktop-first | Heavy compute and/or full 3D scene     | ≤ 12 ms on discrete / Apple Silicon |

## Architecture sketch

```
watertemple/
  index.html
  vite.config.ts
  src/
    main.ts            bootstrap: one shared GPU session, mounts an effect
    gpu/               device, canvas sizing, frame loop, shaders, uniforms, targets, passes,
                       plus scene/ (camera, meshes, forward pass), post/ (bloom, tonemap) and
                       sim/ (heightfield, stable fluids 2D and 3D, shallow water, obstacle
                       masks, live caustics, pigment, displacement warp)
    wgsl/              shared include chunks: noise, sdf, colour, and the optics library
                       (fresnel, sky, specular, refraction, absorption, foam, normals, flow, ssr)
    effects/<id>/      meta.ts (eager), index.ts + *.wgsl (lazy chunk), notes.md, poster.webp
    shell/             app frame, router, gallery, effect page and its panels, design tokens
    bench/             benchmark harness
  tests/               vitest on Dawn's Node bindings: real WebGPU, no browser
  docs/                research output, e.g. support-matrix.md
  public/              textures, environment maps
```

Effect contract (every effect implements this; the shell knows nothing else about it):

```ts
// src/effects/<id>/index.ts exports `effect`; meta.ts exports `meta`.
export interface EffectModule<S extends ParamSchema> {
  meta: { id: string; title: string; epic: string; tier: 'A' | 'B' | 'C'; tags: string[] };
  params: S; // drives controls, docs table, URL state
  presets?: Record<string, Partial<ParamValues<S>>>;
  create(ctx: EffectContext<S>): Promise<EffectInstance>; // ctx: device, format, canvas, live params
}

export interface EffectInstance {
  resize(width: number, height: number, dpr: number): void;
  frame(time: FrameTime, encoder: GPUCommandEncoder, target: GPUTextureView): void;
  destroy(): void;
}
```

The full types are in `src/effects/types.ts`. `tests/effects.test.ts` runs every registered effect headlessly, so a new effect is covered as soon as its directory exists.

## Definition of done (per effect story)

1. Runs inside its tier budget on the reference devices, with numbers recorded.
2. Parameters exposed through the schema, with at least two presets.
3. Handles resize, DPR changes, device loss, offscreen pause, and `prefers-reduced-motion`.
4. Shown in at least one realistic web context.
5. `notes.md` written: how it works, cost, pitfalls, when to use it, references.
6. Scorecard filled in and fallback poster captured.

## Milestones

| Milestone             | Scope                                  | Proves                                           |
| --------------------- | -------------------------------------- | ------------------------------------------------ |
| **M0** Vertical slice | Epic 1, stories 2.1–2.3, effect 4.1    | Full pipeline: tooling → effect → shell → deploy |
| **M1** Drop-in set    | Epic 3 basics, Epic 4, stories 5.1–5.3 | Tier A library and shared optics                 |
| **M2** Compute sims   | Rest of Epic 5, Epic 6                 | Where compute shaders pay off                    |
| **M3** Liquid UI      | Epic 9, stories 12.1–12.4              | Web-native usefulness                            |
| **M4** Showpieces     | Epic 8, Epic 7                         | Upper bound of what the browser can do           |
| **M5** Findings       | Epics 10, 11, 13, rest of 12           | Published conclusions                            |

Story format: `ID Title [tier] (size)`. Sizes: S ≈ a day or less, M ≈ a few days, L ≈ a week or more.

---

## Epic 1 — Foundation and tooling

Everything needed before the first effect can exist.

- [x] **1.1 Vite scaffold** (S) — `vanilla-ts` template, strict TS, `@webgpu/types`, ESLint + Prettier, `dev` / `build` / `preview` / `typecheck` scripts.
- [x] **1.2 WebGPU bootstrap** (S) — adapter and device request, canvas configuration with `getPreferredCanvasFormat()`, premultiplied alpha so canvases composite over the DOM, capture of adapter features and limits for the HUD.
- [x] **1.3 Frame loop and resize** (S) — `ResizeObserver` with `devicePixelContentBoxSize`, DPR clamp, rAF loop, fixed-timestep accumulator for simulations.
- [x] **1.4 WGSL pipeline** (M) — `?raw` imports, `#include` preprocessor, HMR that rebuilds pipelines without a page reload, compile errors from `getCompilationInfo()` shown in an overlay with correct line mapping.
- [x] **1.5 Effect contract and registry** (S) — the interface above, discovery through `import.meta.glob`, one lazy chunk per effect.
- [x] **1.6 GPU helpers** (M) — fullscreen triangle, ping-pong targets, uniform struct writer, image/video texture loading, mip generation, blit / blur / downsample passes.
- [x] **1.7 Shared WGSL library** (M) — hash, value/simplex noise, FBM, Voronoi, curl noise, SDF primitives and smooth-min, colour-space and tonemap functions.
- [x] **1.8 Failure handling** (S) — no-WebGPU screen with poster, `device.lost` recovery, uncaptured-error reporting.
- [ ] **1.9 Build and deploy** (S) — static build with configurable base path, CI running typecheck + lint + build, preview deploys. _Partly done: base path (`BASE_PATH`) and a CI workflow exist. The workflow has never run (no git remote yet) and preview deploys wait on a hosting choice._
- [x] **1.10 Support matrix check** (S) — confirm current WebGPU status across Chrome, Edge, Safari, Firefox on desktop and mobile; record which optional features each exposes. Feeds 12.6. _Result: [docs/support-matrix.md](docs/support-matrix.md)._

## Epic 2 — Site shell and research UX

The frame the research is presented in.

- [x] **2.1 Gallery home** (M) — effect cards filterable by tier and tag; live thumbnails only while in view, all canvases sharing one `GPUDevice`.
- [x] **2.2 Effect page** (M) — canvas, generated controls, presets, shareable URL state, fullscreen toggle.
- [x] **2.3 Performance HUD** (M) — FPS, CPU frame time, per-pass GPU time through `timestamp-query` where available, resolution scale, estimated texture memory.
- [x] **2.4 Source viewer** (S) — highlighted WGSL and TS for the effect, with a copyable minimal snippet.
- [x] **2.5 Write-up template** (S) — markdown rendered alongside each demo, same headings for every effect.
- [x] **2.6 Practicality scorecard** (S) — tier, GPU cost, bundle size, mobile-safe, needs compute, optional features required, fallback, best use cases.
- [x] **2.7 In-context mode** (M) — a set of mock page layouts (hero, card grid, nav, footer) that any effect can be mounted into.
- [x] **2.8 Site visual design** (M) — a look that lets the effects carry the page; light and dark themes.

## Epic 3 — Optics and shading building blocks

Shared WGSL that most later effects depend on. Each is proven in a small test scene.

- [x] **3.1 Fresnel and sky reflection** (S) — Schlick approximation, procedural sky and cubemap variants.
- [x] **3.2 Screen-space refraction** (S) — IOR-based offset of a backdrop texture, with chromatic dispersion.
- [x] **3.3 Depth absorption** (S) — Beer–Lambert tint and scattering colour by water depth.
- [x] **3.4 Detail normals** (S) — two scrolling normal maps with a correct blend (whiteout / reoriented).
- [x] **3.5 Flow maps** (M) — two-phase flow blending for directed currents.
- [x] **3.6 Sun glint and sparkle** (S) — tight specular lobe plus stochastic glitter.
- [x] **3.7 Depth-based foam and edge fade** (S) — shoreline foam and soft intersection with geometry.
- [x] **3.8 Wave-crest subsurface scattering** (S) — cheap view/light-dependent approximation.
- [x] **3.9 Reflection strategies** (M) — planar reflection pass versus screen-space reflection, compared on cost and artefacts.
- [x] **3.10 HDR post chain** (M) — `rgba16float` targets, bloom, tonemapping; spike on HDR canvas output.

## Epic 4 — Shader-only 2D water

No simulation state. The cheapest and most broadly usable group.

- [x] **4.1 Procedural caustics** [A] (S) — tileable animated caustic overlay; compare warped-Voronoi and iterated-sine approaches.
- [x] **4.2 Layered wave divider** [A] (S) — SDF sine layers with parallax and a foam line, for section breaks and footers.
- [x] **4.3 Water distortion over image or video** [A] (S) — noise-driven refraction and specular on any media element.
- [x] **4.4 Rain on glass** [A] (M) — droplet grid with trails refracting a blurred backdrop.
- [x] **4.5 Underwater ambience** [A] (M) — god rays, drifting particulates, depth fog, gentle wobble.
- [x] **4.6 Pool floor** [A] (S) — top-down tiles, caustics, and refraction.
- [x] **4.7 Stylised toon water** [A] (M) — cell-noise foam, banded colour, hard highlights.
- [x] **4.8 Puddles and wet surfaces** [A] (M) — masked reflections with rain ripple rings.
- [x] **4.9 Ink wash background** [A] (S) — domain-warped FBM that reads as watercolour without simulating it.
- [x] **4.10 Mist and steam layer** [A] (S) — animated FBM with scroll parallax.

## Epic 5 — Interactive heightfield ripples

A 2D wave equation: the classic "touch the water" effect and the first use of persistent GPU state.

- [x] **5.1 Heightfield solver** [A] (M) — ping-pong height/velocity textures; implement as both a render pass and a compute pass and compare.
- [x] **5.2 Pointer and touch input** [A] (S) — drops on tap, velocity-scaled impulses on drag, multi-touch.
- [x] **5.3 Backdrop refraction** [A] (S) — normals from height gradients, refracted image plus specular.
- [x] **5.4 Caustics from the heightfield** [B] (M) — projected light-mesh technique driven by the live surface.
- [x] **5.5 DOM obstacles** [B] (M) — element rects rasterised into a boundary mask so ripples reflect off cards and headings.
- [x] **5.6 Ambient disturbance** [A] (S) — random rain drops and idle motion so the surface is never dead.
- [x] **5.7 3D pool scene** [B] (L) — walls, floor, a movable object, ray-traced refraction and reflection.
- [x] **5.8 Scroll and audio excitation** [A] (S) — drive the surface from scroll velocity or an audio analyser.

## Epic 6 — Grid fluids (Eulerian, compute)

Velocity-field simulations: ink, smoke-like swirls, flowing water.

- [x] **6.1 Stable-fluids solver** [B] (L) — advection, divergence, Jacobi pressure solve, projection, all in compute shaders.
- [x] **6.2 Dye injection** [B] (S) — pointer-driven ink with colour palettes and dissipation controls.
- [x] **6.3 Detail preservation** [B] (M) — vorticity confinement; compare semi-Lagrangian, MacCormack, and BFECC advection.
- [x] **6.4 Fluid as page distortion** [B] (M) — the velocity field warps imagery or text instead of showing dye.
- [x] **6.5 Obstacles** [B] (M) — solid boundaries from SDFs and DOM masks.
- [x] **6.6 Ink in water** [B] (M) — dye plus diffusion and paper texture for watercolour bleed.
- [x] **6.7 Shallow-water equations** [B] (L) — height and velocity; dam break and flow over terrain.
- [x] **6.8 Solver cost study** [—] (M) — iterations versus resolution versus quality; low-resolution simulation with high-quality upsampling. _Result: [docs/fluid-cost-study.md](docs/fluid-cost-study.md)._
- [x] **6.9 3D grid fluid** [C] (L) — stretch goal: volumetric solve with ray-marched rendering.

## Epic 7 — Particle fluids (Lagrangian, compute)

Splashing, pouring, sloshing liquid. The clearest case for WebGPU over WebGL.

- [x] **7.1 GPU particle framework** [B] (M) — storage buffers, compute integration, instanced billboard rendering.
- [x] **7.2 Neighbour search** [—] (L) — uniform grid with counting sort and prefix sum in compute.
- [x] **7.3 2D SPH** [B] (L) — density, pressure, and viscosity kernels with boundaries and pointer forces.
- [x] **7.4 Position-based fluids** [C] (L) — 2D first, then 3D.
- [x] **7.5 MLS-MPM in 3D** [C] (L) — particle-to-grid with atomics, grid update, grid-to-particle.
- [x] **7.6 Screen-space fluid rendering** [C] (L) — sphere-impostor depth, edge-preserving smoothing, normal reconstruction, thickness pass, refraction and absorption composite.
- [x] **7.7 Whitewater** [C] (M) — secondary spray, foam, and bubble particles. _Implemented as a bounded secondary GPU store in the 3D PBF tank; occupancy-based classification, flow advection, buoyancy and finite lifetimes._
- [x] **7.8 Container interaction** [C] (S) — drag, shake, and device-tilt gravity. _3D PBF presets and controls; sensor opt-in, gesture permission and listener teardown. Physical-phone validation remains pending._
- [x] **7.9 Scale study** [—] (M) — particle count against frame time per device class. _Harness, 22 local configurations and explicit coverage matrix: [docs/particle-scale-study.md](docs/particle-scale-study.md), with raw JSON and bundle sizes. Apple M1 Max measured; other device classes remain unmeasured and are tracked by 13.2._

## Epic 8 — Ocean and open water

Large 3D surfaces.

- [x] **8.1 Gerstner ocean** [B] (M) — sum of waves in the vertex stage with analytic normals.
- [x] **8.2 FFT ocean** [C] (L) — spectrum generation, compute-shader inverse FFT, displacement and derivative maps, multiple cascades.
- [x] **8.3 Foam from surface folding** [C] (M) — Jacobian-based whitecaps with temporal accumulation.
- [x] **8.4 Ocean shading** [B] (M) — combines 3.1, 3.6, 3.8 with horizon fog.
- [x] **8.5 Mesh strategy** [—] (M) — projected grid versus clipmap LOD, compared.
- [x] **8.6 Buoyancy** [B] (M) — floating objects that follow the surface; GPU readback versus a CPU mirror of the wave function.
- [x] **8.7 Shoreline** [B] (M) — breaking edge waves, depth foam, wet sand.
- [x] **8.8 River** [B] (M) — flow-mapped current with foam around rocks.
- [x] **8.9 Waterfall and mist** [B] (M) — scrolling noise sheets plus particle spray.
- [x] **8.10 Waterline transition** [B] (M) — split above/below camera with meniscus and underwater grading.
- [x] **8.11 Calm lake** [B] (M) — planar reflections with light wind ripples.

Implementation: seven demos (`gerstner-ocean`, `fft-ocean`, `shoreline`, `river`, `waterfall`, `waterline`, `calm-lake`), shared source, presets, notes and fallback posters. Numeric GPU tests and an 18-configuration local measurement: [docs/ocean-cost-study.md](docs/ocean-cost-study.md). M1 Max measured; other-device coverage remains with Epic 13.2. The write-ups state the scene approximations and readback/mesh limitations.

## Epic 9 — Liquid UI and web-native motifs

Effects shaped like website components. The most directly applicable group.

- [x] **9.1 Backdrop capture spike** [—] (M) — how to get page pixels into a `GPUTexture`: direct image/video upload, the experimental HTML-in-Canvas APIs, SVG `foreignObject` rasterisation, or re-rendering content inside the canvas. Document trade-offs; several stories below depend on the answer.
- [x] **9.2 Liquid glass panels** [A] (M) — rounded-rect SDF lens with refraction, edge dispersion, specular rim, and blur.
- [x] **9.3 Gooey metaballs** [A] (S) — smooth-min SDF blobs for merging nav items, buttons, and cursor followers.
- [x] **9.4 Liquid buttons** [A] (M) — spring-driven surface wobble on hover, ripple on press.
- [x] **9.5 Liquid transitions** [A] (M) — drip wipe, wave reveal, displacement dissolve between sections or routes.
- [x] **9.6 Liquid image hover** [A] (S) — ripple displacement and channel split on pointer.
- [x] **9.7 Liquid typography** [B] (L) — MSDF text that ripples, melts, or fills with water.
- [x] **9.8 Fill-level components** [A] (M) — progress bars, loaders, and tanks with a sloshing surface.
- [x] **9.9 Ray-marched liquid blob** [B] (M) — 3D SDF metaball hero with environment refraction.
- [x] **9.10 Water cursor trail** [A] (S) — a wet streak that spreads and dries.
- [x] **9.11 Scroll-linked water level** [B] (M) — the page submerges as you scroll.
- [x] **9.12 Droplets on UI** [B] (M) — condensation on a card; drops merge and run under gravity.

Epic 9 completion: eleven registered demos, a runnable `capture-spike.html`, shared renderer/source listings, genuine MSDF atlas and reproducible font generator, presets, fallback posters and measured scorecards. `docs/backdrop-capture-study.md` documents the chosen re-rendered backdrop plus actual canvas/SVG/experimental-API probe results. `docs/ui-cost-study.md` and JSON record 21 local 1080p configurations, active-input cases and the shared bundle cost. `tests/ui.test.ts` covers the component mechanics and lifecycle. Arbitrary live DOM capture is not the shipped path; mobile/other-device budgets remain Epic 13.2, small standalone packaging remains Epic 12.1.

## Epic 10 — Weather, droplets, and bubbles

Small-scale water phenomena.

- [x] **10.1 Rain streaks and splashes** [A] (M) — instanced, depth-layered streaks with wind and ground splashes.
- [x] **10.2 Rain ripple rings** [A] (S) — procedural expanding rings on a surface.
- [x] **10.3 Bubbles** [A] (M) — rising underwater bubbles with wobble and refraction; soap bubbles with thin-film colour.
- [x] **10.4 Splash burst** [A] (S) — droplet burst as click or tap feedback.
- [x] **10.5 Fogged glass** [B] (M) — condensation the pointer can wipe away, which slowly re-fogs.
- [x] **10.6 Frost and melt** [B] (M) — stretch goal: ice growing across glass, then melting into droplets.

Epic 10 completion: six registered weather demos cover all six stories, including the frost-and-melt stretch goal. Instanced rain and depth layers, impact splashes, expanding ripple rings, underwater and soap-film bubbles, press/tap droplet bursts, persistent wipe/refog state, and ice-to-water runoff are implemented. Each has presets, visible shared source and expanded WGSL, five-section notes, scorecards and captured fallback posters. Twelve local 1080p configurations, raw samples and bundle provenance are in [docs/weather-cost-study.md](docs/weather-cost-study.md). Numeric/lifecycle tests cover mechanics and bounds; browser checks cover all 24 page-context mounts and six phone-width layouts. M1 Max timings fit their target tiers; physical-phone and other-device budgets remain Epic 13.2. The glass backdrops are authored textures, frost is a designed coverage model and thin-film optics are approximate, as stated in the write-ups.

## Epic 11 — Other liquids

"Liquid-looking" is wider than water. Reuse existing simulations with different material parameters.

- [ ] **11.1 Viscosity presets** [—] (M) — honey, syrup, and slime variants of the Epic 6 and 7 solvers.
- [ ] **11.2 Liquid metal** [B] (S) — mercury / chrome shading on metaballs and particle fluids.
- [ ] **11.3 Lava** [B] (M) — emissive crust-and-glow shading with slow flow.
- [ ] **11.4 Milk and opaque liquids** [B] (S) — scattering-dominated shading with soft highlights.
- [ ] **11.5 Oil film** [A] (S) — thin-film interference on a water surface.
- [ ] **11.6 Paint mixing** [B] (M) — two or more dyes with pigment-style colour blending.
- [ ] **11.7 Material comparison page** [—] (S) — one simulation, every material, side by side.

## Epic 12 — Production readiness

What it takes to move an effect from demo to real site.

- [ ] **12.1 Embed API** (M) — every effect usable as `mount(canvas, options)` from a standalone ES module; per-effect gzip size reported. _Must also fix shader packaging: the shared WGSL library is one eagerly loaded chunk (45 kB gzipped after Epic 3, 70 kB after Epic 4), so every page pays for every effect's shaders. Each effect should carry only the chunks it includes. Shaders embedded in TypeScript strings (`bake.ts`, `cubemap.ts`) resolve includes at run time, so a build-time inliner alone is not enough._
- [ ] **12.2 Lifecycle hygiene** (S) — pause when offscreen or hidden, shared device across canvases, clean teardown.
- [ ] **12.3 Adaptive quality** (M) — dynamic resolution scaling toward a frame budget; simulation resolution tiers; `powerPreference`.
- [ ] **12.4 Accessibility** (M) — reduced-motion behaviour, a visible pause control, text-contrast guidance over moving backgrounds.
- [ ] **12.5 Fallback ladder** (M) — WebGPU → WebGL2 port (selected Tier A effects) → poster or video → CSS. Decide and document per effect.
- [ ] **12.6 Compatibility notes** (M) — optional features and limits that bite in practice (float filtering, storage formats, timestamp queries), and how to code around each.
- [ ] **12.7 Worker rendering** (M) — `OffscreenCanvas` rendering in a worker; measure main-thread benefit.
- [ ] **12.8 Framework recipes** (M) — React, Vue, and Svelte wrappers; SSR-safe initialisation.
- [ ] **12.9 Layering recipes** (S) — canvas behind content, pointer-event passthrough, alpha compositing, blend modes.
- [ ] **12.10 Three.js / TSL ports** (M) — port two or three effects and compare developer experience and bundle size.

## Epic 13 — Benchmarking and findings

Turn the demos into conclusions.

- [ ] **13.1 Benchmark harness** (M) — scripted run per effect at fixed resolutions; CPU and GPU times exported as JSON. _Method notes from Epic 3: time a frame as first pass start to last pass end, never the sum of passes (they overlap on tile-based GPUs); a trivial pass reads about 1.2 ms on Apple Silicon, so light effects need the workload repeated N times per frame and the slope taken; run with nothing else on the GPU._
- [ ] **13.2 Device matrix** (M) — results from an integrated-GPU laptop, Apple Silicon, a mid-range Android phone, and an iPhone.
- [ ] **13.3 WebGPU versus WebGL2** (L) — the same ripple and stable-fluids effects in both; compare performance and code complexity.
- [ ] **13.4 Thermal behaviour** (S) — frame-time drift over long mobile sessions.
- [ ] **13.5 Decision guide** (M) — "which effect for which job", with budgets and caveats.
- [ ] **13.6 Visual regression** (M) — screenshot tests with a deterministic time seed.
- [ ] **13.7 Bibliography** (S) — every reference, linked from the effects that use it.

---

## Risks and open questions

- **Measurement.** Per-pass GPU timestamps overlap on tile-based GPUs and have a floor of about 1 ms on the development machine, so HUD readings are indicative and light effects cannot be ranked by them. Tier verdicts rest on Apple Silicon until the Epic 13 device matrix exists.
- **HDR output.** An extended-range canvas cut the page's frame rate by more than half on the development machine. It stays opt-in until that is understood on other browsers and displays.
- **Backdrop capture.** Refracting live DOM content is the most requested liquid-UI effect and the least solved. Epic 9.1 chose re-rendered content and stable image/canvas uploads; its prototype found SVG bitmap decoding and experimental HTML capture unavailable on the local browser. Arbitrary live DOM refraction remains an experimental integration.
- **Browser coverage.** Support and optional-feature availability still vary by browser and platform. Story 1.10 pins this down before effects rely on anything optional.
- **Mobile compute limits.** Tier C effects may be desktop-only in practice. Epic 13 should say so plainly rather than tune indefinitely.
- **Scope.** There are over a hundred stories. M0–M3 form a complete, useful site; M4 and M5 can be trimmed without leaving holes.

## Starting references

- Evan Wallace, _WebGL Water_ — heightfield ripples and caustics.
- Jos Stam, _Stable Fluids_ (1999); Mark Harris, _Fast Fluid Dynamics Simulation on the GPU_ (GPU Gems, ch. 38).
- Pavel Dobryakov, _WebGL Fluid Simulation_ — a widely used web implementation of the above.
- Jerry Tessendorf, _Simulating Ocean Water_ (2001) — FFT ocean.
- Mark Finch, _Effective Water Simulation from Physical Models_ (GPU Gems, ch. 1) — Gerstner waves.
- Alex Vlachos, _Water Flow in Portal 2_ (SIGGRAPH 2010) — flow maps.
- Müller, Charypar, Gross, _Particle-Based Fluid Simulation for Interactive Applications_ (2003) — SPH.
- Macklin and Müller, _Position Based Fluids_ (2013).
- Hu et al., _A Moving Least Squares Material Point Method_ (2018) — MLS-MPM.
- van der Laan, Green, Sainz, _Screen Space Fluid Rendering with Curvature Flow_ (2009); Truong and Yuksel, _A Narrow-Range Filter for Screen-Space Fluid Rendering_ (2018).
- Ihmsen et al., _Unified Spray, Foam and Bubbles for Particle-Based Fluids_ (2012) — whitewater.
- matsuoka-601, _WebGPU-Ocean_ — MLS-MPM and screen-space rendering in WebGPU.
- Martijn Steinrucken, _Heartfelt_ (Shadertoy) — rain on glass.
- WebGPU and WGSL specifications; webgpufundamentals.org.

# Writing an effect

How to add an effect or a shared shader chunk to this site. `src/effects/hello/` is the smallest
working example; read it alongside this page.

## Files

```
src/effects/<id>/
  meta.ts      export const meta: EffectMeta    (bundled eagerly; keep it tiny, no imports but types)
  index.ts     export const effect = defineEffect({ meta, params, presets, create })
  *.wgsl       the effect's own shaders, imported with `?raw`
  notes.md     the write-up; headings fixed by docs/effect-notes-template.md
```

`<id>` is the directory name, the `meta.id`, and the URL (`?effect=<id>`). The registry discovers the
directory; nothing else needs registering. `meta.epic` is the epic number in plan.md as a string.

## The contract

`create(ctx)` receives `{ device, format, canvas, params, profiler }` and returns
`{ resize(width, height, dpr), frame(time, encoder, target), destroy() }`.

- `params` is a live object. Read it every frame; never copy it at creation.
- `frame` encodes passes into the encoder it is given and must finish by drawing into `target`
  (a view of the canvas texture, in `format`). The shell submits.
- `resize` is called before the first frame and whenever the drawing buffer changes. Size-dependent
  targets are (re)created there.
- `destroy` releases every GPU object and listener the effect made.
- Give every pass `timestampWrites: profiler.pass('label')` so the HUD can time it.
- Use `time.delta` to advance your own phase (see hello) so a `speed` parameter never makes the
  picture jump. Simulations step with `createFixedStepper` from `src/gpu/loop.ts`.
- Colours in `params` are linear RGB. Output to the canvas must be sRGB-encoded
  (`linear_to_srgb` from `color.wgsl`) because the canvas format is not an `-srgb` format.
- The canvas uses premultiplied alpha. An opaque effect writes alpha 1. A see-through effect
  (an overlay, a divider, mist) writes `vec4f(rgb * a, a)` and the page shows through it; check it
  over both a light and a dark page.
- There is no DOM in the tests: guard any use of `window`, `document` or `matchMedia`, and take
  input only through the `canvas` you are given.

## Helpers (use these before writing your own)

| Need | Where |
|---|---|
| Fullscreen fragment pass | `src/gpu/fullscreen.ts` + `#include "fullscreen.wgsl"` (uv origin top-left) |
| Uniform structs that match WGSL | `src/gpu/uniforms.ts` (`defineUniformStruct`, inject `Struct.wgsl` as a chunk) |
| Render targets, ping-pong | `src/gpu/targets.ts` |
| Textures from data / images, mipmaps | `src/gpu/texture.ts`, `src/gpu/mipmap.ts` |
| Blit, Gaussian blur, downsample | `src/gpu/passes/` |
| Pointer position and drags | `src/gpu/pointer.ts` |
| Shader modules (always) | `createShader` from `src/gpu/shader.ts`; never `device.createShaderModule` |
| Noise, FBM, Voronoi, curl, SDF, colour, maths | `src/wgsl/*.wgsl`; each file's header lists its functions |

## Shared WGSL chunks (`src/wgsl/`)

- Pure libraries: `const`, `fn`, and sparingly `struct`. No entry points, no bindings, no
  module-scope `var`. A chunk that needs a texture takes it as a function parameter.
- One flat namespace across all chunks: snake_case, a distinctive prefix per file, private helpers
  start with `_`. Each file starts with a header listing every public function, its meaning and
  ranges. `tests/wgsl-library.test.ts` enforces the header, compiles every chunk alone and all
  chunks together, so a name collision fails the suite.
- Each chunk `#include`s what it depends on.
- WGSL reserves many ordinary words as identifiers (`patch`, `crate`, `pass`, `filter`, `mod`,
  `target`, …). An "unexpected token" on a plain variable name usually means it is one of them.
- Prove behaviour with numbers: `tests/wgsl-eval.ts` runs WGSL on the GPU from a test
  (`gpuF32`, `scalars`).

## Budgets and limits

- Tier budgets (GPU time at 1080p): A ≤ 2 ms on an integrated GPU, B ≤ 6 ms, C ≤ 12 ms on a
  discrete or Apple Silicon GPU. Pick the tier the effect actually fits.
- Stay inside WebGPU core features and default limits. An optional feature is feature-detected
  (`device.features.has(...)`) and has a fallback path; say so in the scorecard.
- `rgba16float` is renderable and filterable everywhere. 32-bit float textures are not filterable
  without `float32-filterable`. See docs/support-matrix.md.
- No downloaded assets. Textures are procedural (generated in a shader or from typed-array data).

## Definition of done

1. Two or more presets, and parameters that are worth exposing (not every constant).
2. `meta.scorecard` filled in honestly. Leave `gpuMs` and `bundleKb` out: they come from benchmark
   runs, not from a glance at the HUD.
3. `notes.md` follows the template. Put rough HUD readings in "Cost", labelled with the device,
   resolution and "indicative".
4. Works in the page contexts (`&layout=hero`, `cards`, `band`, `backdrop`) at their sizes, and at
   phone width.
5. `npx vitest run` passes: `tests/effects.test.ts` runs every effect headlessly for three frames
   and fails on any validation error or a blank result; `tests/content.test.ts` checks the notes.
6. `npx tsc --noEmit -p tsconfig.json`, `npx eslint`, `npx prettier --check` are clean.

## Simulations

`src/gpu/sim/heightfield.ts` is the model for anything with persistent state (see
`src/effects/ripples/` and its notes):

- Keep the grid independent of the canvas size and step it with a fixed timestep, so it behaves
  the same at 60 Hz, 120 Hz and after the tab was in the background.
- Know the scheme's stability condition and clamp to it; a simulation that can explode will.
- State lives in `rgba16float`: it is renderable, filterable and a valid write-only storage format
  on every device. Ping-pong; never assume read-write storage textures.
- Expose a `generation` counter that bumps when textures are re-created, so consumers know to
  rebuild their bind groups.
- A render-pass solver can be ported to WebGL2 as a fallback; a compute solver cannot. Default to
  render passes unless compute is clearly needed, and set `scorecard.compute` accordingly.

## Reading the HUD honestly

- The HUD's GPU figure is the time from the frame's first timed pass starting to its last one
  ending. Never add up the per-pass figures: on tile-based GPUs (Apple Silicon, most phones) the
  intervals overlap, and eight tiny passes can sum to six times the real total.
- There is a floor. On the Apple Silicon development machine a pass that draws almost nothing
  reads about 1.2 ms, so anything at or below roughly 1 ms means "too cheap to measure here",
  not "costs 1 ms".
- Readings move with whatever else the GPU is doing. Quote them as indicative, with the device and
  resolution, and leave scorecard numbers to the benchmark runs.
- An HDR (extended-range) canvas made the whole page drop from 120 fps to 35-50 fps on that same
  machine. It is opt-in per visit (`&hdr=1`) and never used for thumbnails.

## Keeping a fragment effect cheap

What Epic 3 measured, as rules of thumb for a Tier A effect:

- Per-pixel noise is the budget. Five octaves of domain-warped FBM alone put `hello` at the Tier A
  limit. Count octaves and warps per pixel and keep the total small (roughly six noise evaluations).
- A texture fetch is far cheaper than noise: three layers of a baked normal map cost about a
  quarter of the equivalent procedural noise. Bake what does not need to change every frame
  (`src/gpu/bake.ts`, or render once into a target in `create`/`resize`).
- Skip work early for pixels that cannot show it (outside a glitter path, away from a waterline).
- Blur with mips or the downsample chain, not a wide kernel at full resolution.

## Checking in the browser

`npm run dev`, then `?effect=<id>&hud=1`. The HUD shows per-pass GPU time against the tier budget.
Shader errors appear in an overlay with the original file and line; edits hot-reload without a
page refresh.

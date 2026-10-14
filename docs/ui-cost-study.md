# Liquid UI study — Epic 9

Eleven effect modules implement stories 9.2–9.12. The [backdrop spike](backdrop-capture-study.md) resolves 9.1. All use fragment rendering; only the persistent cursor trail needs two per-frame passes. Measurements below are local Dawn/Metal GPU timings, not a cross-device certification.

## Method

Run `npm run study:ui`, with optional `UI_STUDY_DEVICE` and `UI_STUDY_CLASS` labels. Adapter identifiers, exact parameters and raw samples are recorded in [ui-cost-study.json](ui-cost-study.json). The benchmark is skipped in normal tests. Each configuration warms eight units, then alternates one and four repetitions for nine rounds; the reported GPU workload slope is median `(span(4) - span(1)) / 3`. Each span extends from the first pass begin to the last pass end, without summing overlapping intervals. If timestamps are unavailable, the metric is labelled **CPU encode + queue completion wall time**, not GPU time.

The live browser was navigated away before timing. Dawn/Metal timestamps in this run are quantised around 0.0655 ms; small differences between the cheap effects, and reversals between single-span and slope values, are below a useful ranking threshold. Frame phase advances through the workload. Multiple frames encoded into one command buffer use the last uniform queue write for that batch; this is a workload-throughput measurement, not a movie playback timing test. Startup shader compilation, atlas decompression, backdrop baking, user-image decoding/upload and presentation are excluded. The button default is idle, the image-hover default has no pointer, and scroll-water defaults to an unsubmerged page; active-input cases are included separately.

## Local complete frames at 1080p

Device: **Apple M1 Max, 32-core GPU**, macOS 26.6.2, Dawn Node WebGPU/Metal. All tests use rgba8unorm output at 1920 × 1080.

| Configuration                          | Input segments | Passes | Median slope (ms) | Median single span (ms) |
| -------------------------------------- | -------------: | -----: | ----------------: | ----------------------: |
| liquid-glass                           |              0 |      1 |             0.306 |                   0.328 |
| gooey-metaballs                        |              0 |      1 |             0.175 |                   0.197 |
| liquid-buttons                         |              0 |      1 |             0.153 |                   0.197 |
| liquid-transitions                     |              0 |      1 |             0.109 |                   0.131 |
| liquid-image-hover                     |              0 |      1 |             0.175 |                   0.262 |
| liquid-typography                      |              0 |      1 |             0.262 |                   0.262 |
| fill-level                             |              0 |      1 |             0.175 |                   0.262 |
| liquid-blob                            |              0 |      1 |             0.437 |                   0.524 |
| water-cursor                           |              0 |      2 |             0.175 |                   0.262 |
| scroll-water                           |              0 |      1 |             0.153 |                   0.197 |
| ui-droplets                            |              0 |      1 |             1.049 |                   1.114 |
| liquid-transitions / style=wave        |              0 |      1 |             0.087 |                   0.131 |
| liquid-transitions / style=dissolve    |              0 |      1 |             0.153 |                   0.197 |
| fill-level / style=bar                 |              0 |      1 |             0.197 |                   0.197 |
| fill-level / style=loader              |              0 |      1 |             0.197 |                   0.262 |
| liquid-typography / style=melt         |              0 |      1 |             0.218 |                   0.262 |
| liquid-typography / style=fill         |              0 |      1 |             0.197 |                   0.262 |
| water-cursor / moving coalesced path   |             16 |      2 |             0.218 |                   0.262 |
| liquid-image-hover / pointer hover     |              1 |      1 |             0.131 |                   0.197 |
| liquid-buttons / pointer hover + press |              1 |      1 |             0.153 |                   0.197 |
| liquid-glass / strength=2, blur=24     |              0 |      1 |             0.262 |                   0.328 |

All local configurations fit their numeric target tiers on this adapter. The one-pass droplet shader is the costliest default because it evaluates 48 lens slots per pixel. These results do **not** establish integrated-GPU or mobile budgets. Android, iPhone, other integrated and discrete GPUs remain unmeasured under Epic 13.2. No optional WebGPU feature is required; timestamp-query improves instrumentation only. In-app browser checks exercise capture, source display, presets, pointer/keyboard interaction and page layouts on this machine.

## Resources and bounded work

- Ordinary demos allocate one 1024 × 640 rgba16float section: **5 MiB**. Transitions allocate two: **10 MiB**. The atlas is 384 × 144 rgba8unorm distance data: **216 KiB**. The tiny inert textures and uniform buffers add little. These are GPU allocations, not compressed download sizes.
- Water cursor adds two half-resolution r16float fields: **1.98 MiB** at 1080p. Its diffusion/decay and display passes run once per frame; path deposition is capped at 16 segments. Resize clears the wet field. The active-path timing includes all sixteen segments.
- The 3D blob caps the entry march at **72** steps and the interior exit march at **40**. It uses an analytic environment and a screen-space backdrop, rather than a captured HDR panorama or scene-depth buffer.
- Droplets start with **32**, cap at **48**, and update a bounded pairwise CPU merge loop with up to 120 Hz substeps. GPU timing does not include this CPU loop, input processing, browser layout reads or command encoding. The shader still evaluates all 48 slots. This population is appropriate for a card, not a large particle simulator.
- Blur is a five-tap approximation. Backdrop resolution is fixed for the demonstration. User-image upload uses the stable external-image source path and may allocate an additional image texture.

## Text and packaging

The text atlas is real RGB MSDF output from msdfgen's edge-coloured font outlines, generated from Anton (OFL). Glyphs use four-pixel signed-distance range and fixed cell advances. All generation inputs and the pinned revision are in [scripts/ui](../scripts/ui/README.md). The runtime does not fetch a font; it decompresses the packed atlas once through DecompressionStream and uploads linear distance data. It is a fixed display-word demo, not a text shaping/layout engine.

The build keeps the renderer and atlas shared between eleven demos. `node scripts/ui-bundle-sizes.mjs` records [ui-bundle-sizes.json](ui-bundle-sizes.json). Effect code is roughly **54 kB gzipped**, including its shared UI implementation, atlas, local shaders and metadata. A wrapper alone is about 0.1 kB and is not a standalone component. The cold static dependency graph is roughly **151 kB**, including the common WGSL shader registry and generic GPU helpers. Those sums gzip each actual built file; they are not a predicted HTTP transfer under a warm browser cache, and they exclude gallery CSS/posters. Dedicated small embeds belong to Epic 12.1.

## Browser and interaction limits

The default backdrop re-renders a designed coastal illustration and labels. It does not capture the surrounding DOM. Each study has at least two presets, visible source/notes, a measured scorecard and a captured fallback poster. The shared host handles device loss, resize, DPR, hidden/offscreen pause and reduced motion. The liquid button is keyboard-operable; the fill component exposes its percentage through a progressbar role. Other images remain decorative, and an integration should keep real links/headings/content in HTML. The mocked Hero can cover the button or image input area: use Canvas context for unobstructed interaction and Backdrop context for nested-scroll testing.

The capture prototype successfully uploaded a 2D canvas on this browser; its SVG bitmap decode failed and experimental HTML capture methods were absent. See the [capture study](backdrop-capture-study.md) for exact observations and primary references.

## Validation and reproducibility

`tests/ui.test.ts` covers exact spring integration across time steps, volume-conserving coalescence, capillary pinning, swept collisions, drainage, genuine independent MSDF channels, source-file listing, all presets in linear half-float output, portrait/DPR reallocations, idempotent destruction, trail persistence/drying, pointer image hover, keyboard button input, exact transition endpoints and fill accessibility. The universal effect lifecycle tests additionally run all eleven at small resolutions. Capture outputs and a contact sheet are in `assets/epic-9`; the native browser capture evidence is `capture-spike.jpg`.

## References

- [WebGPU external image copies](https://www.w3.org/TR/webgpu/#dom-gpuqueue-copyexternalimagetotexture)
- [WICG HTML-in-Canvas proposal](https://wicg.github.io/html-in-canvas/)
- [Chrome HTML-in-Canvas API updates](https://developer.chrome.com/blog/html-in-canvas-ot-changes)
- [msdfgen — multichannel signed distance fields](https://github.com/Chlumsky/msdfgen)

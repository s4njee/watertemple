# Ocean and open-water study — Epic 8

Measured on 11 October 2026 (America/Chicago), Apple M1 Max, Metal through Dawn Node WebGPU. Raw samples, adapter identifiers and exact parameters are in [ocean-cost-study.json](ocean-cost-study.json). Seven demos implement stories 8.1–8.11. The visual techniques deliberately favor plausible website scenes over offline physical accuracy.

## Method and coverage

Run `npm run study:ocean`. `OCEAN_STUDY_DEVICE` and `OCEAN_STUDY_CLASS` supply a human-readable device label; identifiers are recorded separately. The normal test suite skips this measurement workload. Each configuration warms for eight units, then alternates one and four repetitions for nine rounds. The table reports median `(span(4) - span(1)) / 3`. A span runs from the first timed pass to the last, never a sum of overlapping pass intervals. Single-span medians and every sample remain in the JSON. Without timestamp queries the harness explicitly reports encoding plus queue-completion wall time; those values must not be compared to GPU milliseconds.

The browser demo was navigated away during measurement. Other desktop GPU users can still introduce noise; values below a millisecond and small ordering reversals do not establish a reliable winner. A preliminary run produced 0.59–1.20 ms default slopes, versus the final values below. These are costs of complete configured scenes, rather than isolated shader microbenchmarks or quality-matched comparisons. Animation phase progresses through the run. The local results do not establish a phone or generic integrated-GPU budget.

| Device class          | Coverage                                                             |
| --------------------- | -------------------------------------------------------------------- |
| Apple Silicon         | M1 Max measured through Dawn/Metal; in-app browser functional checks |
| Other integrated GPUs | Unmeasured; run the same study on target hardware                    |
| Discrete GPUs         | Unmeasured                                                           |
| Android / iPhone      | Unmeasured, including browser behavior and sustained thermal cost    |

Other-device measurement belongs to Epic 13.2. No optional feature is required by the demos; timestamps improve measurement only.

## Default complete frames at 1080p

| Effect         | Nominal triangles | Median slope (ms) | Median single span (ms) |
| -------------- | ----------------: | ----------------: | ----------------------: |
| gerstner-ocean |            18,432 |              0.39 |                    0.39 |
| fft-ocean      |            18,432 |              0.83 |                    0.98 |
| shoreline      |            18,432 |              0.72 |                    0.66 |
| river          |            18,432 |              0.55 |                    0.66 |
| waterfall      |            18,432 |              0.79 |                    0.85 |
| waterline      |            18,432 |              1.22 |                    1.25 |
| calm-lake      |            18,432 |              0.52 |                    0.52 |

The FFT default uses three 64² cascades, wave strength 0.6 and camera height 4.5 m. The other defaults use a projected 96-cell mesh. All local default slopes fit their numeric tier targets on this adapter. Tier B still calls for an integrated-GPU measurement; mobile remains **untested**. The waterfall includes 1,536 mist particles and the lake includes its half-resolution reflection pass.

## Projected grid versus clipmap

| Gerstner mesh | Cells | Nominal triangles | Median slope (ms) |
| ------------- | ----: | ----------------: | ----------------: |
| projected     |    96 |            18,432 |              0.39 |
| projected     |    48 |             4,608 |              0.39 |
| projected     |   160 |            51,200 |              0.66 |
| clipmap       |    48 |            18,432 |              0.63 |
| clipmap       |    96 |            73,728 |              0.55 |
| clipmap       |   160 |           204,800 |              0.83 |

Both meshes call the same vertex-stage wave function. **Projected** vertices start on a padded screen lattice and intersect a horizontal reference plane. The cap at 600 m prevents infinite horizon coordinates; a minimum reference height avoids collapse at camera y=0. This concentrates samples in the view and is the default for a fixed website camera. Fast pitch changes, steep crests and large horizontal chop can expose its finite coverage. Padding limits those edge holes rather than solving general frustum projection.

**Clipmap** geometry has five concentric levels at 0.5, 1, 2, 4 and 8 m spacing. Each outer edge collapses odd vertices onto the next lattice; coarser rings omit their central hole. The common center snaps in 8 m increments. Nominal draw triangles equal eight times cells², versus twice cells² for the projected mesh. The geometry remains stable under pitch changes and supports an all-around camera, but does more offscreen work and coarse-center snapping can pop. This simplified clipmap has no independent ring offsets, transition morph or streamed terrain cache. At 48 cells its far boundary is only 192 m away; fog hides much of that boundary.

Use **mesh** view to inspect the colored rings and **normals** view to compare sampling. Their triangle densities are not quality-equivalent. The noisy local timing curve supports choosing by camera behavior and image quality, rather than declaring either strategy universally faster. For moving cameras extend the ring alignment/morphing; for this site's fixed heroes use the projected default.

## FFT resolution and storage

| FFT samples per cascade | Mesh      | Median slope (ms) | Spectrum + field + map storage (MiB) |
| ----------------------- | --------- | ----------------: | -----------------------------------: |
| 64                      | projected |              0.83 |                                 1.22 |
| 128                     | projected |              0.92 |                                 4.88 |
| 256                     | projected |              1.46 |                                19.50 |
| 64                      | clipmap   |              0.61 |                                 1.22 |

The sizes above exclude depth, geometry, uniforms and the output canvas. Doubling resolution adds two FFT stages and quadruples storage and dispatch work. Three disjoint frequency bands prevent counting the same energy repeatedly. Changing resolution rebuilds seeds and maps, resets foam history and changes the band boundaries. Normals come from central differences of displacement. Per-cascade Jacobians inject persistent whitecaps; the combined surface uses maximum foam, so this approximates folding of the complete summed field. No current advection or physically conserved foam is claimed.

## Buoyancy: mirror versus readback

| Surface probes | CPU mirror median (ms) | GPU query slope (ms) | Submit/copy/map wall median (ms) |
| -------------- | ---------------------: | -------------------: | -------------------------------: |
| 6              |                  0.022 |                0.022 |                            0.305 |
| 96             |                  0.373 |                0.044 |                            0.297 |
| 768            |                  3.030 |                0.044 |                            0.302 |

CPU samples run the same six waves and eight bounded inverse iterations as WGSL; each timing averages twenty batches and records a checksum. The GPU columns measure compute-only query workload and synchronous end-to-end readback separately. GPU slopes are close to the timestamp floor for small batches. End-to-end numbers include command encoding, submission, buffer copy and `mapAsync`, without a browser's rendering queue or a frame interval.

The two visible buoys require six probes total. The CPU mirror is simpler and returns current values immediately. GPU computation becomes attractive as probe count grows, or when only a GPU-resident surface is available. The interactive GPU path has one readback in flight, initiates mapping on the next frame after submission, and reuses the latest completed positions meanwhile. It therefore adds at least a frame of temporal lag even though the isolated map workload is fast. GPU samples include their query time. Buoys are kinematic followers with tilted signal tips, without rigid-body forces, collision response or wakes. The GPU query example samples Gerstner waves; it does not claim to read the FFT maps.

## Validation and visual limits

Real-GPU tests compare analytic tangents with finite differences, invert displaced world coordinates, compare CPU/GPU buoyancy, compare all three FFT displacement channels with a direct complex DFT, and check foam accumulation and lifetime decay. Lifecycle tests exercise every preset, both meshes, allocation changes from 64 to 128 to 256 and back, resize, underwater heights including exactly zero, repeated frames in one encoder and teardown. Shared WGSL chunks compile independently and together.

All seven scenes have captured WebP posters, at least two presets, generated controls, readable shared source and a five-section write-up. The shoreline uses bounded shoaling and a procedural wetness mask; the river uses baked potential flow and foam masks; the waterfall uses a procedural sheet and particle spray; the waterline uses a near-camera meniscus test and display-space grading; the lake reflects procedural distant silhouettes in a mirrored-camera pass. These are documented scene approximations. They do not replace a fluid volume solver, physical shoreline breaking, Snell-window ray tracing or a complete rigid-body system.

## Built bundle cost

Build, then run `node scripts/ocean-bundle-sizes.mjs`. [ocean-bundle-sizes.json](ocean-bundle-sizes.json) lists reachable files. The tiny entry wrappers select a scene. The scorecard counts the wrapper, metadata, shared ocean implementation and its local FFT/scene shader chunks as effect code; the JSON lists those files separately. Shared camera, general GPU helpers and the eager WGSL library are excluded from that figure and included in cold static imports. The roughly 7 kB shared implementation alone is insufficient to render a scene. Cold shared totals sum gzip bytes once, excluding the shell, source viewer, notes and posters. They represent this site build, not an extracted standalone embed.

| Effect         | Effect code (gzip kB) | Entry wrapper (gzip kB) | Cold static imports (gzip kB) |
| -------------- | --------------------: | ----------------------: | ----------------------------: |
| gerstner-ocean |                 13.30 |                    0.11 |                        120.18 |
| fft-ocean      |                 13.29 |                    0.11 |                        120.17 |
| shoreline      |                 13.29 |                    0.11 |                        120.17 |
| river          |                 13.29 |                    0.11 |                        120.17 |
| waterfall      |                 13.28 |                    0.11 |                        120.16 |
| waterline      |                 13.28 |                    0.11 |                        120.16 |
| calm-lake      |                 13.28 |                    0.11 |                        120.16 |

The eager shared WGSL library accounts for most of the cold import cost. The common renderer also reaches FFT and spray helpers for scenes that do not execute them. Per-effect shader extraction and tree-shaking remain the Epic 12.1 packaging follow-up. A small wrapper is not evidence of a small standalone download.

## References

- Mark Finch, [Effective Water Simulation from Physical Models](https://developer.nvidia.com/gpugems/gpugems/part-i-natural-effects/chapter-1-effective-water-simulation-physical-models), GPU Gems, 2004.
- Jerry Tessendorf, [Simulating Ocean Water](https://evasion.inrialpes.fr/Membres/Fabrice.Neyret/NaturalScenes/fluids/water/waves/fluids-nuages/waves/Jonathan/articlesCG/simulating-ocean-water-01.pdf), course notes, 1999–2004.
- Arul Asirvatham and Hugues Hoppe, [Terrain Rendering Using GPU-Based Geometry Clipmaps](https://developer.nvidia.com/gpugems/gpugems2/part-i-geometric-complexity/chapter-2-terrain-rendering-using-gpu-based-geometry), GPU Gems 2, 2005.

# Particle scale study — Epic 7

Measured on 11 October 2026 (America/Chicago), Apple M1 Max, Metal through Dawn Node WebGPU. Only this machine was measured. Raw samples and the exact effect parameter objects are in [particle-scale-study.json](particle-scale-study.json).

## Method

`npm run study:particles` runs the opt-in workload in `tests/particle-study.test.ts`. Set `PARTICLE_STUDY_DEVICE` and `PARTICLE_STUDY_CLASS` when naming another machine; the adapter identifiers are recorded independently. The normal test suite skips the study. It uses core WebGPU; timestamp queries are optional. If unavailable, the report clearly labels the metric as CPU encode plus queue-completion wall time, which must not be mixed with GPU times.

Each configuration warms up for eight units, then alternates one and four repetitions in one submission for nine rounds. A GPU span starts at the first pass and ends at the last pass. The per-unit workload slope is `(span(4) - span(1)) / 3`; the table reports its median, not a sum of overlapping per-pass timestamps. Individual samples and the median one-unit span are retained. The browser demo was navigated away during measurement; this is still a desktop session with WindowServer and other background work, not an isolated GPU laboratory. Differences below roughly a millisecond and small reversals in the iteration curve are noise-sensitive. Negative slopes indicate a timing floor or contention, never negative cost.

The solver workload is a stationary regular lattice with gravity and artificial pressure disabled, to avoid progressively compressing the pool during the repeated measurement. It measures neighbour and iteration work at approximately rest occupancy. It is not a fidelity comparison and does not predict a dense impact or collapsed clump. Counts are actual filled lattice counts. A substep is a grid rebuild, prediction, density/correction iterations and viscosity; multiply by the number of substeps only as a rough estimate of simulation cost.

Complete frames use each effect’s actual default scene and parameters at 1920 × 1080, including its rendering chain. PBF 3D includes 4096 secondary whitewater slots. The MLS-MPM wave tank has different geometry and liquid volume from the PBF dam break, so the two are examples of shipping configurations rather than a controlled solver-only comparison. Requested 3D counts can exceed the initial live count because whole lattice rows must fit or a pour is still filling. Changing the count also changes spacing/rest occupancy, not just a dispatch dimension.

## PBF solver scaling

| Solver | Live particles | Iterations | Median slope, ms/substep | Median single span, ms |
| --- | ---: | ---: | ---: | ---: |
| pbf-2d | 4,096 | 2 | 0.42 | 0.39 |
| pbf-2d | 4,096 | 4 | 0.94 | 0.59 |
| pbf-2d | 16,384 | 2 | 0.39 | 0.46 |
| pbf-2d | 16,384 | 4 | 0.85 | 0.79 |
| pbf-2d | 65,536 | 2 | 0.85 | 1.38 |
| pbf-2d | 65,536 | 4 | 2.16 | 1.84 |
| pbf-3d | 3,375 | 2 | 0.66 | 0.72 |
| pbf-3d | 3,375 | 4 | 1.49 | 1.25 |
| pbf-3d | 15,625 | 2 | 0.94 | 1.11 |
| pbf-3d | 15,625 | 4 | 1.57 | 2.03 |
| pbf-3d | 64,000 | 2 | 2.18 | 2.42 |
| pbf-3d | 64,000 | 4 | 4.39 | 3.80 |

## Complete 1080p frames

| Effect | Requested count / spacing | Median slope, ms/frame | Median single span, ms |
| --- | --- | ---: | ---: |
| particles | 250,000 | 1.38 | 2.03 |
| sph-2d | 3,000 | 2.08 | 1.64 |
| pbf-2d | 12 mm | 1.70 | 1.70 |
| pbf-3d | 10,000 | 4.33 | 3.47 |
| pbf-3d | 40,000 | 7.86 | 6.49 |
| pbf-3d | 100,000 | 13.48 | 13.43 |
| mpm-3d | 10,000 | 2.01 | 2.36 |
| mpm-3d | 40,000 | 3.30 | 3.15 |
| mpm-3d | 100,000 | 4.81 | 4.59 |
| fluid-surface | 60,000 | 2.73 | 2.62 |

The starting configurations were tuned from this study: SPH now requests 3,000 particles, 2D PBF starts at 12 mm spacing, and 3D PBF at 10,000 particles. The larger presets remain available. The PBF tank at 100,000 exceeds the 12 ms Tier C target in both runs. At 40,000 it measured 7.86 ms in the final run and 17.24 ms earlier; at 10,000 it measured 4.33 and 11.71 ms. That variation is why the default stays at 10,000. MLS-MPM at 100,000 measured 4.81 ms in the final run and 8.89 ms earlier. Its fixed grid remains a cost at smaller counts. Earlier one-unit spans were sometimes higher than the workload slopes, another sign of shared-desktop contention. The earlier summary is retained in the JSON; the full raw samples are from the latest run. No count is a universal frame-rate guarantee. Tier B additionally needs an integrated-GPU measurement, and mobile safety has not been established.

## Device-class coverage

| Class | Result | Remaining validation |
| --- | --- | --- |
| Apple Silicon desktop/laptop | M1 Max measured above | Repeat on a base M-series integrated GPU and in browsers |
| Other integrated laptop GPUs | Not measured | Run the same workload on the target hardware |
| Discrete desktop GPUs | Not measured | Run on the target driver/browser |
| Android phone | Not measured | Count sweep, compatibility and thermal drift |
| iPhone | Not measured | Count sweep, Safari coverage and tilt permission |

The harness and the local count curves complete the scale-study implementation. Collecting the other device classes is tracked by Epic 13.2; the blank classes are not estimates. Real device-tilt behavior also needs a physical phone even though its mapping, permission lifecycle and teardown are tested headlessly.

## Built bundle cost

Run `npm run build` followed by `node scripts/particle-bundle-sizes.mjs`. The JSON report is [particle-bundle-sizes.json](particle-bundle-sizes.json). “Own module” counts the entry chunk and any effect-module re-export wrapper; shared solver/runtime imports are excluded there. “With shared imports” sums gzip bytes of all statically reachable JavaScript chunks once. It excludes the shell, optional source viewer, notes and posters, and assumes a cold cache. This is the current site build, not an extracted standalone embed.

| Effect | Own module, gzip kB | With shared imports, gzip kB |
| --- | ---: | ---: |
| particles | 4.46 | 134.15 |
| sph-2d | 18.76 | 143.32 |
| pbf-2d | 4.70 | 140.69 |
| pbf-3d | 6.68 | 161.68 |
| mpm-3d | 20.06 | 156.56 |
| fluid-surface | 3.94 | 132.98 |

The shared WGSL chunk remains large and contains unrelated effects. Shader packaging is the existing Epic 12.1 follow-up; the small effect-module number alone is not the cost of embedding a particle fluid.

## Validation and artifacts

The regular tests cover the storage framework, scan/counting sort, neighbour exactness, SPH and PBF physical invariants, MLS-MPM transfers and atomics, screen-space surface reconstruction, whitewater classification/emission/lifetime, and container gravity/permission lifecycle. Every registered effect is exercised headlessly; each Epic 7 effect has two or more presets, a scorecard, template headings and a captured poster. The 3D tank was also checked in the browser, in a hero context and at 390 px width without horizontal overflow. The host now pauses for visibility, intersection and reduced motion, and schedules a static redraw when a paused canvas resizes. Posters are actual rendered frames, not illustrations.

Scratch parameter sweeps and manual capture probes are preserved under `experiments/`; they no longer block the production typecheck or run as part of the regression suite.

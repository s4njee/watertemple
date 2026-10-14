## How it works

A fountain of up to a million spray particles, and the framework under it (`src/gpu/particles/`, stories 7.1 and 7.2): the pieces every particle effect needs, so the fluid solvers that follow write only physics. The demo is deliberately not a fluid: the particles do not interact. It exists to prove the framework and to make the neighbour search visible.

**Lantern behind the page.** [Open the standalone lantern demo](lantern.html). The same fountain physics gets an orange speed palette, a fixed nozzle and a front orthographic camera. Shared camera coordinates anchor the spray to the open rim of the sourced Maui Torch photograph as the viewport resizes. The fixed scene has `pointer-events: none` and sits behind ordinary DOM text, links and opaque cards. The lantern renders linear radiance into an `rgba16float` scene, applies bloom and then maps to an extended-range HDR canvas (up to the selected 1–6× SDR white), with SDR fallback. CSS blending is avoided so the compositor can retain the extended range. Auto HDR/SDR, glow, HDR light, pause and relight controls demonstrate the output. Reduced motion starts paused; missing WebGPU shows a static glow.

**The particle system** (`system.ts`) owns the buffers in the `ParticleSet` layout (`array<vec4f>` positions with a per-particle scalar in `w`, velocities, plus any named extra arrays) and everything that is not physics: filling from a function or typed arrays, emitting from an emitter description (box, sphere, disc, point; a cone of directions; a life), recycling the oldest particles in a ring, growing and shrinking the count, changing the capacity (which bumps `generation`), and reordering every array at once through a permutation. `system.particles` is a live `ParticleSet` that always points at the current buffers. **The integrator** (`integrator.ts`) is semi-implicit Euler with gravity, damping, a force field (a WGSL function or an acceleration texture), a box with restitution and friction, and respawn: `velocities.w` is the life, and a particle whose life ran out is re-seeded from the emitter inside the shader. It runs fixed 1/120 s steps from a clock, capped per frame, all in one dispatch. `compute.ts` is the thin helper under every pass: pipelines through `createShader`, cached per device, bind groups cached per buffer identity (a reorder changes the buffers every step), dispatch sizing, profiler labels.

**The renderer** (`billboards.ts`) draws a `ParticleSet` as one camera-facing quad per particle, sized in world units with a floor in pixels, coloured through a palette by the scalar, additive, alpha-blended or opaque (lit sphere impostors that write the depth of the sphere, so intersecting spheres are right with no sorting), for perspective and orthographic cameras. The particle data reaches the vertex stage as an instance-step vertex buffer, not a storage buffer: compatibility mode gives vertex stages no storage buffers at all (docs/support-matrix.md, 5.1), the renderer then needs no bind group for the particles (a reordered or ping-ponged buffer is simply set per draw), and it is the same speed. The colour can come from `positions.w` (a stride-16 attribute at offset 12 of the same buffer) or from a separate f32 buffer, which is how the neighbour views colour particles.

**The neighbour search** (`neighbor-grid.ts`, `scan.ts`, `src/wgsl/neighbors.wgsl`) is a uniform grid with cell side at least the search radius, built every step by a counting sort:

1. `cells`: each particle computes its cell and takes a ticket with `atomicAdd` on the cell's counter. The ticket is its rank in the cell, which fixes its slot later (one atomic pass, no second pass with a cursor).
2. `scan`: an exclusive prefix sum over the cell counters, in place (`scan.ts`). Each workgroup of 256 scans a tile of 1024 elements in workgroup memory (1 KiB of the 16 KiB every device has); the tile totals are scanned the same way, recursively, and added back: 3 dispatches for a 64³ grid, 5 for 128³, exact for up to 67 M elements. A decoupled look-back scan would be one pass, but needs forward-progress guarantees between workgroups that WebGPU does not give.
3. `scatter`: particle i goes to slot `start[cell] + rank`, giving `sorted[slot] = i`. Plain stores.
4. `reorder` (optional): every per-particle array is gathered through `sorted` and the buffers are swapped, so slot k _is_ particle k.

Cells are numbered x-fastest, so the particles of three x-adjacent cells are contiguous, and a neighbour loop reads three ranges in 2D and nine in 3D instead of 9 or 27 cells. The domain is a bounded box (positions outside are clamped into the edge cells, which keeps the search exact); a spatial hash would allow an unbounded domain, but its collisions put strangers in the loop and break the contiguity, and every solver here runs in a box. A solver includes `neighbors.wgsl` and the grid's generated bindings chunk and writes the loop shown in the header of that file, which is also what `tests/neighbors.test.ts` extracts, compiles and checks against brute force. The demo's colour-by-neighbour-count pass (`neighbor-count.wgsl`) is that loop unchanged.

**What the demo does each frame.** Integrate (a curl-noise force field from `curl.wgsl`, 3D or animated 2D, plus a vortex around the pointer in the 2D scene); respawn at the nozzle, which follows the pointer's ray on the floor; if a neighbour view is on (or "build the grid every frame" is), build the grid; for `neighbour count`, run the reference loop into a scalar buffer; for `neighbours of picked`, find the particle nearest the pointer on screen (two atomic passes over the particles: the smallest screen distance, then the lowest index at it), and mark its neighbours from the grid, so a hole in the highlighted ball would be a search bug; then the backdrop and the box, then the billboards.

## Cost

The maintained Epic 7 measurements are in `docs/particle-scale-study.md`, with raw samples in `docs/particle-scale-study.json`. Those complete-frame numbers use the current defaults at 1080p; the earlier studies below use the configurations stated there. A captured poster is available without WebGPU.

Measured on an Apple M1 Max (Metal through Dawn in Node) with timestamp queries, by the method of `docs/effect-authoring.md`: the cost of a call is the **slope** of the GPU span between a small and a large number of repetitions recorded into one command buffer, so the ~1 ms floor of a single pass drops out. The machine was shared with a Firefox GPU process, WindowServer, Chrome, the Claude desktop app, another agent's tests and a virtual machine; a pass that another process pre-empts can read 13 ms for 0.1 ms of work, so each figure is the slope between the **quietest** of 40 alternating small runs and the quietest of 40 large ones (the slope between the median runs, which a busy machine shows, is in brackets in the studies, 10 to 50 percent higher and above that for the tiniest costs). The timer resolution is about 66 µs. Reproduce with `PARTICLE_STUDY=/tmp/grid.txt npx vitest run tests/neighbors.test.ts -t study` and the same for `tests/particles.test.ts`. Indicative only; scorecard `gpuMs` is left to the benchmark runs.

Particles in a fluid-like pool (the lower 40 percent of a unit box), **in random order in memory** (the worst order for an index list, and what a freshly spawned system looks like), a radius that gives about 32 neighbours in 3D and 16 in 2D, a position and a velocity (32 bytes) moving together. Milliseconds per call:

| Dim | Particles | Cells   | Build, index | Count, index | Sum   | Build, reorder | Count, reorder | Sum   |
| --- | --------- | ------- | ------------ | ------------ | ----- | -------------- | -------------- | ----- |
| 3D  | 10 000    | 3 375   | 0.048        | 0.176        | 0.225 | 0.051          | 0.041          | 0.093 |
| 3D  | 50 000    | 17 576  | 0.066        | 1.300        | 1.365 | 0.069          | 0.100          | 0.169 |
| 3D  | 100 000   | 32 768  | 0.080        | 2.327        | 2.406 | 0.085          | 0.191          | 0.275 |
| 3D  | 250 000   | 85 184  | 0.144        | 6.652        | 6.796 | 0.111          | 0.393          | 0.504 |
| 3D  | 500 000   | 166 375 | 0.255        | 14.69        | 14.95 | 0.176          | 0.786          | 0.963 |
| 3D  | 1 000 000 | 328 509 | 0.426        | 38.38        | 38.81 | 0.382          | 1.595          | 1.977 |
| 2D  | 10 000    | 5 041   | 0.050        | 0.050        | 0.100 | 0.054          | 0.013          | 0.067 |
| 2D  | 50 000    | 24 649  | 0.071        | 0.300        | 0.371 | 0.064          | 0.034          | 0.098 |
| 2D  | 100 000   | 49 284  | 0.094        | 0.557        | 0.652 | 0.071          | 0.062          | 0.133 |
| 2D  | 250 000   | 123 201 | 0.139        | 1.507        | 1.646 | 0.108          | 0.127          | 0.235 |
| 2D  | 500 000   | 246 016 | 0.246        | 3.419        | 3.665 | 0.176          | 0.246          | 0.422 |
| 2D  | 1 000 000 | 491 401 | 0.448        | 9.262        | 9.710 | 0.382          | 0.481          | 0.863 |

"Count" is the reference neighbour-counting pass (the first loop of any SPH or PBF step: visit every candidate, test the distance, count). **Reordering is the whole story**: the build is the cheap part either way (under half a millisecond for a million particles), and the pass that follows it is 4 to 24 times faster on reordered data (13 times at 50 000 particles, 24 at a million), because the candidates of a particle are adjacent in memory and the index indirection is gone. Moving the 32 bytes of every particle costs 0.2 ms at a million.

**Where the build goes** (what each stage adds to the cumulative span, ms; noise about ±0.03). Left, particles in random order (index mode always sees this); right, reorder mode in steady state (the data is already nearly in cell order, so the atomics hit nearby addresses):

| Dim | Particles | cells | + scan | + scatter | cells | + scan | + scatter | + gather |
| --- | --------- | ----- | ------ | --------- | ----- | ------ | --------- | -------- |
| 3D  | 100 000   | 0.054 | 0.017  | 0.008     | 0.050 | 0.011  | 0.004     | 0.019    |
| 3D  | 250 000   | 0.100 | 0.007  | 0.037     | 0.061 | 0.016  | 0.007     | 0.027    |
| 3D  | 1 000 000 | 0.242 | 0.028  | 0.156     | 0.135 | 0.032  | 0.015     | 0.201    |
| 2D  | 1 000 000 | 0.249 | 0.041  | 0.158     | 0.121 | 0.046  | 0.025     | 0.191    |

`cells` includes the buffer clear, the cell computation and the atomic count. The scan is a few hundredths of a millisecond for the grids that go with up to a million particles; the cost is the two passes over the particles, and below 100 000 particles the whole build is a handful of dispatches at the timer's resolution. A grid much finer than the particles need (4 million cells) makes the scan as costly as the particle passes. The scan alone, in place over u32: 4 097 elements (16³) 0.011 ms in 3 dispatches, 32 769 (32³) 0.012, 262 145 (64³) 0.027, 2 097 153 (128³) 0.112 in 5 dispatches, 4 096 001 (about the default cap of 4.19 M cells) 0.197.

**What a neighbour costs.** 3D, a pool, ms for the counting pass:

| Particles | Neighbours | Random order | Reordered | Index on coherent data |
| --------- | ---------- | ------------ | --------- | ---------------------- |
| 250 000   | 8          | 1.82         | 0.197     | 0.281                  |
| 250 000   | 32         | 8.60         | 0.404     | 0.721                  |
| 250 000   | 128        | 25.1         | 1.42      | 2.11                   |
| 1 000 000 | 8          | 11.4         | 0.776     | 1.20                   |
| 1 000 000 | 32         | 51.3         | 2.01      | 3.62                   |
| 1 000 000 | 128        | 184.9        | 7.72      | 9.62                   |

The cost is linear in the neighbour count (about 27/(4π/3) = 6.4 candidates are visited per true neighbour): reordered, the GPU tests about a hundred billion candidates a second, on random data about six billion. "Index on coherent data" is index mode after the data was reordered earlier: it is 1.3 to 1.8 times slower than reordered and 6 to 19 times faster than index on random data, so reordering every few steps and building an index list in between is a real middle path. **Contention**: 250 000 particles in one cell build in 0.42 ms against 0.12 to 0.14 for spread-out ones; the atomics serialise a little, nothing more. The loop in that cell is quadratic, of course.

**What it means for the solvers** (extrapolation, not measured: the multiplier for a real kernel is a guess of 3 to 4 times the distance test, and a solver has more arrays to read). With reordering, a build plus one pass of 32 neighbours costs 0.17 ms at 50 000 particles, 0.5 ms at 250 000 and 2 ms at a million, so SPH (about two such passes plus integration) is under 3 ms at 200 000 and about 15 ms at a million; position-based fluids with four constraint iterations (two passes each) are about 7 ms at 200 000 and out of reach at a million. Without reordering the same passes are 10 to 20 times more expensive and the 200 000 case is already over the Tier B budget.

**The demo itself**, ms per frame at 1280×720 with the fountain's default settings (additive, colour by speed, 2 sub-steps per frame), quietest of 24 alternating runs:

| Particles         | Frame | Integrate pass        | Draw pass |
| ----------------- | ----- | --------------------- | --------- |
| 100 000           | 0.46  | 0.0 (below the timer) | 0.39      |
| 250 000 (default) | 1.11  | 0.07                  | 1.05      |
| 500 000           | 2.47  | 0.20                  | 2.23      |
| 1 000 000         | 4.87  | 0.39                  | 5.18      |

**What limits the particle count**: first the draw, then (when a view uses it) the neighbour loop, and the simulation and the sort last. The integrator, with a curl-noise field evaluated once per frame, is 0.4 ms for a million particles. The grid build is 0.4 ms. The draw is 5 ms for a million sprites of the default size, and it does not depend on the sprite size until they are large: at a million particles the frame is 4.8 to 5.8 ms for sizes 0.25 to 1 (a sprite is already covered by its one-pixel floor), 7 ms at twice the size and 14 to 16 ms at four times (about 10 px radius), where the overdraw of additive blending takes over. A canvas with a quarter of the pixels is slower at a million particles (6.6 ms at 640×360 against 4.9), because the same sprites pile onto fewer pixels, so the cost is the rasterising and blending of a million tiny overlapping quads, not the vertex shader or the sort. Reordering helps the draw a little too (the particles are then drawn in cell order, so neighbouring sprites hit neighbouring pixels): 5.4 ms with the grid reordering against 5.5 without (inside the noise; a first run read 4.7 against 7.0), and 7.0 with an index list, 1.4 ms more than no grid, which is more than the build's 0.4 ms and which I did not explain. The neighbour views are different: colour by neighbour count at a million particles is 18 ms, 11 of them in the counting pass, because the fountain is far from uniform: the jet holds thousands of neighbours per particle at the default radius of 0.07. **Cost scales with local density times radius cubed, not with the particle count alone**, which is the lesson for the solvers: they keep the density near rest density, which is why their cost is predictable and this demo's is not. In the browser (Chrome pane, DPR 2, 968×976, a shared and busy GPU, indicative): 250 000 particles 2.6 ms GPU at about 107 fps, a million 7.3 ms GPU at about 100 fps; the neighbour views at 250 000 were 28 to 77 ms in the dense core because of the same density effect, which is why the default colour is speed and the neighbour views are opt-in.

The default (250 000 particles, speed) is about 1.1 ms here, **Tier B** by the shape of the work (compute with no WebGL2 path, scales with count) and the reference-device caveat: the verdict rests on Apple Silicon, and a million particles is a desktop number. Memory: positions and velocities are 32 bytes per particle, plus 32 more while the data is reordered (the back buffers), the grid 12 bytes per particle and 4 bytes per cell. For scale: a million particles are 32 MB of particle data (64 MB with the back buffers a reordering allocates), 12 MB of per-particle grid scratch and 1.4 MB of cells at a 70³ grid.

## Pitfalls

- **Index mode on unordered data is the slow path**, by 10 to 20 times at the particle counts that matter, and the failure is silent. A freshly spawned system is in random order. Use reorder mode, or reorder every few steps.
- **After a reorder, particle indices change.** Everything per particle must live in the system's attributes (`extras`) so it moves too; a loose buffer keyed by particle index is wrong the moment the grid reorders. `system.permute` moves all attributes in one call so none can be forgotten, but it cannot move what it does not own. To follow one particle, keep an id attribute.
- **The order inside a cell is not deterministic** (atomics hand out the ranks), so sums over neighbours differ in the last bits between runs. Harmless in a fluid.
- **The grid is bounded.** Positions outside the box are clamped into the edge cells, which stays exact, but a fluid that escapes piles into the edge cells and the loops there grow. A box too big or a radius too small for the cell budget makes the grid coarsen (bigger cells, still exact, more candidates) instead of failing; `layout.coarsened` says so. The cell side is padded by 2^-12 so two particles exactly a radius apart are never split by rounding; positions are exact to about 1e-4 of a cell up to a thousand cells from the origin.
- **Do not trust per-pass HUD figures here.** The passes of a frame overlap on this GPU; the grid's passes read 0.0 to 0.1 ms in the HUD while the whole frame is milliseconds. Use the span.
- **A uniform written with `queue.writeBuffer` lands before the whole command buffer.** The integrator rotates through four uniform buffers so a warm-up and a frame can share a submit; the renderer's camera and style uniforms are written once per draw, so two draws with different cameras in one submit would both see the last. Submit between them.
- **Dawn on Metal misbehaves when a pass is given only one of its two timestamps** (it ran out of memory); the cost study gives every timed pass both, with dummy slots for the unwanted ones.
- Additive sprites are summed in the encoded (sRGB) space, which is the usual convention for glow but not physically additive; a dense cloud saturates to white, so brightness is a parameter. Alpha-blended sprites are not depth-sorted: with a reordering grid the draw order changes every frame and overlaps shimmer. Use additive or opaque with a grid.
- Limits to stay inside: 8 storage buffers per stage (the search takes 1 in reorder mode and 2 in index mode, plus a uniform), 256 invocations per workgroup (the scan uses 256, with 1 KiB of workgroup memory), 128 MiB per storage binding (8.4 M particles of a vec4 array, 33.5 M cells), and 65 535 workgroups per dispatch dimension (4.19 M particles at workgroup size 64; `runCount` throws beyond it instead of dropping particles).

## When to use it

The cheapest honest particle effect for hundreds of thousands of non-interacting sprites (mist, spray, snow, embers) on a desktop-class GPU, and the foundation of the particle fluids (Epic 7): build a solver by registering its per-particle arrays as extras of a `ParticleSystem`, calling `grid.build` each step, and writing the documented loop. Tier B: compute with no WebGL2 fallback, and 250 000 particles are about a millisecond on an M1 Max, a million about five. For a hero that must reach phones or integrated GPUs, cut the count by ten, or use a vertex-shader flow (no state, no compute) or a pre-rendered loop; the neighbour views are a research tool, not a feature. Reach for the screen-space fluid surface (story 7.6) when the particles should read as a liquid, and for the grid solvers of Epic 6 when the volume matters more than the splashes.

## References

- Simon Green, _Particle Simulation using CUDA_ (NVIDIA, 2010): the uniform grid sorted by cell, the origin of the counting-sort build.
  https://developer.download.nvidia.com/assets/cuda/files/particles.pdf
- Rama Hoetzlein, _Fast Fixed-Radius Nearest Neighbors: Interactive Million-Particle Fluids_ (GTC 2014): counting sort and prefix sum for fluids, and the case for reordering the data.
  http://on-demand.gputechconf.com/gtc/2014/presentations/S4117-fast-fixed-radius-nearest-neighbor-gpu.pdf
- Mark Harris, Shubhabrata Sengupta, John Owens, _Parallel Prefix Sum (Scan) with CUDA_, GPU Gems 3, chapter 39: the work-efficient scan the tile scan follows.
  https://developer.nvidia.com/gpugems/gpugems3/part-vi-gpu-computing/chapter-39-parallel-prefix-sum-scan-cuda
- Duane Merrill, Michael Garland, _Single-pass Parallel Prefix Scan with Decoupled Look-back_ (NVIDIA, 2016): the single-pass alternative, and why it needs forward-progress guarantees.
  https://research.nvidia.com/publication/2016-03_single-pass-parallel-prefix-scan-decoupled-look-back
- Robert Bridson, Jim Hourihan, Marcus Nordenstam, _Curl-Noise for Procedural Fluid Flow_ (SIGGRAPH 2007): the divergence-free field that carries the spray.
  https://www.cs.ubc.ca/~rbridson/docs/bridson-siggraph2007-curlnoise.pdf
- Matthias Müller, David Charypar, Markus Gross, _Particle-Based Fluid Simulation for Interactive Applications_ (2003), and Miles Macklin, Matthias Müller, _Position Based Fluids_ (2013): the solvers this neighbour search is built for.
- WebGPU specification, limits and compatibility mode; docs/support-matrix.md in this repository for the default limits used above.

## How it works

An incompressible 2D flow solved on a grid, in the Stable Fluids formulation (Stam 1999; Harris, GPU Gems ch. 38), all in compute shaders. One simulation step is: **advect the velocity** by itself, **add forces** (vorticity confinement, buoyancy, body force), optionally **diffuse** (viscosity), compute the **divergence**, **solve the pressure equation** with Jacobi iterations, **subtract the pressure gradient** so the flow is divergence-free again, then **advect the dye** by the result. Pointer strokes and idle drops enter as _splats_ (capsules of momentum and dye) in an extra pass before the steps, and a _resolve_ pass writes the texture a shader reads. The solver is `src/gpu/sim/fluid.ts` (`fluid-core.wgsl` the grid and samplers, `fluid-advect.wgsl`, `fluid-forces.wgsl`, `fluid-project.wgsl` the passes); `src/gpu/sim/fluid-stir.ts` turns pointers into splats; `src/wgsl/fluid.wgsl` decodes the output; `display.wgsl` in this folder is the only shading.

**The grid is staggered (MAC).** Velocity components live on the faces between cells (u on the left face of a cell, v on its top face) in one `rgba16float` texture of (W+1) x (H+1) texels; pressure, divergence and vorticity live at cell centres. A collocated grid (everything at the centre) is the easy choice and what most web fluids use, but then divergence and the pressure gradient are wide central differences that do not match the compact 5-point pressure Laplacian: the projected field keeps a residual at short wavelengths, the chequerboard mode is invisible to the solver, and a one-cell wall leaks. On the MAC grid divergence, gradient and Laplacian are the same compact differences, so projection is exact up to the linear solve (the tests take a Gaussian source down to under half a percent of its starting divergence with SOR, and to 2 percent with Jacobi at 1024 iterations), a solid face is exactly zero (a test shows a one-cell wall stopping dye and flow completely), and a wall is a velocity of zero rather than a mirrored ghost. What it costs: every interpolation of velocity is a few more taps (u and v are sampled at different offsets, `fluid_velocity_at` does it with two filtered taps), and consumers read a `fields` texture (cell-centred velocity, vorticity, pressure) that one resolve pass makes for them. Boundaries: `wall` is closed and **free-slip** (normal velocity zero, tangential free), `open` lets the flow and the dye leave (pressure zero outside, clear water enters), `wrap` is periodic. The obstacle mask is `src/gpu/sim/obstacles.ts`, shared with the heightfield solver: solid cells block their faces in advection, divergence and projection, and solid neighbours are skipped in the interpolation, so the fluid slides along an obstacle instead of being dragged to a stop.

**Two resolutions.** The velocity grid is coarse (128 cells on the short side by default); the dye is a separate `rgba16float` grid (512 on the short side) advected by the coarse velocity, which is where the visible detail comes from. All state is `rgba16float` (renderable, filterable and write-only storage in core on every device); pressure, divergence and vorticity are `r32float` (core storage, read with `textureLoad`, since it is not filterable without a feature). No pass uses read-write storage: each reads a sampled texture and writes a different one, and the solver rotates through a pool of three identical textures (hence `fluid.dye.slot`). Every pass binds exactly one storage texture (the default limit is four) and at most five sampled textures.

**Pass structure per step** (three compute passes, one dispatch per kernel, all pressure iterations in the same pass):

| Phase    | Dispatches (semi-Lagrangian / MacCormack / BFECC)                             | Storage textures written |
| -------- | ----------------------------------------------------------------------------- | ------------------------ |
| velocity | advect 1 / 2 / 3, curl + blur 2 (confinement on), forces 1, viscosity 12 (on) | 1 per dispatch           |
| pressure | divergence 1, Jacobi or SOR N, project 1                                      | 1 per dispatch           |
| dye      | advect 1 / 2 / 3                                                              | 1 per dispatch           |

Once per update (not per step): the splat pass (1 or 2 dispatches) and the resolve pass (1). Pipelines are cached per device; bind groups are cached by the textures they hold.

**Advection schemes** (`fluid-advect.wgsl`). Semi-Lagrangian: sample the field where the parcel came from (a midpoint back-trace by default). It is unconditionally stable and smears everything. MacCormack (Selle et al. 2008): advect forward, advect that result backward, add half the difference to the forward result, then **clamp to the smallest and largest of the source samples the forward trace interpolates**. BFECC: forward, backward, form `f + (f - back) / 2`, advect that, then the same clamp. The limiter is what makes both stable and free of new extrema; it can be switched off (`limiter: false`) only to show what it prevents (table below). Velocity and dye choose their scheme independently.

**Vorticity confinement** (Fedkiw, Stam, Jensen 2001) adds an acceleration `eps * h * (N x w)` that turns the flow round the nearest vortex centre in the sense it already turns, with `N` the unit gradient of `|w|` and `h` the cell size. It puts back the small vortices that interpolation in advection smooths away. `N` is taken from a 3 x 3 blurred `|w|`, not the raw one (see Pitfalls).

**Time.** The solver steps at a fixed rate (60 Hz) with `createFixedStepper`, capped at four steps per frame, so it looks the same at 60, 120 and 144 Hz and after a stalled tab (tests compare them); the trace distance is capped at `maxCfl` cells, velocities are clamped to the matching speed and NaN or Inf are replaced by zero, so absurd forces (tests use 3e38 and 256 splats at once) leave a finite field.

**Input** (`fluid-stir.ts`): each pointer's path since the last frame is cut into a few straight capsule splats, so a fast drag is continuous; the capsules have a smooth ramp at both ends, so consecutive frames' capsules add up to a stroke of uniform density with no beads at the joins (the test checks the density along a stroke stays within a factor of two of its mean). The fluid is given the pointer's own velocity times `Push`, capped; a pressed pointer that stands still drips; a tap puffs (six small outward pushes, which the projection turns into a ring of vortices). A press-drag stirs; by default a hover pushes at a third of the strength and paints faintly; touch drags **scroll the page** unless `Touch drags` is `paints`, in which case the canvas takes the pointer (`touch-action: none`), the same policy as `ripples`. The colour of a stroke steps through the palette per press (golden-ratio steps) and drifts along the stroke; idle strokes (two emitters on slow Lissajous paths and the occasional drop) keep the canvas alive and ease off while someone is touching it.

**Colour.** Dye is HDR-ish and linear. On a dark ground it is emissive: `rgb = colour * amount`, shaded as `e * (1 - exp(-peak)) / peak`, a soft shoulder on the brightest channel that keeps the hue (plain clipping washes bright blobs to white, and averaging complementary colours gives grey mud; this keeps saturation, and a toe fades the faintest haze to nothing). On a light ground it is subtractive: the dye holds **optical density** (`-ln` of the pigment colour times the amount), densities add, the page is multiplied by `exp(-density)`, so blue over yellow is green and thin ink fades to the paper. A dye-gradient normal gives the optional sheen. The debug views show velocity (hue is direction), pressure, the divergence the projection removed, and vorticity.

## Cost

Measured on the Apple Silicon development laptop (Metal through Dawn). The per-step figures come from GPU timestamp queries in `tests/fluid.test.ts` (run `FLUID_STUDY=/tmp/fluid.txt npx vitest run tests/fluid.test.ts -t study`): the slope of the span between the first and last timed pass between 4 and 44 steps, median of 7 submits, which removes the ~1 ms floor of a single pass. They are **indicative**, not benchmark-run numbers (the scorecard leaves `gpuMs` empty), and they moved by a factor of two to three between runs while another tab was using the GPU: the quietest consistent pair of runs is quoted.

| One step, default settings (MacCormack both, 30 Jacobi iterations) | ms  |
| ------------------------------------------------------------------ | --- |
| velocity 128² cells, dye 512² (the demo)                           | 1.1 |
| velocity 128², dye 1024²                                           | 2.8 |
| velocity 256², dye 512²                                            | 1.8 |
| velocity 256², dye 1024²                                           | 3.4 |

What dominates, at the demo setting: the dye advection (0.59 ms of 1.1, MacCormack at 512²), then 30 pressure iterations (0.24 ms), then the velocity advection (0.2 ms); forces, divergence and projection are about 0.06 ms together. At 1024² the dye is two thirds of the step.

| Advection of one field | Dispatches | Bilinear samples per texel (four loads each; velocity counted per component) | Dye 512² / 1024² | Velocity 128² / 256²                                  |
| ---------------------- | ---------- | ---------------------------------------------------------------------------- | ---------------- | ----------------------------------------------------- |
| semi-Lagrangian        | 1          | 5 (4 velocity, 1 field)                                                      | 0.17 / 0.67 ms   | 0.11 / 0.23 ms (whole step with dye 64², 1 iteration) |
| MacCormack             | 2          | 11 (8 velocity, 3 field)                                                     | 0.59 / 2.2 ms    | 0.21 / 0.50 ms                                        |
| BFECC                  | 3          | 16 (12 velocity, 4 field)                                                    | 0.62 / 2.3 ms    | 0.21 / 0.49 ms                                        |

Pressure iterations cost 0.008, 0.018 and 0.076 ms per dispatch at 128², 256² and 512² cells (Jacobi and SOR alike). Below 256² the cost is mostly the fixed price of a dependent dispatch (about 0.5 ns per cell at 128², 0.3 at 256² and up), which is why a tiled kernel that does several iterations per dispatch would pay there and not at 512². Confinement adds two dispatches (about 0.04 to 0.07 ms at 128²); viscosity with 12 iterations about 0.15 ms.

Residual divergence against iterations (rms divergence after one projection from zero, divided by before; Gaussian source of sigma 0.1 on the canvas, then sigma 0.03; Jacobi weight 0.8, SOR with the optimal omega):

| Iterations                 | 4    | 8    | 16   | 32   | 64   | 128  | 256   | 512  | 1024 |
| -------------------------- | ---- | ---- | ---- | ---- | ---- | ---- | ----- | ---- | ---- |
| Jacobi, 64², large source  | 0.94 | 0.89 | 0.80 | 0.67 | 0.48 | 0.30 | 0.16  | 0.07 | 0.02 |
| SOR, 64², large source     | 6.0  | 7.4  | 4.7  | 2.0  | 0.36 | 0.06 | 0.01  | 0    | 0    |
| Jacobi, 128², large source | 0.99 | 0.97 | 0.94 | 0.89 | 0.80 | 0.67 | 0.48  | 0.30 | 0.16 |
| SOR, 128², large source    | 6.4  | 9.9  | 10.8 | 6.2  | 2.3  | 0.74 | 0.17  | 0.05 | 0.01 |
| Jacobi, 128², small source | 0.86 | 0.74 | 0.58 | 0.39 | 0.22 | 0.11 | 0.05  | 0.02 | 0.01 |
| SOR, 128², small source    | 5.0  | 4.3  | 2.5  | 1.1  | 0.21 | 0.03 | 0.005 | 0    | 0    |

Jacobi converges like cos(pi/N)^k, so the long wavelengths of an N-cell grid need of the order of N² iterations and halving the cell size quadruples the work (the 128² rows are the 64² rows shifted by a factor of four in iterations). Red-black SOR needs of the order of N, but is not monotone: below about N/2 iterations the residual of the long wavelengths is _higher_ than before projecting, so it is an option for high iteration counts, not a default. In a running simulation the previous step's pressure is a good start (warm start), and the mean ratio of divergence after to before projection over steps 30 to 90 of a force circling a closed box is:

| Iterations       | 15   | 30   | 60   | 120  |
| ---------------- | ---- | ---- | ---- | ---- |
| Jacobi 0.8, 64²  | 0.60 | 0.34 | 0.13 | 0.05 |
| Jacobi 0.8, 128² | 0.77 | 0.76 | 0.59 | 0.32 |
| SOR, 64²         |      | 0.28 | 0.06 | 0.01 |
| SOR, 128²        |      | 0.70 | 0.24 | 0.11 |

So at 128² with the default 30 iterations a step removes about a quarter of its divergence and the rest is carried (and mostly removed over following steps by the warm start); the flow is visibly incompressible at the scale of the strokes and not at the scale of a few cells. The cure for more is a multigrid solve (not built), not more Jacobi: SOR at 60 to 120 iterations is the available step up.

Memory: velocity is three textures of (W+1) x (H+1) x 8 B (0.4 MB at 128²), the dye two or three of 8 B per texel (2.1 MB each at 512², 8.4 MB at 1024²; the third exists once a corrected scheme runs), plus pressure (two), divergence, vorticity (two) and fields at 4 to 8 B per cell. The HUD read about 7 MB and 12 to 14 textures for the whole page at 128² / 512².

In the browser (Chromium dev pane, canvas 968 x 976 = 0.94 Mpx, DPR 2) the HUD read 120 fps and 2.3 to 2.6 ms of GPU for a whole frame (one step, resolve, shade) with nothing else running; its per-pass figures overlap and sit on the ~1.2 ms floor, so only the repeat-N slope means anything (`Steps per frame` runs N steps per frame: the slope was 2.6 ms per step while another tab was busy, against 1.1 ms in the quiet Node runs above). Tier B budget is 6 ms at 1080p: this is inside it on this machine at the default sizes. Not measured: any integrated GPU other than Apple's, any phone, any browser but Chromium.

### Scheme comparison

One revolution (240 steps at 60 Hz, a quarter turn per second) of a Gaussian blob (sigma 0.03 = 7.7 dye texels) and a hard disc (radius 0.05) round a vortex, velocity grid 64², dye 256², prescribed flow (`dyeOnly`), so only the advection scheme differs. Peak and width are relative to the start; edge is the 10 to 90 percent width of the disc in dye texels; over and under are the largest values above 1 and below 0 the disc ever reaches.

| Scheme                          | Blob peak | Blob width | Blob total | Disc edge    | Overshoot / undershoot       | Dispatches | Cost (512² dye) |
| ------------------------------- | --------- | ---------- | ---------- | ------------ | ---------------------------- | ---------- | --------------- |
| semi-Lagrangian, Euler trace    | 0.54      | x1.21      | 0.78       | 18 texels    | 0 / 0                        | 1          |                 |
| semi-Lagrangian, midpoint trace | 0.55      | x1.29      | 0.92       | 20 texels    | 0 / 0                        | 1          | 0.17 ms         |
| **MacCormack, limited**         | **0.94**  | **x1.01**  | **0.96**   | **7 texels** | **0 / 0**                    | 2          | 0.59 ms         |
| BFECC, limited                  | 0.90      | x1.00      | 0.88       | 6 texels     | 0 / 0                        | 3          | 0.62 ms         |
| MacCormack, NO limiter          | 0.94      | x1.00      | 0.96       | 6 texels     | +0.24 / -0.25 (negative dye) | 2          |                 |
| BFECC, NO limiter               | 0.91      | x1.00      | 0.88       | 5 texels     | 0 / -0.10                    | 3          |                 |

Semi-Lagrangian loses half the peak and widens the blob by 30 percent in one turn; both corrected schemes keep it essentially intact and the disc edge about three times sharper. Without the limiter MacCormack overshoots the disc by 24 percent and undershoots below zero by 25 percent, which as dye is negative ink; the limiter costs a few texels of edge sharpness and nothing else. The midpoint trace does not sharpen anything (the schemes diffuse in amplitude, not phase) but keeps the total (0.92 against 0.78) and the rotation radius true. Vortex retention, a Gaussian vortex (sigma 0.045) on a 64² grid for 3 s without dissipation: peak vorticity retained semi-Lagrangian 19 percent, MacCormack 46, BFECC 45; enstrophy 6, 40 and 34 percent. Dye total after 72 steps of vigorous stirring in a closed box (converged projection): semi-Lagrangian 0.97 to 1.0, MacCormack 1.24, BFECC 1.15; with 30 Jacobi iterations all three drift up by 20 to 50 percent because the under-converged flow compresses.

**Recommendation.** Dye: MacCormack. It keeps the sharpness of BFECC (a 7-texel disc edge against 6, and a higher blob peak), loses less mass in the rigid test (4 percent against 12), costs one dispatch less, and the limiter makes it unconditionally safe. Semi-Lagrangian is the choice when the budget is a phone (a third of the cost, soft ink, which for smoke looks fine). Velocity: MacCormack as well (it retains twice the vorticity of semi-Lagrangian for 0.1 ms at 128²); BFECC has no advantage there either. Keep the midpoint trace on.

## Pitfalls

- **Confinement amplifies grid-scale noise.** The first version took `N` from the raw `|w|`; in quiet water the gradient of cell-scale ripples points at a different cell each step, confinement amplifies whatever it points at, and the vorticity view showed a lattice of spurious vortices spreading over the canvas (seen in the browser; with confinement off the same scene is clean). Blurring `|w|` over 3 x 3 before taking its gradient keeps the vortices of three cells and more and sends the noise away. It is also a feedback loop: an isolated vortex spins up at roughly 0.07 x eps per second (measured), so `eps` must stay below about 14 x (velocity dissipation + 0.3) or a vortex runs away (with no dissipation at all, eps = 12 multiplied the energy of a single vortex by 7.5 in 3 s). The presets obey that rule.
- **Undamped Jacobi has a parity bug.** The textbook iteration multiplies the chequerboard error mode by -1 per iteration, so an odd iteration count over-corrects it: with the same simulation, 30 iterations left 0.24 of the divergence and 31 left 1.11, worse than not projecting at all. The default weight 0.8 shrinks that mode by 0.6 per iteration (0.34 at 30 and 0.32 at 31), at a fifth more iterations for the smooth modes. `jacobiWeight: 1` brings the bug back and is only there to show it.
- **SOR is not a drop-in speed-up.** Optimal over-relaxation makes the residual rise before it falls (table above); from a cold start 32 iterations at 128² left six times the divergence it started with, and even warm-started it removes less than Jacobi below about N/2 iterations. Use it at 60 iterations and more.
- **Dye is a tracer, not a conserved quantity.** It is advected as a scalar by an interpolated velocity whose divergence is zero only cell by cell, so its total drifts: tests bound it at 0.6 to 1.6 of the start in a hard stir, and the table above gives the measured values. For ink this is invisible; do not use it as a mass.
- **Momentum and dye share a stroke.** The push gives the fluid under the pointer the pointer's velocity, and dye is laid in the same fluid, so a fast stroke carries its own ink forward as a bulb with a tail. `Push` 0.6 and a velocity decay of 0.6 per second keep that a plume rather than a jet; at 1.2 with 0.35 a drag across the canvas carries the ink to the far wall.
- **Resizing clears the fluid.** The grids are re-created when the aspect changes, and a re-created grid is empty. The effect only re-sizes the solver when the canvas shape changes by more than 6 percent (in between the fluid is stretched a few percent), so dragging a window edge does not wipe the ink; changing `Velocity cells` or `Dye texels` does.
- **Half-float precision.** Velocity is stored as 16-bit floats: relative error 5e-4. After projection the divergence cannot fall below about a thousandth of the velocity per cell (SOR in the test stops at 5e-3 of the starting divergence), which is invisible and is also why a 32-bit velocity texture was not worth losing filtering for.
- **Textures change under you.** The dye and the velocity ping-pong through three textures; read `fluid.dye` each frame and cache bind groups by `slot`, dropping the cache when `generation` changes. `fluid.fieldsView` is stable. The pressure and divergence textures are `r32float`: `textureLoad` only, and an explicit bind group layout (`sampleType: 'unfilterable-float'`), as `display.wgsl`'s pipeline does.
- **One `update` per submitted command buffer.** The parameters go through one uniform buffer written with `queue.writeBuffer`; two updates encoded before one submit share the last values.
- **Timings move with whatever else uses the GPU.** The HUD varied by a factor of two to three between runs because other work shared the device; quote slopes, not single readings.

## When to use it

An interactive set piece where the visitor stirs ink, smoke or colour into a hero, a card or a full-page background, and an ambient version of the same that never repeats: Tier B on this machine at 128 cells and a 512 dye grid (about 1 ms of GPU per step, one step per frame), 128 / 1024 or 256 / 512 for a showpiece. It needs compute shaders, so there is no WebGL2 fallback without a rewrite as render passes over float targets (which is how the classic web fluids do it); show a looping video or WebP of the idle ink instead. Pick the sizes by what dominates: the dye grid (and its scheme) sets the cost, the velocity grid sets how small the vortices can be, and 30 pressure iterations are right for strokes of a few cells and not for large smooth flows. If you only want the look of flowing colour and nobody touches it, the analytic `ink-wash` background is far cheaper. The solver is also the base for fluid-driven page distortion, obstacles from the DOM, ink with diffusion and a solver cost study: they use `createFluid` (`fields`, `dye`, `obstacles`, `addForce`, `addDye`, `hooks`) and this effect is the smallest consumer.

## References

- Jos Stam, [Stable Fluids](https://pages.cs.wisc.edu/~chaol/data/cs777/stam-stable_fluids.pdf), SIGGRAPH 1999: the semi-Lagrangian step and the projection.
- Mark Harris, [Fast Fluid Dynamics Simulation on the GPU](https://developer.nvidia.com/gpugems/gpugems/part-vi-beyond-triangles/chapter-38-fast-fluid-dynamics-simulation-gpu), GPU Gems chapter 38.
- Harlow and Welch, _Numerical Calculation of Time-Dependent Viscous Incompressible Flow of Fluid with Free Surface_, 1965: the staggered (MAC) grid.
- Robert Bridson, _Fluid Simulation for Computer Graphics_ (2nd ed., 2015): boundary conditions on the MAC grid, solids, and the pressure solve.
- Selle, Fedkiw, Kim, Liu, Rossignac, [An Unconditionally Stable MacCormack Method](https://physbam.stanford.edu/~fedkiw/papers/stanford2008-04.pdf), 2008.
- Dupont and Liu, _Back and forth error compensation and correction methods for removing errors induced by uneven gradients of the level set function_, 2003; Kim, Lee, Ko, [Flowfixer: Using BFECC for Fluid Simulation](https://doi.org/10.2312/NPH/NPH05/051-056), 2005.
- Fedkiw, Stam, Jensen, [Visual Simulation of Smoke](https://web.stanford.edu/class/cs237d/smoke.pdf), SIGGRAPH 2001: vorticity confinement.
- Pavel Dobryakov, [WebGL Fluid Simulation](https://github.com/PavelDoGreat/WebGL-Fluid-Simulation): the widely used web implementation (collocated grid, render passes) this one is compared with.
- Press et al., _Numerical Recipes_ section 19.5: red-black ordering and successive over-relaxation; David Young, _Iterative Solution of Large Linear Systems_, 1971: the optimal omega.
- Inigo Quilez, [Palettes](https://iquilezles.org/articles/palettes/): the cosine palettes of the strokes.
- docs/support-matrix.md section 6: why `rgba16float` and `r32float`.

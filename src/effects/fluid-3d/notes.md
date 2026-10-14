## How it works

A drop of ink released into a closed glass tank of water, simulated as an incompressible 3D flow on a grid and drawn by marching camera rays through the ink. The solver is `src/gpu/sim/fluid3d.ts` (with `fluid3d-core.wgsl`, `-advect.wgsl`, `-forces.wgsl` and `-project.wgsl`): the 2D stable-fluids solver of `fluid-dye` carried into three dimensions, all in compute shaders with state in 3D textures. The ray-marcher is in this folder: `volume-prep.wgsl` (two small compute passes), `march.wgsl` and `composite.wgsl`, with the reusable pieces (ray-box intersection, jitter, the per-step transmittance and in-scatter accumulation, a phase function) in `src/wgsl/volume.wgsl`.

**One step.** Advect the velocity (semi-Lagrangian, midpoint trace), add forces (buoyancy from the ink, gravity, vorticity confinement), take the divergence, solve for the pressure, subtract its gradient, advect the ink (MacCormack with a limiter). Pointer pushes and drops enter as splats (Gaussian capsules of momentum and ink) before the steps. The step is fixed at 60 Hz with `createFixedStepper`, at most three steps per frame, the trace distance is capped at 6 cells, velocities are clamped to match and NaN or Inf are replaced by zero (a test throws 3e38 at it from 128 splats and checks every texel stays finite and bounded).

**Where it differs from the 2D solver, and why.**

- _Staggered (MAC) grid, kept._ Velocity components on cell faces in one `rgba16float` 3D texture of (n+1)³ texels, pressure and divergence at cell centres in `r32float`. The task offered a collocated grid as the simple option. It would read a velocity with one filtered tap instead of three (`fluid3d-core.wgsl`, `vel_at`), which matters in advection, but its divergence and gradient are wide differences that do not match the compact 7-point pressure laplacian, so the projection leaves a residual at short wavelengths, and at 48 to 96 cells short wavelengths are the rolled-up ring. The MAC grid projects exactly (up to the linear solve), and a wall face is exactly zero (a test checks the six walls are bit-exactly zero after a hard shove). The price was not isolated: I did not build the collocated variant, so "how much faster would it be" is not measured.
- _Two resolutions, as in 2D._ The velocity grid is coarse (default 48 x 64 x 48 cells for a 1 x 1.25 x 1 tank), the ink is a separate `rgba16float` grid (72 x 96 x 72) carried by the coarse velocity, which is where the visible detail comes from. The ink is stored as an optical density per unit length in rgb (`-ln` of the colour times the amount, so two inks add: subtractive mixing, blue over yellow is green) and the amount in alpha (what buoyancy and scattering follow).
- _Multigrid pressure, new._ The 2D notes ended with "the cure is a multigrid solve (not built)". In 3D it is needed: damped Jacobi needs of the order of N² sweeps for the long wavelengths of an N-cell grid, and at 64 cells that is hundreds. `pressureSolver: 'multigrid'` is a V(2,2) cycle (`cs_restrict` takes the residual of eight children to a grid of half the size, times 4; `cs_prolong` interpolates the correction back trilinearly; the smoother is the same damped Jacobi, now with the weight 6/7 that is optimal for the 7-point laplacian) with up to three coarsenings (64 to 32, 16, 8). Grid sizes are multiples of 8 so it halves cleanly. Warm start: the fine level starts from last step's pressure, coarse levels from zero. Plain `jacobi` remains an option, and it is what runs while an obstacle sphere is on (obstacles are not represented on the coarse grids).
- _No viscosity, no BFECC._ Neither earned its passes in 2D for what the picture needs; velocity dissipation stands in for friction. MacCormack is available for the velocity too (`velocityScheme`).
- _Splat counts and `queue.writeBuffer`._ The uniform buffer is written with `queue.writeBuffer`, which lands before the whole command buffer. `resolve()` (used by the velocity debug view) first rewrote the buffer with the splat counts reset to zero, which silently deleted every drop that frame (found in the browser: the debug view showed an empty tank). The counts of the last `update` are now remembered and repeated, and a test covers update, resolve and project in one encoder.

**Rendering.** `volume-prep.wgsl` runs two small compute passes: a light volume (half the ink grid's size; for every voxel the transmittance toward the sun, per channel, so a blue ink casts a blue shadow, plus how open the voxel is to the sky straight above it) and an occupancy volume (the largest ink amount in each 4 x 4 x 4 block). `march.wgsl` is a fullscreen pass at a fraction of the canvas resolution (`renderScale`) writing two targets, the radiance scattered toward the eye and the transmittance left over. Each pixel intersects its ray with the tank (`volume_ray_box`), starts one jittered step in, and takes a fixed number of steps through the box; a step looks up the occupancy (empty: skip), the ink, the light volume, and adds `sigma_s * tint * (sun * transmittance * phase + ambient)` through `volume_accumulate`, which integrates the step exactly (so the step count changes noise, not brightness; a test shows 1 step and 64 give the analytic transmittance of a slab), and leaves early when the transmittance is below 0.01. Extinction is coloured absorption (the stored optical density times a gain) plus a grey scattering term, and the light a step scatters is tinted by the ink's own colour `exp(-density / amount)`: ink reads as ink, not as smoke. `composite.wgsl` upsamples the two targets bilinearly at full resolution and puts the backdrop (a studio gradient and a table) behind them and the tank in front: the tank is found analytically from the ray's entry and exit points on the box (edges within a pixel of a box edge), so the far edges are seen through the ink (attenuated by its transmittance) and the near edges over it, with exact depth order and no geometry. Then exposure, ACES tonemapping and sRGB. Debug views replace all of it with a slice through the velocity (hue is direction) or the ink.

**Interaction.** Orbit camera (`stepOrbit`, idle drift). A press whose ray hits the tank starts a push: a tap (little travel, short time) releases a lumpy drop (a main blob and four satellites, so it breaks up) where the ray enters the tank; a drag pushes the fluid along the stroke, in the plane through the entry point facing the camera, with the pointer's own velocity capped and times `Push`. A press that misses the tank orbits (`Dragging: orbits the camera` makes every drag orbit). Pointers are captured and `touch-action` is `none` while the effect lives, a deliberate policy (as `pool-3d`): this is a showpiece, not a scrolling background; `touch-action` and `user-select` are restored on destroy. Left alone, a drop falls every few seconds from a random place near the top (`reducedMotion` makes them rarer and stops the drift).

**What a drop does** (contact sheets rendered headlessly at 64 / 96, one frame every 0.7 s): it sinks as a ball with a tail, develops a mushroom cap, rolls up into a vortex ring while it falls (the ring forms without vorticity confinement too; at 64 / 96 side-by-side sheets with it off and on at 4 differ only in a slightly crisper ring, so I would not claim more for it than that, and it matters more as the cell count falls), the ring hits the floor, spreads as a wrinkled ring along it, climbs the walls and breaks into tendrils. At 96 / 160 the ring has visibly finer wrinkles and the tendrils branch more; at 32 / 64 the ring still forms but is smooth and fat. The smoke preset (negative buoyancy, white ink, a puff every few frames from below) makes a string of rising puff rings.

## Cost

Measured on the Apple Silicon development laptop (Metal through Dawn in Node, `tests/fluid3d.test.ts`, run `FLUID3D_ROUNDS=30 FLUID3D_STUDY=/tmp/fluid3d.txt npx vitest run tests/fluid3d.test.ts -t study`), with timestamp queries and the method of `docs/effect-authoring.md`: the **slope** of the span from the first timed pass to the last, between a small and a large amount of work (8 and 68 steps, or 4 and 20 frames, or 1 and 9 marches), so the ~1 ms floor of a single pass drops out. The GPU was shared with other agents' browser tabs and tests the whole time, and a busy moment only ever adds time, so each figure is the **quietest of many round-robin samples** (the items are sampled in turn, 30 to 70 rounds, so a busy moment hurts them alike); even so the same configuration read 1.1 and 2.6 ms in two runs when the load was heavy. Quote these as indicative, scorecard `gpuMs` is left to the benchmark runs. Tank 1 x 1.25 x 1; `resolution` counts cells along the long (vertical) side.

**The solver, ms per step** (defaults: multigrid x2, confinement 4, MacCormack ink, semi-Lagrangian velocity, workgroup 4 x 4 x 4):

| Velocity cells (ink texels)                     | Resolution / ink setting | ms per step |
| ----------------------------------------------- | ------------------------ | ----------- |
| 24 x 32 x 24, 18k (48 x 64 x 48, 147k)          | 32 / 64 (low-cost)       | 0.43        |
| 40 x 48 x 40, 77k (80 x 96 x 80, 614k)          | 48 / 96                  | 0.95        |
| **48 x 64 x 48, 147k (72 x 96 x 72, 498k)**     | **64 / 96 (default)**    | **1.10**    |
| 48 x 64 x 48, 147k (96 x 128 x 96, 1.18M)       | 64 / 128                 | 1.41        |
| 64 x 80 x 64, 328k (102 x 128 x 102, 1.33M)     | 80 / 128                 | 1.87        |
| 80 x 96 x 80, 614k (107 x 128 x 107, 1.47M)     | 96 / 128                 | 2.96        |
| 80 x 96 x 80, 614k (133 x 160 x 133, 2.83M)     | 96 / 160 (high quality)  | 3.48        |
| 104 x 128 x 104, 1.38M (130 x 160 x 130, 2.70M) | 128 / 160                | 5.64        |

Where a default step goes (64 / 96): pressure is most of it (two V-cycles about 0.7 ms), the ink advection 0.13 to 0.2 ms (MacCormack; 0.04 semi-Lagrangian), the rest (velocity advection, forces, confinement, projection, divergence) about 0.3 ms. Switching the velocity advection to MacCormack costs about 0.3 ms (1.39 against 1.10); semi-Lagrangian is the default and I have not compared the pictures. Confinement is below the noise (1.15 with it off, 1.10 on; its four extra dispatches over 147k cells are about 0.05 ms). A sphere obstacle (Jacobi, 24 sweeps) read 1.05.

**Pressure alone**, ms per Jacobi sweep and per V(2,2) cycle (`project()`), and the residual divergence each leaves:

| Cells                  | Jacobi sweep | V-cycle | A V-cycle costs |
| ---------------------- | ------------ | ------- | --------------- |
| 24 x 32 x 24           | 0.006        | 0.19    | 32 sweeps       |
| 40 x 48 x 40           | 0.010        | 0.25    | 23 sweeps       |
| 48 x 64 x 48 (default) | 0.016        | 0.31    | 20 sweeps       |
| 80 x 96 x 80           | 0.046        | 0.67    | 15 sweeps       |
| 104 x 128 x 104        | 0.109        | 1.20    | 11 sweeps       |

A dependent dispatch has a fixed price of about 0.008 ms here (a V-cycle is about 30 dispatches at 64 cells), which is why a V-cycle costs 20 to 30 sweeps on small grids and 11 on large ones. What the work buys, rms divergence after one projection from zero divided by before (a Gaussian source, sigma 0.12 of the box):

| Work               | 8    | 32   | 128  | 512   |
| ------------------ | ---- | ---- | ---- | ----- |
| Jacobi sweeps, 32³ | 0.78 | 0.43 | 0.12 | 0.009 |
| Jacobi sweeps, 64³ | 0.94 | 0.78 | 0.43 | 0.12  |

| V(2,2) cycles  | 1    | 2     | 3     | 4      | 6      |
| -------------- | ---- | ----- | ----- | ------ | ------ |
| multigrid, 32³ | 0.24 | 0.061 | 0.015 | 0.0039 | 0.0003 |
| multigrid, 64³ | 0.21 | 0.049 | 0.012 | 0.0029 | 0.0002 |

Two things in it. Jacobi at 64³ needs 512 sweeps to reach what two V-cycles do (about 30 times the dispatches and, at 0.016 ms a sweep, 8 ms against 0.6), and halving the cell size multiplies its sweeps by four (the 64³ row is the 32³ row shifted by 4 x in work). The multigrid factor per cycle does not depend on the grid size (0.24 and 0.21) but is only about 0.22, not the 0.1 of textbook multigrid: the smoothing factor of point Jacobi on the 7-point laplacian is 5/7 per sweep at best, so a cycle with four sweeps on the fine level gains about 4, and three smoothing sweeps (V(3,3)) read 0.17, 0.03, 0.006. In a running simulation the warm start helps both: the mean ratio of divergence after to before the projection over steps 20 to 60 of a falling heavy drop at 64 cells (a 48 x 64 x 48 grid) was 0.087 for one V-cycle per step, **0.017 for two (the default)**, 0.007 for three; Jacobi 0.73 for 8 sweeps, 0.24 for 24, 0.062 for 64.

**Workgroup shape.** 4 x 4 x 4, 8 x 4 x 4, 4 x 4 x 8, 8 x 8 x 4, 16 x 4 x 4, 8 x 8 x 2, 32 x 4 x 2 and 16 x 16 x 1 were within the noise of one another at both 64 / 96 and 96 / 160 (about 1.0 to 1.5 ms and 3.8 to 4.8 ms; the noise was larger than any difference I could find), and a test checks they all give bit-identical results. The default is 4 x 4 x 4: 64 invocations, safe under every device's limits, 3D-local. It is a parameter of `createFluid3d` (`workgroup`) if a device measures differently.

**The ray march, ms** at 1920 x 1080 (the tank in its default view, ink from 150 frames of ambient drops). One extra march (slope over 1 to 9 per frame), by march resolution (fraction of the canvas, per axis) and steps per ray:

| Scale                       | 32 steps | 64       | 96   | 128  |
| --------------------------- | -------- | -------- | ---- | ---- |
| 0.25                        | 0.21     | 0.34     | 0.45 | 0.63 |
| 0.35                        | 0.22     | 0.43     | 0.58 | 0.74 |
| **0.5 (default, 64 steps)** | 0.43     | **0.66** | 1.09 | 1.23 |
| 0.75                        | 0.57     | 1.01     | 1.48 | 1.62 |
| 1                           | 0.67     | 1.47     | 1.95 | 2.53 |

Reducing the march to half resolution saves 55 percent (0.66 against 1.47 ms at 64 steps) and the ink, being soft, does not show it; a quarter-scale march (0.34 ms) is visible only at the silhouette of dense ink. The whole frame (light volume and occupancy, march, composite) at 64 steps read 1.4 to 1.6 ms from scale 0.25 to 0.75 and 2.1 ms at 1: below about 1.5 ms the frame is floor, so the march table above is the figure to read. The composite alone (debug view, which skips the march and the light volume) is under the floor (0.05 ms by slope).

**Lighting**, whole frame at scale 0.5, 64 steps: off 1.66 ms; light volume with 6, 12 and 24 steps toward the light 1.75, 1.55 and 1.87 (the pass is at the floor: its cost does not show); per-sample march (the shadow rays inside the march loop) with 6, 12 and 24 steps **2.29, 3.95 and 5.33**. So the light volume is the technique, and per-sample shadows cost two to three times the frame for the same picture at 12 steps. They are still in the effect (`Lighting`) as the comparison.

**Empty-space skipping and jitter.** The occupancy lookup made no measurable difference when the tank is half full of ink (march alone 0.75 on, 0.67 off; frame 1.70 and 1.52, both inside the noise): the saving is the sampled taps in empty steps, and a sample of a small texture per step costs about what it saves. It would pay with sparse ink, or with a loop that jumps over empty runs instead of visiting each step; I did not build the jump, and it is on by default only because it did not lose. Jitter costs nothing measurable (1.56 on, 1.30 off, noise).

**Whole frame, solver and render together**, at 1080p (one step, light volume, march, composite per frame; slope over 4 to 20 frames, ink from 90 frames of drops):

| Configuration                                                          | ms per frame       |
| ---------------------------------------------------------------------- | ------------------ |
| low-cost (32 / 64, 32 steps, scale 0.35, no lighting, semi-Lagrangian) | 1.2 (at the floor) |
| **default (64 / 96, 64 steps, scale 0.5, light volume)**               | **3.9**            |
| smoke plume                                                            | 4.8                |
| 96 / 128, 96 steps, scale 0.75                                         | 9.4                |
| high quality (96 / 160, 112 steps, scale 0.75, 16 shadow steps)        | 10.6               |
| 128 / 160, 128 steps, scale 1                                          | 14.9               |

These are upper bounds: the whole-frame figure includes the dependent passes' floors that the separate tables take out (the default is 1.1 + about 1.7 = 2.8 by the pieces). In the browser pane (Chromium, 1702 x 724 = 1.2 Mpx at DPR 2, the same GPU shared with other tabs) the HUD read 104 to 115 fps and 6.1 to 6.9 ms of GPU for the default frame, its march pass 5.9 to 6.7 ms: **the HUD's per-pass figures overlap on a tile-based GPU and under contention they are not the cost**, and the number to keep is the repeat-N slope above. The browser figure is higher than the Node figure for the same work (Metal through Chromium's GPU process, the page compositor and a shared GPU); I did not separate those.

**Texture memory** (estimated from the formats; the HUD read ≈ 26.1 MB, 26 textures for the default at 1.2 Mpx, which agrees):

| Configuration         | Solver       | Render (light, occupancy, march targets at 1080p) | Total        |
| --------------------- | ------------ | ------------------------------------------------- | ------------ |
| low-cost 32 / 64      | 3.5 MB       | 4.0 MB                                            | 7.5 MB       |
| **default 64 / 96**   | **20.8 MB**  | **8.4 MB**                                        | **29 MB**    |
| high quality 96 / 160 | about 105 MB | 21 MB                                             | about 125 MB |
| 128 / 160             | 150 MB       | 35 MB                                             | 185 MB       |

The big items are the ink grid, three textures of 8 bytes a texel once MacCormack runs (11.9 MB at 72 x 96 x 72, 68 MB at 133 x 160 x 133; semi-Lagrangian needs two), the three velocity textures (3.8 MB at 48 x 64 x 48), and the confinement scratch (curl, blur, force: 3 MB). 3D float textures are large: the high-quality ink alone is more than the whole default.

**Verdict, tiers.**

- **Default (64 / 96 / 64 steps / half-resolution march): Tier C, and comfortably.** 3.9 ms per 1080p frame on this Apple Silicon laptop against Tier C's 12 ms; it also reads under Tier B's 6 ms, but that budget is for an integrated GPU and none was measured, and an integrated GPU a few times slower than this one (Intel and AMD parts are, for bandwidth-bound compute) would land past 6 ms. So I do not claim Tier B for it. The browser HUD's 6 to 7 ms (shared GPU, 1.2 Mpx at DPR 2) is consistent with that margin shrinking in practice.
- **High quality (96 / 160, 112 steps, scale 0.75): 10.6 ms, the upper edge of Tier C** on this machine. 128 / 160 at full resolution is 14.9 ms and does not fit; it is there to show where the line is.
- **Cut-down (`low-cost`: 32 / 64, 32 steps, scale 0.35, no lighting, semi-Lagrangian ink): 1.2 ms, the floor.** On this machine that is Tier A by number, but the solver is compute (no WebGL2 fallback) and a few times slower on an integrated GPU would still be a few milliseconds, so I put it at **Tier B**. Is it worth showing? The ring still forms and the picture is smooth and soft: less detail, no per-voxel shadows (so less form), but it reads as ink in water and it is the version a landing page could carry on a laptop. On a phone: untested; the arithmetic (a GPU a fraction of this one, 7.5 MB of 3D textures, compute) does not promise it, hence `mobile: 'untested'`.

## Pitfalls

- **A second parameter upload in one encoder replaces the first.** `queue.writeBuffer` is ordered before the whole command buffer, so `resolve()` after `update()` rewrote the uniforms with zero splat counts and the injection pass saw none: with the velocity debug view on, drops vanished. Found in the browser, fixed by repeating the last counts, covered by a test. (The 2D solver documents the same hazard as "one update per submitted command buffer"; the lesson is that _any_ method that uploads parameters counts.)
- **Point Jacobi is a weak multigrid smoother in 3D.** The smoothing factor is 5/7 per sweep, so a V(2,2) cycle gains a factor of about 4.5, not 10. Two cycles per step is the default; a better smoother (red-black Gauss-Seidel, or a Chebyshev polynomial of Jacobi) would be the next step and is not built.
- **Cycles after the first started from zero.** The first version applied the cold start to every V-cycle, so any number of cycles gave the result of one (found in a test: 1, 2, 3 and 8 cycles all read 0.2619). Only the first cycle of a cold solve starts from zero now.
- **Multigrid and obstacles.** The coarse grids know nothing of the sphere, so with an obstacle on the solver falls back to Jacobi sweeps (`pressureIterations` then means sweeps, and 24 is much weaker than two cycles).
- **Semi-Lagrangian is not conservative where the flow compresses.** A prescribed flow into a wall (a test pushes ink at x = 1) does not pile the ink up: a passive scalar is copied, not accumulated, so its total falls. In the projected flow the divergence is nearly zero and the total stays within a few percent; the walls test checks 0.8 to 1.25 of the start.
- **Ink fades from the floor, not from the picture.** A tank with a drop every few seconds fills the bottom with a muddy mix of every colour. `Ink fade` (0.2 per second by default) and the drop rate are the dials; a real tank would do the same.
- **A short tank, a fast fall.** A heavy drop reaches the floor in about a second (the ring forms on the way, and the spreading ring on the floor is the second act). `Flow speed` below 1 slows time (and costs less), but the solver then steps on only some of the frames, so slow motion judders slightly; it is not a substitute for physics at a different scale. `Ink weight` is the other dial: a lighter ink (0.9 in my trials) sinks more slowly as an elongated blob, and its ring forms late, on the floor.
- **`atan2(0, 0)` is NaN on this GPU, and NaN survives `mix(a, b, 0)`.** The velocity debug view was black in still water until the arguments were kept off the origin.
- **A line one pixel wide dashes.** The tank's edges, drawn at one pixel, stepped in and out of coverage as they slid across pixel centres and read as dashes; two soft pixels do not.
- **Half-float precision.** Velocity, ink and vorticity are 16-bit floats (relative error 5e-4); after projection the divergence cannot fall below about a thousandth of the velocity per cell. Invisible, as in 2D.
- **The ink is a tracer, not a mass.** MacCormack with the limiter keeps a blob's total within 5 percent in a test, and the walls test bounds a hard stir at 0.8 to 1.25 of the start. Do not use it as a quantity.
- **Resizing the grid clears the fluid.** Changing `Velocity cells` or `Ink texels` re-creates every texture (`generation` bumps; the effect rebuilds its volumes); the canvas size does not touch the solver, so resizing the window keeps the ink.
- **Dense ink goes black and the in-scatter stops helping.** A very strong ink (`Ink strength` high, or a black ink) leaves a dark core whose lit edge is thin; the march exits early there (transmittance below 0.01), which is also what makes it cheap. A white ink with high milkiness is the smoke look.
- **Timings under contention are not repeatable.** The same configuration read 1.1 ms in one run and 2.6 ms in another with other agents' GPU work running; only quietest-of-many round-robin samples are quoted, and the workgroup comparison could not rank the shapes at all.

## When to use it

A dedicated page or landing experience where ink in water is the subject, on a desktop: a drink, a dye, a paint or a water brand, an installation, a "look what the browser can do" page. At the default it costs about 4 ms of GPU at 1080p on Apple Silicon and 29 MB of textures, and two to three times that on a weaker GPU is the honest expectation, so it is **only on a dedicated page and on a desktop GPU**. It takes the pointer (touch-action none, a drag pushes the fluid), so it is not a background behind text and not a thing to put on a scrolling page; and it needs compute shaders and 3D storage textures, so there is no WebGL2 version.

What it is not for: a phone (untested, and the arithmetic is against it), a hero behind copy (the tank is the subject; the hero context in this site crops it), or a page that only needs the look of ink: the 2D `fluid-dye` does that for a thirtieth of the work and works with a finger scrolling, and `ink-wash` is analytic and far cheaper. For a fixed film of an ink drop, render the sequence offline and play a video: it costs nothing to play and looks the same. The fallback in the scorecard says as much.

Dials, in the order I would turn them for cost: `Render scale` (0.5 to 0.35 saves a third of the march, invisible on soft ink), steps (64 to 32), lighting (the volume costs nothing; per-sample shadows do), `Pressure work` (2 V-cycles to 1: 1.10 to 0.78 ms at the default, visibly looser at the ring), then the grids (`Ink texels` 96 to 64 is the cheapest cut in the solver; the velocity cells set how small a vortex can be). The `low-cost` preset sets all of them.

## References

- Jos Stam, [Stable Fluids](https://pages.cs.wisc.edu/~chaol/data/cs777/stam-stable_fluids.pdf), SIGGRAPH 1999: the semi-Lagrangian step and the projection.
- Harlow and Welch, _Numerical Calculation of Time-Dependent Viscous Incompressible Flow of Fluid with Free Surface_, 1965: the staggered (MAC) grid.
- Robert Bridson, _Fluid Simulation for Computer Graphics_ (2nd ed., 2015): the MAC grid in 3D, boundary conditions, the pressure solve and multigrid-preconditioned conjugate gradients, the step this solver would take next.
- Briggs, Henson, McCormick, _A Multigrid Tutorial_ (2nd ed., SIAM 2000): V-cycles, restriction and prolongation, and the smoothing factors of weighted Jacobi.
- Selle, Fedkiw, Kim, Liu, Rossignac, [An Unconditionally Stable MacCormack Method](https://physbam.stanford.edu/~fedkiw/papers/stanford2008-04.pdf), 2008: the limited correction.
- Fedkiw, Stam, Jensen, [Visual Simulation of Smoke](https://web.stanford.edu/class/cs237d/smoke.pdf), SIGGRAPH 2001: buoyancy and vorticity confinement in 3D.
- Jimenez, _Next Generation Post Processing in Call of Duty: Advanced Warfare_, SIGGRAPH 2014: the interleaved gradient noise used to jitter the rays.
- Henyey and Greenstein, _Diffuse radiation in the Galaxy_, 1941: the phase function in `volume_phase_hg`.
- `src/effects/fluid-dye/notes.md`: the 2D solver this one follows, and its measured costs.
- docs/support-matrix.md section 6: why `rgba16float` and `r32float`.

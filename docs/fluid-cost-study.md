# Solver cost study (story 6.8)

What the stable-fluids solver (`src/gpu/sim/fluid.ts`) costs, what each of its knobs buys, and which
settings to ship at a given GPU budget. The numbers are in the **Results** section at the end, produced by
`tests/fluid-study.test.ts`; the text below says what they mean and cites the table by its heading. When the
script is run again on a quiet machine, paste its markdown over that section and re-read the prose against it:
every figure quoted above comes from those tables.

Reproduce:

```
FLUID_STUDY=/tmp/fluid-study.json npx vitest run tests/fluid-study.test.ts -t "fluid cost study"
```

It writes the JSON (everything, including the per-repetition slopes), a markdown rendering beside it
(`/tmp/fluid-study.md`), and prints the markdown. Optional: `FLUID_STUDY_ONLY=pressure,omega,warm,vortex,resolution,upsample,advection,presets,convergence`
(sections), `FLUID_STUDY_QUICK=1` (fewer repetitions, smaller candidate lists), `FLUID_STUDY_PASSES=n`
(times each cost table is measured end to end; default 2), `FLUID_STUDY_IMAGES=<dir>` (contact sheets as PNG),
`FLUID_STUDY_NOTE="..."` (what else was running; recorded in the output). The full run took about five to
nine minutes on the machine below.

## Method, and its limits

- **Cost** is GPU time from timestamp queries: the span from the first timed pass starting to the last one
  ending, taken at a small and at a large number of repeats of the workload (a step, a pressure iteration, a
  V-cycle), and reported as the **slope**, one per repetition: median with min-max over the repetitions. Never
  one pass, never a sum of passes (they overlap on tile-based GPUs, and a single pass has a floor of about a
  millisecond here). The large repeat count is calibrated so the workload spans about 60 ms, which puts the
  jitter of a submit well below the signal. Each table is measured twice, one pass over everything after the
  other, and the passes are pooled, so a burst of other work on the GPU spoils one sample and not a table.
- **"Fastest"** is the slope between the fastest span seen at each end. Other work on the GPU only ever adds
  time, so on a busy machine this is nearer the truth than the median; on a quiet one they agree. The
  derived tables (time to reduce the divergence, the configuration picks) use it; the raw cost tables show
  the median and the range.
- **Quality** is measured, not judged: residual divergence (rms after / before a projection), retained peak
  vorticity and kinetic energy of a vortex, the 10-90 percent edge of an advected disc, and the error of a
  stirred dye pattern against a high-resolution, converged run of the same scripted scenario (relative L2
  at 128 x 128 for the large-scale placement and at 256 x 256 including fine structure). The scenario is
  two scripted strokes stirring a pattern of stripes, a disc and a square in a closed box for a second, then
  coasting for a second; every grid gets the same strokes in canvas units. Two different good runs (256² / 1024²
  and 192² / 768², three V-cycles each) differ by the "floor" quoted in the tables; an error near it is
  indistinguishable from the reference.
- **Limits.** One Apple Silicon GPU (Metal through Dawn, headless, Node); not an integrated Intel or AMD GPU, not
  a phone, not a browser (the browser adds a compositor and, on a phone, thermal limits). Costs scale with
  memory bandwidth and dispatch latency, which differ a lot between devices: read the ratios, not the
  milliseconds. The machine was shared with other work for the whole run (see "Machine state" in the Results);
  that is why the tables give ranges, and why the figures should be re-measured on a quiet machine before
  anything is promised. The stirred-pattern error is one scenario (one flow, one pattern): it ranks
  configurations reliably and says little about absolute visual quality, so the contact sheets
  (`FLUID_STUDY_IMAGES`) were looked at as well; in them the resolutions and solvers that the table calls
  "close" are close.

## What was added to the solver

All behind options whose defaults are today's behaviour; every existing export, default and test is unchanged
(`tests/fluid.test.ts` passes untouched).

- `pressureSolver: 'multigrid'` (`fluid-multigrid.wgsl`, `fluid.ts`): a V-cycle over a pyramid of grids half as
  wide (any size, obstacles, open and periodic boundaries). Smoothing is the existing damped Jacobi sweep run
  on each level with that level's own uniform block and mask, so every boundary rule carries over; the residual
  is restricted by summing 2 x 2 children, the correction is prolonged bilinearly (renormalised over non-solid
  coarse cells); the coarsest level (up to 32 x 32) is solved by red-black SOR inside **one workgroup** in
  shared memory (one dispatch instead of dozens of dependent ones). Parameters `multigridCycles` (per step, the
  unit `pressureIterations` is for the other solvers), `multigridSmooth` (sweeps each way, default 1),
  `multigridLevels` (cap; 0 = down to the workgroup), `multigridCoarse` (coarsest sweeps, default 24).
  `project(encoder, n)` counts V-cycles for this solver. Exports: `multigridLevelSizes`, `pressureSolveWork`
  (dispatches and cell visits of a solve, machine independent), `MULTIGRID_COARSEST`.
- `dyeVelocityFilter: 'hardware-bilinear' | 'bspline' | 'catmull-rom'` (`fluid-core.wgsl`
  `vel_at_cubic`, used by the dye passes in `fluid-advect.wgsl`): how the staggered velocity is read for the
  dye. `hardware-bilinear` is the same function as today's bilinear through the texture filter (two taps);
  `bspline` (C2) and `catmull-rom` (C1) are cubic reconstructions through the filter (four and nine taps per
  component). They apply without obstacles and without the `wrap` boundary; otherwise the dye is carried by
  today's path. Default `bilinear` (today's).
- Exports `FluidUniforms` and `fluidShaderChunks` (so tests and consumers can write kernels against the grid),
  two uniform fields (`dye_filter`, `mg_sweeps`) and one more binding on the dye passes (a sampler).
- Tests in `tests/fluid-study.test.ts` (normal suite): multigrid removes the divergence far faster than Jacobi
  for the same cell visits (64² and 128²), converges to the same pressure field as a converged SOR (relative
  difference under 2 percent), works with odd sizes, 16:9, open, periodic, obstacles, and beats 30 Jacobi
  iterations inside a running simulation; the cubic filters reproduce a linear velocity field to half-float
  precision, are continuous across cell borders and have a smaller slope kink than bilinear, and carry dye the
  distance they should; the filtered bilinear is the loaded one's function.

## Findings

Table names refer to headings in the Results section.

### 1. The pressure solve

- **Jacobi** is the cheapest per iteration and the worst per unit of work, and it gets worse with the grid:
  its slowest mode shrinks like cos(pi/N)^k, so the iterations needed grow with N². "Cold start: residual
  divergence against iterations" shows 30 iterations removing a third of a smooth divergence at 64² and about
  one percent at 512²; "what it takes to reduce the divergence" shows more than a thousand iterations for a
  factor of ten at 128² and above. The default (30 iterations) is a decision that the picture needs only
  _plausible_ incompressibility and the warm start does the rest.
- **Red-black SOR** converges like N rather than N², but it is not monotone: the residual is _above_ its
  starting value until roughly N/2 to N iterations (ratios of 7 to 17 at 8 to 32 iterations), and from a cold
  start it is worse than doing nothing for a long time. The optimal omega matters a lot and the solver
  caps it: `sorOmega()` returns at most 1.9, but Young's optimum is 1.91, 1.95, 1.98 and 1.99 at 64², 128², 256²
  and 512²; at 256² and 2N iterations the residual is 0.72 with the cap and 0.33 at 1.95, the largest value the
  `sorOmega` parameter accepts ("Red-black SOR: the optimal omega per grid"). SOR is a solver for runs of
  hundreds of iterations, with the optimum omega, warm-started.
- **Multigrid** _leaves_ about 3.5 percent of the divergence (V(1,1)) or 2.5 to 3.2 percent (V(2,2)) per cycle, **independent of the grid size** (64² to 512², same two numbers), and two cycles
  reach a tenth of a percent. A V(1,1) cycle costs about 8 to 18 Jacobi iterations' worth of time (dominated by the
  number of dependent dispatches below 256² and by bandwidth above), so to reduce the divergence tenfold it is
  about 18 times cheaper than Jacobi at 64², and at 128² and above Jacobi does not get there in a thousand
  iterations ("what it takes to reduce..."). Smoothing once each way (V(1,1)) is the better trade: a quarter to a
  half cheaper per cycle than V(2,2) for a residual of 0.035 instead of 0.03 or less. Capping the pyramid
  at three levels leaves a coarsest grid of 64² or more solved by relaxation, and the benefit goes (residual per
  cycle 0.7 to 0.9 at 256² and 512² in the `V(1,1), 3 levels` rows): go all the way down.
- **Warm start** ("Warm start in a running simulation", compare the rates): it helps Jacobi at 64² and barely at
  128² and above (the leftover divergence is mostly the new force, not last step's error), it is essential
  for SOR (cold SOR with 30 iterations leaves four to seven times the divergence of warm), and multigrid hardly needs
  it (cold and warm differ by under 25 percent from 128², 60 percent at 64²). It does not change the ranking: multigrid, then SOR above
  60 iterations, then Jacobi. At 128² and 256² no Jacobi iteration count up to 120 gets near one multigrid
  cycle.
- **What an under-converged solve looks like** ("Under-converged pressure", the `convergence` contact sheet):
  with 30 Jacobi iterations at 128² the stirred pattern is _plausible but different_, with less curling than the
  converged one (error many times the floor); the push of a stroke stays partly compressive, so ink is shoved
  ahead instead of rolling up, and the kinetic energy comes out 7 to 40 percent too high because the
  divergent part of the push is not removed. The dye total drifts ("mass") by at most 2 percent at 128² but by
  30 percent at 256² with 5 iterations and 13 percent with 10 (1 to 2 percent from 20 to 30): that is the
  visible "piling up" and "breathing". Vortices
  _in an undisturbed fluid_ do not decay faster with a bad solve ("Vortex retention": identical to three
  digits for Jacobi 30 and multigrid), they decay from the advection scheme (below); the pressure matters
  when something is pushed. How many iterations does the eye need? Nothing in a still picture of a single
  scene looks broken at 30 Jacobi iterations at 64² to 128²; the difference shows when stirring (ink that
  will not curl) and in how the picture changes when the grid is raised without raising the iterations (256²
  with 30 iterations is visibly worse than 128² with 30). One V-cycle gets the error to where two or three
  do (0.088 against 0.080 at 128²; what is left is the resolution, not the solve); Jacobi has not got there
  after 120 iterations at 128² (0.39).

### 2. Velocity grid against dye grid

"Velocity grid against dye grid": the **dye grid** is what buys detail (the "detail" column: 0.6 of the
reference's fine structure at 256², 0.84 at 512², 0.99 at 1024²) and, because it is where the pattern is
placed, most of the error; the **velocity grid** sets how small a vortex can be (error falls from 0.27 at 64²
to 0.09 at 128² and 0.06 to 0.07 at 192 to 256, at 512²). A 128² velocity grid with a 512² to 1024² dye grid is
within a factor of 1.5 of the best the 256² grid does and, in the `resolution` contact sheet (reference, then
64², 128², 256² velocity at 512² dye), cannot be told from it by eye; 64² is visibly softer in the curls and
its error plateaus (0.27 at every dye size) because the velocity cannot carry the structure. So: 128² is the
default for a reason; 96² is acceptable for a phone; above 192² the cost of the velocity grid is hard to
justify unless the dye is also 1024². The cost of a step is dominated by the dye grid from 512² up.

### 3. Upsampling: how the velocity is read for the dye

"Reading the velocity between its faces", "Streaklines" and "Dye advection alone":

- **Accuracy.** Catmull-Rom reproduces the velocity 5 to 17 times more accurately than bilinear on a smooth
  flow (rms error 0.07 against 1.0 thousandths at 32²) and has no slope kink beyond the field's own curvature;
  the cubic **B-spline** is smoother (C2) but _less_ accurate than bilinear (it is a smoothing, not an
  interpolating, filter), and it is the worst in the streakline test at a coarse grid.
- **Visibility.** Carrying a thin dye line through a smooth flow for a second and comparing with the exact
  characteristics, bilinear's centreline error is 3.5, 2.3 and 2.1 texels at 16², 32² and 64² velocity grids
  against 2.0 for Catmull-Rom: below two texels of a 512² dye grid at 32² and above, and the kinks are not
  visible in the `streakline` contact sheet at any of the three. The cubic reconstruction is a correction for
  velocity grids of 16 to 24 cells, not for the 64 to 256 that the effects use.
- **Cost, which is the surprise.** The filtered paths are **several times cheaper** than today's bilinear
  path, because the loaded one makes eight loads with the obstacle and boundary bookkeeping per sample and
  the filtered ones make two (hardware bilinear), four or nine taps per component through the texture
  hardware. Dye advection at 1024², MacCormack: about 4.3 ms with loads, 0.86 ms hardware-bilinear, 1.2 ms
  B-spline, 1.8 ms Catmull-Rom; the whole step at 128² / 1024² goes from 4.4 to 1.5 ms. So the
  best settings are **hardware-bilinear (same picture, a fifth of the dye cost)** or **Catmull-Rom
  (a little more accurate, about twice that)** whenever there are no obstacles and the boundary is not
  periodic.

### 4. Advection scheme cost, and where MacCormack stops being worth it

"Dye advection alone" and "Sharpness". Per step, MacCormack costs about 3.5 to 4 times the
semi-Lagrangian pass at every size with today's loads (it runs three velocity traces per texel to
semi-Lagrangian's one) and about 3.5 to 5 times with the filtered velocity; BFECC costs the same as
MacCormack here and sharpens the disc edge further (5 texels against 10 at 1024²) at the price of a worse
peak and mass. What it buys is in "Sharpness": a semi-Lagrangian disc edge is 18 to 20 texels wide at every
dye size and loses 4 to 18 percent of its peak, which at 256² is a smear a twelfth of the image wide;
MacCormack keeps the peak and the edge at 8 to 10 texels. At equal cost, then: semi-Lagrangian at the next
dye size up is about as sharp as MacCormack (edge 17.6 thousandths of the height at 1024² against 15.6 at 512²)
and costs about the same (1.2 ms against 1.1 ms with loads; 0.23 against 0.22 with hardware bilinear), but
with a 4x larger dye texture (memory and bandwidth for everything that reads it) and 4 to 6 percent
less peak and mass. MacCormack for the dye is worth its pass when the dye is already at the largest size you will
afford (it then buys the last factor of two in edge width, and its price is the same as the next size up);
with a smaller dye grid and a tight budget, spend the pass on the next dye size instead. For
**velocity** it is worth it most on coarse grids: a vortex keeps 2.4 times the peak vorticity of
semi-Lagrangian at 64², 1.5 times at 128², 1.1 times at 256² ("Vortex retention"), for a pass that is small at 128².

### 5. Putting it together

"Configurations": 192 combinations of velocity grid, dye grid, advection schemes, pressure solver and dye
velocity filter, each measured for cost and for error against the reference; the first table picks the best
under each budget, the second the best that today's solver and dye path can do. Under the budgets of the
brief, today's options reach nothing better than a 64² / 512² configuration with a 0.37 error whatever the
budget: with 30 Jacobi iterations a larger velocity grid _looks worse_ (error 0.41 at 64², 0.58 at 128², 0.48 at
192², "What each change buys"), because the solve does not converge at the grid it is given. With the new options a
128² / 512² configuration fits under 0.5 ms and a 192² / 1024² one under 2 ms. The presets below are read off those tables and the frontier.

## Recommended settings

Per-step GPU time on the machine in the Results; "fastest" figures (see Method). Pressure is one multigrid
V-cycle (`pressureSolver: 'multigrid'`, `multigridCycles: 1`, `multigridSmooth: 1`) in every row, and the dye is
read through the texture filter (`dyeVelocityFilter`) as marked; use `bilinear` with obstacles or a periodic
boundary, which costs several times more on the dye.

| preset     | budget | velocity | dye   | velocity / dye advection | dye velocity      | error | for                                               |
| ---------- | ------ | -------- | ----- | ------------------------ | ----------------- | ----- | ------------------------------------------------- |
| phone      | 0.5 ms | 128²     | 512²  | MacCormack / MacCormack  | hardware-bilinear | 0.09  | ambient ink on a phone                            |
| laptop     | 1 ms   | 192²     | 768²  | MacCormack / MacCormack  | hardware-bilinear | 0.05  | the default for an interactive hero               |
| showpiece  | 2 ms   | 192²     | 1024² | MacCormack / MacCormack  | catmull-rom       | 0.05  | a dedicated page                                  |
| showpiece+ | 4 ms   | 192²     | 1024² | MacCormack / MacCormack  | catmull-rom       | 0.05  | no further gain: the search stops at 192² / 1024² |

(The rows are the picks of "Configurations" in the Results below, from the re-run made after the other agents had finished; the first run, on a busier GPU, fitted less into each budget.
A phone's GPU is several times slower than this one at the same work: scale the budget, not the settings
below it. If the effect has to fit 0.5 ms on a phone, take the 64² / 256² / semi-Lagrangian row of the
frontier (0.3 ms) and expect soft ink with about half the vortex energy.)

## Recommended changes to the solver defaults

Recommendation 1 has since been applied (`dyeVelocityFilter` now defaults to `hardware-bilinear`) and `fluid-dye` offers `multigrid` in its solver control. The others are still recommendations. The study recommends:

1. **Dye velocity filter: change the default from `bilinear` to `hardware-bilinear`** when there are no
   obstacles and the boundary is not periodic (the code already falls back otherwise). It is the same
   function, within the precision of the filter weights (the interpolation table shows the same error and
   kink to three digits), and takes the dominant pass of the step to a fifth of its cost.
2. **Pressure: default to `multigrid`, one V(1,1) cycle, instead of 30 Jacobi iterations**, at 96² and up. Error
   against the converged run drops by a factor of 3 to 7 from 96² (1.5 at 64², where Jacobi's warm start works)
   ("What each change buys") at equal or lower cost.
3. **`sorOmega()`: raise the cap from 1.9** (and the `sorOmega` parameter's 1.95) towards Young's optimum
   (about 1.97 to 1.99 above 256²), if SOR is kept at all; with multigrid there is little reason to.
4. **`fluid-dye`'s `Pressure iterations` / `Pressure solver` controls**: add `multigrid`, and note that
   `sor` below N/2 iterations is worse than Jacobi.
5. Minor: the comment above `sor()` in `fluid-project.wgsl` ("The first iteration of a solve that does not
   warm-start...") is attached to the wrong function.

## Results

Generated by `tests/fluid-study.test.ts`, run on the shared development machine (Apple Silicon, Metal through
Dawn, Node, timestamp queries); numbers elsewhere in this page are read from these tables. To refresh: re-run the
command at the top on a quiet machine and replace everything below this line.

Run 2026-10-09T01:48:11.299Z; timestamp queries true; 5 repetitions per timing (median, min-max).
Machine state: re-run after all agents finished; desktop apps still using the GPU

### Pressure solve, cold start: cost of one iteration (Jacobi, SOR) or V-cycle (multigrid), ms

| solver                     | 64²                    | 128²                   | 256²                   | 512²                   |
| -------------------------- | ---------------------- | ---------------------- | ---------------------- | ---------------------- |
| jacobi                     | 0.0083 (0.0074-0.0092) | 0.0104 (0.0091-0.0119) | 0.0262 (0.0255-0.0323) | 0.0885 (0.0823-0.1060) |
| sor                        | 0.0111 (0.0095-0.0122) | 0.0129 (0.0120-0.0139) | 0.0276 (0.0257-0.0288) | 0.0877 (0.0822-0.0929) |
| multigrid V(1,1)           | 0.1209 (0.1151-0.1619) | 0.2206 (0.2093-0.2610) | 0.3241 (0.2985-0.3541) | 0.6108 (0.5770-0.6524) |
| multigrid V(2,2)           | 0.1380 (0.1305-0.1922) | 0.2900 (0.2555-0.3128) | 0.4455 (0.4097-0.5359) | 0.9040 (0.8645-1.1015) |
| multigrid V(1,1), 3 levels | 0.1212 (0.1173-0.1779) | 0.2189 (0.1784-0.2814) | 0.3808 (0.3618-0.4147) | 0.6559 (0.6317-0.6961) |

### Pressure solve, cold start: residual divergence (rms after / before) against iterations or cycles, large source (sigma 0.1)

**64²**

| solver                     | count: residual                                                                            |
| -------------------------- | ------------------------------------------------------------------------------------------ |
| jacobi                     | 8: 0.893, 16: 0.804, 32: 0.665, 64: 0.483, 128: 0.298, 256: 0.156, 512: 0.068, 1024: 0.022 |
| sor                        | 8: 7.449, 16: 4.744, 32: 1.982, 64: 0.357, 128: 0.056, 256: 0.010, 512: 0.000, 1024: 0.000 |
| multigrid V(1,1)           | 1: 0.035, 2: 0.003, 3: 0.001, 4: 0.000, 6: 0.000                                           |
| multigrid V(2,2)           | 1: 0.025, 2: 0.001, 3: 0.000, 4: 0.000, 6: 0.000                                           |
| multigrid V(1,1), 3 levels | 1: 0.035, 2: 0.003, 3: 0.001, 4: 0.000, 6: 0.000                                           |

**128²**

| solver                     | count: residual                                                                             |
| -------------------------- | ------------------------------------------------------------------------------------------- |
| jacobi                     | 8: 0.971, 16: 0.944, 32: 0.893, 64: 0.804, 128: 0.665, 256: 0.483, 512: 0.298, 1024: 0.156  |
| sor                        | 8: 9.854, 16: 10.795, 32: 6.192, 64: 2.338, 128: 0.741, 256: 0.169, 512: 0.052, 1024: 0.011 |
| multigrid V(1,1)           | 1: 0.036, 2: 0.003, 3: 0.001, 4: 0.000, 6: 0.000                                            |
| multigrid V(2,2)           | 1: 0.030, 2: 0.002, 3: 0.000, 4: 0.000, 6: 0.000                                            |
| multigrid V(1,1), 3 levels | 1: 0.036, 2: 0.003, 3: 0.001, 4: 0.000, 6: 0.000                                            |

**256²**

| solver                     | count: residual                                                                               |
| -------------------------- | --------------------------------------------------------------------------------------------- |
| jacobi                     | 8: 0.992, 16: 0.985, 32: 0.971, 64: 0.944, 128: 0.893, 256: 0.804, 512: 0.665, 1024: 0.483    |
| sor                        | 8: 10.600, 16: 14.127, 32: 13.331, 64: 8.310, 128: 4.253, 256: 1.952, 512: 0.723, 1024: 0.180 |
| multigrid V(1,1)           | 1: 0.036, 2: 0.004, 3: 0.001, 4: 0.000, 6: 0.000                                              |
| multigrid V(2,2)           | 1: 0.031, 2: 0.002, 3: 0.000, 4: 0.000, 6: 0.000                                              |
| multigrid V(1,1), 3 levels | 1: 0.725, 2: 0.556, 3: 0.445, 4: 0.366, 6: 0.265                                              |

**512²**

| solver                     | count: residual                                                                                 |
| -------------------------- | ----------------------------------------------------------------------------------------------- |
| jacobi                     | 8: 0.998, 16: 0.996, 32: 0.992, 64: 0.985, 128: 0.971, 256: 0.944, 512: 0.893, 1024: 0.804      |
| sor                        | 8: 10.797, 16: 15.147, 32: 16.913, 64: 15.007, 128: 11.327, 256: 7.336, 512: 4.056, 1024: 1.917 |
| multigrid V(1,1)           | 1: 0.036, 2: 0.004, 3: 0.001, 4: 0.000, 6: 0.000                                                |
| multigrid V(2,2)           | 1: 0.032, 2: 0.002, 3: 0.000, 4: 0.000, 6: 0.000                                                |
| multigrid V(1,1), 3 levels | 1: 0.916, 2: 0.843, 3: 0.780, 4: 0.724, 6: 0.630                                                |

### Pressure solve, cold start: what it takes to reduce the divergence by a factor (time on this GPU)

| grid | source | solver                     | to 0.5                         | to 0.1                         | to 0.01                        |
| ---- | ------ | -------------------------- | ------------------------------ | ------------------------------ | ------------------------------ |
| 64²  | large  | jacobi                     | 59.3 iterations, 0.440 ms      | 371.5 iterations, 2.752 ms     | > 1024 iterations (> 7.59 ms)  |
| 64²  | large  | sor                        | 55.8 iterations, 0.528 ms      | 103.0 iterations, 0.974 ms     | 253.8 iterations, 2.401 ms     |
| 64²  | large  | multigrid V(1,1)           | 1.0 cycles, 0.119 ms           | 1.0 cycles, 0.119 ms           | 1.4 cycles, 0.169 ms           |
| 64²  | large  | multigrid V(2,2)           | 1.0 cycles, 0.135 ms           | 1.0 cycles, 0.135 ms           | 1.2 cycles, 0.168 ms           |
| 64²  | large  | multigrid V(1,1), 3 levels | 1.0 cycles, 0.118 ms           | 1.0 cycles, 0.118 ms           | 1.4 cycles, 0.169 ms           |
| 64²  | small  | jacobi                     | 8.0 iterations, 0.059 ms       | 33.7 iterations, 0.250 ms      | 191.9 iterations, 1.422 ms     |
| 64²  | small  | sor                        | 32.6 iterations, 0.308 ms      | 61.8 iterations, 0.585 ms      | 101.6 iterations, 0.961 ms     |
| 64²  | small  | multigrid V(1,1)           | 1.0 cycles, 0.119 ms           | 1.3 cycles, 0.155 ms           | 3.0 cycles, 0.361 ms           |
| 64²  | small  | multigrid V(2,2)           | 1.0 cycles, 0.135 ms           | 1.0 cycles, 0.138 ms           | 2.1 cycles, 0.281 ms           |
| 128² | large  | jacobi                     | 237.5 iterations, 2.153 ms     | > 1024 iterations (> 9.28 ms)  | > 1024 iterations (> 9.28 ms)  |
| 128² | large  | sor                        | 153.9 iterations, 1.843 ms     | 348.0 iterations, 4.167 ms     | > 1024 iterations (> 12.26 ms) |
| 128² | large  | multigrid V(1,1)           | 1.0 cycles, 0.212 ms           | 1.0 cycles, 0.212 ms           | 1.4 cycles, 0.306 ms           |
| 128² | large  | multigrid V(2,2)           | 1.0 cycles, 0.257 ms           | 1.0 cycles, 0.257 ms           | 1.3 cycles, 0.335 ms           |
| 128² | large  | multigrid V(1,1), 3 levels | 1.0 cycles, 0.178 ms           | 1.0 cycles, 0.178 ms           | 1.4 cycles, 0.258 ms           |
| 128² | small  | jacobi                     | 20.8 iterations, 0.189 ms      | 134.1 iterations, 1.216 ms     | 758.9 iterations, 6.881 ms     |
| 128² | small  | sor                        | 44.3 iterations, 0.530 ms      | 82.1 iterations, 0.984 ms      | 188.6 iterations, 2.258 ms     |
| 128² | small  | multigrid V(1,1)           | 1.0 cycles, 0.212 ms           | 1.4 cycles, 0.300 ms           | 3.3 cycles, 0.691 ms           |
| 128² | small  | multigrid V(2,2)           | 1.0 cycles, 0.257 ms           | 1.1 cycles, 0.285 ms           | 2.3 cycles, 0.595 ms           |
| 256² | large  | jacobi                     | 950.1 iterations, 24.227 ms    | > 1024 iterations (> 26.11 ms) | > 1024 iterations (> 26.11 ms) |
| 256² | large  | sor                        | 615.4 iterations, 15.829 ms    | > 1024 iterations (> 26.34 ms) | > 1024 iterations (> 26.34 ms) |
| 256² | large  | multigrid V(1,1)           | 1.0 cycles, 0.302 ms           | 1.0 cycles, 0.302 ms           | 1.5 cycles, 0.443 ms           |
| 256² | large  | multigrid V(2,2)           | 1.0 cycles, 0.410 ms           | 1.0 cycles, 0.410 ms           | 1.3 cycles, 0.542 ms           |
| 256² | large  | multigrid V(1,1), 3 levels | 2.4 cycles, 0.878 ms           | > 6 cycles (> 2.17 ms)         | > 6 cycles (> 2.17 ms)         |
| 256² | small  | jacobi                     | 83.4 iterations, 2.128 ms      | 535.6 iterations, 13.657 ms    | > 1024 iterations (> 26.11 ms) |
| 256² | small  | sor                        | 76.8 iterations, 1.976 ms      | 203.3 iterations, 5.229 ms     | 768.4 iterations, 19.764 ms    |
| 256² | small  | multigrid V(1,1)           | 1.0 cycles, 0.302 ms           | 1.4 cycles, 0.436 ms           | 3.3 cycles, 1.001 ms           |
| 256² | small  | multigrid V(2,2)           | 1.0 cycles, 0.410 ms           | 1.1 cycles, 0.466 ms           | 2.4 cycles, 0.977 ms           |
| 512² | large  | jacobi                     | > 1024 iterations (> 84.35 ms) | > 1024 iterations (> 84.35 ms) | > 1024 iterations (> 84.35 ms) |
| 512² | large  | sor                        | > 1024 iterations (> 84.21 ms) | > 1024 iterations (> 84.21 ms) | > 1024 iterations (> 84.21 ms) |
| 512² | large  | multigrid V(1,1)           | 1.0 cycles, 0.578 ms           | 1.0 cycles, 0.578 ms           | 1.5 cycles, 0.853 ms           |
| 512² | large  | multigrid V(2,2)           | 1.0 cycles, 0.867 ms           | 1.0 cycles, 0.867 ms           | 1.3 cycles, 1.152 ms           |
| 512² | large  | multigrid V(1,1), 3 levels | > 6 cycles (> 3.79 ms)         | > 6 cycles (> 3.79 ms)         | > 6 cycles (> 3.79 ms)         |
| 512² | small  | jacobi                     | 333.9 iterations, 27.505 ms    | > 1024 iterations (> 84.35 ms) | > 1024 iterations (> 84.35 ms) |
| 512² | small  | sor                        | 258.1 iterations, 21.228 ms    | 805.8 iterations, 66.262 ms    | > 1024 iterations (> 84.21 ms) |
| 512² | small  | multigrid V(1,1)           | 1.0 cycles, 0.578 ms           | 1.4 cycles, 0.837 ms           | 3.3 cycles, 1.919 ms           |
| 512² | small  | multigrid V(2,2)           | 1.0 cycles, 0.867 ms           | 1.1 cycles, 0.993 ms           | 2.4 cycles, 2.083 ms           |

### Red-black SOR: residual after 2N iterations for several omega (cold, large source)

| grid | iterations | Young optimum -> residual     | omega: residual                                                                              | best omega tried |
| ---- | ---------- | ----------------------------- | -------------------------------------------------------------------------------------------- | ---------------- |
| 64²  | 128        | 1.906 (used: 1.900) -> 0.0559 | 1.5: 0.2362, 1.7: 0.1589, 1.8: 0.0906, 1.9: 0.0559, 1.92: 0.0621, 1.94: 0.1596, 1.95: 0.3010 | 1.9              |
| 128² | 256        | 1.952 (used: 1.900) -> 0.1688 | 1.5: 0.5303, 1.7: 0.4644, 1.8: 0.3752, 1.9: 0.1688, 1.92: 0.1278, 1.94: 0.1161, 1.95: 0.1114 | 1.95             |
| 256² | 512        | 1.976 (used: 1.900) -> 0.7235 | 1.5: 0.9949, 1.7: 1.0228, 1.8: 0.9731, 1.9: 0.7235, 1.92: 0.6014, 1.94: 0.4271, 1.95: 0.3269 | 1.95             |

### Warm start in a running simulation (mean over steps 30 to 90 of a force circling a closed box), warm start / cold start

Each pair of columns: divergence after / before the projection, then the divergence left in the flow as a rate (rms of the divergence times the cell count: how fast a parcel of fluid changes area, 1 / s). The ratio flatters a warm start, whose 'before' still holds the last step's leftovers, so compare the rates.

| solver                     | 64² ratio   | 64² rate      | 128² ratio  | 128² rate      | 256² ratio  | 256² rate      |
| -------------------------- | ----------- | ------------- | ----------- | -------------- | ----------- | -------------- |
| jacobi 15                  | 0.60 / 0.61 | 3.609 / 4.544 | 0.77 / 0.70 | 6.533 / 6.719  | 0.71 / 0.71 | 6.020 / 5.835  |
| jacobi 30 (default)        | 0.34 / 0.50 | 1.753 / 3.117 | 0.76 / 0.67 | 5.295 / 5.443  | 0.80 / 0.73 | 5.375 / 5.068  |
| jacobi 60                  | 0.13 / 0.43 | 0.627 / 2.083 | 0.59 / 0.63 | 2.841 / 4.076  | 0.88 / 0.76 | 4.081 / 4.207  |
| jacobi 120                 | 0.05 / 0.30 | 0.237 / 1.304 | 0.32 / 0.55 | 1.296 / 2.904  | 0.82 / 0.76 | 3.132 / 3.351  |
| sor 30                     | 0.28 / 1.14 | 1.363 / 7.337 | 0.70 / 1.80 | 3.135 / 22.767 | 1.68 / 1.90 | 7.716 / 33.501 |
| sor 60                     | 0.06 / 0.36 | 0.286 / 1.729 | 0.24 / 1.43 | 0.982 / 9.407  | 1.09 / 1.81 | 3.046 / 19.078 |
| sor 120                    | 0.01 / 0.08 | 0.037 / 0.363 | 0.11 / 0.73 | 0.454 / 3.261  | 0.55 / 1.68 | 1.313 / 9.664  |
| multigrid V(1,1), 1 cycle  | 0.08 / 0.12 | 0.365 / 0.584 | 0.16 / 0.19 | 0.659 / 0.812  | 0.26 / 0.26 | 0.608 / 0.634  |
| multigrid V(2,2), 1 cycle  | 0.04 / 0.07 | 0.178 / 0.351 | 0.12 / 0.14 | 0.478 / 0.587  | 0.21 / 0.22 | 0.497 / 0.517  |
| multigrid V(1,1), 2 cycles | 0.02 / 0.03 | 0.079 / 0.119 | 0.09 / 0.10 | 0.391 / 0.413  | 0.20 / 0.20 | 0.462 / 0.464  |

### Vortex retention: a Gaussian vortex (sigma 0.045 of the canvas height) after 3 s, no dissipation, no confinement

| velocity grid | pressure                    | velocity advection | peak vorticity kept | kinetic energy kept |
| ------------- | --------------------------- | ------------------ | ------------------- | ------------------- |
| 64²           | jacobi 30                   | semi-lagrangian    | 0.191               | 0.099               |
| 64²           | jacobi 30                   | maccormack         | 0.459               | 0.476               |
| 64²           | multigrid 1                 | semi-lagrangian    | 0.191               | 0.099               |
| 64²           | multigrid 1                 | maccormack         | 0.460               | 0.475               |
| 64²           | multigrid 1 + confinement 8 | semi-lagrangian    | 1.561               | 0.703               |
| 64²           | multigrid 1 + confinement 8 | maccormack         | 1.835               | 2.676               |
| 128²          | jacobi 30                   | semi-lagrangian    | 0.278               | 0.185               |
| 128²          | jacobi 30                   | maccormack         | 0.404               | 0.474               |
| 128²          | multigrid 1                 | semi-lagrangian    | 0.276               | 0.183               |
| 128²          | multigrid 1                 | maccormack         | 0.403               | 0.470               |
| 128²          | multigrid 1 + confinement 8 | semi-lagrangian    | 2.257               | 0.529               |
| 128²          | multigrid 1 + confinement 8 | maccormack         | 3.450               | 1.120               |
| 256²          | jacobi 30                   | semi-lagrangian    | 0.359               | 0.326               |
| 256²          | jacobi 30                   | maccormack         | 0.396               | 0.487               |
| 256²          | multigrid 1                 | semi-lagrangian    | 0.350               | 0.313               |
| 256²          | multigrid 1                 | maccormack         | 0.391               | 0.469               |
| 256²          | multigrid 1 + confinement 8 | semi-lagrangian    | 2.733               | 0.621               |
| 256²          | multigrid 1 + confinement 8 | maccormack         | 3.521               | 0.780               |

### Velocity grid against dye grid: cost of a step, and the stirred pattern against the reference run

Reference: velocity 256², dye 1024², multigrid x3. A second good run (velocity 192², dye 768²) differs from it by 0.027 (128²) and 0.036 (256²): errors near that are indistinguishable from it. Pressure: one multigrid V-cycle; MacCormack for velocity and dye. Error: relative L2 of the dye at 128² (large-scale placement) and 256² (including fine structure); energy and detail (mean squared gradient at 256²) relative to the reference.

| velocity | dye   | ms per step         | error 128² | error 256² | energy | detail |
| -------- | ----- | ------------------- | ---------- | ---------- | ------ | ------ |
| 64²      | 256²  | 0.566 (0.502-0.595) | 0.282      | 0.310      | 0.97   | 0.62   |
| 64²      | 512²  | 1.052 (1.013-1.195) | 0.271      | 0.303      | 0.97   | 0.83   |
| 64²      | 1024² | 3.065 (2.856-3.211) | 0.271      | 0.306      | 0.97   | 0.98   |
| 96²      | 256²  | 0.633 (0.582-0.788) | 0.180      | 0.210      | 0.98   | 0.62   |
| 96²      | 512²  | 1.149 (1.056-1.262) | 0.144      | 0.170      | 0.98   | 0.84   |
| 96²      | 1024² | 3.195 (2.761-3.834) | 0.136      | 0.163      | 0.98   | 0.99   |
| 128²     | 256²  | 0.635 (0.596-0.704) | 0.147      | 0.177      | 0.99   | 0.61   |
| 128²     | 512²  | 1.138 (1.038-1.466) | 0.088      | 0.110      | 0.99   | 0.84   |
| 128²     | 1024² | 3.260 (2.897-3.665) | 0.073      | 0.089      | 0.99   | 0.99   |
| 192²     | 256²  | 0.834 (0.756-1.011) | 0.143      | 0.173      | 0.99   | 0.61   |
| 192²     | 512²  | 1.360 (1.295-1.459) | 0.071      | 0.089      | 0.99   | 0.84   |
| 192²     | 1024² | 3.366 (3.288-3.760) | 0.049      | 0.058      | 0.99   | 0.99   |
| 256²     | 256²  | 1.079 (1.041-1.447) | 0.137      | 0.167      | 1.00   | 0.61   |
| 256²     | 512²  | 1.616 (1.472-1.709) | 0.060      | 0.077      | 1.00   | 0.84   |
| 256²     | 1024² | 3.529 (3.315-4.216) | 0.035      | 0.040      | 1.00   | 1.00   |

### Reading the velocity between its faces: a cell flow of speed 0.3 sampled on a coarse grid

Error is against the analytic velocity at 2000 interior points, in 1e-3 image heights per second; the kink is the jump of the slope across a face, per cell, in the same unit (a C1 reconstruction has none).

| velocity grid | filter            | rms error | max error | slope kink |
| ------------- | ----------------- | --------- | --------- | ---------- |
| 16²           | bilinear          | 3.948     | 11.231    | 17.633     |
| 16²           | hardware-bilinear | 3.950     | 11.298    | 17.578     |
| 16²           | bspline           | 7.373     | 15.185    | 5.252      |
| 16²           | catmull-rom       | 0.227     | 0.741     | 5.700      |
| 32²           | bilinear          | 1.011     | 2.737     | 3.886      |
| 32²           | hardware-bilinear | 1.016     | 2.771     | 3.885      |
| 32²           | bspline           | 1.909     | 3.929     | 1.178      |
| 32²           | catmull-rom       | 0.072     | 0.258     | 1.216      |
| 64²           | bilinear          | 0.257     | 0.802     | 1.020      |
| 64²           | hardware-bilinear | 0.261     | 0.800     | 0.858      |
| 64²           | bspline           | 0.486     | 1.119     | 0.313      |
| 64²           | catmull-rom       | 0.051     | 0.256     | 0.355      |

### Streaklines: a thin dye line (sigma 6 texels at a 512² dye grid) carried through the cell flow for 1 s, against the exact characteristics

Centreline error is the rms distance of the line's centroid from the exact one, in dye texels; wiggle is the part of it with a period shorter than two velocity cells (the kinks of the interpolation).

| velocity grid | filter            | centreline error (texels) | wiggle (texels) |
| ------------- | ----------------- | ------------------------- | --------------- |
| 16²           | bilinear          | 3.514                     | 2.626           |
| 16²           | hardware-bilinear | 3.515                     | 2.627           |
| 16²           | bspline           | 5.001                     | 3.518           |
| 16²           | catmull-rom       | 2.002                     | 1.657           |
| 32²           | bilinear          | 2.337                     | 1.525           |
| 32²           | hardware-bilinear | 2.337                     | 1.525           |
| 32²           | bspline           | 2.719                     | 1.665           |
| 32²           | catmull-rom       | 1.964                     | 1.383           |
| 64²           | bilinear          | 2.054                     | 0.800           |
| 64²           | hardware-bilinear | 2.054                     | 0.800           |
| 64²           | bspline           | 2.147                     | 0.815           |
| 64²           | catmull-rom       | 1.960                     | 0.784           |

### Dye advection alone (dyeOnly), ms per step, velocity grid 128²

| scheme                                                | 256²                | 512²                | 768²                | 1024²               |
| ----------------------------------------------------- | ------------------- | ------------------- | ------------------- | ------------------- |
| semi-lagrangian                                       | 0.059 (0.055-0.082) | 0.200 (0.192-0.207) | 0.431 (0.409-0.518) | 0.746 (0.707-0.813) |
| maccormack                                            | 0.213 (0.197-0.261) | 0.713 (0.689-0.721) | 1.521 (1.334-1.585) | 2.608 (2.370-2.908) |
| bfecc                                                 | 0.214 (0.203-0.266) | 0.664 (0.616-0.699) | 1.447 (1.245-1.766) | 2.574 (2.460-2.840) |
| semi-lagrangian + hardware-filtered bilinear velocity | 0.010 (0.009-0.013) | 0.041 (0.034-0.047) | 0.084 (0.080-0.094) | 0.155 (0.141-0.200) |
| maccormack + hardware-filtered bilinear velocity      | 0.054 (0.048-0.063) | 0.158 (0.149-0.192) | 0.331 (0.306-0.345) | 0.599 (0.528-0.761) |
| maccormack + b-spline velocity                        | 0.080 (0.073-0.082) | 0.239 (0.223-0.245) | 0.486 (0.460-0.599) | 0.861 (0.829-0.923) |
| maccormack + catmull-rom velocity                     | 0.088 (0.084-0.106) | 0.273 (0.253-0.292) | 0.578 (0.521-0.620) | 0.996 (0.933-1.265) |
| semi-lagrangian + catmull-rom velocity                | 0.020 (0.014-0.025) | 0.067 (0.065-0.075) | 0.145 (0.141-0.174) | 0.259 (0.254-0.267) |

### Sharpness: a disc (radius 0.05) carried once round a vortex (240 steps), velocity grid 64²

Edge is the 10 to 90 percent width of the disc edge, in texels and in thousandths of the canvas height (comparable across dye sizes); peak the largest value; mass the total over the disc area.

| dye   | scheme          | edge (texels) | edge (1/1000 height) | peak  | mass  |
| ----- | --------------- | ------------- | -------------------- | ----- | ----- |
| 256²  | semi-lagrangian | 20            | 78.1                 | 0.824 | 0.936 |
| 256²  | maccormack      | 8             | 31.3                 | 1.000 | 1.011 |
| 256²  | bfecc           | 6             | 23.4                 | 0.975 | 0.946 |
| 512²  | semi-lagrangian | 18            | 35.2                 | 0.945 | 0.935 |
| 512²  | maccormack      | 8             | 15.6                 | 1.000 | 0.984 |
| 512²  | bfecc           | 6             | 11.7                 | 0.978 | 0.950 |
| 1024² | semi-lagrangian | 18            | 17.6                 | 0.959 | 0.944 |
| 1024² | maccormack      | 10            | 9.8                  | 1.000 | 0.981 |
| 1024² | bfecc           | 5             | 4.9                  | 0.980 | 0.962 |

### Configurations: the best-looking one under a GPU budget per simulation step

Candidates: velocity grid 64, 96, 128, 192; dye grid 256, 512, 768, 1024; advection velocity/dye (SL semi-Lagrangian, MC MacCormack); pressure J30 (30 Jacobi iterations, today's default) or MG(1,1) (one multigrid V-cycle, one smoothing sweep each way); dye velocity filter bilinear (today's: loads), hardware-bilinear or catmull-rom (J30 only with bilinear). 192 configurations. Cost is the slope per step of the whole step: 'fastest' is the slope between the fastest spans seen over 2 pass(es) of 3 repetitions (what the picks rank by), the median and range are in the JSON and the frontier table. Error is the relative L2 difference of the stirred pattern at 128² from the reference run.

| budget (ms/step) | velocity / dye | advection, pressure, dye velocity | fastest ms | error | energy kept |
| ---------------- | -------------- | --------------------------------- | ---------- | ----- | ----------- |
| 0.5              | 128² / 512²    | MC/MC, MG(1,1), hardware-bilinear | 0.50       | 0.088 | 0.99        |
| 1                | 192² / 768²    | MC/MC, MG(1,1), hardware-bilinear | 0.89       | 0.054 | 0.99        |
| 2                | 192² / 1024²   | MC/MC, MG(1,1), catmull-rom       | 1.71       | 0.046 | 0.99        |
| 4                | 192² / 1024²   | MC/MC, MG(1,1), catmull-rom       | 1.71       | 0.046 | 0.99        |

The same budgets with today's solver and dye path only (30 Jacobi iterations, bilinear loads):

| budget (ms/step) | velocity / dye | advection, pressure, dye velocity | fastest ms | error | energy kept |
| ---------------- | -------------- | --------------------------------- | ---------- | ----- | ----------- |
| 0.5              | 64² / 256²     | MC/SL, J30, bilinear              | 0.49       | 0.386 | 1.03        |
| 1                | 64² / 512²     | MC/SL, J30, bilinear              | 0.66       | 0.372 | 1.03        |
| 2                | 64² / 512²     | MC/SL, J30, bilinear              | 0.66       | 0.372 | 1.03        |
| 4                | 64² / 512²     | MC/SL, J30, bilinear              | 0.66       | 0.372 | 1.03        |

### Configurations: the cost / error frontier (no cheaper configuration looks better)

| velocity | dye   | advection | pressure | dye velocity      | fastest ms | median ms (min-max) | error | energy kept |
| -------- | ----- | --------- | -------- | ----------------- | ---------- | ------------------- | ----- | ----------- |
| 64²      | 256²  | SL/SL     | MG(1,1)  | hardware-bilinear | 0.27       | 0.27 (0.26-0.34)    | 0.576 | 0.51        |
| 64²      | 256²  | SL/SL     | MG(1,1)  | catmull-rom       | 0.28       | 0.29 (0.28-0.31)    | 0.572 | 0.51        |
| 96²      | 256²  | SL/SL     | MG(1,1)  | catmull-rom       | 0.32       | 0.33 (0.32-0.38)    | 0.502 | 0.63        |
| 128²     | 256²  | SL/SL     | MG(1,1)  | catmull-rom       | 0.34       | 0.35 (0.31-0.37)    | 0.436 | 0.72        |
| 64²      | 256²  | MC/SL     | MG(1,1)  | hardware-bilinear | 0.35       | 0.37 (0.35-0.39)    | 0.330 | 0.97        |
| 64²      | 256²  | MC/SL     | MG(1,1)  | catmull-rom       | 0.38       | 0.38 (0.37-0.40)    | 0.323 | 0.97        |
| 96²      | 256²  | MC/SL     | MG(1,1)  | hardware-bilinear | 0.38       | 0.42 (0.38-0.51)    | 0.299 | 0.98        |
| 128²     | 256²  | MC/SL     | MG(1,1)  | hardware-bilinear | 0.39       | 0.44 (0.39-0.47)    | 0.292 | 0.99        |
| 64²      | 512²  | MC/SL     | MG(1,1)  | hardware-bilinear | 0.40       | 0.40 (0.39-0.44)    | 0.263 | 0.97        |
| 64²      | 768²  | MC/SL     | MG(1,1)  | hardware-bilinear | 0.40       | 0.44 (0.40-0.48)    | 0.256 | 0.97        |
| 128²     | 512²  | MC/SL     | MG(1,1)  | hardware-bilinear | 0.40       | 0.45 (0.40-0.51)    | 0.162 | 0.99        |
| 96²      | 768²  | MC/SL     | MG(1,1)  | hardware-bilinear | 0.43       | 0.51 (0.43-0.59)    | 0.148 | 0.98        |
| 128²     | 256²  | MC/MC     | MG(1,1)  | hardware-bilinear | 0.45       | 0.47 (0.45-0.51)    | 0.147 | 0.99        |
| 128²     | 256²  | MC/MC     | MG(1,1)  | catmull-rom       | 0.47       | 0.50 (0.46-0.54)    | 0.142 | 0.99        |
| 128²     | 768²  | MC/SL     | MG(1,1)  | hardware-bilinear | 0.49       | 0.52 (0.49-0.56)    | 0.112 | 0.99        |
| 128²     | 512²  | MC/MC     | MG(1,1)  | hardware-bilinear | 0.50       | 0.58 (0.49-0.66)    | 0.088 | 0.99        |
| 128²     | 512²  | MC/MC     | MG(1,1)  | catmull-rom       | 0.66       | 0.67 (0.66-0.81)    | 0.083 | 0.99        |
| 192²     | 512²  | MC/MC     | MG(1,1)  | hardware-bilinear | 0.72       | 0.78 (0.71-0.84)    | 0.071 | 0.99        |
| 192²     | 512²  | MC/MC     | MG(1,1)  | catmull-rom       | 0.86       | 0.88 (0.81-0.91)    | 0.068 | 0.99        |
| 192²     | 768²  | MC/MC     | MG(1,1)  | hardware-bilinear | 0.89       | 0.98 (0.89-1.09)    | 0.054 | 0.99        |
| 192²     | 768²  | MC/MC     | MG(1,1)  | catmull-rom       | 1.20       | 1.33 (1.18-1.43)    | 0.051 | 0.99        |
| 192²     | 1024² | MC/MC     | MG(1,1)  | hardware-bilinear | 1.23       | 1.31 (1.22-1.39)    | 0.049 | 0.99        |
| 192²     | 1024² | MC/MC     | MG(1,1)  | catmull-rom       | 1.71       | 1.82 (1.52-2.05)    | 0.046 | 0.99        |

### What each change buys at a fixed grid (MacCormack both): fastest cost per step and error

| velocity / dye | Jacobi 30, loaded bilinear (today) | multigrid, loaded bilinear | multigrid, hardware bilinear | multigrid, Catmull-Rom |
| -------------- | ---------------------------------- | -------------------------- | ---------------------------- | ---------------------- |
| 64² / 512²     | 1.08 ms, error 0.410               | 1.01 ms, error 0.271       | 0.53 ms, error 0.271         | 0.63 ms, error 0.257   |
| 64² / 1024²    | 2.79 ms, error 0.414               | 2.81 ms, error 0.271       | 0.89 ms, error 0.271         | 1.23 ms, error 0.258   |
| 96² / 512²     | 1.03 ms, error 0.459               | 1.07 ms, error 0.144       | 0.53 ms, error 0.144         | 0.61 ms, error 0.135   |
| 96² / 1024²    | 3.29 ms, error 0.465               | 3.03 ms, error 0.136       | 0.95 ms, error 0.136         | 1.53 ms, error 0.128   |
| 128² / 512²    | 1.12 ms, error 0.577               | 1.02 ms, error 0.088       | 0.50 ms, error 0.088         | 0.66 ms, error 0.083   |
| 128² / 1024²   | 3.41 ms, error 0.584               | 3.21 ms, error 0.073       | 1.05 ms, error 0.073         | 1.42 ms, error 0.068   |
| 192² / 512²    | 1.48 ms, error 0.477               | 1.29 ms, error 0.071       | 0.72 ms, error 0.071         | 0.86 ms, error 0.068   |
| 192² / 1024²   | 3.74 ms, error 0.485               | 3.23 ms, error 0.049       | 1.23 ms, error 0.049         | 1.71 ms, error 0.046   |

### Under-converged pressure: the stirred pattern after 2 s against the reference run (dye 512²)

Error: relative L2 difference of the dye from the reference (floor between two good runs: 0.027 at 128², 0.036 at 256²). Mass: total dye over its start (1 for an incompressible flow and a conservative scheme; ink piles up or thins where the flow compresses). Energy: kinetic energy relative to the reference. Divergence: rms divergence left in the flow at the end, as the rate at which a parcel's area changes, 1 / s.

| velocity | pressure            | error 128² | error 256² | mass  | energy | divergence (1/s) |
| -------- | ------------------- | ---------- | ---------- | ----- | ------ | ---------------- |
| 128²     | jacobi 5            | 0.688      | 0.708      | 1.023 | 1.38   | 0.381            |
| 128²     | jacobi 10           | 0.569      | 0.590      | 1.004 | 1.32   | 0.274            |
| 128²     | jacobi 20           | 0.561      | 0.589      | 0.994 | 1.17   | 0.172            |
| 128²     | jacobi 30 (default) | 0.577      | 0.606      | 0.981 | 1.07   | 0.094            |
| 128²     | jacobi 60           | 0.456      | 0.487      | 0.991 | 1.12   | 0.073            |
| 128²     | jacobi 120          | 0.389      | 0.423      | 0.986 | 1.06   | 0.051            |
| 128²     | sor 60              | 0.078      | 0.100      | 0.986 | 1.00   | 0.003            |
| 128²     | multigrid 1 cycle   | 0.088      | 0.110      | 0.986 | 0.99   | 0.003            |
| 128²     | multigrid 2 cycles  | 0.080      | 0.102      | 0.986 | 1.00   | 0.003            |
| 256²     | jacobi 5            | 1.066      | 1.074      | 1.300 | 1.35   | 0.771            |
| 256²     | jacobi 10           | 0.750      | 0.768      | 1.132 | 1.57   | 0.601            |
| 256²     | jacobi 20           | 0.685      | 0.705      | 1.023 | 1.39   | 0.382            |
| 256²     | jacobi 30 (default) | 0.657      | 0.677      | 1.015 | 1.31   | 0.289            |
| 256²     | jacobi 60           | 0.489      | 0.515      | 0.999 | 1.26   | 0.201            |
| 256²     | jacobi 120          | 0.574      | 0.603      | 0.982 | 1.07   | 0.094            |
| 256²     | sor 60              | 0.093      | 0.115      | 0.986 | 1.00   | 0.045            |
| 256²     | multigrid 1 cycle   | 0.060      | 0.077      | 0.986 | 1.00   | 0.006            |
| 256²     | multigrid 2 cycles  | 0.047      | 0.065      | 0.986 | 1.00   | 0.005            |

## How it works

The shared position-based fluid solver (`src/gpu/sim/pbf.ts`) runs in two dimensions. It predicts positions, bins particles with GPU counting sort and an exclusive prefix sum, then alternates density and position-correction passes. Density is a constraint on positions rather than a stiff pressure force. Poly6 density and spiky-gradient kernels are calibrated against a rest lattice at the selected spacing. Wall density compensates for missing liquid beyond the vessel; box and sphere colliders project particles out of solids. XSPH viscosity and optional vorticity confinement modify velocities after the solve.

The demo offers a dam break, a pour with a shelf, waves and falling drops. Drag through the fluid to push it; tap to add a blob. Liquid splats and a composite pass join the particles into a continuous 2D surface; debug views expose density error, speed and neighbour count. The same parameter schema supplies presets, controls and URL state. Fixed timesteps separate solver behavior from the display refresh rate. Resize changes the rendering targets without resampling the physical particles; the host pauses hidden canvases and handles device recovery and reduced motion.

## Cost

See `docs/particle-scale-study.md` and its raw JSON for the measured particle-count and iteration curves. The study distinguishes one solver substep from a complete 1080p frame. Increasing the count at fixed liquid volume decreases particle spacing; maintaining similar density convergence can require more iterations as well as more particles. Tier B is a target for a modest 2D vessel, with mobile still untested. Optional timestamp queries measure cost but are not needed to run the effect. The captured poster is the no-WebGPU fallback.

## Pitfalls

- A parallel Jacobi solve converges less quickly than a sequential correction. Too few iterations allow compression, particularly at the bottom of a deep pool. Large timesteps can remain bounded while producing poor water.
- The neighbour grid is built from predicted positions once per substep. Large corrections can change the true neighbourhood before the last iteration; smaller substeps reduce that error.
- Artificial pressure prevents clumping but can add expansion and noise if pushed too high. Surface tension and incompressibility are approximations.
- Sorting changes particle indices. All persistent per-particle attributes must move with the particle system; an external array indexed by the old order is incorrect.
- Dense cells make neighbour work quadratic locally. Count alone cannot predict cost for a collapsed or compressed scene.
- This is a two-dimensional slice, with area as particle rest volume. It is not a three-dimensional splash viewed from the side.

## When to use it

Use it for an interactive vessel, filling illustration, hero card or loading scene where water must pour and slosh. Keep the particle count and solver iterations modest and use the debug views when choosing a quality budget. A heightfield is cheaper when only the top surface needs to move. Use the 3D tank when depth and separated splashes matter.

## References

- Miles Macklin and Matthias Müller, [Position Based Fluids](https://mmacklin.com/pbf_sig_preprint.pdf), 2013: density constraints, artificial pressure, viscosity and vorticity correction.
- Matthias Müller, David Charypar and Markus Gross, [Particle-Based Fluid Simulation for Interactive Applications](https://matthias-research.github.io/pages/publications/sca03.pdf), 2003: SPH kernels and the particle-fluid baseline.

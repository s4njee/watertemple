## How it works

The two- and three-dimensional demos share `src/gpu/sim/pbf.ts`. In 3D, each fixed step predicts particle motion, constructs the uniform-grid neighbour search with counting sort and a prefix sum, and applies density constraints through alternating density and position-correction compute passes. Pressure is expressed as a change in position; XSPH viscosity and optional vorticity confinement follow the iterations. The solver supports moving container walls, box and sphere colliders, a pour emitter, pointer impulses and particle blocks.

Three scenes demonstrate a dam break around pillars, a pour and a wave tank with a moving paddle. In **stir** mode, drag the water to push it, tap to add a blob, and drag empty space to orbit. In **container** mode, dragging tilts gravity in the vessel's local frame and adds the opposite of pointer acceleration, so the liquid sloshes against its walls. The shake control adds bounded periodic inertial acceleration. The vessel is drawn in this local frame; its outline stays fixed. **Device tilt** is opt-in; enable it and tap the tank on browsers that require permission from a gesture. Denial or an unavailable sensor leaves pointer controls available. Turn it off to detach the sensor listener.

The surface renderer draws sphere-impostor depth, smooths it with an edge-preserving narrow-range filter, reconstructs normals, accumulates and blurs thickness, and composites reflection, refraction and absorption over the scene. The scene's depth prevents foreground pillars from being painted over. The sphere, speed and foam debug views bypass the surface reconstruction. Rendering resolution is independent of the physical simulation, and the host handles offscreen pause, reduced motion and device recovery.

**Secondary whitewater** (`src/gpu/particles/whitewater.ts`) has a separate bounded store of 4096 particles. Dead slots sample current energetic primary particles, using their foam scalar as an emission probability. A fresh sorted-index neighbour grid classifies each secondary particle from local occupancy: sparse spray integrates gravity, surface foam follows nearby fluid velocity, and submerged bubbles follow flow with an upward buoyancy term. Finite lifetimes recycle the slots; secondary particles never add mass or forces to the primary fluid. Their indices do not follow the primary solver's reorder. White sprites with a blue bubble tint overlay the tank, depth-tested against the solid scene. Disable whitewater to compare its appearance and cost. The shake-and-spray and calm-vessel presets expose both ends of the interaction.

## Cost

See `docs/particle-scale-study.md` and `docs/particle-scale-study.json`: particle count, 2D/3D solver iterations and complete 1080p frames are measured separately. The default 10,000-particle request is a desktop showpiece; a lattice block may contain fewer particles than requested because only whole rows fit. Higher count, more iterations and more substeps all cost simulation time. Surface scale, smoothing iterations and thickness stride primarily change rendering cost. The cut-down preset is a starting point, not an independently measured Tier B guarantee. Mobile is untested; timestamp queries are optional. A captured poster is available without WebGPU.

## Pitfalls

- Bounded motion does not guarantee incompressibility. A deep pool with few iterations visibly compresses; more substeps and iterations increase cost.
- The neighbour grid is reused across correction iterations. Very large corrections can miss newly adjacent particles. Dense clumps also make the search expensive.
- The rendered surface is one depth layer. It cannot correctly refract several disconnected bodies along the same ray, and excessive smoothing can merge nearby droplets.
- Secondary whitewater is a visual approximation inspired by occupancy-based classification. It does not implement the complete entrainment, curvature or energy model of the whitewater paper. Foam is a sprite and bubbles have a tinted disc rather than a resolved refractive shell.
- Whitewater uses a fixed-capacity store and a capped update delta; under long frames it loses elapsed lifetime/motion time. Its emission can saturate the store and is deliberately bounded. It performs an extra neighbour-grid build after the primary solve, so the classification uses the final positions.
- Drag and shake use a translating/tilting frame approximation. The demo does not integrate a rigid-body container or include rotational Coriolis terms. Device orientation is an artistic gravity mapping with beta zero as neutral, not an accelerometer measurement.
- Tilt permission must originate from a canvas gesture on browsers that require it. Sensor availability and permission are separate from WebGPU availability.
- Fixed-step backlog is dropped after long frames. A static poster cannot preserve interaction. Phones and non-Apple integrated GPUs still need measurements.

## When to use it

Use it as a dedicated desktop landing-page set piece, a tank toy, or a product experience whose subject is pouring and splashing. Container controls make sense when the user is meant to move a vessel. The liquid fits hero and card contexts through the common host, but the default simulation is too costly for many simultaneous animated thumbnails. For a decorative surface use ripples or shallow water; use 2D PBF when depth is unnecessary.

## References

- Miles Macklin and Matthias Müller, [Position Based Fluids](https://mmacklin.com/pbf_sig_preprint.pdf), 2013.
- Wladimir J. van der Laan, Simon Green and Miguel Sainz, [Screen Space Fluid Rendering with Curvature Flow](https://doi.org/10.1145/1507149.1507164), 2009.
- Nghia Truong and Cem Yuksel, [A Narrow-Range Filter for Screen-Space Fluid Rendering](https://www.cemyuksel.com/research/papers/narrowrangefilter.pdf), 2018.
- Markus Ihmsen, Nadir Akinci, Gizem Akinci and Matthias Teschner, [Unified Spray, Foam and Bubbles for Particle-Based Fluids](https://doi.org/10.1007/s00371-012-0687-9), 2012: the inspiration for occupancy-based secondary particle types; the simplified implementation is described above.

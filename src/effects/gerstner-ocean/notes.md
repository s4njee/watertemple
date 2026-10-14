## How it works

Six directional Gerstner waves displace a real indexed mesh in the vertex stage. The two analytic tangents produce the normal and horizontal Jacobian. The water combines the shared Fresnel, sky, finite-sun specular and crest scattering functions, then fades into horizon fog. **Normals**, **foam** and **mesh** views isolate the ingredients. Projected and clipmap geometry use the same wave function and shading.

Two signal buoys sample three surface points each, giving height and tilt. **CPU** sampling mirrors the WGSL function and inverts horizontal displacement with bounded Newton steps; sampling height at the original coordinate would let the buoy slide. **GPU** sampling dispatches the same inverse function, copies six positions into a staging buffer, and asynchronously maps a previous submission. The renderer keeps the latest completed samples while waiting, without blocking the frame. The result timestamp is preserved in each sample. Small spheres mark the buoy centers and tilted mast tips.

The Source tab exposes the shared renderer, wave sampling, mesh generation and buoyancy code.

## Cost

The default has 18,432 triangles and six CPU probes. `docs/ocean-cost-study.md` compares six mesh configurations and 6/96/768 CPU versus GPU queries. GPU compute time and readback wall latency are separate metrics. Six CPU probes avoid a frame of readback latency; large query batches can favor GPU computation. Timestamp queries are optional and mobile is untested.

## Pitfalls

Large height/steepness combinations can fold the surface and make its inverse ambiguous. Newton steps are bounded rather than a global intersection solver. Buoys follow the surface kinematically; this does not integrate rigid-body forces or create wakes. Readback trails the current wave phase. Projected geometry pads the view, caps the horizon intersection, and uses a minimum reference height near the surface; large breaking crests can still expose its finite extent. Clipmap levels share a snapped center and stitched edges but can pop when that center moves.

## When to use it

Use a controllable ocean hero, maritime scene or lightweight buoyancy illustration. Prefer CPU samples for a handful of objects. Choose the spectral demo when many wave frequencies and persistent folding foam matter.

## References

- Mark Finch, [Effective Water Simulation from Physical Models](https://developer.nvidia.com/gpugems/gpugems/part-i-natural-effects/chapter-1-effective-water-simulation-physical-models), GPU Gems, 2004.
- Arul Asirvatham and Hugues Hoppe, [Terrain Rendering Using GPU-Based Geometry Clipmaps](https://developer.nvidia.com/gpugems/gpugems2/part-i-geometric-complexity/chapter-2-terrain-rendering-using-gpu-based-geometry), GPU Gems 2, 2005. The study uses a simplified concentric ring mesh.

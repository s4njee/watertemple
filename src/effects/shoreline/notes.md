## How it works

A sloping procedural beach intersects a meshed wave surface. Depth increases offshore; a bounded inverse-square-root shoaling factor raises incoming crests and fades displacement at the dry edge. Alongshore phase variation breaks perfectly parallel crests. Shallow depth and crest phase drive foam, broken up by world-space noise. Fragments beneath the beach are discarded against the same slope used by the background intersection.

Wet sand darkens near the moving wash zone and adds a small reflected-sky lobe. Sand grains, water absorption, sky reflection, sun glints and fog share the scene palette. The gentle surf and breaking shore presets expose different wash heights.

## Cost

One depth-tested scene pass; no simulation textures or compute dispatches. Mesh density primarily controls silhouette quality. The default 1080p frame is measured in `docs/ocean-cost-study.json`. The poster supplies the no-WebGPU fallback. Mobile remains untested.

## Pitfalls

Breaking is a shoaling and crest-shape approximation, not an overturning fluid solve. The beach is a plane, foam is procedural and wetness is a moving mask rather than persistent water saturation. High waves or a very low camera can expose the finite projected mesh. Terrain displacement and water clipping must use the same beach definition.

## When to use it

Use a beach landing page, coastal product hero or a compact example of depth foam and wet material response. Use shallow water or particle fluids for erosion, surge forces and splashes.

## References

- Mark Finch, [Effective Water Simulation from Physical Models](https://developer.nvidia.com/gpugems/gpugems/part-i-natural-effects/chapter-1-effective-water-simulation-physical-models), GPU Gems, 2004.

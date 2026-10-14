## How it works

A signed RG flow texture is baked from potential-flow disturbances around four rocks. The base current moves downstream; each rock diverts velocity tangentially around its circular footprint. Two offset scrolling phases blend continuously, avoiding a texture reset at the end of the cycle. Flow-distorted noise drives the surface detail.

Rock intersections share scene depth with the water. A distance band adds foam around each rock, and downstream masks extend broken wakes. A gently curved channel clips the water against procedural banks. Forest stream and fast current presets change the current's apparent speed and detail.

## Cost

One scene pass and a 128² `rgba16float` signed flow map (128 KiB), generated once. There is no pressure solve. Both mesh strategies remain available for comparison; default cost is recorded at 1080p on the M1 Max. No mobile measurements have been made.

## Pitfalls

Potential-flow disturbances are superimposed and capped, so the multi-rock map is an approximation, without separation or turbulence. Rock positions must match the map, depth scene and foam masks. Wakes are directional masks rather than advected fluid. Detail normals are an artistic noise perturbation, and changing flow speed does not conserve discharge.

## When to use it

Use a river hero, environment stream or a directed-flow demonstration. A baked flow map works well when the river and obstacles are fixed. Use the grid-fluid solver when obstacles or currents must respond dynamically.

## References

- Mark Finch, [Effective Water Simulation from Physical Models](https://developer.nvidia.com/gpugems/gpugems/part-i-natural-effects/chapter-1-effective-water-simulation-physical-models), GPU Gems, 2004.

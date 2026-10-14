## How it works

A vertical sheet uses narrow scrolling noise streaks to suggest falling water. An uneven mossy cliff and rounded rocks give it a depth-tested setting. A rippled plunge pool adds foam near the impact zone.

A separate bounded store contains 1,536 GPU spray particles. Compute integration applies gravity and drag, decrements lifetime, and recycles particles at the foot of the sheet with deterministic randomized launch velocities. Soft alpha billboards supply mist, with the common scene depth hiding particles behind rocks and the cliff. The sheet uses premultiplied alpha and keeps the pool's depth.

## Cost

The scene pass is followed by spray integration and one instanced mist draw. Particle count is fixed; mist overdraw can dominate at close viewpoints. Update delta is capped at 1/30 second. The complete 1080p frame, including mist, is measured in the ocean study. Mobile is untested.

## Pitfalls

This is a procedural sheet and spray model, without a volume solve or conserved flow into the pool. Spray is not depth sorted and densely overlapping sprites can look cloudy. The capped delta drops motion time on slow frames. Cliff shape and noise are intentionally simple. Mist is a translucent sprite rather than volumetric scattering.

## When to use it

Use a waterfall set piece or a compact demonstration of combining procedural geometry with particle mist. Use a particle-fluid solver for close views of water colliding with arbitrary objects.

## References

- Mark Finch, [Effective Water Simulation from Physical Models](https://developer.nvidia.com/gpugems/gpugems/part-i-natural-effects/chapter-1-effective-water-simulation-physical-models), GPU Gems, 2004.

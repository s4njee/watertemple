## How it works

A top-down river in one fullscreen pass, `river surface`. Two baked textures drive it, both made once in `create()` by `src/gpu/bake.ts`:

- **The flow map** (`createFlowMap`, 1024 x 512 `rgba8unorm`, no mips). `rg` is the velocity (byte 127 is exactly zero), `b` a foam mask (rings round the rocks, wakes behind them, a trace in the shallows) and `a` the water depth, where 0 is dry ground. The velocity is the curl of a stream function: a meandering channel with a parabolic cross-section (fast in the middle, still at the banks, faster where it narrows) plus a potential-flow dipole for each of six rocks plus a little curl noise. Velocity from a stream function is divergence-free, so the water goes round the rocks and never piles up or thins out. The map repeats seamlessly along x.
- **A ripple normal map** (`createWaterNormalMap` with the height channel kept, 512 x 512 `rgba16float` with mips).

The shader looks up the flow at the pixel and uses `flow_phases` from `flow.wgsl` (the two-phase technique of Vlachos' Portal 2 talk): the normal map's coordinate is moved along the flow, but only for `cycle` seconds, then snapped back. A second copy runs half a cycle behind it, and each copy's weight is a triangle wave that is zero exactly at its own reset, so the snap is never visible. Each pixel's phase is offset by smooth noise (`flow_phase_noise`) so the resets do not all happen at the same moment. Two ripple scales are advected this way. The two phases are combined with `flow_blend_slope`, which divides by `sqrt(w_a^2 + w_b^2)` so the ripples do not lose contrast at the 50/50 point, and the normal strength follows the local speed (`flow_strength`). Foam is a thresholded version of the map's height channel, advected the same way, masked by the baked foam channel.

Texture lookups use `textureSampleGrad` with the derivatives of the unadvected coordinate. The advected uv jumps at each reset, and hardware derivatives there would pick a blurry mip along thin lines.

The debug views: `flow` colours the direction (hue, exaggerated 2.5x about "to the right") and the speed (brightness); `phase` shows the weight of the first phase, which makes the pulsing visible.

## Cost

One pass, no per-frame textures to update. Indicative, from the HUD in Chrome on this development Mac (shared Apple GPU), at 1702 x 724 (1.2 Mpx, DPR 2): about 1.1 ms for the `fast stream` preset. A pixel does one flow fetch, four normal-map fetches (two scales, two phases each), two for the static ground texture, which is about 7 fetches in all. Scaled to 1080p that is about 2 ms on this GPU, so Tier A is the target rather than a measured result; an integrated GPU has not been tried.

Memory: flow map 2.0 MB, normal map 2.7 MB; the HUD reports about 4.7 MB across two textures. Baking takes about 12 ms for the normal map and 15 ms for the flow map at startup (headless test device, pipeline compile included). Bundle size is not measured yet.

## Pitfalls

- **Cycle too long: stretching.** Each copy drifts for a whole cycle before it resets, and pixels with different flow drift different distances, so the ripples shear along the current. The `cycle too long` preset (8 s at 1.6x speed) shows thin streaks that smear along the flow, worst beside the rocks where the flow bends. Too short and the pattern visibly refreshes. The default 1.6 s looks right at the default scale and speed; the longer the cycle, the slower the water should be.
- **Phase noise off: pulsing.** With `phaseNoise` at 0 every pixel resets at the same instant, so the whole surface fades through the half-way blend and back together, twice per cycle. The `phase` debug view shows it: a uniform grey that breathes. With the noise on it breaks into a marbled map of different phases, and the pulse becomes an even shimmer. I verified the mechanism in the debug view and the weights in the tests; I did not capture the motion.
- **A flow map must not leak.** The baked velocity encodes zero as byte 127, because 0.5 cannot be stored in 8 bits (128/255 decodes to a steady 1/127 drift, which the test documents). Still water would otherwise creep.
- **Seams in the bake.** My first dipole used a nearest-image wrap in x, and its sign flipped half way round the world, leaving a one-texel line of spurious velocity across the river. The dipole now fades out over distance. The test that caught the next problem was flux: the same amount of water must cross every vertical cut of the channel, which the bake satisfies to within 10%, rocks included.
- **Light.** Seen from above, the plain Fresnel reflection is only a few percent, so the ripples vanish. The demo adds a wrapped N.L on the body colour and a broad specular lobe, with the sun low enough that only tilted ripples glint. Put the sun at the mirror direction and the whole river lights up flat.
- **Mip seams** at the phase resets are the reason for `textureSampleGrad`; I added it up front and did not render the broken version, so I have not seen the seams myself.
- **Repetition.** The map repeats every two river-heights. In the wide `band` layout the rocks visibly repeat three or four times.
- **Rock outlines** come from a depth ramp in the bake, so they are as soft as one flow-map texel, about 2 px at 1080p here.

## When to use it

Rivers, streams, canals, drains and any directed current seen from above or at a slant; foam, leaves or oil carried along a baked field (swap the normal map for any texture). It adds about one extra texture fetch per layer and phase to a normal-mapped surface, so it fits Tier A. For an open ocean use a wave simulation instead, for turbulent water a real fluid simulation (Epic 6): a flow map is a fixed field and cannot react to the player.

## References

- Alex Vlachos, [Water Flow in Portal 2](<https://advances.realtimerendering.com/s2010/Vlachos-Waterflow(SIGGRAPH%202010%20Advanced%20RealTime%20Rendering%20Course).pdf>), SIGGRAPH 2010.
- Jasper Flick, [Texture Distortion](https://catlikecoding.com/unity/tutorials/flow/texture-distortion/), a clear walk through the two-phase blend and the per-pixel noise offset.
- Robert Bridson, Jim Hourihan and Marcus Nordenstam, [Curl-Noise for Procedural Fluid Flow](https://www.cs.ubc.ca/~rbridson/docs/bridson-siggraph2007-curlnoise.pdf) (2007), the divergence-free field construction.

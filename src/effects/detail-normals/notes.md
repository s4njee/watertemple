## How it works

A perspective view of a horizontal plane of water. For each pixel the shader intersects the view ray with the plane, which gives a world position `q` on the water and a distance. Up to three layers of one baked ripple normal map are looked up at `q`, each at its own scale, tile rotation, scroll direction and speed, blended into one normal, and lit with a small local model (a sky-gradient reflection with a Schlick Fresnel, a sun lobe, a body colour and distance fog). The lighting is only there to make the normals legible; it is not one of the shared optics chunks.

The texture comes from `createWaterNormalMap` in `src/gpu/bake.ts`, rendered once in `create()`: five octaves of seamlessly tileable 3D gradient noise (`gnoise_3d_tile`, the third axis is the seed), turned into a slope by central differences. It is stored as the signed `xy` of a tangent-space normal in an `rg16float` texture with a full mip chain (z is rebuilt with `normal_decode_signed`). Averaging `xy` averages slopes, so the mips flatten with distance instead of turning to noise.

`normals.wgsl` does the layer work. Each layer is decoded, turned back into world axes with `normal_yaw` (the layer's lookup is rotated, so the stored slopes must be rotated too), and scaled with `normal_scale`. The base layer and each detail layer are then combined with `normal_blend_whiteout`, `_rnm`, `_udn`, `_pd` or `_linear`, chosen by the `blend` parameter. Linear is the naive baseline: it is `normalize(a + b)`, so z is the sum of both z's and every slope comes out roughly halved. Tick `Compare` to put linear on the left half of the view and the chosen blend on the right.

Other switches that exist to show a problem: `Source = procedural` computes the same look per pixel from analytic-gradient noise (the cost the texture saves); `Filter` swaps the sampler between anisotropic, trilinear and mip-0 only; `Debug view` shows the blended normal as colour, in the perspective view or straight down.

## Cost

One fullscreen pass, `water surface`. Indicative, read from the HUD in Chrome on this development Mac (an Apple GPU shared with other work, so treat the figures as rough), at 1702 x 724 (1.2 Mpx, DPR 2), 16x anisotropic:

| Configuration                                 | GPU ms |
| --------------------------------------------- | ------ |
| texture, 3 layers                             | 1.05   |
| texture, 2 layers                             | 1.03   |
| procedural (5-octave FBM per layer), 3 layers | 3.79   |
| procedural, 1 layer                           | 1.83   |

A procedural layer costs about 0.9 ms here and a texture layer too little to separate from the noise, on top of a floor of about 1 ms that is the sky, fog and tonemap, not the normals. That puts the equivalent per-pixel noise at about 3.6x the whole texture version at three layers, and it also aliases at distance, which the texture does not. At 1080p (2.1 Mpx) the texture version scales to about 1.8 ms; not measured on an integrated GPU, so the Tier A claim rests on that extrapolation.

Memory: the baked map is 512 x 512 `rg16float` with 10 mip levels, 1.4 MB (the HUD shows about 1.3 MB). Baking it takes about 12 ms at startup in the headless test device (pipeline compile included). Bundle size is not measured yet.

## Pitfalls

- **Linear blending flattens the detail.** Measured in `tests/optics-detail.test.ts`: blending a normal with a flat detail returns the base for whiteout, RNM and PD, but linear returns a normal whose slope is only about 0.4 to 0.65 of the base's. In the compare view the left half loses its sun glints and its fine ripples at once. UDN has the opposite small flaw: a flat base does not return the detail exactly (up to about 0.13 off at slopes around 0.7).
- **Whiteout versus RNM.** They are hard to tell apart at ripple slopes (for slopes under about 0.2 all four correct blends agree within about 2%). RNM earns its extra ALU only when the base is steep, for example large chop under fine ripples.
- **Rotated layers need rotated normals.** Rotating the tile grid per layer hides the repetition, but the stored slopes belong to the texture's axes, so skipping `normal_yaw` lights the rotated layers from the wrong side.
- **Two textures sliding over each other.** Layers with identical scale and opposite directions (the `sliding layers` preset: ratio 1, spread 180) read as one texture dragged across another. What hides it, all of which are defaults here: a non-integer scale ratio (2.3, so the tile seams never coincide), scroll directions about 70 degrees apart rather than 0 or 180, speeds that are not multiples of each other (0.035, 0.052 and 0.081 tiles per second), a different tile rotation per layer, and falling weights (1, 0.8, 0.6) so no layer dominates. I judged this from stills; the motion itself is not captured here.
- **Tile repetition.** The straight-down debug view shows the base layer's 6-cell tile repeating every few hundred pixels. In the perspective view it is hidden by the foreshortening, the ratio and the rotations; a flat top-down water view would need a bigger tile or a third decorrelated layer.
- **No mips, no peace.** The `no mipmaps` preset (sampler locked to level 0) sparkles and crawls toward the horizon. With mips plus anisotropy the distance goes smooth. The cost of averaging slopes is that the distant water also loses its roughness, so distant glints are weaker than a procedural version would give.
- **Seeds.** Shifting a periodic noise only translates it, so a different seed would have given the same pattern. The bake uses a slice through 3D tile noise instead.
- **Sampling and control flow.** `textureSample` needs uniform control flow, so sky pixels are shaded with a clamped distance and thrown away at the end. A `layers` count read from a uniform may skip work: my first version sampled all three layers regardless, and the HUD showed no saving until the layers were made conditional.

## When to use it

Any time a water surface needs ripple detail and a texture fetch is affordable: lakes, pools, the detail layers on top of Gerstner or FFT waves, a sea seen from a hero. Use whiteout by default, RNM when the base normals are steep. It fits Tier A as a building block. Reach for per-pixel noise only where there is no texture budget at all or the pattern must never repeat, and expect to pay around 1 ms per layer at 1.2 Mpx on a GPU like this one.

## References

- Colin Barre-Brisebois and Stephen Hill, [Blending in Detail](https://blog.selfshadow.com/publications/blending-in-detail/) (2012), the source of UDN, whiteout, partial-derivative and reoriented blending.
- Inigo Quilez, [Gradient noise derivatives](https://iquilezles.org/articles/gradientnoise/), for the analytic gradients behind the procedural comparison.
- Krzysztof Narkowicz, [ACES filmic tone mapping curve](https://knarkowicz.wordpress.com/2016/01/06/aces-filmic-tone-mapping-curve/), the tonemapper used for the demo lighting.

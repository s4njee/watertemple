## How it works

`hello` is a placeholder. It exists to exercise the foundation end to end: shader includes, a generated uniform struct, the parameter schema and presets, the shared WGSL library, and the shell around them. It is not meant to look like water yet.

It is a single fullscreen fragment pass. The fragment stage turns the pixel position into aspect-corrected coordinates scaled by the `scale` parameter, then does three things with them:

1. **Domain warp.** `domain_warp_2d_anim` displaces the coordinates along two animated fractal noise fields. `warp` sets how far, and the field drifts slowly with time.
2. **Pattern.** A 5-octave 3D fractal sum (`fbm_3d`) sampled at the warped coordinates, with time as the third axis, gives a value `n` between 0 and 1. A second, ridged sum gives thin bright creases (`crest`) where the noise crosses zero.
3. **Colour.** `n` indexes a cosine palette whose offset term is the `tint` parameter. The crest term is added as a pale highlight, and the result is tonemapped with ACES and converted from linear to sRGB.

The core of the pattern is two lines of `hello.wgsl`:

```wgsl
let warped = domain_warp_2d_anim(p, params.time * 0.2, params.warp, 3, 2.0, 0.5);
let n = 0.5 + 0.5 * fbm_3d(vec3f(warped, params.time * 0.15), 5, 2.0, 0.5);
```

The `Params` struct is declared once in `index.ts`. Its WGSL is generated from that definition and injected as the `params.wgsl` include, so the CPU-side packing and the shader cannot drift apart. The Source tab shows both that chunk and the fully expanded shader.

Time is accumulated on the CPU (`phase += delta * speed`) rather than computed as `time * speed`, so moving the speed slider never makes the pattern jump.

## Cost

One render pass, no textures, no compute. Per pixel the shader evaluates gradient noise about 14 times by the octave counts in the code: 6 for the warp (two 3-octave fields), 5 for the pattern, 3 for the crests, plus the palette and tonemap.

GPU time per device: not measured yet. The Scorecard tab shows a live reading for your device once the browser reports pass timings. The cost scales with the number of pixels, so the resolution scale matters more than any parameter.

## Pitfalls

- The output is plausible as marble, smoke or ink. Nothing in it reads as water: there is no depth, reflection or refraction.
- The noise is cheap per octave but the count adds up at high resolution. A full-resolution canvas on a high-DPR phone is the case to test first; it has not been tested.

## When to use it

Not for shipping. Use it to check that a change to the shell, the preprocessor, the uniform writer or the shared library still runs an effect from start to finish. As a visual it is a quiet abstract background, and it uses the same warped-noise structure that the planned ink wash background (story 4.9) describes.

## References

- Inigo Quilez, [Domain warping](https://iquilezles.org/articles/warp/).
- Inigo Quilez, [Palettes](https://iquilezles.org/articles/palettes/), the cosine palette form.
- Krzysztof Narkowicz, [ACES filmic tone mapping curve](https://knarkowicz.wordpress.com/2016/01/06/aces-filmic-tone-mapping-curve/).

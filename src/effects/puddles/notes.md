## How it works

A perspective camera looks at an endless ground plane. Two passes.

**Bake** (`ground.wgsl`, once at start and again when the `surface` select changes). One fullscreen pass renders a 6 m seamlessly tiling patch of ground, 1024 x 1024, into two targets with mip chains: `rgba16float` with the dry albedo in rgb and a height in alpha, and `rg16float` with the tangent-space normal. Three surfaces: asphalt (binder, aggregate speckle, hairline cracks, a worn lane line), cobbles (Voronoi stones about 12 cm across with domed tops and dark grout) and flagstones (75 cm slabs, each tilted and tinted). Every pattern is built from the `_tile` noise and Voronoi functions, so the texture repeats without a seam. Height is mostly a metre-scale noise (this is what decides where the low spots are) plus the profile of the stones, spanning 5 cm. The normal comes from finite differences of the height at bake time, so the per-frame pass does not need neighbours.

**Frame** (`puddle.wgsl`). Per pixel it intersects the view ray with the ground and reads both baked textures (trilinear and 4x anisotropic, before any branching because `textureSample` needs uniform control flow). Everything after that is the shared chunks:

- `wetness.wgsl`: `wet_depth` is water level minus ground height; `wet_water_mask` turns it into coverage with a shoreline one pixel wide (the edge width is the ground slope times the pixel footprint, so the edge stays crisp up close and far away); `wet_damp` gives the damp band, 1 at the waterline and falling to the general wetness over a reach of a few centimetres; `wet_albedo` and `wet_roughness` darken, saturate and smooth the material.
- Ground pixels are lit by the sun (GGX glint from `specular.wgsl`, strong only when wet) and the sky, and get a dull sheen of the sky whose weight is `wet_reflectance` (Fresnel times wetness squared times smoothness).
- Water pixels mirror the environment through the ripple normal, weighted by Fresnel from `fresnel.wgsl`, over the ground seen through the water (`wet_through_water`: it darkens and loses red with depth) plus the sun's glint.
- The environment is `sky.wgsl` plus two skylines of lit windows and six glowing lamps, so reflections have shapes and colours to show. The sky pixels above the horizon use the same function, so the reflection matches what is "there". The rain thickens the haze and darkens the sky.
- Rain ripples come from `ripples.wgsl`: expanding cosine wavelets from a hashed drop in each cell, summed over the four nearest cells so rings cross cell borders, with an analytic slope. Two layers at different cell sizes. The slope fades by the pixel footprint so distant rings do not shimmer, and by depth so there are none at the very shore. Neighbouring cells get independent phases and periods, so nothing pulses in step.

Per water pixel at the default settings: two baked-texture reads for everyone, then up to 8 ripple cell lookups (4 per layer; the second layer is skipped where its rings are no longer resolved), one environment evaluation with two skylines and lamps, a handful of Fresnel and GGX terms. Dry ground pixels skip the ripples and the water shading.

## Cost

Indicative HUD readings on an Apple Silicon laptop, 1702x662 canvas (1.1 Mpx, DPR 2), other agents' tabs sharing the GPU: about 1.3 ms with `rain` 0 (the floor of this machine, a near-empty pass reads about 1.2 ms) and about 1.5 ms for `steady rain` with 4x anisotropy. The first version, with 8x anisotropy, a second ripple layer at every pixel and the lamp directions computed per pixel, read about 1.9 ms; the three cuts together bought about 0.4 ms. The bake is not repeated. Textures: 16 MB (1024 x 1024, `rgba16float` plus `rg16float`, with mips). Scaled to 1080p (2.1 Mpx) the rain presets would be around 2.5 to 3 ms by pixel count, so the effect is Tier A on this hardware but likely over the 2 ms integrated-GPU budget at full 1080p with heavy rain and a large flooded area; not measured on an integrated Intel or Mali GPU or a phone. To fit, render into a half-size canvas, drop the second ripple layer, or use `rain` 0 for a still wet street. The cost grows with the share of the screen that is water, and with the rain intensity.

## Pitfalls

- **Rings need something to reflect.** Seen from above the water is 2% mirror (Fresnel), and a flat sky gives the rings nothing to bend, so ripples vanish at high camera tilt. Keep the tilt under about 40 degrees, or put a bright sky gradient or the sun behind the rings. A "top-down flood" preset was dropped for this reason.
- **Derivatives in branches.** `textureSample` must run in uniform control flow, so both ground reads happen at the top of the shader and the sky, ground and water branches only use their results; for pixels above the horizon the ray distance is clamped so the unused read has sane derivatives.
- **Not measured, only designed.** The horizon fade (the ground is forced to the environment over the last degree below the horizon) and the shoreline width (baked slope times pixel footprint) are there to avoid a seam and a blurry or jagged coast; they looked right in every screenshot, but no alternative was compared.
- **Windows as big pale boxes.** The first skyline had windows about 3 degrees wide and bright enough to clip to white after the tone map. Smaller, dimmer windows read as a city.
- **Pink asphalt.** Lit by a low sun the ambient sky colour tints everything pink; the `cobbled night` preset puts the sun below the horizon and lights the ground with a warm street-light fill.
- **Sun gap.** To keep the sun visible the buildings are lowered around its azimuth; this leaves a hard-edged opening in the skyline. It is only applied when the sun is visible (little rain).
- **Aliasing.** The ground texture repeats every 6 m; far away the repetition is hidden by the haze and the anisotropic mips. Ripple rings are the one detail that would shimmer, hence their fade.
- **Texture format.** The height is stored in a 16-bit float, which is plenty for a 5 cm range; an 8-bit height would show contour steps along the shoreline.

## When to use it

A wet street or courtyard in a hero, a game menu or a weather app, anywhere an angled ground plane with reflections will hold attention. It is a good fit for Tier A on desktop at modest sizes and a candidate for phones with a half-size canvas, but that is untested. The pieces are meant to be lifted out: `wetness.wgsl` for any ground you already render (darker, glossier, mirror in the hollows) and `ripples.wgsl` for story 10.2's standalone ripple rings. If you only need a still image, a photograph of a wet street is cheaper and better. If the water needs to react to touch, or to hold waves and splashes, use the heightfield ripple simulations (Epic 5), not these stateless rings. There is no real-time reflection of scene geometry here: the environment is procedural and infinitely far away, so a reflected object that is meant to be nearby will look wrong.

## References

- Sebastien Lagarde, [Water drop 3b - Physically based wet surfaces](https://seblagarde.wordpress.com/2013/04/14/water-drop-3b-physically-based-wet-surfaces/) (2013): darker albedo, lower roughness, puddles as flat Fresnel mirrors.
- Christophe Schlick, An Inexpensive BRDF Model for Physically-based Rendering (1994): the Fresnel approximation in `fresnel.wgsl`.
- Ripple rings from a hashed grid summed across the neighbouring cells is the common real-time scheme for rain on water; the wavelet and its analytic slope here are written from scratch.
- Mark Jarzynski and Marc Olano, "Hash Functions for GPU Rendering" (JCGT 2020): the integer hash behind every random choice.
- Steven Worley, A Cellular Texture Basis Function (1996): the Voronoi stones and slabs.

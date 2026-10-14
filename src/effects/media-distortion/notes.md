## How it works

The picture or video is a texture, and the water is one fragment pass over it. `src/gpu/media.ts` turns an `<img>`, `<canvas>`, `ImageBitmap` or `<video>` into an `rgba8unorm-srgb` texture with a mip chain, so sampling returns linear light, filtering and mips blend in linear space, and the shader's last step is `linear_to_srgb`. Row 0 is the top of the picture, so the `uv` of `fullscreen.wgsl` shows it upright; at strength 0 the output is bit-identical to the source (checked in Chrome by rendering the demo into an `rgba8unorm` target at the picture's own size and comparing every byte).

`media-distortion.wgsl` builds a surface normal from two layers of the baked ripple map (`createWaterNormalMap`, 3 smooth octaves, 512 px, baked once), turned and scaled apart and blended with the whiteout blend. No noise is evaluated per pixel. The pointer adds a lens bump that follows it, rings that run out from its movement, and a drop ring on every press. The normal becomes a uv offset with `refraction_offset_cheap`, in canvas space. The fit (`cover` or `contain`, the CSS `object-fit` rules, computed once per frame by `fitScale` and mirrored by `media_fit_scale` in `media.wgsl`) maps canvas uv to media uv, the offset is scaled the same way, and the picture is sampled with `refraction_safe_uv`: the part of the offset that points at a border fades to nothing before the border and the rest is mirror-folded, so the image never smears or wraps at its edges, and `contain` never pulls its bars into the picture. Dispersion is three taps with per-channel offsets (`refraction_sample_rgb`) or one tap when it is 0. The mip level comes from `media_lod`, derived from the picture-to-canvas size ratio instead of derivatives, so a 1080p video shown small does not alias. A highlight (`pow(n.h, 600)`) and a faint relief shading (slopes facing the light brighten) make it read as a surface. `compare` splits the canvas into original and water.

Interaction modes: `ambient` ignores the pointer, `ripple` adds the lens, rings and drops, `hover` keeps the water dead still until the pointer moves over it, agitates it in a blob around the pointer in proportion to its speed, and settles over a second or two after it stops or leaves. The pointer is read, never captured, so touch still scrolls the page.

Sources, none of them a file in the repository. `still` is a harbour painted onto a 2D canvas at startup (`sources.ts`): long straight lines (pier planks, posts, the horizon, a ruler along the frame), recognisable shapes, readable text and fine grain. `video` is an animated 2D canvas turned into a real `<video>` with `canvas.captureStream()`, so the genuine video path is exercised. **A file dropped on the canvas** (image or video) replaces either; it is read in the page through `createImageBitmap` or an object URL and goes nowhere. Choosing a different `source` brings the demo picture back. Without a DOM (the tests) a procedural test card stands in for both.

**On a real site** you pass your own element:

```ts
const media = await createMediaTexture(device, document.querySelector('img')!); // decoded for you
// or a video:
await whenVideoReady(video);
await video.play();
const media = await createMediaTexture(device, video);
// each frame, before drawing:
media.update(encoder); // uploads a frame only if a new one was presented, mips included
// bind media.view; rebuild the bind group when media.version changes (the video changed size)
```

Images must be same-origin or served with CORS (`crossOrigin = 'anonymous'`), or the copy throws a `SecurityError`; videos need `muted` and `playsinline` to autoplay.

**Why a video is copied, not imported.** WebGPU can read a video two ways: `copyExternalImageToTexture` into a normal texture, or `importExternalTexture`, a zero-copy `texture_external` that is valid only for the current task. The import saves one GPU copy of the frame. It costs what this effect needs: `texture_external` is read with `textureSampleBaseClampToEdge` only (no mip levels, so no blur or minification filtering, no wrap mode), the bind group must be rebuilt every frame, the shader needs a second code path for stills, it returns sRGB-encoded values to be decoded by hand, and none of the refraction helpers take it. Support is also thinner: it is in Chrome and Safari 26 but, as of the sources checked, not in Firefox. So the demo copies, and uploads only when `requestVideoFrameCallback` says a new frame was presented (24 or 30 fps video on a 60 to 120 Hz display is a new frame every second to fourth render). Reasoned, not measured head to head: the import would win for a full-screen 4K video under a trivial shader.

## Cost

Indicative, read from the HUD on an Apple Silicon laptop (Metal), 1702x932 (1.6 Mpx), DPR 2, with other agents' tabs sharing the GPU:

| Source                | `water over media` pass |
| --------------------- | ----------------------- |
| still, 3 taps         | about 1.0 to 1.3 ms     |
| 640x360 video, 3 taps | about 0.9 ms            |

Both are at or near the floor of about 1 ms that the HUD cannot resolve below, so read them as "too cheap to measure here", not as 1 ms. The video upload is not in the GPU figure: the frame copy is a queue operation and the 10 mip passes are untimed. CPU cost of an upload (copy plus recording the mips) was about 0.7 ms in a page loop, once per presented frame (77 uploads in 352 renders over three seconds for a 30 fps source). Memory: the 1280x720 still is 1280 x 720 x 4 x 1.33, about 4.9 MB, plus 1.4 MB for the ripple map; a 4096 square picture would be about 90 MB, which is why stills are downscaled to 4096 on the long side. Tier A on this machine. Nothing was measured on an integrated Intel or Mali GPU or on a phone, so the scorecard says `untested` for mobile.

What scales the cost: pixels, then taps (3 with dispersion, 1 without), then the video's upload size.

## Pitfalls

- **`contain` was wrong in the first draft**: taking the reciprocal of the cover scale stretches a portrait picture across a wide canvas instead of letterboxing it. The contain scale is the cover scale divided by its smaller component. A test now checks a 2:1 canvas around a 1:2 picture.
- **Fine noise looks like cracked glass.** The default five-octave ripple map, scaled to the picture, tore text apart. Three octaves, lowest period 3, and gentle layer weights read as water. Mip bias on the map is another way to the same calm.
- **Distortion in canvas space, sampling in media space.** The offset is multiplied by the fit scale and the safe-uv fold runs after the fit. Doing the fold in canvas space leaves the `contain` bars in the picture.
- **`loadTexture` read wide-gamut files in the wrong colour space** (`colorSpaceConversion: 'none'` for colour photos: a Display P3 (0.8, 0.3, 0.3) came out 204,77,77 instead of 221,64,71). Fixed in `texture.ts`; `media.ts` decodes with the default conversion.
- **`copyExternalImageToTexture` with an `<img>` is Chrome-only** in the sources checked, so `createMediaTexture` decodes every image element through `createImageBitmap` first; that is also where oversized pictures are downscaled.
- **A submit per frame.** `updateTextureFromSource` used to submit its own command buffer for the mips; it now takes the frame's encoder.
- **Background tabs.** Neither rendering nor `requestVideoFrameCallback` runs in a hidden tab, so a dropped file loads when the tab next draws.
- **Drop target.** Only the canvas itself receives the drop. In the hero layout the page content sits over it, so a real page needs its own drop zone (or listeners on the container) if it wants one.
- **Highlights.** A tight exponent on a busy normal makes sparkly confetti; the demo uses 600 and a small gain.

## When to use it

A hero image that ripples gently, a product shot that wobbles on hover, a looping background video under water. It is Tier A: one pass, three texture taps, no per-pixel noise. A modest `strength` (0.2 to 0.6) is what reads as water; above 1 it reads as a funhouse mirror. Keep the picture's important content away from where the pointer will sit, and keep text out of the distorted layer, because the text will wobble too. Where WebGPU is missing the page should just show the `<img>` or `<video>` underneath; an SVG `feDisplacementMap` filter is a rough CPU-bound substitute and gets slow on large or animated elements. To refract the live page (HTML) instead of one image or video, see story 9.1 in plan.md: that is an open problem, and this effect covers only media the page can hand to the GPU.

## References

- [Chrome: What's new in WebGPU (118)](https://developer.chrome.com/blog/new-in-webgpu-118): `HTMLImageElement` and `ImageData` as sources of `copyExternalImageToTexture`.
- [MDN: GPUQueue.copyExternalImageToTexture()](https://developer.mozilla.org/en-US/docs/Web/API/GPUQueue/copyExternalImageToTexture) and [GPUDevice.importExternalTexture()](https://developer.mozilla.org/en-US/docs/Web/API/GPUDevice/importExternalTexture).
- [MDN: HTMLVideoElement.requestVideoFrameCallback()](https://developer.mozilla.org/en-US/docs/Web/API/HTMLVideoElement/requestVideoFrameCallback).
- [toji.dev: WebGPU img, canvas and video textures](https://toji.dev/webgpu-best-practices/img-textures.html).
- `src/wgsl/refraction.wgsl` and `docs/support-matrix.md` in this repository.

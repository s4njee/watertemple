# Backdrop capture spike — Epic 9.1

The liquid UI demos **re-render their own content into a texture**. They do not sample pixels from the surrounding DOM. This decision makes the glass, image hover, trails, drops and section transitions reproducible in the browser and headless WebGPU tests. A host page can supply a decoded image, video frame or canvas instead. The gallery's image-hover demo accepts dropped local images.

## Four routes

| Route                                | What reaches the GPU                                                                                 | Updates and cost                                                                                                | Limits                                                                                                                                                                                                   | Decision                                                                                                 |
| ------------------------------------ | ---------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------- |
| Direct image / video / canvas upload | A decoded external image source through `copyExternalImageToTexture`                                 | Once for still content; every new video frame for animation. Upload bandwidth and mip regeneration can dominate | The source must be decoded and origin-clean. DOM elements are not accepted. Video sizing and colour conversion need care                                                                                 | Stable path for owned visual content; existing `gpu/texture.ts` and `gpu/media.ts`, plus `ui/capture.ts` |
| HTML-in-Canvas                       | Browser-generated snapshots of drawable canvas descendants, uploaded with experimental GPUQueue APIs | Paint-driven snapshot + upload; hit-testing geometry must stay in sync                                          | Experimental API, changing method/attribute names, origin trial or flags, drawable ancestry and snapshot preconditions. No ordinary-page screenshot capability                                           | Capability probe only in this release; do not make it a production requirement                           |
| SVG `foreignObject`                  | Serialized, self-contained XHTML in an SVG, decoded as an image/bitmap                               | Serialisation, style/font/image embedding, asynchronous decode and upload on each refresh                       | Computed styles are not cloned automatically; external resources/scripts are restricted in SVG image contexts. Decode success can still leave blank HTML. Engine-specific security and painting behavior | Tested as a small prototype; unsuitable as the default live backdrop                                     |
| Re-render content inside canvas      | A designed illustration/layout rendered to a linear GPU texture                                      | One bake for a static section, one additional texture for transitions; re-bake when owned content changes       | Duplicates layout/content authoring. Does not reproduce arbitrary HTML/CSS, forms, video or selection                                                                                                    | Default for these research scenes; explicit semantic control labels remain in DOM                        |

The WebGPU specification lists the permitted [external image sources and copy rules](https://www.w3.org/TR/webgpu/#dom-gpuqueue-copyexternalimagetotexture). SVG's [image processing modes](https://www.w3.org/TR/SVG/conform.html#secure-static-mode) restrict scripting and external resources. A `foreignObject` snapshot is a different rendering path from a live DOM subtree.

## Local spike and evidence

Open [capture-spike.html](../capture-spike.html), or serve the built `capture-spike.html`, and click **Run capture probes**. `src/demos/capture-spike/main.ts` exercises an actual 480 × 240 canvas upload and a self-contained SVG bitmap decode, and probes GPUQueue's current and legacy HTML capture methods. It does not request a capture permission, turn on flags, enable an origin trial, or inspect other windows.

On the in-app browser on the local M1 Max:

- Canvas → sRGB texture → linear sample → sRGB display **passed**, with the text and circle visibly present. The observed 6.0 ms is cold wall time including first pipeline use and queue completion, not a steady-state GPU frame budget.
- `createImageBitmap` of this SVG `foreignObject` source **failed to decode**. This establishes the failure of the tested bitmap path on this browser, not a universal SVG incompatibility. An `<img>` plus 2D canvas path could behave differently and is not claimed as tested.
- `drawElementImageToTexture` and `copyElementImageToTexture` were **unavailable**. No experimental DOM upload was executed or timed.

Screenshot: [capture-spike.jpg](../assets/epic-9/capture-spike.jpg). These are local functional observations, not cross-browser support claims.

## Experimental API state

The current [Chrome HTML-in-Canvas update](https://developer.chrome.com/blog/html-in-canvas-ot-changes) describes `drawElementImageToTexture` replacing `copyElementImageToTexture`, `<canvas content="drawable">` replacing `layoutsubtree`, drawable descendants, and explicit `updateElementGeometry()` for WebGPU hit-testing/accessibility. Older versions use other signatures. The [WICG proposal](https://wicg.github.io/html-in-canvas/) requires a recorded paint snapshot of an eligible drawable element; method presence alone is insufficient. A production integration should feature-test, handle exceptions and keep a stable source/fallback route.

## Integration consequences

The generated coastal image is an illustration, not a photo or a screenshot of the website. Backdrop textures are **linear `rgba16float`**; uploaded image textures are **`rgba8unorm-srgb`**, decoded to linear by sampling. The text atlas is linear distance data. Only the final display pass encodes to sRGB. Section 0 occupies about 5 MiB at 1024 × 640; only the transition study allocates section 1.

For a real site, keep links, headings, input semantics and progress values in HTML, and treat the shader as decoration. The demo button supports pointer, Enter and Space; the fill component exposes its percentage through a progressbar role. Typography currently renders a fixed Latin display word with fixed advances; it is not a shaping or selectable-text engine. Native content should remain available with the fallback poster and reduced-motion/offscreen lifecycle provided by the shared host.

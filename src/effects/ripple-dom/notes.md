## How it works

A WebGPU canvas cannot see the DOM's pixels, but it can know where elements are, and for a wall that is enough. The page marks the elements that should be solid with an attribute, `data-wt-obstacle`. `src/gpu/dom-rects.ts` finds them, measures their boxes, converts each to the canvas's uv space with the corner radius read from computed style, and hands the list to the heightfield solver's obstacle mask (`solver.obstacles.setRects`). The solver then treats those cells as walls (height held at zero, neighbours see a reflecting boundary), so a ring spreading from the pointer bends around a card, bounces off a headline and laps at the edge of the content. Nothing on the page is changed, drawn into or copied: the elements are only measured.

**Measuring.** `trackDomRects(canvas)` is read once per frame from `update`, before the effect writes anything: one `getBoundingClientRect` for the canvas and one per marked element, no writes, so the reads never force a second layout. Observers were considered and rejected on correctness, not only cost. `ResizeObserver` reports a change of size, not of position, so it stays silent while a card scrolls over a sticky canvas (the `backdrop` context does exactly that), while a transform animates, or while a sibling pushes the layout down. `IntersectionObserver` reports coarse overlap thresholds, not boxes. Scroll listeners would have to sit on every scrollable ancestor (the window never hears an inner scroller) and still miss CSS transitions and JS-driven layout. A `MutationObserver` on the search root is used for the one thing it is good at: noticing that a marked element was added, removed or re-marked, so the element list is rebuilt on the next frame instead of waiting for the periodic requery (every 60 frames). Computed style, the expensive read (corner radius, `display`, `visibility`, `opacity`), is cached per element and refreshed when the element changes size and every 20 frames.

**Where it looks.** Not the whole document, since several effects can share a page. The root is `options.root`, else the nearest ancestor of the canvas with `data-wt-obstacle-root`, else the nearest ancestor (stopping before `<body>`) that contains a marked element, else the canvas's parent. Only elements that overlap the canvas become rects.

**Geometry** (pure, in `boxToObstacle` and friends, unit-tested): the box is mapped to canvas uv; the corner radius is the largest of the four computed radii, limited to half the shorter side and expressed in image heights so corners stay round; the rect is clipped to the canvas widened by the radius, so a card that runs off the edge keeps its real rounded corner outside the canvas instead of getting a fake one at the edge; hidden, transparent, zero-size and non-overlapping elements are skipped. `rectsChanged` compares with a threshold of a quarter of a grid cell, so the mask is re-rasterised (about 0.24 ms for a canvas-high plate at 256 cells, 0.85 ms at 512, on the development machine) only when something moved enough to change it.

**Moving elements.** When a card scrolls through the pool or is dragged, its cells turn solid in front and are freed behind; freed cells start flat and the neighbouring waves flow in, which is stable (the solver clamps heights anyway). On top of that the moving edge pushes water: for each edge whose normal speed is above a small threshold, `edgeWake` queues a few drops just outside it, raised where the edge advances and dented where it retreats, in proportion to the distance the edge travelled this frame (so it does not depend on the frame rate) and saturating with `tanh` so a flick is a big wave and not a huge one. Edges on the canvas boundary are not real edges and make nothing, and a jump of more than 0.3 uv in one frame (a layout change at a breakpoint) is treated as a teleport and not as motion. The `wake` parameter scales it; 0 turns it off.

**Shading** (`water.wgsl`, one pass). The same refraction, glint and relief as `ripples`, then two things for the page. The physics mask is as sharp as the grid (about 6 CSS pixels per cell at 256 cells over 1500 px), so the shading pass does not use it for the picture: it draws the elements from their exact rounded rectangles (a uniform array of up to 24 boxes, `sd_rounded_box_2d` from `sdf.wgsl`), which lines the cut-out up with the real element to the pixel at any size. Under an element the output is transparent (premultiplied alpha 0), so the page's own background shows through and the element looks exactly as it would without a canvas; nothing is drawn that could fight its colour. Outside it, the `edge` parameter adds a soft darkening that falls off over about 0.014 image heights and a thin meniscus line a few pixels from the wall, brighter where a crest is arriving. The `mask` view overlays the solver's mask (red) and the exact outlines (green) on the water, which is how to check the alignment on a real page; the mask is the one place the mask texture is read.

**Pointer.** Hover wake and press-drag as in `ripples`, read from the canvas through `trackPointers`. The mock page content is `inert` and sits above the canvas, so events fall through to it; a real page must do the same (`pointer-events: none` on the overlay, or listen on a parent).

**Marking up a real page.**

```html
<section class="hero" data-wt-obstacle-root>
  <!-- optional: scopes the search -->
  <canvas id="water"></canvas>
  <nav data-wt-obstacle>...</nav>
  <div class="plate" data-wt-obstacle>...</div>
  <!-- border-radius is read from CSS -->
  <h1 data-wt-obstacle style="width: fit-content">...</h1>
  <!-- the box, not the glyphs -->
</section>
```

`data-wt-obstacle="off"` opts an element out without removing the marker. In the bare Canvas layout there is no page around the canvas, so the effect inserts three real elements of its own next to it (a card, a pill and a draggable card, `aria-hidden` apart from the draggable one, which is a labelled group that also moves with the arrow keys) and removes them on destroy or when the page has obstacles of its own (`demo`: auto, always, never).

**What it cannot do.** Rectangles and rounded rectangles only: no circles, no glyph shapes (text is its box), a rotated element is its bounding box, an element clipped by an `overflow` ancestor still counts at its full size. It sees no pixels, so the water cannot refract the page's content; it refracts its own backdrop texture. The experimental HTML-in-Canvas API (docs/support-matrix.md section 5.2, a Chromium origin trial that is not shipped anywhere else) would add exactly that: with it the page's elements could be copied into a texture each frame and refracted, and their alpha could make the mask. Until it ships and in every other browser, this approach, which needs only `getBoundingClientRect`, works everywhere.

## Cost

Indicative readings from the HUD, Apple Silicon development laptop (Chromium, Metal, the GPU shared with another tab), 1702 x 724 drawing buffer (1.2 Mpx, DPR 2), 256-cell grid, hero layout: GPU about 0.8 ms per frame (shade about 0.5 to 0.65, splat, step and resolve about 0.3 each, overlapping), CPU about 0.3 ms per frame, 5.4 MB of textures. Rain at 30 drops per second with a 512 grid, an undamped reflecting pool and a card being dragged around read 1.2 ms GPU at the same size. That is inside Tier A on this machine; not measured: any integrated GPU other than Apple's, any phone, any browser other than Chromium. The scorecard leaves `gpuMs` empty for that reason.

The page measurement, timed in the page by calling `update` 2000 times in the `cards` layout (6 marked elements, 1 overlapping the canvas): about 11 microseconds per call with a clean layout. With 66 marked elements (26 overlapping): about 115 microseconds. With a style change written to the page before every call, which forces a layout each time and is not what a frame loop does: about 370 microseconds. A moving element adds the mask rasterisation (above) and an upload of one byte per cell (29 KB at 256 x 113). A static page costs the measurement and nothing else: the mask is not touched until a rect changes.

## Pitfalls

- **The mask is a cell grid.** Walls are rasterised at cell centres, so the physics edge is within half a cell (about 3 CSS pixels at 256 cells over 1500 px) of the element. The picture is exact because it is drawn from the rectangles, so the two can disagree by those few pixels, which is invisible except in the `mask` view. A finer grid narrows it at the cost of more steps per second.
- **Overlays must let the pointer through.** If the content above the canvas takes the pointer, the water never hears it. The mock pages use `inert`; a real page needs `pointer-events: none` on the overlay or a listener on a common parent. Dragging the demo card is a different case: the card takes the pointer by design, and its moving edges stir the water instead.
- **The wall margin is physics only.** `outset` moves the solver's wall away from (or into) the element. The water is still cut out at the element's own box, so a margin shows as a collar of calm backdrop around the element, and a negative one lets waves lap under it.
- **Re-gridding clears the water but not the walls.** `solver.resize` re-creates the grid and the surface goes flat; the rects given to `setRects` are remembered in uv and rasterised again, so the walls survive. Raw bytes from `obstacles.set` would not, which is why this effect uses rects.
- **A layout change is not motion.** A breakpoint can move a card by half the canvas in one frame. Wake treats any edge that jumps by more than 0.3 uv as a teleport; without that, a window resize splashed the whole pool.
- **Wake strength is tuned by eye.** The first value (3 times what ships) turned a slowly dragged card into a storm; the shipped gain gives a visible wake at 250 px per second and a large one at speeds that a scroll flick reaches.
- **Cheap per frame, not free on a cold page.** The first update, and the one after each requery or a mutation of the marked elements, runs `querySelectorAll` and `getComputedStyle` for every marked element; keep marked elements to the tens, not the thousands.
- **Hidden by an ancestor is not detected.** `display: none` on an ancestor gives a zero box, which is skipped; `visibility: hidden` and `opacity` are read from the element's own style, so an ancestor's opacity is invisible to it.
- **Synthetic input and screenshots.** Pointer events dispatched from script have the same timestamp unless spaced out, and a screenshot taken a fraction of a second after a tap shows a young ripple; waves need half a second to read.

## When to use it

A hero, a section or an article background whose water should visibly respect the content in front of it, and respond when that content moves: Tier A as measured here. It costs a few microseconds of measurement per marked element per frame and a small amount of mask upload when something moves, on top of the plain ripple surface. Use `ripples` when nothing sits over the water. If the layout is static and you only want a few shapes, build the mask yourself with `solver.obstacles.setRects` and skip the tracker. If the water has to refract or distort the page's own text and images, this approach cannot do it: that needs the page's pixels in a texture (HTML-in-Canvas where it exists, or a pre-rendered image of the content) and is the subject of the liquid-UI epic. Where WebGPU is missing, show the backdrop image with the page on top; nothing in the layout depended on the canvas.

## References

- `src/gpu/dom-rects.ts`: the tracker and the pure geometry; `src/gpu/sim/obstacles.ts`: the mask and its rasteriser; `src/wgsl/sdf.wgsl`: `sd_rounded_box_2d`.
- MDN, [`Element.getBoundingClientRect()`](https://developer.mozilla.org/en-US/docs/Web/API/Element/getBoundingClientRect), and [`ResizeObserver`](https://developer.mozilla.org/en-US/docs/Web/API/ResizeObserver) (content-box size changes only), [`IntersectionObserver`](https://developer.mozilla.org/en-US/docs/Web/API/Intersection_Observer_API), [`MutationObserver`](https://developer.mozilla.org/en-US/docs/Web/API/MutationObserver).
- Paul Irish, [What forces layout/reflow](https://gist.github.com/paulirish/5d52fb081b3570c81e3a): why reads are batched before writes.
- WICG, [HTML-in-Canvas](https://github.com/WICG/html-in-canvas), and docs/support-matrix.md section 5.2 for its status and the two name sets.
- Inigo Quilez, [2D distance functions](https://iquilezles.org/articles/distfunctions2d/): the rounded box.
- Evan Wallace, [WebGL Water](https://madebyevan.com/webgl-water/): the height-velocity surface and moving-footprint displacement the wake follows in spirit.

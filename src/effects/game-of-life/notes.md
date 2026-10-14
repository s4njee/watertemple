# Game of Life

## How it works

A standalone interactive page lives at `game-of-life.html`. It uses the same GPU host as the other field studies and has its own toolbar, without entering the orbit Variation selector.

The simulation uses Conway’s B3/S23 rules: births with three neighbours; survival with two or three. All cells update synchronously into a separate `r32uint` texture. Viewport edges wrap. An integer packs live state, age and a nine-generation afterimage; only the live bit contributes to neighbour counts. Conway’s rules govern each generation. Every 18 seconds of active playback, three to eight small, random seed patches add life to the existing world without clearing it. This injection timer stops while paused, resets with a new world, and can be disabled by setting `reseedInterval` to zero in the gallery.

The default Random world seed uses reproducible 23% occupancy. Glider garden combines Gosper glider guns with travelling gliders; oscillators contains blinkers and toads. Clicking exposed background stamps a glider without advancing the generation. Pause freezes both simulation and rendering; Step advances exactly one generation. Reduced motion starts paused, with manual Play and Step available.

All live cells share the ink orbit’s six-second, eased red → green → blue cycle in linear RGB. Crisp cores sit inside two soft glow radii; a shared hue-preserving brightness ceiling keeps dense clusters saturated. The standalone page enables Auto HDR by default, requesting an extended-range `rgba16float` canvas on supported displays and browsers. The default Light multiplier is 2×, applied to neon emission only; unsupported devices use brighter saturated SDR output. A toolbar output selector and HDR Light slider allow adjustment. Afterimages use the same hue at lower intensity, and Pause holds the color cycle as well as the simulation.

The canvas is fixed behind ordinary HTML. Opaque text and card surfaces keep the page readable while the automaton evolves underneath, and moving the card reveals the simulation. Resizing the grid, changing cell size or changing seed resets the world. GPU tests verify stable blocks, period-two blinkers, glider translation, edge wrapping, isolated-cell death, stamp behavior and separation of visual state from life state. Phone performance is unmeasured.

## Cost

One integer-texture compute update per generation plus a fullscreen neon display pass. The grid follows the chosen CSS cell size. GPU timings and phone budgets for this demo remain unmeasured.

## Pitfalls

Resizing or changing the seed/cell size resets the world. Random injection is an explicit decorative extension; Conway’s B3/S23 rules still govern each generation between injections. HDR requires support from the display and browser.

## When to use it

A playful cellular background behind real HTML content. Pause, Step and reduced-motion controls keep it inspectable.

## References

- Conway’s B3/S23 automaton, implemented directly in `life.wgsl`; seed shapes and tests are in `seeds.ts` and `tests/life.test.ts`.

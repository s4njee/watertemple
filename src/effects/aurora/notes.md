## How it works

A separate variation at `aurora.html`, linked from the ink orbit demo. Both pages have a toolbar dropdown labelled Variation that switches between Ink orbit and Aurora ribbons, opening the corresponding page with its own defaults. It shares the measured DOM obstacles, deterministic head motion, and stable fluid solver with ink orbit through `createOrbitEffect`. Its metadata and shader remain separate, so the original demo keeps its existing rendering and settings.

Seven slower heads lay narrow emerald, cyan and violet trails. Less wander, restlessness, stirring and vorticity preserve longer, calmer ribbons; each head drifts through its own palette phase while deposited dye retains its colour.

The display shader shades fine folds using advected density and a slowly changing phase. Five samples beneath each pixel form a faint upward curtain, and the density gradient adds a small luminous rim. Hue-preserving emission compression and a midnight background leave dark space between ribbons. This is a stylised fluid visualization, not a physical model of atmospheric aurora.

The standalone page retains the existing pause/reset, HDR brightness, palette, reverse, forces, obstacle outlines, draggable card, pointer stirring and reduced-motion behavior. Reduced motion starts paused and freezes the shader's animated fold phase. The shared clock advances only while the effect is rendering.

## Cost

One shading pass over the shared simulation, with ten dye samples per output pixel including the four gradient samples. GPU lifecycle validation is covered by the registered-effects test; mobile performance is unmeasured.

## Pitfalls

This is a stylised visualization rather than a physical model of aurora. Bright overlap is compressed to preserve hue, and mobile timings remain unmeasured.

## When to use it

A quiet luminous curtain behind a hero or editorial page. Use the original Ink orbit for restless movement, or this preset for slower narrow trails.

## References

- Shared renderer, fluid solver and obstacle tracking: `src/effects/ink-orbit/notes.md`.

## How it works

Glowing ink blots, tapering luminous lines and faint afterimages.

A separate orbit material selected through the Variation toolbar or `neon-calligraphy.html`. Motion and palettes are in `src/effects/orbit-variants/catalog.ts`; the rendering branch is in `display.wgsl` in that directory. The same obstacle-aware fluid solver, pointer routing and draggable card are shared with ink orbit. See that directory's `notes.md` for rendering costs and limitations.

## Cost

This material adds a fullscreen shading pass to the shared fluid simulation. Neon calligraphy additionally renders bounded stroke history. Standalone pages expose the existing quality controls. GPU timings and phone budgets for these material variations remain unmeasured; see the shared orbit notes for bounds.

## Pitfalls

These are screen-space material approximations. Dichroic glass and chrome use authored analytic lighting rather than capturing arbitrary live page pixels. Reduced motion freezes autonomous animation and the standalone page starts paused.

## When to use it

Use this variation as an atmospheric hero or page backdrop. Keep readable content on solid surfaces and use the standalone pause control.

## References

- Shared implementation and limits: `src/effects/orbit-variants/notes.md` and `src/effects/ink-orbit/notes.md`.

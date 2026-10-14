# Neon mycelium

## How it works

A standalone Physarum garden at `neon-mycelium.html`, linked from Neon script. It has its own Growth, Palette, Pause/Grow, Regrow, Show boundaries and HDR controls. It is deliberately absent from the orbit Variation dropdown.

The simulation adapts the three-sensor agents, fixed-point atomic deposits and ping-pong diffusion/decay from the sibling HoloGPU implementation, `packages/effects/src/physarum/index.ts`. This is actual trail-following Physarum behavior rather than a fluid shader or a drawn network. Every step, agents compare the trails ahead and to either side, turn toward the stronger path, deposit colored trails, and diffuse/fade the accumulated field.

Page adaptations:

- Up to 24 visible DOM obstacles are measured with the shared page tracker. Their rounded rectangles are rasterized to a signed-distance texture when geometry changes. Agents avoid occupied cells, project out of moved elements, and do not deposit there. Diffusion ignores solid neighbors, and covered trail cells are cleared.
- A quarter of initial agents start near page outlines; the remainder seed open space. A weak optional nutrient band encourages roots near content without forcing a rigid orbit.
- The viewport has periodic agent and trail boundaries rather than reflecting walls, with a soft fade at the visible window edges. It therefore does not accumulate a closed perimeter outline. Each agent also has a small chance to restart at a fresh location, keeping established paths exploratory.
- Trail decay, thinner deposits and less diffusion prevent mature networks from becoming broad static bands. Pressing on open ground seeds fresh local trail concentration. Moving the card, scrolling or rearranging notes rebuilds the page boundaries.
- Colored trail deposits are retained, with neutral suppression and hue-preserving exposure compression to keep overlaps saturated. Palette changes regrow with the selected colors. HDR is optional, with SDR and a usable static-page fallback.

The initial 48 simulation steps show a young colony immediately, including a static composition when reduced motion is enabled. Growth runs at an accumulated fixed step rate with a bounded catch-up count, freezes while paused, and is disabled under reduced motion. Resolution/count changes rebuild the colony; small viewport resizing preserves it.

## Cost

Default budget: 8,192 agents and a 512-cell short-side grid. Two RGBA16F trail textures, one RGBA16F distance texture, 16-byte atomic deposit cells and 16-byte agents are bounded by the selected resolution/count. Timings and phone performance remain unmeasured. Source typechecks, scoped lint, Vite build, a headless effect lifecycle/pixel test and a GPU integration test cover growth, movement of page walls, exclusion of ink/agents from solid content, continued trail change, and avoidance of a perimeter-dominated network.

## Pitfalls

Physarum trails can settle into persistent networks; periodic relocation and decay are the implemented controls. This is a decorative trail-following simulation, not biological growth or a physical fluid. Resolution/count changes recreate the colony.

## When to use it

A separate interactive landing-page garden whose network follows measured content boundaries. Use the existing pause control or reduced-motion still for a quiet reading surface.

## References

- Adapted local algorithm: sibling HoloGPU `packages/effects/src/physarum/index.ts`.

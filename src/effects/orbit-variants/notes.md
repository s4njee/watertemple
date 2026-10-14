# Orbit material studies

## How it works

Six separate material variations of Ink orbit. The Variation menu on all eight standalone pages navigates between them; original Ink orbit and Aurora remain available. Each material is also registered in the main gallery, with the same parameter controls and DOM-aware fluid simulation.

| Variation               | Material and behavior                                                                                                                                                                                       |
| ----------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Neon calligraphy        | Larger irregular emissive ink heads, 1.7-second vector stroke history with tapering widths, and a dim fluid afterimage. Strokes use instanced quads with alpha blending to retain saturated overlap colors. |
| Dichroic glass          | Density-derived normals, angular teal/magenta thin-film color, sharp highlights and thickness at fluid crossings. Refracts a procedural illuminated background; it does not capture or distort DOM text.    |
| Bioluminescent currents | Cyan and green heads respond to angular velocity and pointer proximity. Excitation decays exponentially over time, and also brightens the dye deposited in their wakes.                                     |
| Liquid chrome           | Analytic moving studio reflection, narrow strip highlights, dark metallic body and small spectral edge flashes. Alternative palettes tint the alloy.                                                        |
| Silk embers             | Warm filaments cool toward crimson as density fades. Up to 32 actual detached sparks coast, rise, dim and expire independently of the heads.                                                                |
| Eclipse halos           | Rounded-rectangle rings follow the measured page outlines. Moving bright arcs bow away from their rings, then return. Dim fluid carries escaped head trails.                                                |

`catalog.ts` is the source for palettes, material IDs, page links and motion defaults. `material.ts` supplies those defaults to registered modules. `display.wgsl` supplies six shading branches. The shared Ink orbit factory supplies retained dye, obstacle tracking, heads, live pointer response and bounded sparks; `strokes.ts` adds precise geometry only for Neon calligraphy.

Reduced motion starts the standalone pages paused on a seeded composition. The gallery also freezes autonomous material animation and disables spark spawning when reduced motion is enabled. Existing pause, reset, force, reverse, palette, output and card keyboard controls continue to work. With WebGPU unavailable, the ordinary page remains usable over a static accent gradient.

## Cost

The fullscreen material pass loops over at most 48 heads, 32 sparks and 24 measured outlines; only the relevant material executes its head/spark loops. Neon stores at most 2048 stroke segments and draws them with six vertices per instance. These add to the existing fluid solver's cost. GPU timing and phone budgets have not been measured; metadata therefore leaves mobile support untested. Glass and chrome use an analytic surface approximation rather than a volumetric or physically traced material.

Validation: source typecheck, scoped lint, production Vite build, existing head dynamics tests, turn/pointer pulse and spark lifecycle tests, catalog/default consistency, and headless WebGPU lifecycle/pixel checks for all eight orbit effects. Browser verification covers each material and the shared toolbar.

## Pitfalls

Shared fluid costs apply in addition to the material pass. Analytic glass/chrome optics do not refract arbitrary DOM pixels, and saturation compression trades peak luminance for stable hue.

## When to use it

Select a material through the standalone Variation menu or the gallery. Preserve real HTML content and use pause/reduced-motion behavior for continuous backgrounds.

## References

- Base solver and rendering references: `src/effects/ink-orbit/notes.md`.

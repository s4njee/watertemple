## How it works

A deterministic six-armed crystal field with diagonal side branches defines local ice shape and growth delays. Persistent coverage approaches that field as the freeze front advances. A freeze/hold/thaw cycle turns coverage loss into the memory texture’s water channel; a downward semi-Lagrangian sample transports it. The display derives icy refraction from coverage gradients and forms small bead lenses from accumulated meltwater. Freeze, winter-cycle and explicit thaw presets expose the stages; the thaw preset starts with a developed pane.

## Cost

Two steady render passes, an aspect-matched, at-most-1,024×256 ping-pong field and the same cached scene backdrop as Fogged Glass. Growth, freeze/thaw and explicit melt configurations are measured in [the cost study](../../docs/weather-cost-study.md). Crystal pattern evaluation affects the small history pass rather than the full display resolution.

Local Apple M1 Max (Dawn Metal), 1920×1080: conservative complete-frame cost **0.590 ms**, using the larger of median span and workload slope over the measured configurations. Owned effect code is **9.85 kB gzipped**. Shared runtime and WGSL imports bring the transitive load to about 104 kB; standalone packaging remains Epic 12.1. Phones and other GPU classes remain unmeasured.

## Pitfalls

This is an authored dendrite shape plus persistent coverage and approximate runoff, not diffusion-limited crystal growth or a thermal energy solver. Meltwater is advected with numerical diffusion and evaporates; bead lenses are a visual representation of that channel, not mass-conserving discrete water drops. Wiping removes ice directly. The automatic cycle repeats every 22 animation seconds.

## When to use it

Tier B winter hero windows and controlled freeze-to-water transitions. Use Slow Frost for gradual growth, Winter Cycle for ambient thawing, or Thaw the Pane for a staged melt.

## References

- [Ken Libbrecht, Snow Crystal Branching](https://www.snowcrystals.com/branching/branching.html) — the morphology reference; no physical crystal-growth solver is claimed.

- [WebGPU specification: render passes and premultiplied compositing](https://www.w3.org/TR/webgpu/).

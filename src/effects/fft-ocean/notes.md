## How it works

Deterministic Gaussian seeds sample a wind-directed Phillips spectrum on the CPU. Three periodic domains (192, 48 and 12 meters) occupy disjoint frequency bands. The GPU evolves each complex mode with deep-water dispersion and its conjugate partner. A radix-two, two-dimensional inverse FFT uses bit-reversed input, row butterflies, then column butterflies. Height and both horizontal displacements are packed into two complex channel pairs. Stage uniforms are immutable, so consecutive dispatches cannot accidentally read the last stage's settings.

A packing pass normalizes by N² and writes three-layer `rgba16float` displacement and derivative maps. Periodic central differences produce slope normals and the horizontal deformation Jacobian. Folding injects whitecaps below the chosen Jacobian threshold. Ping-pong derivative maps preserve foam with exponential lifetime decay. The final vertex stage combines all three cascades on either surface mesh. **Foam** view, folding threshold and lifetime expose the history independently of shading.

Tests compare all displacement channels against a direct complex DFT and verify real-valued output and foam accumulation/decay.

## Cost

Choose 64, 128 or 256 samples per cascade. The default runs 12 butterfly passes plus spectrum evolution and map packing; each additional size doubling adds two butterfly passes and quadruples field storage. At 256, seeds, ping-pong fields and three map allocations use about 19.5 MiB, before scene depth and geometry. The cost study measures all three sizes and both meshes at 1080p. Only the M1 Max was measured; Tier C and mobile claims do not extend to other hardware.

## Pitfalls

Domains repeat, even though independent bands reduce obvious tiling. This is a surface synthesis model, without breaking topology, conservation, shore coupling or object wakes. Normals use finite differences rather than spectral derivative transforms. Folding foam is computed per cascade and combined by maximum; it approximates the complete summed-displacement Jacobian. History stays in periodic surface coordinates, without current advection. Changing FFT size resets history and changes band boundaries. Extreme chop can invert triangles. Derivative maps have one mip level, so distant detail can alias.

## When to use it

Use a dedicated ocean showpiece or a study of GPU FFTs and folding foam. Start at 64 samples and increase only when the closer viewpoint needs more detail. Use Gerstner waves for simpler art direction and a smaller simulation cost.

## References

- Jerry Tessendorf, [Simulating Ocean Water](https://evasion.inrialpes.fr/Membres/Fabrice.Neyret/NaturalScenes/fluids/water/waves/fluids-nuages/waves/Jonathan/articlesCG/simulating-ocean-water-01.pdf), course notes, 1999–2004.

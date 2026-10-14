export interface OceanMeshData {
  vertices: Float32Array;
  indices: Uint32Array;
  triangles: number;
}
/** Screen lattice for projected grid; ring lattice for five nested, stitched clipmap levels. */
export function oceanMesh(strategy: 'projected' | 'clipmap', cells = 96): OceanMeshData {
  if (!Number.isInteger(cells) || cells < 8 || cells % 4 !== 0)
    throw new Error('Ocean cells must be a multiple of four, at least eight');
  const vertices: number[] = [],
    indices: number[] = [];
  const levels = strategy === 'projected' ? 1 : 5;
  for (let level = 0; level < levels; level++) {
    const offset = vertices.length / 4,
      half = cells / 2,
      step = 0.5 * 2 ** level;
    for (let y = 0; y <= cells; y++)
      for (let x = 0; x <= cells; x++) {
        let px = x - half,
          pz = y - half;
        // Collapse odd perimeter vertices onto the next coarser lattice, eliminating T junctions.
        if (strategy === 'clipmap' && level < levels - 1) {
          if (x === 0 || x === cells) pz = Math.round(pz / 2) * 2;
          if (y === 0 || y === cells) px = Math.round(px / 2) * 2;
        }
        vertices.push(
          strategy === 'projected' ? (x / cells) * 2 - 1 : px * step,
          strategy === 'projected' ? y / cells : pz * step,
          level,
          step,
        );
      }
    for (let y = 0; y < cells; y++)
      for (let x = 0; x < cells; x++) {
        if (
          strategy === 'clipmap' &&
          level > 0 &&
          x >= cells / 4 &&
          x < (3 * cells) / 4 &&
          y >= cells / 4 &&
          y < (3 * cells) / 4
        )
          continue;
        const a = offset + y * (cells + 1) + x,
          b = a + 1,
          c = a + cells + 1,
          d = c + 1;
        if (strategy === 'projected') indices.push(a, b, c, b, d, c);
        else indices.push(a, c, b, b, c, d);
      }
  }
  return {
    vertices: new Float32Array(vertices),
    indices: new Uint32Array(indices),
    triangles: indices.length / 3,
  };
}

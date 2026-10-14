// The two per-particle kernels of the counting sort that builds the neighbour grid (the prefix sum between
// them is scan.wgsl, the reorder after them is ParticleSystem.permute). See neighbor-grid.ts for the whole
// algorithm and its costs.
//
// Every kernel binds its per-particle arrays with exactly `count` elements, so arrayLength is the live
// particle count; nothing here needs a count uniform.

#include "neighbors.wgsl"

@group(0) @binding(0) var<uniform> grid: NeighborGrid;
@group(0) @binding(1) var<storage, read> positions: array<vec4f>;
// cells + 1 counters. Cleared before this kernel; scan.wgsl turns them into start offsets afterwards.
@group(0) @binding(2) var<storage, read_write> counts: array<atomic<u32>>;
// Per particle: x the cell, y its rank among the particles of that cell (the counter's value before the add).
@group(0) @binding(3) var<storage, read_write> cell_rank: array<vec2u>;

// Stage 1 and 2 fused: find the particle's cell and take a ticket in it. The ticket (the old counter value) is
// the particle's rank within its cell, which fixes its final slot once the starts are known and removes the
// need for a second atomic pass with a per-cell cursor.
@compute @workgroup_size(64)
fn cells(@builtin(global_invocation_id) id: vec3u) {
  let i = id.x;
  if (i >= arrayLength(&positions)) { return; }
  let cell = neighbor_cell_id(grid, positions[i].xyz);
  let rank = atomicAdd(&counts[cell], 1u);
  cell_rank[i] = vec2u(cell, rank);
}

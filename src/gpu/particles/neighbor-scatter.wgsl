// Stage 4 of the counting sort: with every cell's start offset known, particle i goes to slot
// start[cell] + rank. The result `sorted` is a permutation: sorted[k] is the particle that belongs in slot k.

@group(0) @binding(0) var<storage, read> cell_start: array<u32>;
@group(0) @binding(1) var<storage, read> cell_rank: array<vec2u>;
@group(0) @binding(2) var<storage, read_write> sorted: array<u32>;

@compute @workgroup_size(64)
fn scatter(@builtin(global_invocation_id) id: vec3u) {
  let i = id.x;
  if (i >= arrayLength(&cell_rank)) { return; }
  let cr = cell_rank[i];
  sorted[cell_start[cr.x] + cr.y] = i;
}

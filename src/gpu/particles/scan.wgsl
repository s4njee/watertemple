// Exclusive prefix sum over u32, in place, built from two kernels (see scan.ts for the levels).
//
// A workgroup of 256 invocations owns a tile of 1024 elements, four consecutive elements per invocation.
// scan_blocks scans each tile and writes the tile's total into `sums`; add_offsets later adds the
// (already scanned) tile offset back to every element of the tile. Both bind the data and sums arrays
// with exactly the length being scanned, so arrayLength is the element count.

@group(0) @binding(0) var<storage, read_write> data: array<u32>;
@group(0) @binding(1) var<storage, read_write> sums: array<u32>;

// One partial sum per invocation: 256 x 4 bytes = 1 KiB of the 16 KiB workgroup memory every device has.
var<workgroup> partial: array<u32, 256>;

@compute @workgroup_size(256)
fn scan_blocks(@builtin(local_invocation_id) lid: vec3u, @builtin(workgroup_id) wid: vec3u) {
  let n = arrayLength(&data);
  let base = wid.x * 1024u + lid.x * 4u;
  var v = vec4u(0u);
  if (base < n) { v.x = data[base]; }
  if (base + 1u < n) { v.y = data[base + 1u]; }
  if (base + 2u < n) { v.z = data[base + 2u]; }
  if (base + 3u < n) { v.w = data[base + 3u]; }

  // Exclusive prefixes inside this invocation's four elements.
  let e1 = v.x;
  let e2 = e1 + v.y;
  let e3 = e2 + v.z;
  let total = e3 + v.w;

  // Hillis-Steele inclusive scan of the 256 totals. Every step reads, waits, then writes, so no
  // invocation overwrites a value a neighbour has yet to read.
  partial[lid.x] = total;
  workgroupBarrier();
  for (var offset = 1u; offset < 256u; offset = offset << 1u) {
    var t = partial[lid.x];
    if (lid.x >= offset) {
      t += partial[lid.x - offset];
    }
    workgroupBarrier();
    partial[lid.x] = t;
    workgroupBarrier();
  }
  let inclusive = partial[lid.x];
  let prefix = inclusive - total;

  if (base < n) { data[base] = prefix; }
  if (base + 1u < n) { data[base + 1u] = prefix + e1; }
  if (base + 2u < n) { data[base + 2u] = prefix + e2; }
  if (base + 3u < n) { data[base + 3u] = prefix + e3; }
  if (lid.x == 255u) {
    sums[wid.x] = inclusive;
  }
}

@compute @workgroup_size(256)
fn add_offsets(@builtin(local_invocation_id) lid: vec3u, @builtin(workgroup_id) wid: vec3u) {
  let n = arrayLength(&data);
  let base = wid.x * 1024u + lid.x * 4u;
  let offset = sums[wid.x];
  if (base < n) { data[base] += offset; }
  if (base + 1u < n) { data[base + 1u] += offset; }
  if (base + 2u < n) { data[base + 2u] += offset; }
  if (base + 3u < n) { data[base + 3u] += offset; }
}

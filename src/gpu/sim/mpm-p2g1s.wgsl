// mpm-p2g1s.wgsl -- P2G pass 1 for particles that are sorted by cell (the `sorted` mode of src/gpu/sim/mpm.ts).
// The arithmetic is mpm-p2g1.wgsl's; what changes is who does the adding.
//
// An invocation owns one ROW of one CELL of the sort: a cell is the set of particles with one stencil base
// (floor(x / h - 0.5), which is exactly what the sort bins by), so they all touch the same 27 nodes; a row is the
// three nodes along x at one (y, z) of that stencil. The invocation adds the contributions of all the cell's
// particles to its three nodes in registers and writes the sums with one atomic add per value. With about 8
// particles per cell that is 8 times fewer atomics than a thread per particle, which is what pass 1 is bound by,
// and nine invocations per cell keep the GPU fed (a cell alone would be too few threads: a 100 000-particle pool
// touches only 12 000 cells). Neighbouring invocations have neighbouring cells, so at any moment they add to
// different nodes.
//
// A particle the sort binned into a different cell than its own stencil base (the sort pads the cell by 1/4096,
// so a particle within that of a cell face can be off by one) is a stray: the row-0 invocation of its cell adds
// all 27 of its nodes directly, as mpm-p2g1.wgsl does, and the rows skip it.
//
// Bindings: 1 uniform and 7 storage buffers; cell_start is the sort's table (`cells + 1` offsets).

#include "mpm-params.wgsl"
#include "mpm-core.wgsl"

const WG = 64u;

@group(0) @binding(0) var<uniform> P: MpmParams;
@group(0) @binding(1) var<storage, read_write> grid: array<atomic<i32>>;
@group(0) @binding(2) var<storage, read> positions: array<vec4f>;
@group(0) @binding(3) var<storage, read> velocities: array<vec4f>;
@group(0) @binding(4) var<storage, read> affine0: array<vec4f>;
@group(0) @binding(5) var<storage, read> affine1: array<vec4f>;
@group(0) @binding(6) var<storage, read> affine2: array<vec4f>;
@group(0) @binding(7) var<storage, read> cell_start: array<u32>;

fn add_node(node: u32, w: f32, q: vec3f) {
  atomicAdd(&grid[node], mpm_to_fixed(w));
  atomicAdd(&grid[node + 1u], mpm_to_fixed(q.x));
  atomicAdd(&grid[node + 2u], mpm_to_fixed(q.y));
  atomicAdd(&grid[node + 3u], mpm_to_fixed(q.z));
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3u, @builtin(num_workgroups) groups: vec3u) {
  let item = gid.x + gid.y * (groups.x * WG);
  let cell = item / 9u;
  let row = item % 9u;
  let cells = P.dims - vec3u(2u);
  if (cell >= cells.x * cells.y * cells.z) {
    return;
  }
  let first = cell_start[cell];
  let last = cell_start[cell + 1u];
  if (first == last) {
    return;
  }
  let dims = vec3i(P.dims);
  let cbase = vec3i(i32(cell % cells.x), i32((cell / cells.x) % cells.y), i32(cell / (cells.x * cells.y)));
  let k = i32(row / 3u);
  let j = i32(row % 3u);

  var a0 = vec4f(0.0);
  var a1 = vec4f(0.0);
  var a2 = vec4f(0.0);
  for (var p = first; p < last; p++) {
    let g = positions[p].xyz * P.inv_h;
    let base = clamp(vec3i(floor(g - vec3f(0.5))), vec3i(0), dims - vec3i(3));
    let v = velocities[p].xyz;
    let C = mat3x3f(affine0[p].xyz, affine1[p].xyz, affine2[p].xyz);
    let fx = g - vec3f(base);
    let W = mpm_weight_matrix(fx);
    if (any(base != cbase)) {
      // A stray: the first row adds its whole stencil.
      if (row == 0u) {
        for (var kk = 0; kk < 3; kk++) {
          for (var jj = 0; jj < 3; jj++) {
            for (var ii = 0; ii < 3; ii++) {
              let w = W[ii].x * W[jj].y * W[kk].z;
              let d = (vec3f(f32(ii), f32(jj), f32(kk)) - fx) * P.h;
              let node = u32(((base.z + kk) * dims.y + (base.y + jj)) * dims.x + (base.x + ii)) * 4u;
              add_node(node, w, w * (v + C * d));
            }
          }
        }
      }
      continue;
    }
    let wyz = W[j].y * W[k].z;
    let dyz = vec3f(0.0, f32(j) - fx.y, f32(k) - fx.z) * P.h;
    let w0 = W[0].x * wyz;
    let w1 = W[1].x * wyz;
    let w2 = W[2].x * wyz;
    a0 += vec4f(w0, w0 * (v + C * (dyz + vec3f(-fx.x * P.h, 0.0, 0.0))));
    a1 += vec4f(w1, w1 * (v + C * (dyz + vec3f((1.0 - fx.x) * P.h, 0.0, 0.0))));
    a2 += vec4f(w2, w2 * (v + C * (dyz + vec3f((2.0 - fx.x) * P.h, 0.0, 0.0))));
  }
  let node = u32(((cbase.z + k) * dims.y + (cbase.y + j)) * dims.x + cbase.x) * 4u;
  add_node(node, a0.x, a0.yzw);
  add_node(node + 4u, a1.x, a1.yzw);
  add_node(node + 8u, a2.x, a2.yzw);
}

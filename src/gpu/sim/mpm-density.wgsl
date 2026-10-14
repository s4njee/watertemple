// mpm-density.wgsl -- the first half of P2G pass 2 in the sorted mode (src/gpu/sim/mpm.ts): the density ratio of
// every particle, gathered from the mass that pass 1 left on the grid, with the wall correction. The arithmetic is
// the first half of mpm-p2g2.wgsl (read its header); the ratio goes to affine0.w, where the scatter half
// (mpm-p2g2s.wgsl) reads it. A thread per particle: all of them are independent, and the gather is a few loads.
//
// Bindings: 1 uniform and 3 storage buffers.

#include "mpm-params.wgsl"
#include "mpm-core.wgsl"

@group(0) @binding(0) var<uniform> P: MpmParams;
@group(0) @binding(1) var<storage, read_write> grid: array<atomic<i32>>;
@group(0) @binding(2) var<storage, read> positions: array<vec4f>;
@group(0) @binding(3) var<storage, read_write> affine0: array<vec4f>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let p = gid.x;
  if (p >= P.count) {
    return;
  }
  let dims = vec3i(P.dims);
  let g = positions[p].xyz * P.inv_h;
  let base = clamp(vec3i(floor(g - vec3f(0.5))), vec3i(0), dims - vec3i(3));
  let fx = g - vec3f(base);
  let W = mpm_weight_matrix(fx);

  var mass = 0.0;
  for (var k = 0; k < 3; k++) {
    for (var j = 0; j < 3; j++) {
      let row = u32(((base.z + k) * dims.y + (base.y + j)) * dims.x + base.x) * 4u;
      let wyz = W[j].y * W[k].z;
      mass += wyz * (W[0].x * mpm_from_fixed(atomicLoad(&grid[row]))
        + W[1].x * mpm_from_fixed(atomicLoad(&grid[row + 4u]))
        + W[2].x * mpm_from_fixed(atomicLoad(&grid[row + 8u])));
    }
  }
  var fill = 1.0;
  if (P.wall_comp != 0u) {
    let lo = P.bounds_min * P.inv_h;
    let hi = P.bounds_max * P.inv_h;
    for (var axis = 0; axis < 3; axis++) {
      var share = 0.0;
      for (var o = 0; o < 3; o++) {
        let node = f32(base[axis] + o);
        let inside = mpm_kernel_above(lo[axis] - node) - mpm_kernel_above(hi[axis] - node);
        share += W[o][axis] * inside;
      }
      fill *= share;
    }
  }
  let a0 = affine0[p];
  affine0[p] = vec4f(a0.xyz, clamp(mass / (P.rest_ppc * max(fill, 0.05)), 0.0, 64.0));
}

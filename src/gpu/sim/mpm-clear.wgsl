// mpm-clear.wgsl -- step 1 of an MLS-MPM step: zero the grid's fixed-point mass and momentum
// (src/gpu/sim/mpm.ts). One invocation per node, 16 bytes each. The dispatch may be two-dimensional for a
// big grid (more than 65535 workgroups), so the node index is x + y * (workgroups.x * size).

#include "mpm-params.wgsl"

// Workgroup size; mpm.ts rewrites this line and @workgroup_size together.
const WG = 64u;

@group(0) @binding(0) var<uniform> P: MpmParams;
@group(0) @binding(1) var<storage, read_write> grid: array<vec4<i32>>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3u, @builtin(num_workgroups) groups: vec3u) {
  let node = gid.x + gid.y * (groups.x * WG);
  if (node >= P.dims.x * P.dims.y * P.dims.z) {
    return;
  }
  grid[node] = vec4<i32>(0);
}

// mpm-p2g1.wgsl -- P2G, pass 1 of an MLS-MPM step (src/gpu/sim/mpm.ts): every particle scatters its mass and its
// momentum, including the APIC affine term, to the 27 grid nodes its quadratic B-spline touches.
//
//     node mass     m_i  = sum_p w_ip
//     node momentum q_i  = sum_p w_ip (v_p + C_p (x_i - x_p))       (every particle has mass 1)
//
// Many particles write one node in parallel, so the sums are atomic adds on i32 fixed point (mpm-core.wgsl
// documents the scale and range). A node is 4 consecutive i32: mass, momentum x, y, z. Bindings: 1 uniform and
// 6 storage buffers (the limit is 8).

#include "mpm-params.wgsl"
#include "mpm-core.wgsl"

@group(0) @binding(0) var<uniform> P: MpmParams;
@group(0) @binding(1) var<storage, read_write> grid: array<atomic<i32>>;
@group(0) @binding(2) var<storage, read> positions: array<vec4f>;
@group(0) @binding(3) var<storage, read> velocities: array<vec4f>;
@group(0) @binding(4) var<storage, read> affine0: array<vec4f>;
@group(0) @binding(5) var<storage, read> affine1: array<vec4f>;
@group(0) @binding(6) var<storage, read> affine2: array<vec4f>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let p = gid.x;
  if (p >= P.count) {
    return;
  }
  let x = positions[p].xyz;
  let v = velocities[p].xyz;
  let C = mat3x3f(affine0[p].xyz, affine1[p].xyz, affine2[p].xyz);

  let dims = vec3i(P.dims);
  let g = x * P.inv_h;
  let base = clamp(vec3i(floor(g - vec3f(0.5))), vec3i(0), dims - vec3i(3));
  let fx = g - vec3f(base);
  let W = mpm_weight_matrix(fx);

  for (var k = 0; k < 3; k++) {
    for (var j = 0; j < 3; j++) {
      for (var i = 0; i < 3; i++) {
        let w = W[i].x * W[j].y * W[k].z;
        let d = (vec3f(f32(i), f32(j), f32(k)) - fx) * P.h;
        let q = w * (v + C * d);
        let node = u32(((base.z + k) * dims.y + (base.y + j)) * dims.x + (base.x + i)) * 4u;
        atomicAdd(&grid[node], mpm_to_fixed(w));
        atomicAdd(&grid[node + 1u], mpm_to_fixed(q.x));
        atomicAdd(&grid[node + 2u], mpm_to_fixed(q.y));
        atomicAdd(&grid[node + 3u], mpm_to_fixed(q.z));
      }
    }
  }
}

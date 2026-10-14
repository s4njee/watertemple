// mpm-p2g2s.wgsl -- the scatter half of P2G pass 2 for particles sorted by cell (the `sorted` mode of
// src/gpu/sim/mpm.ts): the stress impulses of mpm-p2g2.wgsl (read its header for the equation of state, the
// viscous stress and the walls), added per row of a cell as mpm-p2g1s.wgsl adds mass and momentum (read its header
// for the layout and the strays). The densities were gathered by mpm-density.wgsl and sit in affine0.w.
//
// Bindings: 1 uniform and 5 storage buffers.

#include "mpm-params.wgsl"
#include "mpm-core.wgsl"

const WG = 64u;

@group(0) @binding(0) var<uniform> P: MpmParams;
@group(0) @binding(1) var<storage, read_write> grid: array<atomic<i32>>;
@group(0) @binding(2) var<storage, read> positions: array<vec4f>;
@group(0) @binding(3) var<storage, read> affine0: array<vec4f>;
@group(0) @binding(4) var<storage, read> affine1: array<vec4f>;
@group(0) @binding(5) var<storage, read> affine2: array<vec4f>;
@group(0) @binding(6) var<storage, read> cell_start: array<u32>;

fn add_momentum(node: u32, q: vec3f) {
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

  let c2g = P.sound_speed * P.sound_speed / P.gamma;
  let scale = 4.0 * P.inv_h * P.inv_h * P.dt;
  var q0 = vec3f(0.0);
  var q1 = vec3f(0.0);
  var q2 = vec3f(0.0);
  for (var p = first; p < last; p++) {
    let g = positions[p].xyz * P.inv_h;
    let base = clamp(vec3i(floor(g - vec3f(0.5))), vec3i(0), dims - vec3i(3));
    let a0 = affine0[p];
    let C = mat3x3f(a0.xyz, affine1[p].xyz, affine2[p].xyz);
    let rc = clamp(a0.w, 0.001, P.r_max);
    let pressure = max(c2g * (pow(rc, P.gamma) - 1.0), -P.tension * c2g);
    let inv_r = 1.0 / max(rc, 0.25);
    let pressure_over_r = pressure * inv_r;
    let strain = (C + transpose(C)) * (P.viscosity * inv_r);
    let fx = g - vec3f(base);
    let W = mpm_weight_matrix(fx);
    if (any(base != cbase)) {
      if (row == 0u) {
        for (var kk = 0; kk < 3; kk++) {
          for (var jj = 0; jj < 3; jj++) {
            for (var ii = 0; ii < 3; ii++) {
              let w = W[ii].x * W[jj].y * W[kk].z;
              let d = (vec3f(f32(ii), f32(jj), f32(kk)) - fx) * P.h;
              let node = u32(((base.z + kk) * dims.y + (base.y + jj)) * dims.x + (base.x + ii)) * 4u;
              add_momentum(node, (w * scale) * (pressure_over_r * d - strain * d));
            }
          }
        }
      }
      continue;
    }
    let wyz = W[j].y * W[k].z * scale;
    let dyz = vec3f(0.0, f32(j) - fx.y, f32(k) - fx.z) * P.h;
    let d0 = dyz + vec3f(-fx.x * P.h, 0.0, 0.0);
    let d1 = dyz + vec3f((1.0 - fx.x) * P.h, 0.0, 0.0);
    let d2 = dyz + vec3f((2.0 - fx.x) * P.h, 0.0, 0.0);
    q0 += (W[0].x * wyz) * (pressure_over_r * d0 - strain * d0);
    q1 += (W[1].x * wyz) * (pressure_over_r * d1 - strain * d1);
    q2 += (W[2].x * wyz) * (pressure_over_r * d2 - strain * d2);
  }
  let node = u32(((cbase.z + k) * dims.y + (cbase.y + j)) * dims.x + cbase.x) * 4u;
  add_momentum(node, q0);
  add_momentum(node + 4u, q1);
  add_momentum(node + 8u, q2);
}

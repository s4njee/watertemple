// pbf-correct.wgsl -- step 3b of a position-based fluid sub-step, once per solver iteration (story 7.4): the position
// correction, the container, and the velocity.
//
//   dp_i = 1/n0 * sum_j (lambda_i + lambda_j + s_corr_ij) grad W_spiky(x_i - x_j)       (+ the walls: wall_k lambda_i g_wall)
//   x_i  = contain(x_i + relax * dp_i)
//   v_i  = (x_i - previous_i) / dt
//
// s_corr = -k (W(r) / W(dq))^n is the artificial pressure of the paper: a small repulsion between close pairs that
// stops the clumping a negative lambda would cause at the free surface (the ratio is capped at 1, so closer pairs than
// dq do not push harder and a large dq cannot blow the fluid up). k arrives in units of lambda at rest
// (params.scorr_k = k / S0), so it means the same thing at any particle spacing and in 2D and 3D.
//
// JACOBI. Every particle reads the positions of the previous iterate and writes a new buffer (`positions_out`, the
// back buffer of the position attribute, swapped by the solver afterwards). Nothing depends on the order in which
// particles are processed, which is what makes this parallel, and it converges more slowly than the Gauss-Seidel
// order of the CPU paper: the sum over about 30 neighbours over-corrects (every neighbour's constraint pushes on i),
// which `eps` and `relax` damp. See pbf.ts for the measured convergence.
//
// The velocity is written every iteration, from the position change since the start of the sub-step, so the last
// iteration leaves the velocity that goes with the final position. The foam amount (positions.w) is raised by fast
// particles with a low density, and decays.

#include "pbf-core.wgsl"
#include "neighbor_bindings.wgsl"

@group(0) @binding(2) var<storage, read> positions: array<vec4f>;
@group(0) @binding(3) var<storage, read_write> velocities: array<vec4f>;
@group(0) @binding(4) var<storage, read> previous: array<vec4f>;
@group(0) @binding(5) var<storage, read> solve: array<vec2f>;
@group(0) @binding(6) var<storage, read_write> positions_out: array<vec4f>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3u) {
  let i = id.x;
  if (i >= params.count) { return; }
  let pw = positions[i];
  let p = pw.xyz;
  let prev = previous[i].xyz;
  let state = solve[i];
  let lambda_i = state.x;
  let c = neighbor_cell_of(neighbor_grid, p);
  var dp = vec3f(0.0);
  let use_scorr = params.scorr_k != 0.0;
  for (var row = 0u; row < neighbor_grid.rows; row++) {
    let span = neighbor_row_span(c, row);
    for (var k = span.x; k < span.y; k++) {
      let j = neighbor_particle(k);
      if (j == i) { continue; }
      var r = p - positions[j].xyz;
      var r2 = neighbor_dist2(neighbor_grid, r);
      if (r2 >= params.h2) { continue; }
      if (r2 < 1e-12) {
        r = pbf_nudge(i, j, params.h);
        r2 = dot(r, r);
      }
      var scorr = 0.0;
      if (use_scorr) {
        let ratio = min(pbf_poly6(r2, params.h2, params.poly6) * params.inv_w_dq, 1.0);
        // The exponent is a small integer (4 in the paper): repeated products, not a pow() per neighbour.
        var power = ratio * ratio;
        power *= power;
        if (params.scorr_n != 4u) {
          power = ratio;
          for (var q = 1u; q < params.scorr_n; q++) {
            power *= ratio;
          }
        }
        scorr = -params.scorr_k * max(power - params.scorr_floor, 0.0);
      }
      let other = solve[j];
      if (use_scorr && params.gate_scale > 0.0) {
        // Optional (gate_scale > 0). Only where the fluid is under pressure (both lambdas negative, about 2 % compression or more): a free blob, a
        // jet or a free surface has nothing pushing back, and a repulsion there only blows it up.
        scorr *= clamp(-0.5 * (lambda_i + other.x) * params.gate_scale, 0.0, 1.0);
      }
      dp += (lambda_i + other.x + scorr) * pbf_spiky_grad(r, params.h, params.spiky);
    }
  }
  if (params.boundary_density != 0u) {
    dp += params.wall_k * lambda_i * pbf_wall_terms(p).grad;
  }
  dp *= params.inv_n0 * params.relax;
  let dp_len = length(dp);
  if (dp_len > params.max_correction) {
    dp *= params.max_correction / dp_len;
  }

  var x = pbf_contain(pbf_finite3(p + dp, p), prev, 1.0);
  var v = (x - prev) * tick.inv_dt;
  let speed = length(v);
  if (speed > params.max_speed) {
    v *= params.max_speed / speed;
  }
  v = pbf_finite3(v, vec3f(0.0));

  // Foam: fast and sparse. The density ratio comes from the density pass of this iteration.
  let sparse = clamp((0.8 - state.y) / 0.45, 0.0, 1.0);
  let fast = clamp((speed - 0.5 * params.foam_speed) / params.foam_speed, 0.0, 1.0);
  let foam = clamp(max(sparse * fast, pw.w * exp(-params.foam_decay * tick.dt)), 0.0, 1.0);

  positions_out[i] = vec4f(x, foam);
  velocities[i] = vec4f(v, 0.0);
}

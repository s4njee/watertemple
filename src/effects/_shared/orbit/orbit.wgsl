// orbit.wgsl -- the current: a slow drift of the fluid round each of the page's elements, and the water
// a moving element carries with it. One compute pass per simulation step, run from the solver's
// `hooks.forces` (after its own forces, before the pressure solve): it reads the staggered velocity and
// writes it back nudged towards a target field. The ink trails themselves are laid by the heads
// (heads.ts); this is what keeps their ink turning after the head has moved on.
//
// THE FIELD. Each element is a rounded box with a signed distance d to its edge. The current runs
// ALONG the level sets of d (the element's outline, pushed outwards), at a speed that depends on d
// alone:
//
//     target(p) = spin * speed * f(d / lane) * tangent(p)        f(x) = x * exp(1 - x)
//
// f is 0 on the element (no slip), 1 at d = lane and fades beyond it. A flow along the level sets of
// a distance field at a speed that is a function of the distance is exactly divergence free (its
// stream function is a function of d), so the pressure solve that follows has nothing to remove: what
// is asked for is what the fluid does. A pull TOWARDS the element would be the opposite case, a pure
// gradient, which an incompressible fluid cancels completely: the turning is the part of an
// attractor a fluid can feel, which is why the pull acts on the heads and not on the water.
//
// RELAXATION, NOT A PUSH. The velocity is moved a fraction of the way to the target, by
// 1 - exp(-grip * dt) where the current is strongest and less where it is weak. That is
// unconditionally stable and independent of the step rate, it cannot spin the fluid up past `speed`,
// and away from the elements the fluid is left to itself (the heads' wakes, pointer strokes, vortices).
//
// A MOVING ELEMENT adds its own velocity to the target near its edge, so a dragged card carries the
// water with it.
//
// Lengths are image heights (canvas height = 1), positions have the origin at the top left with y
// down, matching src/gpu/sim/fluid.ts. The velocity texture is (W + 1) x (H + 1): u sits on the left
// face of a cell, v on its top face (see fluid-core.wgsl); each component is evaluated where it lives.
#include "orbit_params.wgsl"

@group(0) @binding(0) var<uniform> P: OrbitParams;
@group(0) @binding(1) var src: texture_2d<f32>;
@group(0) @binding(2) var dst: texture_storage_2d<rgba16float, write>;

// xy: the velocity the current wants at p. z: how firmly, 0..1.
fn orbit_target(p: vec2f) -> vec3f {
  var velocity = vec2f(0.0);
  var hold = 0.0;
  for (var k = 0u; k < P.count; k++) {
    let box = P.boxes[k];
    let extra = P.extras[k];
    let radius = extra.x;
    let rel = p - box.xy;
    let q = abs(rel) - (box.zw - vec2f(radius));
    let outside = max(q, vec2f(0.0));
    let d = length(outside) + min(max(q.x, q.y), 0.0) - radius;
    if (d <= 0.0) {
      // Inside the element. The solver holds solid faces still; nothing to ask for.
      continue;
    }
    // The outward normal: radial around a corner, axis aligned along a side.
    var n = vec2f(0.0, 1.0);
    if (q.x > 0.0 && q.y > 0.0) {
      n = outside / length(outside);
    } else if (q.x > q.y) {
      n = vec2f(1.0, 0.0);
    }
    n *= vec2f(select(-1.0, 1.0, rel.x >= 0.0), select(-1.0, 1.0, rel.y >= 0.0));
    // Clockwise on screen (y down) for a positive spin.
    let tangent = vec2f(-n.y, n.x);

    let x = d / P.lane;
    let lane = x * exp(1.0 - x);
    let near = exp(-x);
    let carried = clamp(length(extra.zw) / max(P.speed, 1e-3), 0.0, 1.0) * near;
    velocity += tangent * (extra.y * P.speed * lane) + extra.zw * near;
    hold = max(hold, max(lane, carried));
  }
  return vec3f(velocity, clamp(hold, 0.0, 1.0));
}

@compute @workgroup_size(8, 8)
fn cs_orbit(@builtin(global_invocation_id) id: vec3u) {
  if (any(id.xy >= P.grid + vec2u(1u))) {
    return;
  }
  let c = vec2i(id.xy);
  let g = vec2i(P.grid);
  let old = textureLoad(src, c, 0);
  // The padding row of u and column of v repeat the last real face.
  let at_u = vec2f(f32(c.x), f32(min(c.y, g.y - 1)) + 0.5) * P.cell;
  let at_v = vec2f(f32(min(c.x, g.x - 1)) + 0.5, f32(c.y)) * P.cell;
  let want_u = orbit_target(at_u);
  let want_v = orbit_target(at_v);
  let rate = 1.0 - exp(-P.grip * P.dt);
  var u = mix(old.x, want_u.x, rate * want_u.z);
  var v = mix(old.y, want_v.y, rate * want_v.z);
  // Nothing crosses a closed canvas edge.
  if (P.closed != 0u) {
    if (c.x == 0 || c.x == g.x) {
      u = old.x;
    }
    if (c.y == 0 || c.y == g.y) {
      v = old.y;
    }
  }
  textureStore(dst, c, vec4f(u, v, old.zw));
}

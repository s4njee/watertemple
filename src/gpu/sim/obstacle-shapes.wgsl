// obstacle-shapes.wgsl -- analytic solid shapes for the fluid solver's obstacle mask
// (src/gpu/sim/obstacle-shapes.ts), as a library chunk: no bindings, only functions of a shape's
// three vec4f records. The rasteriser (obstacle-raster.wgsl), the moving-boundary pass
// (obstacle-drive.wgsl) and an effect's own shading pass all include this, so the cells the solver
// treats as solid and the outline the viewer sees come from the same numbers.
//
// COORDINATES. `p` is a point in IMAGE HEIGHTS with y down: (uv.x * aspect, uv.y). Distances are in
// the same unit. Negative is inside.
//
// THE RECORD (what `T.data[3 * i + k]` holds, see ObstacleTable in obstacle-shapes.ts):
//   geo     (cx, cy, cos a, sin a)    centre in image heights and the rotation a. A positive angle turns
//                                     the shape CLOCKWISE on the screen (y is down).
//   size    (hx, hy, radius, kind)    half extents in the shape's own frame (x along its long axis),
//                                     a corner radius, and the kind (the OBSTACLE_* constants below)
//   motion  (vx, vy, spin, 0)         velocity of the centre (image heights / s) and angular velocity
//                                     (rad / s, positive clockwise on screen)
//
//   obstacle_local(geo, p) -> vec2f                p in the shape's own frame (centre at the origin)
//   obstacle_sd(geo, size, p) -> f32               signed distance to the shape: exact for the circle, box,
//                                                  rounded box, capsule and the wing (a tapered
//                                                  capsule); a true distance outside, within a fraction
//                                                  of it inside
//   obstacle_sd_wing(q, half_chord, half_thickness) -> f32   the wing alone, in its own frame
//   obstacle_normal(geo, size, p) -> vec2f         unit outward normal (central differences of obstacle_sd)
//   obstacle_velocity(geo, motion, p) -> vec2f     velocity of the solid at p: v + spin x (p - centre)
//
// SHAPES. circle: radius hx. box: half extents (hx, hy). rounded-box: a box with corners of `radius`.
// capsule: a stadium, outer half extents (hx, hy), end caps of radius hy. wing: a symmetric teardrop
// with a round leading edge of radius hy at -x and a sharp trailing edge at +x, half chord hx. It is
// the convex hull of two circles (the leading edge and a tiny one at the tail), which is the
// "uneven capsule" distance of Inigo Quilez; rotate it for an angle of attack.
//
// Cost: a few dozen ALU per shape.

#include "sdf.wgsl"

const OBSTACLE_CIRCLE: f32 = 0.0;
const OBSTACLE_BOX: f32 = 1.0;
const OBSTACLE_ROUNDED_BOX: f32 = 2.0;
const OBSTACLE_CAPSULE: f32 = 3.0;
const OBSTACLE_WING: f32 = 4.0;

fn obstacle_local(geo: vec4f, p: vec2f) -> vec2f {
  let d = p - geo.xy;
  return vec2f(d.x * geo.z + d.y * geo.w, -d.x * geo.w + d.y * geo.z);
}

fn obstacle_sd_wing(q: vec2f, half_chord: f32, half_thickness: f32) -> f32 {
  let r1 = half_thickness;
  let r2 = max(half_thickness * 0.05, 1e-4);
  let c = max(half_chord, r1 * 1.5);
  // Circle 1 (leading edge) at x1, circle 2 (the tail) at x2: both on the axis.
  let x1 = -c + r1;
  let x2 = c - r2;
  let h = x2 - x1;
  let p = vec2f(abs(q.y), q.x - x1);
  let b = (r1 - r2) / h;
  let a = sqrt(max(1.0 - b * b, 0.0));
  let k = dot(p, vec2f(-b, a));
  if (k < 0.0) {
    return length(p) - r1;
  }
  if (k > a * h) {
    return length(p - vec2f(0.0, h)) - r2;
  }
  return dot(p, vec2f(a, b)) - r1;
}

fn obstacle_sd(geo: vec4f, size: vec4f, p: vec2f) -> f32 {
  let q = obstacle_local(geo, p);
  let kind = size.w;
  if (kind < 0.5) {
    return sd_circle(q, size.x);
  }
  if (kind < 1.5) {
    return sd_box_2d(q, size.xy);
  }
  if (kind < 2.5) {
    return sd_rounded_box_2d(q, size.xy, size.z);
  }
  if (kind < 3.5) {
    let r = min(size.x, size.y);
    let tail = max(size.x - r, 0.0);
    return sd_capsule_2d(q, vec2f(-tail, 0.0), vec2f(tail, 0.0), r);
  }
  return obstacle_sd_wing(q, size.x, size.y);
}

fn obstacle_normal(geo: vec4f, size: vec4f, p: vec2f) -> vec2f {
  let e = 0.0015;
  let g = vec2f(
    obstacle_sd(geo, size, p + vec2f(e, 0.0)) - obstacle_sd(geo, size, p - vec2f(e, 0.0)),
    obstacle_sd(geo, size, p + vec2f(0.0, e)) - obstacle_sd(geo, size, p - vec2f(0.0, e)),
  );
  let l = length(g);
  return select(vec2f(1.0, 0.0), g / l, l > 1e-9);
}

fn obstacle_velocity(geo: vec4f, motion: vec4f, p: vec2f) -> vec2f {
  let r = p - geo.xy;
  return motion.xy + motion.z * vec2f(-r.y, r.x);
}

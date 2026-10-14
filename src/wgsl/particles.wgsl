// particles.wgsl -- helpers shared by particle renderers and particle emitters (story 7.1): billboard
// geometry, sprite edges, a palette for the per-particle scalar, and random emission. The framework in
// src/gpu/particles/ uses all of it; a custom particle shader can pick what it needs.
//
// Constants
//   PARTICLE_SHAPE_POINT, PARTICLE_SHAPE_BOX, PARTICLE_SHAPE_SPHERE, PARTICLE_SHAPE_DISC   u32 emitter shapes
//
// Billboards (camera-facing quads, drawn as a 4-vertex triangle strip)
//   particle_corner(vertex_index) -> vec2f                 strip corner in (-1, 1)^2: (-1,-1) (1,-1) (-1,1) (1,1)
//   particle_size(radius, clip_w, proj_scale, viewport_h, min_pixels) -> vec2f
//                                                          x: the radius to draw, in world units, never smaller
//                                                          than min_pixels on screen; y: an alpha factor in
//                                                          (0, 1] that conserves energy for an enlarged sprite
//                                                          (so subpixel particles do not flare). proj_scale is
//                                                          proj[1][1]; clip_w is the centre's clip w (1 for an
//                                                          orthographic camera). Works for both projections.
//   particle_disc_alpha(d, softness, aa) -> f32            coverage of a round sprite at distance d from its
//                                                          centre (d = 1 is the rim): 1 inside, 0 outside, an
//                                                          edge `aa` wide (one pixel in sprite units) centred
//                                                          on the rim for softness 0, a full smooth falloff for
//                                                          softness 1. Draw the quad (1 + aa) times the radius.
//   particle_sphere_z(d2) -> f32                           height of a sphere impostor above its disc, in radii,
//                                                          at squared distance d2 (0 at the rim, 1 at the centre)
//
// Colour
//   particle_palette(stops, count, t) -> vec3f             piecewise-linear colour through `count` (2..8) stops
//                                                          (rgb in xyz), t clamped to 0..1
//
// Random emission (a pcg hash, no state: the same index and salt give the same numbers on every device)
//   particle_random_3d(index, salt) -> vec3f               three independent uniform numbers in [0, 1)
//   particle_emit_position(shape, center, extent, r, dimension) -> vec3f
//                                                          a point in the emitter shape from r in [0, 1)^3:
//                                                          POINT the centre; BOX center + (2r - 1) * extent
//                                                          (extent is the half-size); SPHERE uniformly in the
//                                                          ball of radius extent.x; DISC uniformly in the
//                                                          horizontal (xz) disc of radius extent.x. In 2D (dimension
//                                                          2u) z is 0, SPHERE is a disc in xy and DISC a segment
//                                                          along x.
//   particle_emit_velocity(direction, speed, spread, jitter, r, dimension) -> vec3f
//                                                          direction (unit) * speed, bent by a random angle up to
//                                                          `spread` radians (a cone in 3D, a fan in 2D), with the
//                                                          speed varied by up to +-jitter (a fraction, 0..1)
//
// Cost: every function is a few ALU ops; particle_random_3d is one pcg_3d (about 20 integer ops).

#include "hash.wgsl"
#include "math.wgsl"

const PARTICLE_SHAPE_POINT: u32 = 0u;
const PARTICLE_SHAPE_BOX: u32 = 1u;
const PARTICLE_SHAPE_SPHERE: u32 = 2u;
const PARTICLE_SHAPE_DISC: u32 = 3u;

fn particle_corner(vertex_index: u32) -> vec2f {
  return vec2f(f32(vertex_index & 1u) * 2.0 - 1.0, f32((vertex_index >> 1u) & 1u) * 2.0 - 1.0);
}

fn particle_size(
  radius: f32,
  clip_w: f32,
  proj_scale: f32,
  viewport_h: f32,
  min_pixels: f32,
) -> vec2f {
  let pixels = radius * proj_scale / max(clip_w, 1e-6) * 0.5 * viewport_h;
  if (pixels >= min_pixels || pixels <= 0.0) {
    return vec2f(radius, 1.0);
  }
  let grow = min_pixels / pixels;
  return vec2f(radius * grow, 1.0 / (grow * grow));
}

fn particle_disc_alpha(d: f32, softness: f32, aa: f32) -> f32 {
  // The edge ramp is centred on the rim (d = 1) when hard, so a disc keeps its true area; the quad is drawn one
  // pixel larger than the rim to leave room for the outer half.
  let top = 1.0 + 0.5 * aa;
  let edge = clamp(mix(aa, top, softness), 1e-4, top);
  let t = clamp((top - d) / edge, 0.0, 1.0);
  return t * t * (3.0 - 2.0 * t);
}

fn particle_sphere_z(d2: f32) -> f32 {
  return sqrt(max(1.0 - d2, 0.0));
}

fn particle_palette(stops: array<vec4f, 8>, count: u32, t: f32) -> vec3f {
  let last = max(count, 2u) - 1u;
  let x = clamp(t, 0.0, 1.0) * f32(last);
  let i = min(u32(floor(x)), last - 1u);
  return mix(stops[i].xyz, stops[i + 1u].xyz, x - f32(i));
}

fn particle_random_3d(index: u32, salt: u32) -> vec3f {
  return to_unorm_3d(pcg_3d(vec3u(index, salt, 0x9e3779b9u)));
}

fn particle_emit_position(
  shape: u32,
  center: vec3f,
  extent: vec3f,
  r: vec3f,
  dimension: u32,
) -> vec3f {
  var offset = vec3f(0.0);
  if (shape == PARTICLE_SHAPE_BOX) {
    offset = (r * 2.0 - 1.0) * extent;
  } else if (shape == PARTICLE_SHAPE_SPHERE) {
    if (dimension == 2u) {
      let a = r.x * TAU;
      offset = vec3f(cos(a), sin(a), 0.0) * sqrt(r.y) * extent.x;
    } else {
      // Uniform in the ball: a direction from two numbers, a radius from the cube root of the third.
      let z = r.x * 2.0 - 1.0;
      let a = r.y * TAU;
      let s = sqrt(max(1.0 - z * z, 0.0));
      offset = vec3f(s * cos(a), z, s * sin(a)) * pow(r.z, 1.0 / 3.0) * extent.x;
    }
  } else if (shape == PARTICLE_SHAPE_DISC) {
    if (dimension == 2u) {
      offset = vec3f((r.x * 2.0 - 1.0) * extent.x, 0.0, 0.0);
    } else {
      let a = r.x * TAU;
      offset = vec3f(cos(a), 0.0, sin(a)) * sqrt(r.y) * extent.x;
    }
  }
  var p = center + offset;
  if (dimension == 2u) {
    p.z = 0.0;
  }
  return p;
}

fn particle_emit_velocity(
  direction: vec3f,
  speed: f32,
  spread: f32,
  jitter: f32,
  r: vec3f,
  dimension: u32,
) -> vec3f {
  let s = speed * (1.0 + (r.z * 2.0 - 1.0) * jitter);
  if (dimension == 2u) {
    let angle = (r.x * 2.0 - 1.0) * spread;
    return vec3f(rotate_2d(direction.xy, angle), 0.0) * s;
  }
  // Uniform in the cone around `direction`: cos(theta) between cos(spread) and 1.
  let cos_theta = mix(1.0, cos(spread), r.x);
  let sin_theta = sqrt(max(1.0 - cos_theta * cos_theta, 0.0));
  let phi = r.y * TAU;
  let up = select(vec3f(0.0, 1.0, 0.0), vec3f(1.0, 0.0, 0.0), abs(direction.y) > 0.99);
  let t1 = normalize(cross(direction, up));
  let t2 = cross(direction, t1);
  return (direction * cos_theta + (t1 * cos(phi) + t2 * sin(phi)) * sin_theta) * s;
}

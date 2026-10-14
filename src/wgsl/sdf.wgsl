// sdf.wgsl -- signed distance functions, combinators and smooth min/max.
// Sign convention: negative inside, 0 on the surface, positive outside. Distances are exact (Lipschitz-1)
// except after the boolean/smooth combinators, which keep them correct near the surface only.
//
// 2D primitives (p: point relative to the shape's centre; rotate/translate p yourself)
//   sd_circle(p, radius) -> f32
//   sd_box_2d(p, half_size: vec2f) -> f32
//   sd_rounded_box_2d(p, half_size: vec2f, radius) -> f32   outer extent is half_size; radius is clamped to fit
//   sd_segment_2d(p, a, b) -> f32                           unsigned distance to the segment ab (>= 0)
//   sd_capsule_2d(p, a, b, radius) -> f32                   segment inflated by radius
//
// 3D primitives
//   sd_sphere(p, radius)                      sd_box_3d(p, half_size: vec3f)
//   sd_rounded_box_3d(p, half_size, radius)   sd_capsule_3d(p, a, b, radius)
//   sd_torus(p, major, minor) -> f32          ring in the xz plane (axis y); major = ring radius,
//                                             minor = tube radius
//
// Combinators on distances (all f32)
//   sd_union(a, b) = min          sd_intersect(a, b) = max          sd_subtract(a, b) = a minus b = max(a, -b)
//   sd_round(d, r)  grow the shape by r (rounds convex corners; negative r shrinks)
//   sd_onion(d, thickness)  hollow shell: |d| - thickness (the shell is 2 * thickness thick)
//   sd_coverage(d, aa) -> f32   anti-aliased fill: 1 inside, 0 outside, a ramp `aa` wide centred on the
//                               surface. Pass aa = size of one pixel in the same units as d.
//
// Smooth minimum (metaballs, gooey merging, filleted joins). `k` is the blend size, in distance units.
// All are symmetric, never exceed min(a, b), and approach min as k -> 0. Max deviation at a == b:
//   smin_poly(a, b, k)       quadratic, C1, local: blends only where |a-b| < k; deviation k/4. Cheapest.
//   smin_cubic(a, b, k)      cubic, C2 (smoother shading), local like poly; deviation k/6
//   smin_exp(a, b, k)        exponential, C-infinity, associative (order of merging many blobs does not
//                            matter) but never exactly min; deviation k ln 2. Costs an exp and a log.
//   smin_poly_blend / smin_cubic_blend / smin_exp_blend (a, b, k) -> vec2f   (value, w): the same value
//                            plus w in [0, 1] = weight of b, i.e. d(value)/d(b). Gradient of the result =
//                            mix(grad a, grad b, w), and w blends material colours across the join.
//                            a << b gives w = 0, a == b gives 0.5.
//   smax_poly / smax_cubic / smax_exp(a, b, k)   smooth maximum, never below max(a, b)
//   Smooth intersection is smax_*(a, b, k); smooth subtraction is smax_*(a, -b, k).
//
// Cost: primitives are 5-15 ALU ops each. Smooth-min follows Inigo Quilez's articles on smooth minimum;
// polynomial forms re-derived here with k as the full blend width.

// ---- 2D primitives ------------------------------------------------------------------------------

fn sd_circle(p: vec2f, radius: f32) -> f32 {
  return length(p) - radius;
}

fn sd_box_2d(p: vec2f, half_size: vec2f) -> f32 {
  let q = abs(p) - half_size;
  return length(max(q, vec2f(0.0))) + min(max(q.x, q.y), 0.0);
}

fn sd_rounded_box_2d(p: vec2f, half_size: vec2f, radius: f32) -> f32 {
  let r = clamp(radius, 0.0, min(half_size.x, half_size.y));
  return sd_box_2d(p, half_size - vec2f(r)) - r;
}

fn sd_segment_2d(p: vec2f, a: vec2f, b: vec2f) -> f32 {
  let pa = p - a;
  let ba = b - a;
  let h = clamp(dot(pa, ba) / max(dot(ba, ba), 1e-20), 0.0, 1.0);
  return length(pa - ba * h);
}

fn sd_capsule_2d(p: vec2f, a: vec2f, b: vec2f, radius: f32) -> f32 {
  return sd_segment_2d(p, a, b) - radius;
}

// ---- 3D primitives ------------------------------------------------------------------------------

fn sd_sphere(p: vec3f, radius: f32) -> f32 {
  return length(p) - radius;
}

fn sd_box_3d(p: vec3f, half_size: vec3f) -> f32 {
  let q = abs(p) - half_size;
  return length(max(q, vec3f(0.0))) + min(max(q.x, max(q.y, q.z)), 0.0);
}

fn sd_rounded_box_3d(p: vec3f, half_size: vec3f, radius: f32) -> f32 {
  let r = clamp(radius, 0.0, min(half_size.x, min(half_size.y, half_size.z)));
  return sd_box_3d(p, half_size - vec3f(r)) - r;
}

fn sd_capsule_3d(p: vec3f, a: vec3f, b: vec3f, radius: f32) -> f32 {
  let pa = p - a;
  let ba = b - a;
  let h = clamp(dot(pa, ba) / max(dot(ba, ba), 1e-20), 0.0, 1.0);
  return length(pa - ba * h) - radius;
}

fn sd_torus(p: vec3f, major: f32, minor: f32) -> f32 {
  return length(vec2f(length(p.xz) - major, p.y)) - minor;
}

// ---- Combinators --------------------------------------------------------------------------------

fn sd_union(a: f32, b: f32) -> f32 {
  return min(a, b);
}

fn sd_intersect(a: f32, b: f32) -> f32 {
  return max(a, b);
}

fn sd_subtract(a: f32, b: f32) -> f32 {
  return max(a, -b);
}

fn sd_round(d: f32, r: f32) -> f32 {
  return d - r;
}

fn sd_onion(d: f32, thickness: f32) -> f32 {
  return abs(d) - thickness;
}

fn sd_coverage(d: f32, aa: f32) -> f32 {
  return clamp(0.5 - d / max(aa, 1e-12), 0.0, 1.0);
}

// ---- Smooth minimum / maximum -------------------------------------------------------------------

// value = min - k h^2 / 4 with h = max(k - |a-b|, 0) / k.
fn smin_poly(a: f32, b: f32, k: f32) -> f32 {
  let kk = max(k, 1e-12);
  let h = max(kk - abs(a - b), 0.0) / kk;
  return min(a, b) - 0.25 * kk * h * h;
}

fn smin_poly_blend(a: f32, b: f32, k: f32) -> vec2f {
  let kk = max(k, 1e-12);
  let h = max(kk - abs(a - b), 0.0) / kk;
  let w = select(0.5 * h, 1.0 - 0.5 * h, b < a); // weight of b
  return vec2f(min(a, b) - 0.25 * kk * h * h, w);
}

// value = min - k h^3 / 6.
fn smin_cubic(a: f32, b: f32, k: f32) -> f32 {
  let kk = max(k, 1e-12);
  let h = max(kk - abs(a - b), 0.0) / kk;
  return min(a, b) - kk * h * h * h / 6.0;
}

fn smin_cubic_blend(a: f32, b: f32, k: f32) -> vec2f {
  let kk = max(k, 1e-12);
  let h = max(kk - abs(a - b), 0.0) / kk;
  let w = select(0.5 * h * h, 1.0 - 0.5 * h * h, b < a);
  return vec2f(min(a, b) - kk * h * h * h / 6.0, w);
}

// value = -k ln(exp(-a/k) + exp(-b/k)), evaluated in an overflow-free form.
fn smin_exp(a: f32, b: f32, k: f32) -> f32 {
  let kk = max(k, 1e-12);
  return min(a, b) - kk * log(1.0 + exp(-abs(a - b) / kk));
}

fn smin_exp_blend(a: f32, b: f32, k: f32) -> vec2f {
  let kk = max(k, 1e-12);
  let e = exp(-abs(a - b) / kk);
  let small = e / (1.0 + e); // weight of the larger input, in (0, 0.5]
  let w = select(small, 1.0 - small, b < a);
  return vec2f(min(a, b) - kk * log(1.0 + e), w);
}

fn smax_poly(a: f32, b: f32, k: f32) -> f32 {
  return -smin_poly(-a, -b, k);
}

fn smax_cubic(a: f32, b: f32, k: f32) -> f32 {
  return -smin_cubic(-a, -b, k);
}

fn smax_exp(a: f32, b: f32, k: f32) -> f32 {
  return -smin_exp(-a, -b, k);
}

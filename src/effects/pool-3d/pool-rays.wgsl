// pool-rays.wgsl -- the analytic ray helpers of the 3D pool: ray against a sphere, and where a ray that
// starts inside the open box leaves it. No bindings, no includes: a test (tests/pool-3d.test.ts) runs
// these on the GPU against their TypeScript twins in maths.ts. World space, +Y up.
//
//   pl_ray_sphere(o, d, c, r) -> f32     distance to where the ray ENTERS the sphere, or -1 for a miss, a
//                                        sphere behind the origin, or an origin already inside it. d is unit.
//   pl_exit(o, d, y_lo, y_hi) -> PlExit  where a ray starting inside |x|, |z| <= POOL_HALF, y_lo <= y <= y_hi
//                                        leaves the box: t (distance) and n (that face's normal, pointing in).
//                                        A face the ray runs parallel to is never chosen.
const POOL_HALF: f32 = 1.0;
const PL_INF: f32 = 1e9;

struct PlExit {
  t: f32,
  n: vec3f,
}

fn pl_ray_sphere(o: vec3f, d: vec3f, c: vec3f, r: f32) -> f32 {
  let oc = o - c;
  let b = dot(oc, d);
  let k = dot(oc, oc) - r * r;
  let disc = b * b - k;
  if (disc < 0.0) {
    return -1.0;
  }
  let t = -b - sqrt(disc);
  if (t <= 1e-4) {
    return -1.0;
  }
  return t;
}

fn pl_exit(o: vec3f, d: vec3f, y_lo: f32, y_hi: f32) -> PlExit {
  var t = PL_INF;
  var n = vec3f(0.0, 1.0, 0.0);
  if (d.x > 1e-6) {
    let s = (POOL_HALF - o.x) / d.x;
    if (s < t) { t = s; n = vec3f(-1.0, 0.0, 0.0); }
  } else if (d.x < -1e-6) {
    let s = (-POOL_HALF - o.x) / d.x;
    if (s < t) { t = s; n = vec3f(1.0, 0.0, 0.0); }
  }
  if (d.z > 1e-6) {
    let s = (POOL_HALF - o.z) / d.z;
    if (s < t) { t = s; n = vec3f(0.0, 0.0, -1.0); }
  } else if (d.z < -1e-6) {
    let s = (-POOL_HALF - o.z) / d.z;
    if (s < t) { t = s; n = vec3f(0.0, 0.0, 1.0); }
  }
  if (d.y > 1e-6) {
    let s = (y_hi - o.y) / d.y;
    if (s < t) { t = s; n = vec3f(0.0, -1.0, 0.0); }
  } else if (d.y < -1e-6) {
    let s = (y_lo - o.y) / d.y;
    if (s < t) { t = s; n = vec3f(0.0, 1.0, 0.0); }
  }
  return PlExit(max(t, 0.0), n);
}

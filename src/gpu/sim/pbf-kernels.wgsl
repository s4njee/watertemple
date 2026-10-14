// pbf-kernels.wgsl -- the smoothing kernels of the position-based fluid solver (story 7.4), as pure functions with
// every constant passed in, so the tests can evaluate them on the GPU and compare with the CPU twin in pbf.ts.
// Not part of the shared library (src/wgsl): the SPH solver keeps its own kernels in sph-kernels.wgsl, and the
// two could be merged once both have settled; every name here starts with `pbf_` so they cannot collide.
//
// Macklin and Mueller, "Position Based Fluids" (2013), use Poly6 to estimate density and the gradient of Spiky to
// move particles. Both are normalised to integrate to 1 over the disc (2D) or ball (3D) of radius h, so a sum of
// kernel values over neighbours is a NUMBER density (particles per unit area or volume). The normalisations differ by
// dimension and nothing else does:
//
//   Poly6    W(r) = c6 (h^2 - r^2)^3                c6 = 315 / (64 pi h^9)  (3D)    4 / (pi h^8)  (2D)
//   Spiky    W(r) = cs (h - r)^3                    cs = 15 / (pi h^6)      (3D)    10 / (pi h^5) (2D)
//   Spiky gradient, with respect to the first particle, for r = p_i - p_j:
//            grad W = -3 cs (h - |r|)^2 r / |r|     3 cs = 45 / (pi h^6)    (3D)    30 / (pi h^5) (2D)
//
// pbf_poly6(r2, h2, c6) -> f32                      Poly6 value at squared distance r2 (0 beyond h)
// pbf_spiky_grad(r, h, c3) -> vec3f                 Spiky gradient for the separation r, with c3 = 3 cs; zero
//                                                   beyond h and for coincident particles (the caller nudges those)
// pbf_nudge(i, j, h) -> vec3f                       a tiny deterministic separation for two particles in the same
//                                                   place, so the solver can push them apart instead of dividing by 0
// pbf_hash(x) -> u32                                32-bit integer hash used by pbf_nudge

fn pbf_poly6(r2: f32, h2: f32, c6: f32) -> f32 {
  let d = h2 - r2;
  return select(0.0, c6 * d * d * d, d > 0.0);
}

fn pbf_spiky_grad(r: vec3f, h: f32, c3: f32) -> vec3f {
  let len = length(r);
  let d = h - len;
  if (d <= 0.0 || len < 1e-7) {
    return vec3f(0.0);
  }
  return r * (-c3 * d * d / len);
}

fn pbf_hash(x: u32) -> u32 {
  var v = x * 747796405u + 2891336453u;
  v = ((v >> ((v >> 28u) + 4u)) ^ v) * 277803737u;
  return (v >> 22u) ^ v;
}

fn pbf_nudge(i: u32, j: u32, h: f32) -> vec3f {
  let a = pbf_hash(i * 2654435761u + j);
  let b = pbf_hash(a ^ 0x9e3779b9u);
  let c = pbf_hash(b ^ 0x85ebca6bu);
  let d = vec3f(f32(a & 0xffffu), f32(b & 0xffffu), f32(c & 0xffffu)) / 32768.0 - vec3f(1.0);
  let len = max(length(d), 1e-3);
  // A fraction of a thousandth of the radius: enough to pick a direction, far too little to be seen.
  return d / len * (h * 1e-3);
}

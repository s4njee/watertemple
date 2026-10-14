// The force field of the particles demo: a divergence-free curl-noise flow that swirls the spray, plus, in the
// 2D scene, a vortex around the pointer. Included by the integrator as `force_field.wgsl` (see
// src/gpu/particles/integrator.ts); it returns an acceleration.
//
//   params.user0: x flow strength (acceleration per unit of curl), y noise frequency (lattice cells per world
//                 unit), z drift speed (lattice cells per second the noise scrolls)
//   params.user1: xy pointer position in the world, w swirl strength of the pointer vortex (2D only)

#include "curl.wgsl"

fn force_field(p: vec3f, v: vec3f, time: f32, index: u32) -> vec3f {
  let flow = params.user0.x;
  let k = params.user0.y;
  let drift = params.user0.z;
  if (params.dimension == 2u) {
    var a = vec3f(curl_2d_anim(p.xy * k, time * drift, 2, 2.0, 0.5) * flow, 0.0);
    let d = p.xy - params.user1.xy;
    // A vortex around the pointer with a softened core, so nothing is flung out at r = 0.
    a += vec3f(-d.y, d.x, 0.0) * (params.user1.w / (dot(d, d) + 0.05));
    return a;
  }
  return curl_3d(p * k + vec3f(0.0, drift * time, 0.0)) * flow;
}

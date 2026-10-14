// caustics-jacobian.wgsl -- caustics without a mesh: a per-pixel BACKWARD estimate from the
// slope field's Jacobian. One fullscreen pass over the floor texture.
//
// The surface -> floor map is q(p) = p + D(p), where D is how far the refracted ray travels sideways
// before it reaches the floor (caustics_displacement). Light from a patch of surface of area dA lands
// on a patch of floor of area |det J| dA, J = I + dD/dp, so the floor is brighter by 1 / |det J|.
// For each floor pixel q we need the surface point p that sends light there (a short fixed-point
// iteration p <- q - D(p), exact to first order; D is small compared with the features, so two or
// three rounds do) and the Jacobian there (central differences of D, four more taps). About 8 taps
// and 8 refractions per pixel, and nothing else: no vertices, no blending, no derivatives.
//
// What it cannot do: the formula sees ONE ray per floor point. Past the first focus the map folds and
// several surface points land on the same floor point; the mesh method adds them all, this one counts
// only the one it found. Where det J reaches 0 (a caustic) the intensity is capped by the smooth
// regulariser 1 / sqrt(det^2 + eps^2), normalised so that a flat surface (det = 1) is exactly 1.
// Measured against the mesh on a rain-stirred surface (see src/effects/ripple-caustics/notes.md): they
// agree well (correlation 0.84, mean within 1%) on gentle waves and drift apart as the surface steepens
// and folds the light, where this one adds energy (mean 1.06 at the demo's default steepness, 1.16 at
// twice that). For a wave field strong enough to fold the light the mesh is the more faithful one.
#include "fullscreen.wgsl"
#include "caustics_core.wgsl"

fn jacobian_intensity(q: vec2f, ior: f32) -> f32 {
  var p = q;
  for (var k = 0u; k < P.iterations; k++) {
    p = q - caustics_displacement(caustics_surface(p), ior);
  }
  let inside = all(p >= vec2f(0.0)) && all(p <= vec2f(1.0));
  if (P.extend < 0.5 && !inside) {
    return 0.0;
  }

  let e = P.texel;
  let s = caustics_surface(p);
  let d_xp = caustics_displacement(caustics_surface(p + vec2f(e.x, 0.0)), ior);
  let d_xm = caustics_displacement(caustics_surface(p - vec2f(e.x, 0.0)), ior);
  let d_yp = caustics_displacement(caustics_surface(p + vec2f(0.0, e.y)), ior);
  let d_ym = caustics_displacement(caustics_surface(p - vec2f(0.0, e.y)), ior);
  let ddu = (d_xp - d_xm) / (2.0 * e.x);
  let ddv = (d_yp - d_ym) / (2.0 * e.y);
  let det = (1.0 + ddu.x) * (1.0 + ddv.y) - ddv.x * ddu.y;

  let t = caustics_transmit(s, ior);
  let z = s.x * P.cell_size * P.height_scale;
  let travel = max(z + P.depth, 0.0) / (-t.dir.z);
  let lit = caustics_lit(vec3f(p.x * P.aspect, p.y, z), t.dir, travel);

  let regular = sqrt(det * det + P.eps * P.eps);
  let value = t.weight * lit * sqrt(1.0 + P.eps * P.eps) / regular;
  return min(value, P.max_intensity);
}

@fragment fn caustics_jacobian_fs(in: FullscreenVarying) -> @location(0) vec4f {
  let q = in.uv;
  if (P.dispersive == 0u) {
    return vec4f(vec3f(jacobian_intensity(q, P.ior_rgb.y)), 1.0);
  }
  return vec4f(
    jacobian_intensity(q, P.ior_rgb.x),
    jacobian_intensity(q, P.ior_rgb.y),
    jacobian_intensity(q, P.ior_rgb.z),
    1.0,
  );
}

// normals.wgsl -- tangent-space normals: decoding, strength, layer blending, height gradients,
// and the step into world space for a horizontal water surface.
//
// Conventions
//  * A tangent-space normal is a unit vec3f with +z pointing OUT of the surface, so a flat surface is
//    (0, 0, 1). Its xy are the negated height slopes: n.xy = -(dh/dx, dh/dy) / sqrt(1 + |grad h|^2).
//    "x" and "y" are the texture's u and v axes (v grows downwards on screen, like uv everywhere here).
//  * Maps are stored as xy only, z rebuilt from sqrt(1 - |xy|^2). That is cheaper (2 channels), and
//    it makes mip chains behave: averaging xy averages the SLOPES, so a minified normal map flattens
//    with distance (shorter xy, z -> 1) instead of turning to noise. Water-ripple maps from
//    createWaterNormalMap (src/gpu/bake.ts) are signed xy in rg of an rg16float/rgba16float texture.
//
// Decode / encode
//   normal_from_xy(xy: vec2f) -> vec3f          z = sqrt(1 - |xy|^2); |xy| > 1 (filtering overshoot or
//                                               bad data) is scaled back to length 1 (z = 0)
//   normal_decode_signed(texel: vec4f) -> vec3f   texel.rg are the signed xy (float-format maps)
//   normal_decode_unorm(texel: vec4f) -> vec3f    texel.rg are xy * 0.5 + 0.5 (8-bit maps; blue is
//                                               ignored, so BC5/RG8 two-channel maps work)
//   normal_encode_unorm(n: vec3f) -> vec3f        n * 0.5 + 0.5, for writing maps and debug colours
//
// Strength
//   normal_scale(n, strength) -> vec3f          multiplies the SLOPE (xy / z) by `strength` and
//                                               renormalises: 0 gives (0, 0, 1), 1 returns n, 2 doubles
//                                               the slope. Scaling xy alone, or lerping toward
//                                               (0, 0, 1) and renormalising, are both common and
//                                               both wrong: the former overshoots |xy| > 1 at high
//                                               strength, the latter is not linear in slope.
//
// Blending two layers. `a` is the BASE (large waves), `b` the DETAIL (ripples), both unit tangent-space
// normals; the result is unit length. Apply normal_scale to each layer BEFORE blending to weight them.
//   normal_blend_linear(a, b)    normalize(a + b). The naive baseline (what averaging two normal-map
//                                RGB values does). Its z is the SUM of both z's, so every slope is halved
//                                relative to the sum: flat + detail gives half the detail, base + flat
//                                gives half the base. Detail washes out. Do not ship.
//   normal_blend_udn(a, b)       normalize(a.xy + b.xy, a.z). Unreal Developer Network blend: slopes add
//                                exactly, detail is not rotated by the base, and flat detail returns the
//                                base. Detail is slightly too steep where the base is steep. Cheapest.
//   normal_blend_whiteout(a, b)  normalize(a.xy + b.xy, a.z * b.z). Like UDN but z multiplies, which
//                                keeps the detail from turning too steep on steep bases. Flat on either
//                                side returns the other. The default choice for water.
//   normal_blend_pd(a, b)        partial-derivative blend: normalize(a.xy * b.z + b.xy * a.z, a.z * b.z).
//                                Adds the height gradients (slope = xy / z), the mathematically right sum
//                                of two heightfields. For small slopes identical to whiteout.
//   normal_blend_rnm(a, b)       reoriented normal mapping (Barre-Brisebois and Hill): rotates the
//                                detail into the base's frame, so the detail follows the base's tilt.
//                                Flat detail returns the base and flat base returns the detail, exactly.
//                                Use when the base is steep (chop, large waves) and the detail should
//                                lean with it.
//   Which to use: whiteout for almost everything; RNM for strong base waves; PD when you want the true
//   heightfield sum; UDN when the extra ALU does not matter and you want the slopes to add plainly;
//   linear only to show what goes wrong. Cost: all five are 10-20 ALU ops with one normalize (RNM the
//   most, one dot and one divide). They are cheap enough that a texture fetch costs more, so choose
//   on correctness, not speed. For small slopes the four good ones agree: within about 2% for |xy| < 0.2, 0.5% below 0.05.
//
// Height gradients
//   normal_from_gradient(grad: vec2f, strength: f32) -> vec3f
//       normal of the surface h(x, y) whose gradient is `grad` (dh/dx, dh/dy), scaled:
//       normalize(-grad * strength, 1). Feed it fbm_2d_d(...).yz or a height texture's finite
//       differences. The slope is NOT bounded: pick the height units so typical |grad| is below ~1.
//
// World space (horizontal water, up = +y, the usual y-up convention)
//   normal_to_world_up(n: vec3f) -> vec3f
//       tangent (x, y, z) -> world (x, z, y): u runs along world +x, v along world +z, the normal along
//       world +y. This is the exact mapping for the surface y = h(x, z), which is a mirror image of a
//       rotation, so it is NOT the cross-product frame (that would flip the tilt direction).
//   normal_to_world(n: vec3f, tangent: vec3f, bitangent: vec3f, normal: vec3f) -> vec3f
//       general frame: n.x * tangent + n.y * bitangent + n.z * normal, normalised. Use it for a
//       sloped surface or a mesh with real tangents.
//   normal_yaw(n: vec3f, angle: f32) -> vec3f
//       rotates the tangent frame about the surface normal by `angle` radians. Needed whenever a
//       layer's uv is rotated: the uv rotation turns the pattern, so the normals stored in it must
//       be turned too, or the lighting of the rotated layer is wrong.
//
// Technique: the blends are those compared in Colin Barre-Brisebois and Stephen Hill, "Blending in
// Detail" (2012): UDN, whiteout, partial derivatives and reoriented normal mapping. Code written
// from the published formulas.

fn normal_from_xy(xy: vec2f) -> vec3f {
  let l2 = dot(xy, xy);
  // Past the unit disc there is no real z: pull xy back onto the circle rather than return NaN.
  let k = select(1.0, inverseSqrt(l2), l2 > 1.0);
  let v = xy * k;
  return vec3f(v, sqrt(max(1.0 - dot(v, v), 0.0)));
}

fn normal_decode_signed(texel: vec4f) -> vec3f {
  return normal_from_xy(texel.rg);
}

fn normal_decode_unorm(texel: vec4f) -> vec3f {
  return normal_from_xy(texel.rg * 2.0 - 1.0);
}

fn normal_encode_unorm(n: vec3f) -> vec3f {
  return n * 0.5 + 0.5;
}

fn normal_scale(n: vec3f, strength: f32) -> vec3f {
  // Slope is xy / z, so scaling the slope by s is (s * xy, z) up to length.
  return normalize(vec3f(n.xy * max(strength, 0.0), max(n.z, 1e-4)));
}

fn normal_blend_linear(a: vec3f, b: vec3f) -> vec3f {
  return normalize(a + b);
}

fn normal_blend_udn(a: vec3f, b: vec3f) -> vec3f {
  return normalize(vec3f(a.xy + b.xy, a.z));
}

fn normal_blend_whiteout(a: vec3f, b: vec3f) -> vec3f {
  return normalize(vec3f(a.xy + b.xy, a.z * b.z));
}

fn normal_blend_pd(a: vec3f, b: vec3f) -> vec3f {
  return normalize(vec3f(a.xy * b.z + b.xy * a.z, a.z * b.z));
}

fn normal_blend_rnm(a: vec3f, b: vec3f) -> vec3f {
  let t = vec3f(a.x, a.y, a.z + 1.0);
  let u = vec3f(-b.x, -b.y, b.z);
  // t.z = a.z + 1 >= 1 for any upward-facing base; guard the divide for a degenerate one.
  return normalize(t * (dot(t, u) / max(t.z, 1e-4)) - u);
}

fn normal_from_gradient(grad: vec2f, strength: f32) -> vec3f {
  return normalize(vec3f(-grad * strength, 1.0));
}

fn normal_to_world_up(n: vec3f) -> vec3f {
  return vec3f(n.x, n.z, n.y);
}

fn normal_to_world(n: vec3f, tangent: vec3f, bitangent: vec3f, normal: vec3f) -> vec3f {
  return normalize(n.x * tangent + n.y * bitangent + n.z * normal);
}

fn normal_yaw(n: vec3f, angle: f32) -> vec3f {
  let s = sin(angle);
  let c = cos(angle);
  return vec3f(c * n.x - s * n.y, s * n.x + c * n.y, n.z);
}

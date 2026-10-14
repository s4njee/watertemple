// pigment-water.wgsl -- adds water: capsule splats of wetness, the brush of the pigment layer
// (src/gpu/sim/pigment.ts). The same capsule footprint as the solver's splats (fluid-core.wgsl): a
// segment a -> b in uv inflated by a Gaussian of radius `shape.x` image heights, with a smooth start
// and end ramp `shape.y`, so segments laid end to end add up to a uniform stroke.
//
// Bindings: 0 W (WaterUniforms), 1 wet_in (r32float), 2 splats (read-only storage), 3 wet_out
// (r32float storage). Entry point cs_water: wet_out = min(1, wet_in + the splats' water). An amount
// above 1 makes a flat-topped film with a steeper edge.
//
// RAGGED EDGES. Paper does not wet evenly, so a film's outline is never a circle. The water a splat
// adds at a point is scaled by 1 + W.ragged * n, with n two octaves of gradient noise of the canvas (fixed
// to the sheet, the same for every splat, about 12 and 30 features per image height): where the paper takes
// water more readily the film reaches further. 0 gives the plain Gaussian capsules.

#include "noise.wgsl"
#include "water_uniforms.wgsl"

struct Splat {
  seg: vec4f,
  shape: vec4f,
  value: vec4f,
}

@group(0) @binding(0) var<uniform> W: WaterUniforms;
@group(0) @binding(1) var wet_in: texture_2d<f32>;
@group(0) @binding(2) var<storage, read> splats: array<Splat>;
@group(0) @binding(3) var wet_out: texture_storage_2d<r32float, write>;

fn splat_weight(uv: vec2f, s: Splat) -> f32 {
  let k = vec2f(W.aspect, 1.0);
  let a = s.seg.xy * k;
  let ab = s.seg.zw * k - a;
  let p = uv * k;
  let len = length(ab);
  let ramp = s.shape.y;
  var along = 1.0;
  var d = length(p - a);
  if (len > 1e-6) {
    let dir = ab / len;
    let rel = p - a;
    let t = dot(rel, dir);
    if (ramp > 0.0) {
      along = smoothstep(-ramp, ramp, t) * (1.0 - smoothstep(len - ramp, len + ramp, t));
      d = length(rel - dir * t);
    } else {
      d = length(rel - dir * clamp(t, 0.0, len));
    }
  }
  let x = d / max(s.shape.x, 1e-5);
  if (x > 3.0) {
    return 0.0;
  }
  return along * exp(-x * x);
}

@compute @workgroup_size(16, 16)
fn cs_water(@builtin(global_invocation_id) id: vec3u) {
  if (any(id.xy >= W.size)) {
    return;
  }
  let t = vec2i(id.xy);
  let uv = (vec2f(t) + 0.5) / vec2f(W.size);
  var w = textureLoad(wet_in, t, 0).x;
  var rough = 1.0;
  if (W.ragged != 0.0) {
    let p = uv * vec2f(W.aspect, 1.0);
    rough = max(1.0 + W.ragged * (0.7 * gnoise_2d(p * 12.0 + vec2f(3.1, 7.7)) + 0.4 * gnoise_2d(p * 30.0 + vec2f(9.3, 1.9))), 0.0);
  }
  for (var k = 0u; k < W.count; k++) {
    let s = splat_weight(uv, splats[k]);
    w += s * splats[k].value.x * rough;
  }
  textureStore(wet_out, t, vec4f(clamp(w, 0.0, 1.0), 0.0, 0.0, 0.0));
}

#include "fullscreen.wgsl"
#include "color.wgsl"
#include "sdf.wgsl"
#include "mycelium_params.wgsl"
@group(0) @binding(0) var<uniform> P: MyceliumParams;
@group(0) @binding(1) var trail: texture_2d<f32>;
@group(0) @binding(2) var samp: sampler;
@group(0) @binding(3) var walls: texture_2d<f32>;
@fragment fn fs(in: FullscreenVarying) -> @location(0) vec4f {
  let uv = in.uv;
  let ink = textureSampleLevel(trail, samp, uv, 0);
  let texel = 1.0 / vec2f(P.grid);
  var soft = vec3f(0);
  for (var i = 0u; i < 4u; i++) {
    let angle = f32(i) * 1.5707963;
    soft += textureSampleLevel(trail, samp, uv + vec2f(cos(angle),sin(angle)) * texel * 3.0, 0).rgb;
  }
  var emission = ink.rgb + soft * 0.055;
  // A saturated neon finish even where differently colored networks join.
  emission -= vec3f(min(emission.r, min(emission.g, emission.b)) * 0.82);
  let peak = max(emission.r, max(emission.g, emission.b));
  emission *= 0.72 * (1.0 - exp(-peak * P.exposure / 0.72)) / max(peak, 1e-5);
  let p = uv * vec2f(P.aspect,1.0);
  var distance = 2.0;
  for (var i = 0u; i < P.rect_count; i++) {
    let box = P.boxes[i];
    distance = min(distance, sd_rounded_box_2d(p-box.xy,box.zw,P.extras[i].x));
  }
  let clear = clamp(0.5 + distance / P.pixel, 0.0, 1.0);
  let ground = vec3f(0.001518, 0.002125, 0.005605); // #05070f
  // The colony continues beyond this window; avoid a luminous frame at the viewport edge.
  let edge = min(min(uv.x, 1.0-uv.x), min(uv.y, 1.0-uv.y));
  emission *= smoothstep(0.0, 0.055, edge);
  let vignette = 1.0 - 0.18 * smoothstep(0.3,0.9,length(uv-0.5));
  return vec4f(linear_to_srgb((ground + emission * clear * vignette) * P.light), 1);
}

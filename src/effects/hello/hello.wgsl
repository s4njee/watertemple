#include "fullscreen.wgsl"
#include "fbm.wgsl"
#include "color.wgsl"
#include "params.wgsl"

@group(0) @binding(0) var<uniform> params: Params;

@fragment fn fs(in: FullscreenVarying) -> @location(0) vec4f {
  let p = vec2f(in.uv.x * params.aspect, in.uv.y) * params.scale;
  let warped = domain_warp_2d_anim(p, params.time * 0.2, params.warp, 3, 2.0, 0.5);
  let n = 0.5 + 0.5 * fbm_3d(vec3f(warped, params.time * 0.15), 5, 2.0, 0.5);
  let crest = ridged_2d(warped * 1.7 + 3.0, 3, 2.0, 0.5);

  let base = palette_cosine(n, params.tint, vec3f(0.1, 0.25, 0.3), vec3f(1.0), vec3f(0.55, 0.6, 0.7));
  let lit = base + vec3f(0.9, 0.95, 1.0) * pow(crest, 6.0) * 0.5;
  return vec4f(linear_to_srgb(tonemap_aces(lit)), 1.0);
}

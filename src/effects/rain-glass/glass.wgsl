// The window pane. Per pixel: one droplet_field evaluation (shared chunk droplets.wgsl), then
//   - outside any drop: the BLURRED scene (one bilinear tap of a tiny texture) with a milky veil;
//   - along a drop's trail: the glass is wiped, so the SHARP scene shows through;
//   - inside a drop: the sharp scene seen through a lens, so it is flipped, plus a rim and a glint.
// Only drop and trail pixels take the second (mip-mapped) tap.
#include "fullscreen.wgsl"
#include "color.wgsl"
#include "refraction.wgsl"
#include "droplets.wgsl"
#include "glass_params.wgsl"

@group(0) @binding(0) var<uniform> P: GlassParams;
@group(0) @binding(1) var sharp_tex: texture_2d<f32>;
@group(0) @binding(2) var blur_tex: texture_2d<f32>;
@group(0) @binding(3) var samp: sampler;

const LIGHT: vec3f = vec3f(-0.42, -0.62, 0.66);

@fragment fn fs(in: FullscreenVarying) -> @location(0) vec4f {
  let uv = in.uv;
  let p = vec2f(uv.x * P.aspect, uv.y);
  let s = droplet_field(p, P.time, P.rain, P.size, P.layers, P.aa);

  // Misted glass: blurred scene, a little desaturated and lifted. Little fog shows more of the sharp scene.
  let blurred = textureSampleLevel(blur_tex, samp, uv, 0.0).rgb;
  var col = mix(blurred, vec3f(luminance(blurred)), 0.25 * P.fog) + vec3f(0.006, 0.007, 0.009) * P.fog;
  if (P.fog < 0.3) {
    let clear_view = textureSampleLevel(sharp_tex, samp, uv, P.sharp_lod).rgb;
    col = mix(clear_view, col, smoothstep(0.0, 0.3, P.fog));
  }

  if (s.clear > 0.003 || s.mask > 0.0) {
    // Wiped trail: clean glass, the sharp scene.
    if (s.mask < 0.999) {
      let trail = textureSampleLevel(sharp_tex, samp, uv, P.sharp_lod).rgb;
      col = mix(col, trail, s.clear);
    }
    if (s.mask > 0.0) {
      // The drop is a lens: the offset is -local * radius * strength, so the picture is flipped.
      let lens_uv = droplet_lens(s, P.refraction) / vec2f(P.aspect, 1.0);
      let seen = textureSampleLevel(sharp_tex, samp, refraction_mirror_uv(uv + lens_uv), P.sharp_lod).rgb;
      let n = droplet_normal(s, 0.85);
      let rim = droplet_rim(s);
      var drop = seen * (1.15 - 0.45 * rim);
      // Light from behind collects in a crescent along the lower edge; a window glints on the upper left.
      let crescent = smoothstep(0.55, 1.0, length(s.local)) * smoothstep(0.0, 0.9, s.local.y);
      drop += blurred * 0.9 * crescent;
      drop += vec3f(1.0, 0.97, 0.92) * droplet_highlight(n, normalize(LIGHT), 80.0) * 0.9;
      col = mix(col, drop, s.mask);
    }
  }

  let mapped = tonemap_aces(col * P.exposure);
  return vec4f(linear_to_srgb(mapped), 1.0);
}

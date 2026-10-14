// Water behind a real page: the ripples surface (refraction of a backdrop through the live heightfield
// plus a sun glint), cut away where the page's own elements are, with a contact edge where water meets
// them.
//
// WHERE THE ELEMENTS ARE. The solver's obstacle mask is one texel per grid cell, so its edge is only as
// sharp as the grid (about 6 CSS pixels per cell at 256 cells over 1500 px). The physics lives with
// that. This pass instead draws the elements from the exact rounded rectangles in the uniform block
// (`boxes` / `radii`, in image heights with x scaled by the aspect), so the cut-out and the contact
// edge land on the real element to the pixel. The mask is read only by the `mask` debug view, which
// overlays it (red) and the exact outlines (green) on the water.
//
// UNDER THE ELEMENT. Alpha is 0 there (premultiplied), so the page's own background shows through
// and the element looks exactly as it would without a canvas: nothing drawn under it fights its
// colour. Its edge is anti-aliased by one pixel.
//
// THE CONTACT EDGE (`edge` 0..1). Outside a wall, water gets (a) a soft darkening that falls off over
// about 0.014 image heights, like ambient occlusion in the corner between water and wall, and (b) a thin
// bright meniscus line two or three pixels from it, brighter where a crest is arriving (positive
// height), so a wave that hits the wall lights its edge for a moment.
#include "fullscreen.wgsl"
#include "heightfield.wgsl"
#include "refraction.wgsl"
#include "specular.wgsl"
#include "sdf.wgsl"
#include "color.wgsl"
#include "water_params.wgsl"

@group(0) @binding(0) var<uniform> P: WaterParams;
@group(0) @binding(1) var backdrop_tex: texture_2d<f32>;
@group(0) @binding(2) var backdrop_samp: sampler;
@group(0) @binding(3) var field_tex: texture_2d<f32>;
@group(0) @binding(4) var field_samp: sampler;
@group(0) @binding(5) var mask_tex: texture_2d<f32>;

const VIEW_WATER: u32 = 0u;
const VIEW_MASK: u32 = 1u;
const VIEW_HEIGHT: u32 = 2u;
const VIEW_NORMALS: u32 = 3u;

const LIGHT: vec3f = vec3f(-0.4, -0.5, 0.77);
const CONTACT_WIDTH: f32 = 0.014;

fn diverging(x: f32) -> vec3f {
  let t = clamp(x, -1.0, 1.0);
  let cold = vec3f(0.1, 0.3, 0.9);
  let hot = vec3f(0.95, 0.25, 0.1);
  return mix(vec3f(0.5), select(cold, hot, t > 0.0), abs(t));
}

// Signed distance (image heights; negative inside) to the nearest page element.
fn page_distance(p: vec2f) -> f32 {
  var d = 1.0e6;
  for (var i = 0u; i < P.rect_count; i++) {
    let b = P.boxes[i];
    d = min(d, sd_rounded_box_2d(p - b.xy, b.zw, P.radii[i].x));
  }
  return d;
}

@fragment fn fs(in: FullscreenVarying) -> @location(0) vec4f {
  let uv = in.uv;
  let p = vec2f(uv.x * P.aspect, uv.y);
  let aa = P.pixel;
  let d = page_distance(p);

  let s = heightfield_sample(field_tex, field_samp, uv);
  let n = heightfield_normal(s, P.strength);

  let offset_world = refraction_offset_physical(n, vec3f(0.0, 0.0, -1.0), P.ior, P.thickness);
  let offset_uv = refraction_to_uv(offset_world, P.aspect);

  var colour = vec3f(0.0);
  if (P.dispersion < 0.0005) {
    colour = refraction_sample(backdrop_tex, backdrop_samp, uv, offset_uv, 0.0, P.fade);
  } else {
    let scales = refraction_dispersion_scales(P.ior, P.dispersion);
    colour = refraction_sample_rgb(
      backdrop_tex, backdrop_samp, uv, refraction_rgb_offsets(offset_uv, scales), 0.0, P.fade,
    );
  }

  colour *= mix(vec3f(1.0), P.tint, saturate(P.thickness * 1.6));
  colour *= 1.0 + P.relief * 4.0 * dot(n.xy, normalize(LIGHT.xy)) * length(LIGHT.xy);
  let light = normalize(LIGHT);
  let glint = spec_sun(n, vec3f(0.0, 0.0, 1.0), light, 0.04, 0.0093);
  colour += vec3f(1.0, 0.97, 0.9) * glint * P.shine * 30.0;

  // Contact edge, outside the walls only.
  if (P.edge > 0.0 && d > 0.0) {
    let shadow = exp(-d / CONTACT_WIDTH);
    colour *= 1.0 - 0.45 * P.edge * shadow;
    let q = d / (2.5 * aa);
    let rim = exp(-q * q);
    let crest = 0.45 + 1.6 * saturate(heightfield_height(s) * P.height_gain);
    colour += vec3f(0.85, 0.95, 1.0) * (0.28 * P.edge * rim * crest);
  }

  // Coverage of water: 0 under an element, 1 outside, a one pixel ramp on the edge.
  var alpha = clamp(0.5 + d / aa, 0.0, 1.0);

  if (P.view == VIEW_MASK) {
    let dims = vec2f(textureDimensions(mask_tex));
    let cell = vec2i(clamp(uv * dims, vec2f(0.0), dims - vec2f(1.0)));
    let solid = textureLoad(mask_tex, cell, 0).r;
    colour = mix(colour, vec3f(1.0, 0.1, 0.1), 0.5 * solid);
    colour = mix(colour, vec3f(0.1, 1.0, 0.3), 1.0 - smoothstep(0.4 * aa, 1.1 * aa, abs(d)));
    alpha = 1.0;
  } else if (P.view == VIEW_HEIGHT) {
    // Square-root scale: faint, late waves stay visible next to the first splash.
    let h = heightfield_height(s) * P.height_gain;
    colour = srgb_to_linear(diverging(sign(h) * sqrt(abs(h))));
  } else if (P.view == VIEW_NORMALS) {
    colour = srgb_to_linear(n * 0.5 + 0.5);
  }
  if (P.view != VIEW_WATER && P.view != VIEW_MASK) {
    alpha = 1.0;
  }
  return vec4f(linear_to_srgb(clamp(colour, vec3f(0.0), vec3f(1.0))) * alpha, alpha);
}

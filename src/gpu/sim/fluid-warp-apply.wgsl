// fluid-warp-apply.wgsl -- draws content (an image, a video frame, a headline rendered to a texture)
// through a displacement field from src/gpu/sim/fluid-warp.ts, to the canvas. Pure shading: all the
// physics happened in the field.
//
//   colour(uv) = content(fit(uv - D(uv)))     see fluid-warp-core.wgsl for why it is a backward map
//
// REST. Where D is exactly 0 every extra term below is exactly 0 too (the stretch, the film slope and
// the channel split all vanish with D), the source uv is returned untouched, and the output is the
// content sampled at the pixel's own centre: pixel-identical to drawing the content (the test renders
// it and compares bytes), so type stays legible whenever the fluid is calm.
//
// LOOK.
//   chroma  splits the channels along the displacement in proportion to the local STRETCH (the norm of
//           the Jacobian of D): where the content is smeared the red and blue copies part, in a pure
//           translation they do not. 0 takes a single tap.
//   sheen   a liquid film: |D| is treated as a thickness, its gradient tilts a normal, and a soft
//           shade plus a small glint show where the content has been dragged, like wet paint.
//   fade    how far from a border the offset is faded out (uv): the border pixel never smears.
// DEBUG VIEWS (P.view): 0 content, 1 displacement (hue is direction, brightness magnitude), 2 grid (the
// content's own coordinates drawn as lines, so the fold of the map is visible), 3 velocity of the solver.

#include "fullscreen.wgsl"
#include "color.wgsl"
#include "media.wgsl"
#include "fluid.wgsl"
#include "fluid-warp-core.wgsl"
#include "warp_apply_params.wgsl"

@group(0) @binding(0) var<uniform> P: WarpApplyParams;
@group(0) @binding(1) var content: texture_2d<f32>;
@group(0) @binding(2) var content_samp: sampler;
@group(0) @binding(3) var field: texture_2d<f32>;
@group(0) @binding(4) var field_samp: sampler;
@group(0) @binding(5) var fields: texture_2d<f32>;

const VIEW_CONTENT: u32 = 0u;
const VIEW_DISPLACEMENT: u32 = 1u;
const VIEW_GRID: u32 = 2u;
const VIEW_VELOCITY: u32 = 3u;

const LIGHT: vec3f = vec3f(-0.45, -0.55, 0.7);

// The content at a point of the canvas (canvas uv), with the bars of a `contain` fit.
fn content_at(canvas_uv: vec2f, lod: f32) -> vec3f {
  let muv = media_fit_uv(canvas_uv, P.fit);
  let c = textureSampleLevel(content, content_samp, clamp(muv, vec2f(0.0), vec2f(1.0)), lod).rgb;
  return mix(P.bars, c, media_inside(muv, 0.0));
}

fn grid_lines(canvas_uv: vec2f) -> vec3f {
  let p = canvas_uv * vec2f(P.aspect, 1.0) * 10.0;
  let f = abs(fract(p - 0.5) - 0.5) / max(fwidth(p), vec2f(1e-4));
  let line = 1.0 - min(min(f.x, f.y), 1.0);
  let major = select(0.0, 1.0, all(abs(p - round(p / 5.0) * 5.0) < vec2f(0.5)));
  return mix(vec3f(0.02), vec3f(0.9, 0.95, 1.0), line) * (0.7 + 0.3 * major);
}

@fragment fn fs(in: FullscreenVarying) -> @location(0) vec4f {
  let uv = in.uv;
  var d = vec2f(0.0);
  if (P.cubic != 0u) {
    d = fwarp_fetch_cubic(field, field_samp, uv, P.field_size);
  } else {
    d = fwarp_fetch(field, field_samp, uv);
  }
  let lod = media_lod(P.media_px, P.fit, P.canvas_px);

  var colour = vec3f(0.0);
  let at_rest = d.x == 0.0 && d.y == 0.0;
  var g = mat2x2f(vec2f(0.0), vec2f(0.0));
  if (!at_rest) {
    g = fwarp_gradient(field, field_samp, uv, P.field_size, P.aspect);
  }
  let stretch = fwarp_stretch(g);

  if (P.view == VIEW_CONTENT || P.view == VIEW_GRID) {
    let split = P.chroma * min(stretch, 1.5);
    var src = fwarp_source_uv(uv, fwarp_to_uv(d, P.aspect), P.fade);
    if (P.view == VIEW_GRID) {
      colour = grid_lines(src);
    } else if (split > 0.0) {
      let r_uv = fwarp_source_uv(uv, fwarp_to_uv(d * (1.0 + split), P.aspect), P.fade);
      let b_uv = fwarp_source_uv(uv, fwarp_to_uv(d * (1.0 - split), P.aspect), P.fade);
      colour = vec3f(content_at(r_uv, lod).r, content_at(src, lod).g, content_at(b_uv, lod).b);
    } else {
      colour = content_at(src, lod);
    }

    if (P.sheen > 0.0 && !at_rest) {
      let slope = fwarp_film_slope(g, d) * P.sheen * 3.0;
      let n = normalize(vec3f(-slope, 1.0));
      let l = normalize(LIGHT);
      let h = normalize(l + vec3f(0.0, 0.0, 1.0));
      let gate = smoothstep(0.0, 0.05, length(slope));
      let glint = pow(max(dot(n, h), 0.0), 40.0) * gate;
      let shade = dot(n.xy, l.xy);
      colour = colour * (1.0 + 0.8 * shade) + vec3f(glint * 0.35);
    }
  } else if (P.view == VIEW_DISPLACEMENT) {
    colour = vec3f(0.012) + fluid_flow_color(d, P.debug_gain * 12.0);
  } else {
    let s = fluid_sample_fields(fields, field_samp, uv);
    colour = vec3f(0.012) + fluid_flow_color(fluid_velocity(s), 2.0 * P.debug_gain);
  }
  return vec4f(linear_to_srgb(clamp(colour, vec3f(0.0), vec3f(1.0))), 1.0);
}

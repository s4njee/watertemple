// media.wgsl -- fitting an image or video into a canvas of a different shape.
//
// An image or video has its own aspect ratio and the canvas has another. These functions turn canvas uv
// (0..1, top-left origin, as in fullscreen.wgsl) into media uv (0..1 over the media, top-left origin)
// the way CSS `object-fit` does:
//   cover    the media fills the canvas and the overflowing part is cropped. Media uv never leaves [0, 1].
//   contain  the whole media is visible and the canvas shows bars at the sides or top and bottom. Media
//            uv leaves [0, 1] over the bars; media_inside() says where the media is.
// The fit is a pure scale about the centre, so it is described by one vec2f computed once per frame on
// the CPU (media.ts `fitScale` computes the same numbers) and a single mad per pixel.
//
//   media_fit_scale(canvas_aspect: f32, media_aspect: f32, contain: bool) -> vec2f
//       aspects are width / height. Returns s such that media_uv = (uv - 0.5) * s + 0.5. Cover has
//       s <= 1 on one axis and exactly 1 on the other (the media is cropped along its longer side);
//       contain has s >= 1 on one axis and exactly 1 on the other (bars along the shorter side). With
//       r = canvas_aspect / media_aspect: r > 1 gives cover (1, 1/r) and contain (r, 1); r < 1 gives cover
//       (r, 1) and contain (1, 1/r). Equal aspects give (1, 1) either way.
//   media_fit_uv(uv: vec2f, s: vec2f) -> vec2f
//       media_uv for a canvas uv. The centre maps to the centre.
//   media_offset_to_media(offset_uv: vec2f, s: vec2f) -> vec2f
//       a displacement in canvas uv expressed in media uv (offset * s). Distort in canvas space, then
//       apply the fit, so the distortion has the same size on screen for either fit.
//   media_inside(media_uv: vec2f, aa: f32) -> f32
//       1 where the media is, 0 over the bars, with an `aa` wide ramp (media uv units) centred on the
//       border. aa = 0 gives a hard edge. Use it to mask the bars of a `contain` fit.
//   media_lod(media_px: vec2f, s: vec2f, canvas_px: vec2f) -> f32
//       mip level that matches the screen footprint: log2 of how many media texels fall under one canvas
//       pixel (>= 0). Derivative-free, so it is legal anywhere, and it keeps a downscaled video from
//       aliasing. media_px is the media size in texels, canvas_px the canvas size in pixels.
//
// Colour: media is uploaded to an rgba8unorm-srgb texture (src/gpu/media.ts), so sampling returns LINEAR
// light and the shader writes it out through linear_to_srgb (color.wgsl) like everything else.
//
// Cost: a handful of ALU per call. Pitfall: distort in media space AFTER the fit and mirror/fade near the
// media border (refraction_safe_uv), or the water will pull in bar pixels and wrap at the media edges.

fn media_fit_scale(canvas_aspect: f32, media_aspect: f32, contain: bool) -> vec2f {
  let r = max(canvas_aspect, 1e-4) / max(media_aspect, 1e-4);
  // Canvas wider than the media (r > 1): cover crops the media's height, contain adds bars left and right.
  var cover = vec2f(1.0, 1.0 / r);
  if (r < 1.0) {
    cover = vec2f(r, 1.0);
  }
  return select(cover, cover / min(cover.x, cover.y), contain);
}

fn media_fit_uv(uv: vec2f, s: vec2f) -> vec2f {
  return (uv - vec2f(0.5)) * s + vec2f(0.5);
}

fn media_offset_to_media(offset_uv: vec2f, s: vec2f) -> vec2f {
  return offset_uv * s;
}

fn media_inside(media_uv: vec2f, aa: f32) -> f32 {
  let inner = min(min(media_uv.x, 1.0 - media_uv.x), min(media_uv.y, 1.0 - media_uv.y));
  return clamp(inner / max(aa, 1e-6) + 0.5, 0.0, 1.0);
}

fn media_lod(media_px: vec2f, s: vec2f, canvas_px: vec2f) -> f32 {
  let texels = media_px * s / max(canvas_px, vec2f(1.0));
  return log2(max(max(texels.x, texels.y), 1.0));
}

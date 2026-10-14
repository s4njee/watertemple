// Fullscreen triangle, shared by every fragment-only effect and pass:
//
//   #include "fullscreen.wgsl"
//   @fragment fn fs(in: FullscreenVarying) -> @location(0) vec4f { ... }
//
// Convention: `uv` is (0, 0) at the TOP-LEFT of the target and (1, 1) at the BOTTOM-RIGHT, the same
// orientation as texture coordinates. Sampling a texture with `uv` shows it upright, and `uv.y`
// grows downwards; use `1.0 - uv.y` for maths that expects y up. `position.xy` is in framebuffer
// pixels with the same top-left origin.

struct FullscreenVarying {
  @builtin(position) position: vec4f,
  @location(0) uv: vec2f,
}

// One oversized triangle covers the viewport: no vertex buffer, no diagonal seam.
@vertex fn fullscreen_vs(@builtin(vertex_index) index: u32) -> FullscreenVarying {
  let corner = vec2f(f32((index << 1u) & 2u), f32(index & 2u)); // (0,0) (2,0) (0,2)
  var out: FullscreenVarying;
  out.position = vec4f(corner * 2.0 - 1.0, 0.0, 1.0);
  out.uv = vec2f(corner.x, 1.0 - corner.y);
  return out;
}

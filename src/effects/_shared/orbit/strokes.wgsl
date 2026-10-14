#include "color.wgsl"
#include "sdf.wgsl"
#include "display_params.wgsl"
@group(0) @binding(0) var<uniform> P: DisplayParams;
struct StrokeVertex {
  @builtin(position) position: vec4f,
  @location(0) local: vec2f,
  @location(1) point: vec2f,
  @location(2) colour: vec4f,
};
@vertex fn vs(@builtin(vertex_index) vertex: u32, @location(0) segment: vec4f,
  @location(1) colour: vec4f, @location(2) shape: vec4f) -> StrokeVertex {
  let corners = array<vec2f, 6>(vec2f(0,-1), vec2f(1,-1), vec2f(0,1), vec2f(0,1), vec2f(1,-1), vec2f(1,1));
  let corner = corners[vertex];
  let direction = normalize(segment.zw - segment.xy);
  let radius = max(P.pixel * 0.65, shape.x);
  let point = mix(segment.xy, segment.zw, corner.x) + vec2f(-direction.y, direction.x) * corner.y * radius * 3.0;
  var out: StrokeVertex;
  out.position = vec4f(point.x / P.aspect * 2.0 - 1.0, 1.0 - point.y * 2.0, 0, 1);
  out.local = vec2f(corner.x, corner.y * 3.0);
  out.point = point;
  out.colour = colour;
  return out;
}
@fragment fn fs(in: StrokeVertex) -> @location(0) vec4f {
  var clear = 1.0;
  for (var i = 0u; i < P.rect_count; i++) {
    let box = P.rects[i];
    let distance = sd_rounded_box_2d(in.point - box.xy, box.zw, P.radii[i].x);
    clear *= clamp(0.5 + distance / P.pixel, 0.0, 1.0);
  }
  let core = exp(-in.local.y * in.local.y * 1.8);
  let glow = exp(-in.local.y * in.local.y * 0.35) * 0.18;
  return vec4f(linear_to_srgb(in.colour.rgb * P.light * 0.65), (core + glow) * in.colour.a * clear);
}

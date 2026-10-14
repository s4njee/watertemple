// The backdrop of the particles demo: a dark water gradient, and the wireframe of the simulation box so the
// orbit has something to read depth from.

#include "fullscreen.wgsl"
#include "camera.wgsl"
#include "color.wgsl"

#include "scene_params.wgsl"

@group(0) @binding(0) var<uniform> cam: Camera;
@group(0) @binding(1) var<uniform> scene: SceneParams;

@fragment
fn fs_background(in: FullscreenVarying) -> @location(0) vec4f {
  let t = in.uv.y;
  let vignette = 1.0 - 0.35 * dot(in.uv - 0.5, in.uv - 0.5) * 2.0;
  let color = mix(scene.top, scene.bottom, smoothstep(0.0, 1.0, t)) * vignette;
  return vec4f(linear_to_srgb(color), 1.0);
}

// 12 edges, two vertices each, as pairs of corner numbers (bit 0 x, bit 1 y, bit 2 z).
@vertex
fn vs_lines(@builtin(vertex_index) index: u32) -> @builtin(position) vec4f {
  let edges = array<u32, 24>(0u, 1u, 2u, 3u, 4u, 5u, 6u, 7u, 0u, 2u, 1u, 3u, 4u, 6u, 5u, 7u, 0u, 4u, 1u, 5u, 2u, 6u, 3u, 7u);
  let corner = edges[index];
  let t = vec3f(f32(corner & 1u), f32((corner >> 1u) & 1u), f32((corner >> 2u) & 1u));
  return cam.view_proj * vec4f(mix(scene.box_min, scene.box_max, t), 1.0);
}

@fragment
fn fs_lines() -> @location(0) vec4f {
  let color = vec3f(0.18, 0.45, 0.6) * scene.line_strength;
  return vec4f(linear_to_srgb(color), 1.0);
}

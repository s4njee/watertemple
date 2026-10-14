// camera.wgsl -- perspective camera uniform and screen-space / depth-buffer helpers.
//
// Conventions (also in src/gpu/scene/camera.ts, which fills the uniform): world space is
// right-handed, +Y up. View space looks down -Z. Clip depth is 0 (near) .. 1 (far), not reversed.
// Screen `uv` is (0, 0) at the TOP-LEFT with v growing downwards, like fullscreen.wgsl.
// View-space "linear depth" means the positive distance along the view axis, in world units
// (clip w), range [near, far]; it is NOT the distance to the eye.
//
// struct Camera       the uniform block (view, proj, view_proj and their inverses, eye position,
//                     near, far, viewport size in pixels). Declare `@group(g) @binding(b) var<uniform>
//                     cam: Camera;` in the shader. Filled by `setCameraUniform` in
//                     src/gpu/scene/camera.ts; tests/scene.test.ts checks that the two layouts match.
//
// Depth
//   camera_linear_depth(cam, depth) -> f32        depth-buffer value -> view-space distance along -Z
//   camera_depth_from_linear(cam, z) -> f32       the inverse, z in [near, far]
//
// Screen <-> clip
//   camera_uv_to_ndc(uv) -> vec2f                 uv (top-left origin) -> ndc (y up)
//   camera_ndc_to_uv(ndc) -> vec2f
//
// Reconstruction (uv plus the depth-buffer value at that pixel)
//   camera_view_pos(cam, uv, depth) -> vec3f      view-space position
//   camera_world_pos(cam, uv, depth) -> vec3f     world-space position
//   camera_world_ray(cam, uv) -> vec3f            unit world-space direction from the eye through uv
//
// Projection
//   camera_world_to_view(cam, p) -> vec3f         world -> view
//   camera_view_to_uv(cam, p) -> vec3f            view-space point -> (uv, depth); z is -1 behind the eye
//   camera_world_to_uv(cam, p) -> vec3f           world point -> (uv, depth); z outside [0, 1] means
//                                                 clipped, -1 means behind the eye
//
// A mirrored camera (src/gpu/scene/camera.ts `mirrorCamera`) has an oblique near plane, so its
// depth range is skewed: `camera_linear_depth`, `camera_depth_from_linear` and the depth results of
// `camera_view_pos` / `camera_world_pos` are wrong for it. `camera_world_ray`, the uv of
// `camera_world_to_uv` and `camera_world_to_view` stay correct.
//
// Cost: each function is a matrix-vector product (16 multiply-adds) and a divide, or a few ALU ops.

struct Camera {
  view: mat4x4f,
  proj: mat4x4f,
  view_proj: mat4x4f,
  inv_view: mat4x4f,
  inv_proj: mat4x4f,
  inv_view_proj: mat4x4f,
  position: vec3f,
  near: f32,
  far: f32,
  viewport: vec2f,
}

fn camera_linear_depth(cam: Camera, depth: f32) -> f32 {
  return cam.near * cam.far / (cam.far - depth * (cam.far - cam.near));
}

fn camera_depth_from_linear(cam: Camera, z: f32) -> f32 {
  return cam.far * (z - cam.near) / (z * (cam.far - cam.near));
}

fn camera_uv_to_ndc(uv: vec2f) -> vec2f {
  return vec2f(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0);
}

fn camera_ndc_to_uv(ndc: vec2f) -> vec2f {
  return vec2f(ndc.x * 0.5 + 0.5, 0.5 - ndc.y * 0.5);
}

fn camera_view_pos(cam: Camera, uv: vec2f, depth: f32) -> vec3f {
  let h = cam.inv_proj * vec4f(camera_uv_to_ndc(uv), depth, 1.0);
  return h.xyz / h.w;
}

fn camera_world_pos(cam: Camera, uv: vec2f, depth: f32) -> vec3f {
  let h = cam.inv_view_proj * vec4f(camera_uv_to_ndc(uv), depth, 1.0);
  return h.xyz / h.w;
}

fn camera_world_ray(cam: Camera, uv: vec2f) -> vec3f {
  // Built from the x and y scales of the projection, which an oblique near plane leaves alone, so
  // it is also right for a mirrored camera and never touches the depth range.
  let ndc = camera_uv_to_ndc(uv);
  let view_dir = vec3f(ndc.x / cam.proj[0][0], ndc.y / cam.proj[1][1], -1.0);
  return normalize((cam.inv_view * vec4f(view_dir, 0.0)).xyz);
}

fn camera_world_to_view(cam: Camera, p: vec3f) -> vec3f {
  return (cam.view * vec4f(p, 1.0)).xyz;
}

fn camera_view_to_uv(cam: Camera, p: vec3f) -> vec3f {
  let h = cam.proj * vec4f(p, 1.0);
  if (h.w <= 0.0) {
    return vec3f(0.0, 0.0, -1.0);
  }
  let ndc = h.xyz / h.w;
  return vec3f(camera_ndc_to_uv(ndc.xy), ndc.z);
}

fn camera_world_to_uv(cam: Camera, p: vec3f) -> vec3f {
  let h = cam.view_proj * vec4f(p, 1.0);
  if (h.w <= 0.0) {
    return vec3f(0.0, 0.0, -1.0);
  }
  let ndc = h.xyz / h.w;
  return vec3f(camera_ndc_to_uv(ndc.xy), ndc.z);
}

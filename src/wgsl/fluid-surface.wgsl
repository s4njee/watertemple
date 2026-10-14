// fluid-surface.wgsl -- the geometry and filter helpers of screen-space fluid rendering: sphere
// impostors, depth reconstruction, the depth-smoothing kernels and the normal-from-depth rule.
// The passes themselves live in src/gpu/fluid-surface/; this chunk is what they share and what the
// tests evaluate on their own.
//
// CONVENTIONS. View space looks down -Z. "Linear depth" is the positive distance along the view
// axis (clip w), the same quantity as `camera_linear_depth` in camera.wgsl. A ray is a UNIT
// direction `dir` leaving the eye (the view-space origin). A background pixel in a depth image
// holds 0, so "valid" means depth > 0.
//
// Sphere impostors (a particle drawn as a camera-facing quad, shaded as the sphere it stands for)
//   fluid_surface_billboard_half_size(centre: vec3f, radius: f32, near: f32) -> f32
//        half width of the view-space quad, placed on the plane z = centre.z, that covers the whole
//        projected sphere (the silhouette of an off-axis sphere is an ellipse: r / cos(theta + a)).
//        -1 when the sphere reaches the near plane or is behind it: cull it
//   fluid_surface_sphere_span(dir: vec3f, centre: vec3f, radius: f32) -> vec2f
//        distances along the ray where it enters and leaves the sphere (x < y); (-1, -1) for a miss
//        or a sphere behind the eye. x may be negative when the eye is inside the sphere
//   fluid_surface_sphere_depth(dir: vec3f, centre: vec3f, radius: f32) -> f32
//        linear depth of the near surface point, -1 for a miss. This is the exact depth a real
//        sphere would have at that pixel, written to the depth buffer so impostors intersect
//
// Reconstruction
//   fluid_surface_view_pos(cam: Camera, uv: vec2f, linear_depth: f32) -> vec3f
//        view-space position of the surface seen at `uv` (top-left origin) with that linear depth.
//        Uses the x/y scale and offset of the projection, so it is exact for a perspective camera
//        and costs 8 ALU instead of a matrix inverse
//   fluid_surface_pixel_radius(cam: Camera, linear_depth: f32, radius: f32, height_px: f32) -> f32
//        projected radius in pixels of a world-space `radius` at that depth, for a target
//        `height_px` tall
//
// Smoothing
//   fluid_surface_filter_radius(pixel_radius: f32, scale: f32, max_radius: f32) -> f32
//        kernel radius in pixels: the projected particle radius times `scale`, within [1, max_radius]
//        so the filter follows the particle size as the camera moves and cannot run away when a
//        particle fills the screen
//   fluid_surface_kernel_weight(offset: f32, radius: f32) -> f32
//        Gaussian spatial weight with sigma = radius / 2 (truncated at two sigma); 1 at offset 0
//   fluid_surface_bilateral_weight(spatial: f32, dz: f32, range: f32) -> f32
//        the classic bilateral weight: spatial * exp(-dz^2 / (2 range^2)). Far depths get ~0 weight,
//        so they are dropped, which is what leaves "curtains" on slanted surfaces
//   fluid_surface_narrow_clamp(z: f32, centre: f32, window: f32) -> f32
//        the narrow-range rule: a sample is kept but clamped to centre +- window, so a foreign
//        surface can move the result by at most `window`
//
// Normals
//   fluid_surface_pick_difference(forward: vec3f, backward: vec3f, forward_ok: bool, backward_ok: bool) -> vec3f
//        of two one-sided differences of view-space position (the next pixel minus this one, this
//        one minus the previous), the one with the smaller depth change, or the only valid one, or
//        zero. Crossing a silhouette is what makes a central difference produce garbage normals
//   fluid_surface_normal(dx: vec3f, dy_up: vec3f) -> vec3f
//        unit normal facing the camera from a difference towards +x and one towards +y (up) in view
//        space; (0, 0, 1) when they are degenerate
//
// Thickness
//   fluid_surface_chord(dir: vec3f, centre: vec3f, radius: f32, near: f32, limit: f32) -> f32
//        length of the ray inside the sphere, clipped to [near, limit] along the ray: the thickness
//        one particle adds to a pixel. `limit` is the distance to a scene surface that hides the
//        rest of the sphere (use a large number for none)
//
// Cost: every function is a handful of ALU ops (the sphere ones one sqrt, view_pos two divides);
// none samples a texture. The per-pixel cost of a fluid surface is in the loops that call them.
//
// References: van der Laan, Green and Sainz, "Screen Space Fluid Rendering with Curvature Flow"
// (2009), for the impostor and the smoothing idea; Truong and Yuksel, "A Narrow-Range Filter for
// Screen-Space Fluid Rendering" (2018), for the clamped range.

#include "camera.wgsl"

fn fluid_surface_billboard_half_size(centre: vec3f, radius: f32, near: f32) -> f32 {
  let depth = -centre.z;
  let dist = length(centre);
  if (depth - radius < near || dist <= radius) {
    return -1.0;
  }
  let sin_a = radius / dist;
  let cos_a = sqrt(1.0 - sin_a * sin_a);
  let cos_t = depth / dist;
  let sin_t = sqrt(max(1.0 - cos_t * cos_t, 0.0));
  let cos_ta = cos_t * cos_a - sin_t * sin_a;
  return radius / max(cos_ta, 0.05);
}

fn fluid_surface_sphere_span(dir: vec3f, centre: vec3f, radius: f32) -> vec2f {
  // The perpendicular offset form: b * b - |c|^2 + r^2 cancels two numbers of the same size and
  // loses the rim of the sphere to rounding, this does not.
  let b = dot(dir, centre);
  let offset = centre - dir * b;
  let disc = radius * radius - dot(offset, offset);
  if (disc < 0.0 || b + sqrt(disc) < 0.0) {
    return vec2f(-1.0, -1.0);
  }
  let s = sqrt(disc);
  return vec2f(b - s, b + s);
}

fn fluid_surface_sphere_depth(dir: vec3f, centre: vec3f, radius: f32) -> f32 {
  let span = fluid_surface_sphere_span(dir, centre, radius);
  if (span.y < 0.0) {
    return -1.0;
  }
  return -dir.z * max(span.x, 0.0);
}

fn fluid_surface_view_pos(cam: Camera, uv: vec2f, linear_depth: f32) -> vec3f {
  let ndc = camera_uv_to_ndc(uv);
  return vec3f(
    linear_depth * (ndc.x + cam.proj[2][0]) / cam.proj[0][0],
    linear_depth * (ndc.y + cam.proj[2][1]) / cam.proj[1][1],
    -linear_depth,
  );
}

fn fluid_surface_pixel_radius(cam: Camera, linear_depth: f32, radius: f32, height_px: f32) -> f32 {
  return radius * cam.proj[1][1] * 0.5 * height_px / max(linear_depth, 1e-4);
}

fn fluid_surface_filter_radius(pixel_radius: f32, scale: f32, max_radius: f32) -> f32 {
  return clamp(pixel_radius * scale, 1.0, max(max_radius, 1.0));
}

fn fluid_surface_kernel_weight(offset: f32, radius: f32) -> f32 {
  let u = offset / max(radius, 1e-4);
  return exp(-2.0 * u * u);
}

fn fluid_surface_bilateral_weight(spatial: f32, dz: f32, range: f32) -> f32 {
  let u = dz / max(range, 1e-5);
  return spatial * exp(-0.5 * u * u);
}

fn fluid_surface_narrow_clamp(z: f32, centre: f32, window: f32) -> f32 {
  return clamp(z, centre - window, centre + window);
}

fn fluid_surface_pick_difference(
  forward: vec3f,
  backward: vec3f,
  forward_ok: bool,
  backward_ok: bool,
) -> vec3f {
  if (forward_ok && backward_ok) {
    return select(backward, forward, abs(forward.z) < abs(backward.z));
  }
  if (forward_ok) {
    return forward;
  }
  if (backward_ok) {
    return backward;
  }
  return vec3f(0.0);
}

fn fluid_surface_normal(dx: vec3f, dy_up: vec3f) -> vec3f {
  let n = cross(dx, dy_up);
  let len = length(n);
  if (len < 1e-20) {
    return vec3f(0.0, 0.0, 1.0);
  }
  let unit = n / len;
  return select(-unit, unit, unit.z >= 0.0);
}

fn fluid_surface_chord(dir: vec3f, centre: vec3f, radius: f32, near: f32, limit: f32) -> f32 {
  let span = fluid_surface_sphere_span(dir, centre, radius);
  if (span.y < 0.0) {
    return 0.0;
  }
  let t0 = max(span.x, near);
  let t1 = min(span.y, limit);
  return max(t1 - t0, 0.0);
}

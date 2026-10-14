// Thickness pass: each particle adds the length its sphere cuts out of the pixel's ray, summed with
// additive blending and no depth test. Red is the thickness in world units (times a gain); green is
// the same times the particle's scalar `w`, so blurring both and dividing recovers a smooth mean of
// `w` along the ray (the foam hook). The chord is clipped at the scene surface behind which the rest
// of the sphere is hidden.

#include "fluid-frame.wgsl"
#include "camera.wgsl"
#include "fluid-surface.wgsl"

@group(0) @binding(0) var<uniform> cam: Camera;
@group(0) @binding(1) var<uniform> frame: FluidFrame;
@group(0) @binding(2) var scene_depth: texture_depth_2d;

struct ThicknessVarying {
  @builtin(position) position: vec4f,
  @location(0) view: vec3f,
  @location(1) @interpolate(flat) centre: vec3f,
  @location(2) @interpolate(flat) scalar: f32,
}

// A triangle strip: four vertices per particle instead of six.
const QUAD = array<vec2f, 4>(
  vec2f(-1.0, -1.0), vec2f(1.0, -1.0), vec2f(-1.0, 1.0), vec2f(1.0, 1.0),
);

@vertex fn vs(@builtin(vertex_index) vertex: u32, @location(0) particle: vec4f) -> ThicknessVarying {
  var out: ThicknessVarying;
  let centre = (cam.view * vec4f(particle.xyz, 1.0)).xyz;
  let half_size = fluid_surface_billboard_half_size(centre, frame.thick_radius, cam.near);
  out.centre = centre;
  out.scalar = particle.w;
  if (half_size < 0.0) {
    out.position = vec4f(2.0, 2.0, 2.0, 1.0);
    out.view = vec3f(0.0, 0.0, -1.0);
    return out;
  }
  out.view = vec3f(centre.xy + QUAD[vertex] * half_size, centre.z);
  out.position = cam.proj * vec4f(out.view, 1.0);
  return out;
}

@fragment fn fs(in: ThicknessVarying) -> @location(0) vec4f {
  let dir = normalize(in.view);
  let ratio = frame.full_size / frame.thick_size;
  let pixel = clamp(vec2i(in.position.xy * ratio), vec2i(0), vec2i(frame.full_size) - 1);
  let scene = camera_linear_depth(cam, textureLoad(scene_depth, pixel, 0));
  // Distance along the ray to the scene surface: linear depth is the z component of the hit.
  let limit = scene / max(-dir.z, 1e-3);
  let chord = fluid_surface_chord(dir, in.centre, frame.thick_radius, cam.near / max(-dir.z, 1e-3), limit);
  if (chord <= 0.0) {
    discard;
  }
  let thickness = chord * frame.thick_gain;
  return vec4f(thickness, thickness * in.scalar, 0.0, 0.0);
}

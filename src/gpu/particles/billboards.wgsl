// Instanced billboard particles: one camera-facing quad per particle (4 vertices, a triangle strip), the
// particle position and scalar arriving as per-instance vertex attributes. See billboards.ts.
//
// Sizes are in world units with a floor in pixels; the same code serves perspective and orthographic
// cameras because it works from the projection's y scale and the centre's clip w (1 when orthographic).
// The fragment stage draws a round sprite whose edge goes from a one-pixel antialiased rim (softness 0) to a
// smooth falloff over the whole radius (softness 1). In the opaque mode the sprite is a sphere impostor: it
// shades as a lit ball and writes the depth of the sphere's surface, so spheres intersect each other correctly.

#include "camera.wgsl"
#include "particles.wgsl"
#include "color.wgsl"

#include "billboard_params.wgsl"

@group(0) @binding(0) var<uniform> cam: Camera;
@group(0) @binding(1) var<uniform> params: BillboardParams;

struct Varying {
  @builtin(position) position: vec4f,
  @location(0) uv: vec2f,
  // Palette colour in rgb, energy-conserving alpha factor in a.
  @location(1) color: vec4f,
  // View-space centre and radius actually drawn: the fragment stage rebuilds the sphere from them.
  @location(2) view_center: vec3f,
  @location(3) radius: f32,
  // One pixel in sprite units (uv spans the radius), for the antialiased edge.
  @location(4) pixel: f32,
}

@vertex
fn vs(
  @builtin(vertex_index) vertex: u32,
  @location(0) particle: vec4f,
  @location(1) scalar: f32,
) -> Varying {
  var out: Varying;
  let center = (cam.view * vec4f(particle.xyz, 1.0)).xyz;
  let clip = cam.proj * vec4f(center, 1.0);
  // Behind the camera, or not a number: park the quad outside the clip volume.
  if (!(clip.w > 0.0)) {
    out.position = vec4f(2.0, 2.0, 2.0, 1.0);
    out.uv = vec2f(0.0);
    out.color = vec4f(0.0);
    out.view_center = vec3f(0.0);
    out.radius = 1.0;
    out.pixel = 1.0;
    return out;
  }
  let size = particle_size(params.radius, clip.w, cam.proj[1][1], cam.viewport.y, params.min_pixels);
  let corner = particle_corner(vertex);
  // One pixel in sprite units (uv spans the radius): pixels per radius = r * proj_scale / w * viewport_h / 2.
  let pixel = 2.0 * clip.w / max(size.x * cam.proj[1][1] * cam.viewport.y, 1e-9);
  // The quad reaches one pixel past the rim so the antialiased edge is not cut off.
  let reach = 1.0 + pixel;
  out.position = cam.proj * vec4f(center + vec3f(corner * size.x * reach, 0.0), 1.0);
  out.uv = corner * reach;
  let t = (scalar - params.scalar_min) / max(params.scalar_max - params.scalar_min, 1e-9);
  out.color = vec4f(particle_palette(params.palette, params.palette_count, t), size.y);
  out.view_center = center;
  out.radius = size.x;
  out.pixel = pixel;
  return out;
}

struct Fragment {
  @location(0) color: vec4f,
  @builtin(frag_depth) depth: f32,
}

fn encode(rgb: vec3f) -> vec3f {
  return select(rgb, linear_to_srgb(rgb), params.encode_srgb != 0u);
}

@fragment
fn fs(in: Varying) -> @location(0) vec4f {
  let d2 = dot(in.uv, in.uv);
  let alpha = particle_disc_alpha(sqrt(d2), params.softness, in.pixel);
  if (alpha <= 0.0) { discard; }
  var rgb = in.color.rgb;
  if (params.shading > 0.0) {
    let z = particle_sphere_z(d2);
    let normal = vec3f(in.uv, z);
    let light = normalize(vec3f(-0.4, 0.6, 0.7));
    let diffuse = max(dot(normal, light), 0.0);
    let rim = pow(1.0 - z, 3.0) * 0.25;
    rgb = rgb * mix(1.0, 0.25 + 0.85 * diffuse, params.shading) + vec3f(rim * params.shading);
  }
  let a = alpha * in.color.a * params.intensity;
  return vec4f(encode(rgb) * a, a);
}

// The opaque mode: an opaque sphere with the depth of its surface.
@fragment
fn fs_opaque(in: Varying) -> Fragment {
  let d2 = dot(in.uv, in.uv);
  if (d2 >= 1.0) { discard; }
  let z = particle_sphere_z(d2);
  let normal = vec3f(in.uv, z);
  let light = normalize(vec3f(-0.4, 0.6, 0.7));
  let diffuse = max(dot(normal, light), 0.0);
  let rim = pow(1.0 - z, 3.0) * 0.25;
  let rgb = in.color.rgb * mix(1.0, 0.25 + 0.85 * diffuse, params.shading) + vec3f(rim * params.shading);
  var out: Fragment;
  out.color = vec4f(encode(rgb), 1.0);
  // The point of the sphere facing the camera, projected: view space looks down -z, so it is nearer by r * z.
  let surface = in.view_center + vec3f(in.uv * in.radius, z * in.radius);
  let clip = cam.proj * vec4f(surface, 1.0);
  out.depth = clamp(clip.z / clip.w, 0.0, 1.0);
  return out;
}

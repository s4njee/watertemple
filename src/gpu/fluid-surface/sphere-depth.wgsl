// Depth pass of the fluid surface: every particle is a camera-facing quad shaded as the sphere it
// stands for. The fragment intersects the pixel's ray with the sphere, writes the exact depth to the
// depth buffer (so overlapping spheres intersect and the nearest surface wins) and the linear
// view-space depth to a float target. Fragments behind the scene are discarded, so a particle
// hidden by a wall or the floor never reaches the smoothing.

#include "fluid-frame.wgsl"
#include "camera.wgsl"
#include "fluid-surface.wgsl"

@group(0) @binding(0) var<uniform> cam: Camera;
@group(0) @binding(1) var<uniform> frame: FluidFrame;
@group(0) @binding(2) var scene_depth: texture_depth_2d;

struct SphereVarying {
  @builtin(position) position: vec4f,
  // The point on the quad's plane in view space: its direction is the pixel's ray.
  @location(0) view: vec3f,
  @location(1) @interpolate(flat) centre: vec3f,
}

// A triangle strip: four vertices per particle instead of six.
const QUAD = array<vec2f, 4>(
  vec2f(-1.0, -1.0), vec2f(1.0, -1.0), vec2f(-1.0, 1.0), vec2f(1.0, 1.0),
);

@vertex fn vs(@builtin(vertex_index) vertex: u32, @location(0) particle: vec4f) -> SphereVarying {
  var out: SphereVarying;
  let centre = (cam.view * vec4f(particle.xyz, 1.0)).xyz;
  let half_size = fluid_surface_billboard_half_size(centre, frame.radius, cam.near);
  out.centre = centre;
  if (half_size < 0.0) {
    // Culled: a point outside the clip volume produces no fragments.
    out.position = vec4f(2.0, 2.0, 2.0, 1.0);
    out.view = vec3f(0.0, 0.0, -1.0);
    return out;
  }
  out.view = vec3f(centre.xy + QUAD[vertex] * half_size, centre.z);
  out.position = cam.proj * vec4f(out.view, 1.0);
  return out;
}

struct SphereOut {
  @location(0) depth: f32,
  @builtin(frag_depth) frag_depth: f32,
}

// The deepest scene surface among the full-resolution pixels this chain pixel covers, as linear depth.
fn scene_limit(position: vec2f) -> f32 {
  let ratio = frame.full_size / frame.chain_size;
  let top = vec2i(floor((position - 0.5) * ratio + 0.5));
  let last = vec2i(frame.full_size) - 1;
  var deepest = textureLoad(scene_depth, clamp(top, vec2i(0), last), 0);
  if (ratio.x > 1.5) {
    deepest = max(deepest, textureLoad(scene_depth, clamp(top + vec2i(1, 0), vec2i(0), last), 0));
    deepest = max(deepest, textureLoad(scene_depth, clamp(top + vec2i(0, 1), vec2i(0), last), 0));
    deepest = max(deepest, textureLoad(scene_depth, clamp(top + vec2i(1, 1), vec2i(0), last), 0));
  }
  return camera_linear_depth(cam, deepest);
}

@fragment fn fs(in: SphereVarying) -> SphereOut {
  let dir = normalize(in.view);
  let depth = fluid_surface_sphere_depth(dir, in.centre, frame.radius);
  if (depth < 0.0 || depth > scene_limit(in.position.xy)) {
    discard;
  }
  var out: SphereOut;
  out.depth = depth;
  out.frag_depth = camera_depth_from_linear(cam, depth);
  return out;
}

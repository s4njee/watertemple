// The forward renderer's shader: transform, one directional light, hemisphere ambient, distance fog.
// Group 0 is per view (camera and light), group 1 is per object (dynamic offset into one buffer).
// `scene-light.wgsl` and `scene-object.wgsl` are generated from the TypeScript struct definitions.

#include "camera.wgsl"
#include "scene-light.wgsl"
#include "scene-object.wgsl"

@group(0) @binding(0) var<uniform> cam: Camera;
@group(0) @binding(1) var<uniform> light: SceneLight;
@group(1) @binding(0) var<uniform> obj: SceneObject;

struct ForwardVarying {
  @builtin(position) position: vec4f,
  @location(0) world: vec3f,
  @location(1) normal: vec3f,
}

@vertex fn vs(@location(0) position: vec3f, @location(1) normal: vec3f) -> ForwardVarying {
  let world = obj.model * vec4f(position, 1.0);
  var out: ForwardVarying;
  out.position = cam.view_proj * world;
  out.world = world.xyz;
  out.normal = obj.normal * normal;
  return out;
}

@fragment fn fs(in: ForwardVarying) -> @location(0) vec4f {
  let n = normalize(in.normal);
  let ambient = mix(light.ground_color, light.sky_color, n.y * 0.5 + 0.5) * light.ambient;
  let sun = light.sun_color * (light.sun_intensity * max(dot(n, light.sun_dir), 0.0));
  let lit = mix(obj.color.rgb * (ambient + sun), obj.color.rgb, obj.color.a);
  // The distance to a mirrored camera equals the real path length through the mirror.
  let fog = 1.0 - exp(-light.fog_density * length(in.world - cam.position));
  return vec4f(mix(lit, light.horizon_color, fog), 1.0);
}

// Renders the procedural sky into the cubemap. user[0] = (sun elevation, sun azimuth, haze, unused).
#include "sky.wgsl"
#include "waterplane.wgsl"

fn cubemap_source(dir: vec3f) -> vec3f {
  let u = cubemap_inputs.user[0];
  return sky_radiance_ex(dir, waterplane_sun_dir(u.x, u.y), u.z, 1.0);
}

// march.wgsl -- the ray march through the ink. A fullscreen pass at a fraction of the canvas resolution: each
// pixel intersects its camera ray with the tank (volume.wgsl), steps through the ink grid a fixed number of
// times with a per-pixel jitter on the start, and writes TWO targets: the radiance the ink scatters toward the
// eye (rgb) and the transmittance left over for what is behind it (rgb). composite.wgsl puts the tank and the
// backdrop behind them at full resolution, so the soft ink can be rendered small and upsampled.
//
// Per step: extinction sigma_t from the ink sample (ink.wgsl); if the step is not empty, the light that reaches
// it (from the precomputed light volume, from a short march toward the light, or none: `P.lighting`), the
// Henyey-Greenstein phase of the sun, a sky ambient scaled by how open the voxel is above; the step's
// in-scatter is sigma_s * tint * (sun + ambient) and `volume_accumulate` integrates it exactly over the step.
// Empty space is skipped by a lookup in the occupancy texture; the loop ends when the transmittance is below
// 0.01 (early exit).
//
// Bindings: 0 `P`, 1 camera, 2 ink, 3 light volume, 4 occupancy, 5 linear sampler.

#include "fullscreen.wgsl"
#include "camera.wgsl"
#include "volume.wgsl"
#include "ink.wgsl"

@group(0) @binding(1) var<uniform> cam: Camera;
@group(0) @binding(2) var dens: texture_3d<f32>;
@group(0) @binding(3) var light_volume: texture_3d<f32>;
@group(0) @binding(4) var occupancy: texture_3d<f32>;
@group(0) @binding(5) var samp: sampler;

struct MarchOut {
  @location(0) radiance: vec4f,
  @location(1) transmittance: vec4f,
}

// Transmittance toward the light from p by marching `n` taps through the ink (no precomputed volume).
fn inline_light(p: vec3f, n: u32) -> vec3f {
  let span = volume_ray_box(p, P.light_dir, P.box_min, P.box_max);
  let ds = max(span.y, 0.0) / f32(n);
  var tau = vec3f(0.0);
  for (var i = 0u; i < n; i++) {
    let q = p + P.light_dir * ((f32(i) + 0.5) * ds);
    tau += ink_extinction(textureSampleLevel(dens, samp, volume_to_uvw(q, P.box_min, P.box_max), 0.0)) * ds;
  }
  return exp(-tau);
}

@fragment
fn fs(in: FullscreenVarying) -> MarchOut {
  let origin = cam.position;
  let dir = camera_world_ray(cam, in.uv);
  let span = volume_ray_box(origin, dir, P.box_min, P.box_max);
  var out: MarchOut;
  out.radiance = vec4f(0.0);
  out.transmittance = vec4f(1.0);
  if (!volume_hit(span)) {
    return out;
  }

  // An obstacle sphere hides whatever is behind it.
  let t_sphere = sphere_hit(origin, dir);
  let t_end = select(span.y, min(span.y, t_sphere), t_sphere > 0.0);
  if (t_end <= span.x) {
    return out;
  }
  let steps = max(P.steps, 1u);
  let ds = (t_end - span.x) / f32(steps);
  let start = span.x + ds * select(0.5, volume_jitter(in.position.xy, P.frame), P.jitter > 0.0);
  let cos_theta = dot(-P.light_dir, -dir);
  let phase = volume_phase_hg(cos_theta, P.phase_g) * 12.566370614;
  var acc = volume_start();

  for (var i = 0u; i < steps; i++) {
    let p = origin + dir * (start + f32(i) * ds);
    let uvw = volume_to_uvw(p, P.box_min, P.box_max);
    if (P.occupancy != 0u && textureSampleLevel(occupancy, samp, uvw, 0.0).x < 0.004) {
      continue;
    }
    let d = textureSampleLevel(dens, samp, uvw, 0.0);
    let sigma_t = ink_extinction(d);
    if (max(sigma_t.x, max(sigma_t.y, sigma_t.z)) < 1e-3) {
      continue;
    }
    var sun = P.light_color;
    var sky = 1.0;
    if (P.lighting == 1u) {
      let l = textureSampleLevel(light_volume, samp, uvw, 0.0);
      sun *= l.rgb * phase;
      sky = mix(0.3, 1.0, l.a);
    } else if (P.lighting == 2u) {
      sun *= inline_light(p, P.light_steps) * phase;
    } else {
      sun = vec3f(0.0);
    }
    let scatter = d.a * P.scatter;
    let in_scatter = scatter * ink_tint(d) * (sun + P.ambient * sky);
    acc = volume_accumulate(acc, sigma_t, in_scatter, ds);
    if (volume_opaque(acc, 0.01)) {
      break;
    }
  }
  out.radiance = vec4f(acc.radiance, 1.0);
  out.transmittance = vec4f(acc.transmittance, 1.0);
  return out;
}

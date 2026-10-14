// volume.wgsl -- helpers for ray-marching a participating medium (ink, smoke, fog) inside a box: the ray-box
// intersection, per-pixel jitter, the transmittance/in-scatter accumulation of one step, a phase function.
//
// CONVENTIONS. World space, +Y up (camera.wgsl). A medium fills an axis-aligned box [box_min, box_max]; a ray is
// `origin + t * dir` with `dir` a UNIT vector (so t is a distance). Extinction `sigma_t` is per unit of world
// length and per channel (coloured absorption makes ink read as ink rather than smoke), radiance is LINEAR rgb.
// The march is front to back: `VolumeAccum` holds the radiance gathered so far and the transmittance still left.
//
// Box
//   volume_ray_box(origin, dir, box_min, box_max) -> vec2f
//       (t_near, t_far) where the ray enters and leaves the box, t_near clamped to >= 0 (a camera inside the box
//       starts at 0). The ray misses when t_far <= t_near: use volume_hit. Rays parallel to a pair of faces are
//       handled (they hit only if they are between them). Cost: 6 divides, a few min/max.
//   volume_hit(span: vec2f) -> bool                 true when volume_ray_box found a segment of positive length
//   volume_to_uvw(p, box_min, box_max) -> vec3f     world position -> 0..1 texture coordinate inside the box
//
// Stepping
//   volume_jitter(pixel: vec2f, frame: u32) -> f32  [0, 1) interleaved gradient noise of the pixel position
//       (Jimenez 2014); with `frame` = 0 it is a fixed pattern, with a frame counter it moves every frame. Start the
//       march at t_near + jitter * step and the banding of a fixed step count turns into fine noise.
//
// Accumulation (one step of length `ds`)
//   struct VolumeAccum { radiance: vec3f, transmittance: vec3f }
//   volume_start() -> VolumeAccum                   no radiance, full transmittance
//   volume_accumulate(acc, sigma_t, in_scatter, ds) -> VolumeAccum
//       `in_scatter` is the radiance the medium sends toward the eye per unit length (scattering coefficient times
//       the light that reaches it times the phase function; or an emission). The step adds the analytic integral over
//       the step, T * in_scatter * (1 - exp(-sigma_t ds)) / sigma_t, and multiplies T by exp(-sigma_t ds). That
//       is exact for a constant medium over the step, so a few big steps do not brighten or darken it (a plain
//       in_scatter * ds sum would). sigma_t -> 0 gives T * in_scatter * ds.
//   volume_opaque(acc, threshold) -> bool           true when the largest channel of the transmittance is below
//       `threshold`: stop marching (early exit)
//   volume_slab_transmittance(sigma_t, thickness) -> vec3f     exp(-sigma_t * thickness): what a uniform slab leaves
//
// Phase
//   volume_phase_hg(cos_theta, g) -> f32            Henyey-Greenstein phase function, normalised over the sphere
//       (integrates to 1); g in (-1, 1): 0 isotropic, > 0 forward scattering. cos_theta is the cosine of the
//       angle between the direction the light travels and the direction toward the eye.
//
// Cost: the accumulate is 3 exp and a few divides per step; the jitter is one fract and two multiplies.
// Credits: slab method (Kay & Kajiya 1986); interleaved gradient noise from J. Jimenez, "Next Generation Post
// Processing in Call of Duty: Advanced Warfare" (SIGGRAPH 2014); Henyey & Greenstein 1941.

const _VOLUME_EPS: f32 = 1e-8;

struct VolumeAccum {
  radiance: vec3f,
  transmittance: vec3f,
}

fn volume_ray_box(origin: vec3f, dir: vec3f, box_min: vec3f, box_max: vec3f) -> vec2f {
  // A direction component of ~0 would divide by zero; give it a huge reciprocal of the right sign instead,
  // which sends that slab's distances to +-1e30 (or, outside the slab, makes the segment empty).
  let safe = select(dir, vec3f(_VOLUME_EPS), abs(dir) < vec3f(_VOLUME_EPS));
  let inv = 1.0 / safe;
  let t0 = (box_min - origin) * inv;
  let t1 = (box_max - origin) * inv;
  let near = max(max(min(t0.x, t1.x), min(t0.y, t1.y)), max(min(t0.z, t1.z), 0.0));
  let far = min(min(max(t0.x, t1.x), max(t0.y, t1.y)), max(t0.z, t1.z));
  return vec2f(near, far);
}

fn volume_hit(span: vec2f) -> bool {
  return span.y > span.x;
}

fn volume_to_uvw(p: vec3f, box_min: vec3f, box_max: vec3f) -> vec3f {
  return (p - box_min) / (box_max - box_min);
}

fn volume_jitter(pixel: vec2f, frame: u32) -> f32 {
  // The frame shifts the pattern along its own diagonal (5.588238 px per frame), the usual way to animate it.
  let p = pixel + 5.588238 * f32(frame % 64u);
  return fract(52.9829189 * fract(dot(p, vec2f(0.06711056, 0.00583715))));
}

fn volume_start() -> VolumeAccum {
  return VolumeAccum(vec3f(0.0), vec3f(1.0));
}

fn volume_accumulate(acc: VolumeAccum, sigma_t: vec3f, in_scatter: vec3f, ds: f32) -> VolumeAccum {
  let tau = max(sigma_t, vec3f(0.0)) * max(ds, 0.0);
  let step_t = exp(-tau);
  // (1 - exp(-tau)) / sigma_t, with its limit ds as sigma_t -> 0.
  let ratio = select((vec3f(1.0) - step_t) / max(sigma_t, vec3f(1e-12)), vec3f(ds), tau < vec3f(1e-5));
  var out: VolumeAccum;
  out.radiance = acc.radiance + acc.transmittance * in_scatter * ratio;
  out.transmittance = acc.transmittance * step_t;
  return out;
}

fn volume_opaque(acc: VolumeAccum, threshold: f32) -> bool {
  return max(acc.transmittance.x, max(acc.transmittance.y, acc.transmittance.z)) < threshold;
}

fn volume_slab_transmittance(sigma_t: vec3f, thickness: f32) -> vec3f {
  return exp(-max(sigma_t, vec3f(0.0)) * max(thickness, 0.0));
}

fn volume_phase_hg(cos_theta: f32, g: f32) -> f32 {
  let gg = clamp(g, -0.999, 0.999);
  let denom = 1.0 + gg * gg - 2.0 * gg * cos_theta;
  return (1.0 - gg * gg) / (12.566370614 * pow(max(denom, 1e-6), 1.5));
}

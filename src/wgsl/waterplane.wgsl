// waterplane.wgsl -- a small open-water test scene: camera rays, a cheap animated wave surface with
// analytic normals, and a distance fade so the surface does not alias into sparkle near the horizon.
// Built for the optics demos (fresnel-sky, sun-glint, crest-sss); a later ocean effect may reuse it.
// Pure functions, no bindings: the caller owns the uniforms.
//
// World: y is up, the water is the plane y = 0 displaced by the waves, the camera sits at
// (x, height, z) and looks along +z when pitch and yaw are zero. Units are metres and seconds.
//
// Camera
//   waterplane_ray_dir(uv, aspect, pitch, fov_y) -> vec3f
//        unit view ray for a pixel. `uv` has its origin at the TOP-LEFT (fullscreen.wgsl), aspect is
//        width / height, `pitch` is radians (positive looks up, negative looks down at the water),
//        `fov_y` is the vertical field of view in radians
//   waterplane_sun_dir(elevation, azimuth) -> vec3f
//        unit direction toward the sun from angles in radians. Azimuth 0 is straight ahead of the
//        camera (+z), positive turns toward +x (right). Elevation 0 is the horizon
//
// Intersection
//   waterplane_hit(origin, dir) -> f32
//        distance along the ray to the flat plane y = 0, or -1.0 when the ray points at or above the
//        horizon (dir.y >= 0) or starts below the plane. Use `t < 0.0` as the sky test
//   waterplane_footprint(t, dir_y, pixel_angle) -> f32
//        approximate size in metres of one pixel on the water at distance t: t * pixel_angle / |dir_y|.
//        pixel_angle = 2 * tan(fov_y / 2) / height_in_pixels. Grows without bound towards the horizon
//   waterplane_hit_waves(origin, dir, t_plane, time, strength, scale, footprint) -> f32
//        refines the flat-plane distance against the large waves (one fixed-point step with the
//        swell only); returns t_plane when the plane is missed. Cheap and good enough for
//        placing texture/noise lookups on the displaced surface, not an exact ray-surface hit
//
// Waves. Five directional swells shaped like exp(sin(x) - 1) (sharp crests, broad troughs; the sum
// is zero-mean) plus two octaves of gradient-noise ripples that drift across the surface. All
// animated, all with analytic derivatives, no loops and no textures.
//   strength   steepness multiplier, 0 (dead flat) to about 3; 1 is a lively sea (slopes up to ~0.3)
//   scale      wavelength multiplier: 1 gives 25 m down to 1.5 m swells, 2 doubles every wavelength
//   footprint  pixel size on the water in metres (waterplane_footprint). Each wave fades out
//              smoothly as its wavelength drops from 8 pixels to 2.5 pixels, so distant water loses
//              its small waves instead of aliasing. Pass 0.0 for the full unfiltered surface
//   waterplane_wave(p, time, strength, scale, footprint) -> vec3f
//        p is the world xz position. Returns (height, d height / dx, d height / dz), the
//        gradient is analytic and exact
//   waterplane_height(p, time, strength, scale, footprint) -> f32       the height alone
//   waterplane_normal(gradient: vec2f) -> vec3f
//        unit normal (-gx, 1, -gz) / length from the .yz of waterplane_wave
//   waterplane_amplitude(strength, scale) -> f32
//        an upper bound on the swell height in metres (reached only when all five waves line up)
//   waterplane_crest(height, strength, scale) -> f32
//        height as a 0..1 "how crest-like" factor: 0 at and below 8% of waterplane_amplitude (just
//        above the mean level), 1 from 40% of it up. The swell's standard deviation is about a third
//        of the amplitude, so roughly one point in five reads as crest. The input for sss.wgsl
//
// Reflection
//   waterplane_reflect(dir, normal) -> vec3f
//        mirror of the view ray `dir` about the normal, with any downward result folded upward
//        (y -> |y|). A steep wave normal can reflect the ray into the water; the fold makes it sample
//        the sky just above the horizon instead
//
// Cost: waterplane_wave ~ 5 x (sin, cos, exp) + two gnoise_2d_d (8 hashes), about 100 ALU. A wave or
// ripple that has faded out at distance is skipped, so far water is cheaper. waterplane_height costs
// the same (the derivative maths is cheap). Camera ray: 1 normalize, 2 sincos.
//
// Wave shape: exp(sin(x) - 1) is a widely used cheap peaked-crest profile; each wave travels at the
// deep-water gravity speed omega = sqrt(g k). Not a physical spectrum: five hand-picked directions
// and wavelengths, which repeat visibly over a very long view, and a plausible look is the goal.

#include "math.wgsl"
#include "noise.wgsl"

const WATERPLANE_GRAVITY: f32 = 9.81;
// Mean of exp(sin(x) - 1) over a period: I0(1) / e. Subtracted so each wave has zero mean.
const _WATERPLANE_PROFILE_MEAN: f32 = 0.4658;

fn waterplane_ray_dir(uv: vec2f, aspect: f32, pitch: f32, fov_y: f32) -> vec3f {
  let tan_half = tan(0.5 * fov_y);
  let ndc = vec2f((2.0 * uv.x - 1.0) * aspect, 1.0 - 2.0 * uv.y);
  let local = normalize(vec3f(ndc * tan_half, 1.0));
  let c = cos(pitch);
  let s = sin(pitch);
  return vec3f(local.x, local.y * c + local.z * s, local.z * c - local.y * s);
}

fn waterplane_sun_dir(elevation: f32, azimuth: f32) -> vec3f {
  let c = cos(elevation);
  return vec3f(sin(azimuth) * c, sin(elevation), cos(azimuth) * c);
}

fn waterplane_hit(origin: vec3f, dir: vec3f) -> f32 {
  if (dir.y >= 0.0 || origin.y <= 0.0) {
    return -1.0;
  }
  return -origin.y / dir.y;
}

fn waterplane_footprint(t: f32, dir_y: f32, pixel_angle: f32) -> f32 {
  return t * pixel_angle / max(abs(dir_y), 1e-4);
}

// One swell. Returns (height, d/dx, d/dz). `angle` is the travel direction (0 = toward -z, so the
// waves roll toward a camera that looks along +z), `steep` the relative steepness, `phase0` a fixed
// offset that decorrelates the five waves at the origin, `rscale` = inverseSqrt(scale). The speed
// sqrt(g k) has a constant part that the compiler folds, because `wavelength` is a literal.
fn _waterplane_swell_wave(
  p: vec2f, time: f32, angle: f32, wavelength: f32, steep: f32, phase0: f32,
  strength: f32, scale: f32, rscale: f32, footprint: f32,
) -> vec3f {
  let lambda = wavelength * scale;
  let fade = smoothstep(2.5, 8.0, lambda / max(footprint, 1e-6));
  if (fade <= 0.0) {
    return vec3f(0.0);
  }
  let k = TAU / lambda;
  let dir = vec2f(sin(angle), -cos(angle));
  let omega = sqrt(WATERPLANE_GRAVITY * TAU / wavelength) * rscale;
  let phase = k * dot(dir, p) - omega * time + phase0;
  let amp = 0.03 * lambda * steep * strength * fade;
  let e = exp(sin(phase) - 1.0);
  let slope = amp * k * e * cos(phase);
  return vec3f(amp * (e - _WATERPLANE_PROFILE_MEAN), slope * dir.x, slope * dir.y);
}

fn _waterplane_swell(p: vec2f, time: f32, strength: f32, scale: f32, footprint: f32) -> vec3f {
  let r = inverseSqrt(scale);
  return _waterplane_swell_wave(p, time, 0.00, 25.0, 1.00, 0.0, strength, scale, r, footprint) +
    _waterplane_swell_wave(p, time, 0.50, 14.0, 0.90, 1.7, strength, scale, r, footprint) +
    _waterplane_swell_wave(p, time, -0.65, 8.3, 0.80, 3.1, strength, scale, r, footprint) +
    _waterplane_swell_wave(p, time, 1.05, 4.6, 0.70, 4.4, strength, scale, r, footprint) +
    _waterplane_swell_wave(p, time, -1.30, 2.5, 0.55, 0.9, strength, scale, r, footprint);
}

// One ripple octave: 2D gradient noise scrolled across the surface by `drift` (metres per second),
// the cheap way to animate small ripples. `frequency` is in cells per metre.
fn _waterplane_ripple(
  p: vec2f, time: f32, frequency: f32, amp: f32, drift: vec2f, strength: f32, footprint: f32,
) -> vec3f {
  let fade = smoothstep(2.5, 8.0, (1.0 / frequency) / max(footprint, 1e-6));
  let a = amp * min(strength, 1.5) * fade;
  if (a <= 0.0) {
    return vec3f(0.0);
  }
  let n = gnoise_2d_d((p - drift * time) * frequency);
  return vec3f(a * n.x, a * frequency * n.y, a * frequency * n.z);
}

fn waterplane_wave(p: vec2f, time: f32, strength: f32, scale: f32, footprint: f32) -> vec3f {
  let f = 1.0 / scale;
  return _waterplane_swell(p, time, strength, scale, footprint) +
    _waterplane_ripple(p, time, 2.6 * f, 0.014 * scale, vec2f(0.35, -0.5) * scale, strength, footprint) +
    _waterplane_ripple(p + vec2f(31.7, -12.3), time, 6.1 * f, 0.005 * scale, vec2f(-0.6, -0.3) * scale, strength, footprint);
}

fn waterplane_height(p: vec2f, time: f32, strength: f32, scale: f32, footprint: f32) -> f32 {
  return waterplane_wave(p, time, strength, scale, footprint).x;
}

fn waterplane_normal(gradient: vec2f) -> vec3f {
  return normalize(vec3f(-gradient.x, 1.0, -gradient.y));
}

fn waterplane_hit_waves(
  origin: vec3f, dir: vec3f, t_plane: f32, time: f32, strength: f32, scale: f32, footprint: f32,
) -> f32 {
  if (t_plane < 0.0) {
    return t_plane;
  }
  let p = origin + dir * t_plane;
  let h = _waterplane_swell(p.xz, time, strength, scale, footprint).x;
  // The surface is at y = h; the ray reaches that height at t = (h - origin.y) / dir.y.
  return clamp((h - origin.y) / dir.y, 0.5 * t_plane, 2.0 * t_plane);
}

fn waterplane_amplitude(strength: f32, scale: f32) -> f32 {
  // Sum of 0.03 * wavelength * steepness over the five waves, times the largest deviation from the mean.
  return 0.03 * (25.0 + 12.6 + 6.64 + 3.22 + 1.375) * 0.5342 * strength * scale;
}

fn waterplane_crest(height: f32, strength: f32, scale: f32) -> f32 {
  let amplitude = max(waterplane_amplitude(strength, scale), 1e-4);
  return smoothstep(0.08, 0.4, height / amplitude);
}

fn waterplane_reflect(dir: vec3f, normal: vec3f) -> vec3f {
  var r = reflect(dir, normal);
  r.y = abs(r.y);
  return r;
}

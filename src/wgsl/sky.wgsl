// sky.wgsl -- a cheap procedural sky: a function of view direction and sun direction that returns
// LINEAR HDR radiance. Not a physical atmosphere (no scattering integral): a horizon-to-zenith
// gradient, a sun disc with a halo, and a sunset tint driven by the sun's elevation. About 40 ALU ops,
// no textures, no loops. Made to be reflected by water, drawn as a backdrop, or baked into a cubemap
// (src/gpu/cubemap.ts renders any `fn(dir) -> vec3f` into one).
//
// Conventions
//  * World space, y up. `dir` and `sun_dir` are UNIT vectors; `dir` points from the viewer into the
//    sky (for a reflection: the reflected view ray). `sun_dir` points from the scene TOWARD the sun.
//  * Radiance is scene-referred linear RGB: ~0.3 to 1 over the blue sky, up to SKY_SUN_RADIANCE (30)
//    on the sun disc, so apply exposure and a tonemapper (color.wgsl) before display.
//  * Directions below the horizon (dir.y < 0) return a dim, horizon-coloured "ground" so a cubemap
//    has no black hole and a reflection off a steep wave never samples garbage. It matches the
//    horizon colour exactly at dir.y = 0, so the horizon is continuous. Nothing is hidden below it
//    except the sun disc and halo, which fade out across a thin band around the horizon.
//  * The sun passes through the horizon smoothly: elevation 90 degrees is noon (white, sky at its bluest),
//    ~10 to 0 degrees is golden hour to sunset (orange sun, warm horizon glow on the sun's side), below
//    about -10 degrees the sky is almost dark. No clouds.
//
// Constants
//   SKY_SUN_RADIANCE        peak radiance of the sun disc (30); the unit for sun glints, see specular.wgsl
//   SKY_SUN_ANGULAR_RADIUS  radius of the drawn disc, 0.0093 rad (about twice the real sun, for looks)
//
// Sky
//   sky_radiance(dir, sun_dir) -> vec3f         the default look (haze 0.3, full sun)
//   sky_radiance_ex(dir, sun_dir, haze, sun_gain) -> vec3f
//        haze     [0, 1]: 0 clear and deep blue, 1 milky and bright with a wide halo
//        sun_gain [0, ..): scales the sun disc and halo; 0 removes the sun, which is what you want for
//                 a reflection when the sun's reflection is added separately by specular.wgsl
//
// Pieces, for lighting a scene consistently with the sky
//   sky_sun_color(sun_dir) -> vec3f    colour of direct sunlight, max channel <= 1: white at noon, orange near
//                                      the horizon, fading to 0 as the sun sinks below -3 degrees. Multiply by
//                                      SKY_SUN_RADIANCE for the disc's radiance
//   sky_horizon_color(sun_dir, haze) -> vec3f   sky colour at dir.y = 0 away from the sun
//   sky_zenith_color(sun_dir, haze) -> vec3f    sky colour straight up
//   sky_ambient(sun_dir) -> vec3f      cheap hemisphere-average sky radiance, for ambient light and
//                                      for tinting a diffuse body colour
//
// Banding
//   sky_dither(frag_xy) -> f32         triangular-PDF noise in [-1, 1] LSB of an 8-bit channel. Add
//                                      `sky_dither(position.xy) / 255.0` AFTER linear_to_srgb: smooth
//                                      sky gradients otherwise show visible bands in 8-bit output
//
// Cost: sky_radiance ~40 ALU, one pow, two exp, one cross/length. Technique: a two-colour gradient
// with a sunset glow lobe and a Lorentzian sun halo (the common "cheap sky" structure); the sun's
// colour comes from a per-channel exponential extinction over an airmass that grows as 1 / elevation.

#include "hash.wgsl"

const SKY_SUN_RADIANCE: f32 = 30.0;
const SKY_SUN_ANGULAR_RADIUS: f32 = 0.0093;

// Twilight weight: 0 at noon, 1 around the horizon, 0 again when the sun is well below it.
fn _sky_twilight(sun_y: f32) -> f32 {
  return smoothstep(-0.25, 0.0, sun_y) * (1.0 - smoothstep(0.04, 0.5, sun_y));
}

// Day weight: 1 in daylight, 0 at night.
fn _sky_day(sun_y: f32) -> f32 {
  return smoothstep(-0.2, 0.25, sun_y);
}

fn sky_sun_color(sun_dir: vec3f) -> vec3f {
  let e = sun_dir.y;
  let airmass = 1.0 / (max(e, 0.0) + 0.08);
  let transmittance = exp(-airmass * vec3f(0.025, 0.07, 0.17));
  return transmittance * smoothstep(-0.05, 0.04, e);
}

fn sky_zenith_color(sun_dir: vec3f, haze: f32) -> vec3f {
  let day = _sky_day(sun_dir.y);
  let twilight = _sky_twilight(sun_dir.y);
  var c = mix(vec3f(0.002, 0.004, 0.014), vec3f(0.05, 0.16, 0.55), day);
  c = mix(c, vec3f(0.02, 0.035, 0.16), twilight * 0.8);
  return mix(c, sky_horizon_color(sun_dir, 1.0) * 0.7, saturate(haze) * 0.45);
}

fn sky_horizon_color(sun_dir: vec3f, haze: f32) -> vec3f {
  let day = _sky_day(sun_dir.y);
  let twilight = _sky_twilight(sun_dir.y);
  var c = mix(vec3f(0.006, 0.01, 0.025), vec3f(0.42, 0.6, 0.88), day);
  c = mix(c, vec3f(0.7, 0.36, 0.24), twilight * 0.75);
  return mix(c, vec3f(0.85, 0.88, 0.92) * day * day * day + c * (1.0 - day * day * day), saturate(haze) * 0.5);
}

fn sky_ambient(sun_dir: vec3f) -> vec3f {
  return 0.5 * (sky_horizon_color(sun_dir, 0.3) + sky_zenith_color(sun_dir, 0.3));
}

fn sky_radiance_ex(dir: vec3f, sun_dir: vec3f, haze: f32, sun_gain: f32) -> vec3f {
  let h = saturate(haze);
  let up = dir.y;
  let horizon = sky_horizon_color(sun_dir, h);
  let zenith = sky_zenith_color(sun_dir, h);

  // Gradient: steep near the horizon, flattening towards the zenith.
  var col = mix(horizon, zenith, pow(saturate(up), 0.45));

  // Sunset glow: a broad warm lobe on the sun's side that hugs the horizon.
  let cos_s = dot(dir, sun_dir);
  let sun_col = sky_sun_color(sun_dir);
  let twilight = _sky_twilight(sun_dir.y);
  let toward = pow(saturate(0.5 + 0.5 * cos_s), 5.0);
  col += sun_col * vec3f(0.9, 0.4, 0.2) * twilight * toward * exp(-max(up, 0.0) * 4.0) * 1.6;

  // Below the horizon: fade to a darker copy of the horizon colour. Equal to the sky at up = 0.
  col = mix(col, horizon * 0.35, saturate(-up * 4.0));

  // Sun halo (broad Mie-like glow plus a tight Lorentzian) and disc. Both vanish under the horizon.
  let sin_t = length(cross(dir, sun_dir));
  let x = 1.0 - cos_s;
  let near_sun = select(0.0, 1.0, cos_s > 0.0);
  let r = SKY_SUN_ANGULAR_RADIUS;
  let tight = 1.0 / (1.0 + (sin_t / 0.03) * (sin_t / 0.03));
  let halo = (0.9 * tight + 0.18 * exp(-x * 5.0)) * (1.0 + 2.0 * h) * near_sun;
  let disc = (1.0 - smoothstep(0.75 * r, r, sin_t)) * near_sun * SKY_SUN_RADIANCE;
  let above = smoothstep(-0.25, 0.04, up);
  let above_disc = smoothstep(-0.03, 0.03, up);
  col += sun_col * sun_gain * (halo * above + disc * above_disc);

  return max(col, vec3f(0.0));
}

fn sky_radiance(dir: vec3f, sun_dir: vec3f) -> vec3f {
  return sky_radiance_ex(dir, sun_dir, 0.3, 1.0);
}

fn sky_dither(frag_xy: vec2f) -> f32 {
  return hash_2to1(frag_xy) - hash_2to1(frag_xy + vec2f(17.0, 31.0));
}

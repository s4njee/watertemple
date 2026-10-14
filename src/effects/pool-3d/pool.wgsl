// pool.wgsl -- the 3D pool, every draw in one module. Rasterised: a sky, the deck around the pool, the
// pool's inner walls and floor, the ball, and the water surface (a grid displaced by the heightfield).
// The water is where the optics happen, and they are RAY TRACED against the analytic scene instead of
// rasterised: for each water fragment the view ray is reflected and refracted by the wave normal, and
// each ray is intersected with the box and the sphere in closed form (pool-rays.wgsl) to find what it
// sees. Single bounce: a ray that hits something is shaded and not traced further.
//
// Shading under water: tile albedo x (sky ambient + sun x cos x caustics x absorption along the sun's
// path), then absorbed and scattered along the path the eye ray took through the water. The caustics
// come from the live heightfield (src/gpu/sim/caustics.ts) as a floor-space texture; walls and the
// ball look it up by sliding their position along the refracted sun ray to the floor depth.
//
// Bindings, group 0 (shared by every entry point): 0 camera, 1 params, 2 heightfield (rgba16float),
// 3 caustics (rgba16float), 4 filtering sampler.
#include "camera.wgsl"
#include "fullscreen.wgsl"
#include "color.wgsl"
#include "sky.wgsl"
#include "fresnel.wgsl"
#include "absorption.wgsl"
#include "backdrop.wgsl"
#include "pool_rays.wgsl"
#include "pool_params.wgsl"

@group(0) @binding(0) var<uniform> cam: Camera;
@group(0) @binding(1) var<uniform> P: PoolParams;
@group(0) @binding(2) var hf_tex: texture_2d<f32>;
@group(0) @binding(3) var caustic_tex: texture_2d<f32>;
@group(0) @binding(4) var samp: sampler;

const QUALITY_FULL: u32 = 0u;
const VIEW_NORMAL: u32 = 0u;
const VIEW_CAUSTICS: u32 = 1u;

// --- Lighting ---------------------------------------------------------------------------------------

fn pl_sun_irradiance() -> vec3f {
  return sky_sun_color(P.sun) * P.sun_power;
}

fn pl_sky_ambient() -> vec3f {
  return sky_ambient(P.sun) * P.ambient;
}

fn pl_output(c: vec3f, frag: vec2f) -> vec4f {
  let mapped = tonemap_aces(max(c, vec3f(0.0)) * P.exposure);
  let srgb = linear_to_srgb(mapped) + vec3f(sky_dither(frag) / 255.0);
  return vec4f(clamp(srgb, vec3f(0.0), vec3f(1.0)), 1.0);
}

// Light factor (0 shadowed, 1 lit) at an above-water point from the ball's shadow along the sun ray.
fn pl_ball_shadow(pos: vec3f) -> f32 {
  let radius = P.ball.w;
  if (radius <= 0.0) {
    return 1.0;
  }
  let rel = P.ball.xyz - pos;
  let u = max(dot(rel, P.sun), 0.0);
  let d = length(rel - P.sun * u);
  let soft = 0.18 * radius + 0.02;
  return smoothstep(radius - soft, radius + soft, d);
}

// Floor-space caustic light at an underwater point. The caustics texture is indexed by FLOOR position:
// for a point at depth s below the surface, slide it along the refracted sun ray to the floor depth
// (the beam through it enters the surface at p = pos - s shift and would reach the floor at q).
// A beam that entered outside the pool does not exist (the wall's shadow), so that point is dark.
fn pl_caustics_at(pos: vec3f) -> vec3f {
  if (P.caustics_on < 0.5) {
    return vec3f(1.0);
  }
  let s = max(-pos.y, 0.0);
  let entry = pos.xz - s * P.sun_shift;
  let reach = pos.xz + (P.depth - s) * P.sun_shift;
  let uv = clamp(reach / (2.0 * POOL_HALF) + vec2f(0.5), vec2f(0.0), vec2f(1.0));
  let inside = all(abs(entry) <= vec2f(POOL_HALF));
  let c = textureSampleLevel(caustic_tex, samp, uv, 0.0).rgb;
  return select(vec3f(0.0), c, inside);
}

// Irradiance on a surface with unit normal `n` at an underwater point `pos`: sky fill that fades with
// depth, plus the sun after its slanted path down through the water and the caustics.
fn pl_irradiance_water(pos: vec3f, n: vec3f) -> vec3f {
  let s = max(-pos.y, 0.0);
  let travel = s / max(-P.sun_water.y, 0.05);
  let t_sun = absorb_transmittance(P.sigma, travel);
  let facing = max(dot(n, -P.sun_water), 0.0);
  let sun = pl_sun_irradiance() * facing * t_sun * pl_caustics_at(pos);
  let fill = pl_sky_ambient() * absorb_transmittance(P.sigma, s);
  return sun + fill;
}

fn pl_irradiance_air(pos: vec3f, n: vec3f, shadow: f32) -> vec3f {
  return pl_sun_irradiance() * max(dot(n, P.sun), 0.0) * shadow + pl_sky_ambient();
}

// --- Materials --------------------------------------------------------------------------------------

// Pool tiles on the floor (normal up) and the walls, in world units; `aa` is a footprint in world units.
fn pl_tiles(pos: vec3f, n: vec3f, aa: f32) -> vec3f {
  var p = pos.xz;
  if (abs(n.x) > 0.5) {
    p = vec2f(pos.z, pos.y);
  } else if (abs(n.z) > 0.5) {
    p = vec2f(pos.x, pos.y);
  }
  return backdrop_pool_tiles(p * P.tile_scale, min(aa * P.tile_scale, 0.5));
}

// The deck around the pool: big pale slabs, tiles desaturated and warmed.
fn pl_deck_albedo(pos: vec3f, aa: f32) -> vec3f {
  let t = backdrop_pool_tiles(pos.xz * 1.1, min(aa * 1.1, 0.5));
  return vec3f(max(luminance(t), 0.5)) * vec3f(1.1, 0.92, 0.74) * 0.5;
}

// The ball at a surface point: diffuse and a highlight above water; under water the underwater
// irradiance and the water's tint. `v` is the unit vector from the point toward the viewer.
fn pl_ball_colour(pos: vec3f, n: vec3f, v: vec3f) -> vec3f {
  let albedo = P.ball_color;
  if (pos.y > 0.0) {
    let lit = albedo * pl_irradiance_air(pos, n, 1.0);
    let h = normalize(P.sun + v);
    let spec = pow(max(dot(n, h), 0.0), 90.0) * 1.2 * max(dot(n, P.sun), 0.0);
    let fres = fresnel_water(max(dot(n, v), 0.0));
    let sky = sky_radiance_ex(reflect(-v, n), P.sun, 0.3, 0.0) * fres * 0.6;
    return lit + pl_sun_irradiance() * spec + sky;
  }
  return albedo * pl_irradiance_water(pos, n);
}

// --- Tracing ----------------------------------------------------------------------------------------

// What a ray leaving the water surface upward sees: the ball, the walls above the waterline, or the sky.
fn pl_trace_air(o: vec3f, d: vec3f, aa: f32) -> vec3f {
  if (P.quality != QUALITY_FULL) {
    return sky_radiance(d, P.sun);
  }
  let ex = pl_exit(o, d, 0.0, P.rim);
  var t = ex.t;
  var hit_ball = false;
  if (P.ball.w > 0.0) {
    let tb = pl_ray_sphere(o, d, P.ball.xyz, P.ball.w);
    if (tb > 0.0 && tb < t) {
      t = tb;
      hit_ball = true;
    }
  }
  if (!hit_ball && ex.n.y < -0.5) {
    return sky_radiance(d, P.sun);
  }
  let pos = o + d * t;
  if (hit_ball) {
    let n = normalize(pos - P.ball.xyz);
    return pl_ball_colour(pos, n, -d);
  }
  let albedo = pl_tiles(pos, ex.n, aa * (1.0 + t));
  return albedo * pl_irradiance_air(pos, ex.n, pl_ball_shadow(pos));
}

// What a ray going down from the water surface sees: the ball, a wall or the floor, lit by what
// reaches it under water and seen through the water between.
fn pl_trace_water(o: vec3f, d: vec3f, aa: f32) -> vec3f {
  let ex = pl_exit(o, d, -P.depth, 0.0);
  var t = ex.t;
  var hit_ball = false;
  if (P.ball.w > 0.0 && P.quality == QUALITY_FULL) {
    let tb = pl_ray_sphere(o, d, P.ball.xyz, P.ball.w);
    if (tb > 0.0 && tb < t) {
      t = tb;
      hit_ball = true;
    }
  }
  let pos = o + d * t;
  var lit = vec3f(0.0);
  if (hit_ball) {
    let n = normalize(pos - P.ball.xyz);
    lit = P.ball_color * pl_irradiance_water(pos, n);
  } else {
    lit = pl_tiles(pos, ex.n, aa * (1.0 + t)) * pl_irradiance_water(pos, ex.n);
  }
  return absorb_composite(lit, P.sigma, P.turbidity, P.scatter, t);
}

// --- Sky --------------------------------------------------------------------------------------------

@fragment fn sky_fs(in: FullscreenVarying) -> @location(0) vec4f {
  let dir = camera_world_ray(cam, in.uv);
  return pl_output(sky_radiance(dir, P.sun), in.position.xy);
}

// --- Deck, walls and floor --------------------------------------------------------------------------

struct MeshVarying {
  @builtin(position) position: vec4f,
  @location(0) world: vec3f,
  @location(1) normal: vec3f,
}

@vertex fn deck_vs(@location(0) position: vec3f, @location(1) normal: vec3f) -> MeshVarying {
  var out: MeshVarying;
  let world = vec3f(position.x, P.rim, position.z);
  out.world = world;
  out.normal = normal;
  out.position = cam.view_proj * vec4f(world, 1.0);
  return out;
}

@fragment fn deck_fs(in: MeshVarying) -> @location(0) vec4f {
  let footprint = length(fwidth(in.world));
  if (abs(in.world.x) < POOL_HALF && abs(in.world.z) < POOL_HALF) {
    discard;
  }
  let albedo = pl_deck_albedo(in.world, footprint);
  var colour = albedo * pl_irradiance_air(in.world, vec3f(0.0, 1.0, 0.0), pl_ball_shadow(in.world));
  // Fog toward the horizon so the deck's far edge is never seen.
  let dist = length(in.world - cam.position);
  let fog = 1.0 - exp(-0.045 * dist);
  colour = mix(colour, sky_horizon_color(P.sun, 0.3), fog);
  return pl_output(colour, in.position.xy);
}

// The pool's inside: a unit box (position in -0.5..0.5, outward normal) stretched to the pool and
// drawn with the camera-facing side culled, so the near walls are cut away.
@vertex fn box_vs(@location(0) position: vec3f, @location(1) normal: vec3f) -> MeshVarying {
  var out: MeshVarying;
  let world = vec3f(
    position.x * 2.0 * POOL_HALF,
    mix(-P.depth, P.rim, position.y + 0.5),
    position.z * 2.0 * POOL_HALF,
  );
  out.world = world;
  out.normal = -normal;
  out.position = cam.view_proj * vec4f(world, 1.0);
  return out;
}

@fragment fn box_fs(in: MeshVarying) -> @location(0) vec4f {
  let footprint = length(fwidth(in.world));
  let n = normalize(in.normal);
  let albedo = pl_tiles(in.world, n, footprint);
  var colour = vec3f(0.0);
  if (in.world.y > 0.0) {
    colour = albedo * pl_irradiance_air(in.world, n, pl_ball_shadow(in.world));
  } else {
    // Only reached through a crack beside the water mesh: lit and tinted as under water.
    let lit = albedo * pl_irradiance_water(in.world, n);
    colour = absorb_composite(lit, P.sigma, P.turbidity, P.scatter, -in.world.y * 1.3);
  }
  return pl_output(colour, in.position.xy);
}

// --- The ball ---------------------------------------------------------------------------------------

@vertex fn ball_vs(@location(0) position: vec3f, @location(1) normal: vec3f) -> MeshVarying {
  var out: MeshVarying;
  // The mesh is a unit-radius sphere (position == normal).
  let world = P.ball.xyz + normal * P.ball.w;
  out.world = world;
  out.normal = normal;
  out.position = cam.view_proj * vec4f(world, 1.0);
  return out;
}

@fragment fn ball_fs(in: MeshVarying) -> @location(0) vec4f {
  let n = normalize(in.normal);
  let v = normalize(cam.position - in.world);
  var colour = pl_ball_colour(in.world, n, v);
  if (in.world.y <= 0.0) {
    // Seen through a gap beside the water mesh: add the water between.
    colour = absorb_composite(colour, P.sigma, P.turbidity, P.scatter, -in.world.y * 1.2);
  }
  return pl_output(colour, in.position.xy);
}

// --- The water --------------------------------------------------------------------------------------

struct WaterVarying {
  @builtin(position) position: vec4f,
  @location(0) world: vec3f,
  @location(1) uv: vec2f,
}

@vertex fn water_vs(@builtin(vertex_index) index: u32) -> WaterVarying {
  let row = P.mesh_cells + 1u;
  let cell = vec2f(f32(index % row), f32(index / row));
  let uv = cell / f32(P.mesh_cells);
  let height = textureSampleLevel(hf_tex, samp, uv, 0.0).x * P.height_scale;
  let xz = (uv * 2.0 - 1.0) * POOL_HALF;
  var out: WaterVarying;
  out.world = vec3f(xz.x, height, xz.y);
  out.uv = uv;
  out.position = cam.view_proj * vec4f(out.world, 1.0);
  return out;
}

@fragment fn water_fs(in: WaterVarying) -> @location(0) vec4f {
  // World footprint of a pixel on the surface (derivatives first, in uniform control flow).
  let footprint = length(fwidth(in.world));
  let s = textureSampleLevel(hf_tex, samp, in.uv, 0.0);
  let n = normalize(vec3f(-P.wave_scale * s.z, 1.0, -P.wave_scale * s.w));
  let v = normalize(in.world - cam.position);
  let cos_i = clamp(dot(-v, n), 0.0, 1.0);

  var colour = vec3f(0.0);
  if (P.view == VIEW_CAUSTICS) {
    // Debug: the floor's caustics through the surface.
    let refracted = refract(v, n, 1.0 / P.ior);
    let t = pl_exit(in.world, refracted, -P.depth, 0.0);
    let hit = in.world + refracted * t.t;
    colour = pl_caustics_at(hit) * 0.5;
    return pl_output(colour, in.position.xy);
  }

  let fresnel = fresnel_water(cos_i);
  var reflected_dir = reflect(v, n);
  reflected_dir.y = max(reflected_dir.y, 0.02);
  reflected_dir = normalize(reflected_dir);
  let refracted_dir = refract(v, n, 1.0 / P.ior);

  let reflection = pl_trace_air(in.world, reflected_dir, footprint * 1.5);
  var refraction = reflection;
  var weight = 1.0;
  if (dot(refracted_dir, refracted_dir) > 0.5 && refracted_dir.y < -0.01) {
    refraction = pl_trace_water(in.world, normalize(refracted_dir), footprint * 1.5);
    weight = fresnel;
  }
  colour = mix(refraction, reflection, weight);
  return pl_output(colour, in.position.xy);
}

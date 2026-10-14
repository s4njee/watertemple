// shallow.wgsl -- the shallow-water scene, every draw in one module: a sky, the terrain and its skirt as a
// grid displaced by the solver's bed height, and the water surface as a second grid lifted by the depth.
// The solver publishes two textures (src/wgsl/shallow-water.wgsl): water (depth, velocity, foam) and
// ground (bed height, surface slope, wetness). Vertices read the heights, fragments read the same
// textures per pixel, so the shading is smoother than the mesh.
//
// The water is an alpha-blended sheet whose coverage is a soft function of depth, so the shoreline is a
// smooth contour of the depth field and never steps with the cells. Its colour: the bed seen through the
// water (looked up where the refracted ray lands, lit, then absorbed and scattered by the path length,
// absorption.wgsl), a Fresnel-weighted sky reflection, a sun glint, ripples that travel with the flow
// (flow.wgsl, two-phase advection of a noise slope field) and foam: the solver's foam, which is born at
// bores, thin fast sheets and fronts and carried with the flow, plus a faint lace at the waterline.
//
// The terrain: sand that turns to rock where it is steep or high, darkened by the solver's ground
// wetness (which lags behind a retreating shore), lit by the sun with a short shadow march through the
// height field and a sky fill. The same albedo function colours the bed seen through the water.
//
// Bindings, group 0 (shared by every entry point): 0 camera, 1 params, 2 water texture, 3 ground
// texture (both rgba16float), 4 filtering sampler.
#include "camera.wgsl"
#include "fullscreen.wgsl"
#include "color.wgsl"
#include "math.wgsl"
#include "noise.wgsl"
#include "sky.wgsl"
#include "fresnel.wgsl"
#include "specular.wgsl"
#include "absorption.wgsl"
#include "backdrop.wgsl"
#include "foam.wgsl"
#include "flow.wgsl"
#include "wetness.wgsl"
#include "shallow-water.wgsl"
#include "scene_params.wgsl"

@group(0) @binding(0) var<uniform> cam: Camera;
@group(0) @binding(1) var<uniform> P: SceneParams;
@group(0) @binding(2) var water_tex: texture_2d<f32>;
@group(0) @binding(3) var ground_tex: texture_2d<f32>;
@group(0) @binding(4) var samp: sampler;

const VIEW_BEAUTY: u32 = 0u;
const VIEW_DEPTH: u32 = 1u;
const VIEW_SPEED: u32 = 2u;
const VIEW_SURFACE: u32 = 3u;
const VIEW_FOAM: u32 = 4u;
const IOR: f32 = 1.333;

// --- Space ----------------------------------------------------------------------------------------

fn sw_world_xz(uv: vec2f) -> vec2f {
  return (uv - vec2f(0.5)) * P.domain;
}

fn sw_uv(xz: vec2f) -> vec2f {
  return xz / P.domain + vec2f(0.5);
}

fn sw_bed_at(uv: vec2f) -> f32 {
  return textureSampleLevel(ground_tex, samp, uv, 0.0).x;
}

// --- Light ----------------------------------------------------------------------------------------

fn sw_sun_irradiance() -> vec3f {
  return sky_sun_color(P.sun) * P.sun_power;
}

fn sw_sky_ambient() -> vec3f {
  return sky_ambient(P.sun) * 1.25;
}

fn sw_output(c: vec3f, frag: vec2f) -> vec3f {
  let mapped = tonemap_aces(max(c, vec3f(0.0)) * P.exposure);
  return clamp(linear_to_srgb(mapped) + vec3f(sky_dither(frag) / 255.0), vec3f(0.0), vec3f(1.0));
}

// Light factor (0 shadowed, 1 lit) from a short march along the sun's direction through the bed.
fn sw_sun_shadow(p: vec3f) -> f32 {
  let horizontal = max(length(P.sun.xz), 0.05);
  var lit = 1.0;
  for (var i = 1; i <= 9; i++) {
    let t = f32(i) / 9.0;
    let reach = 0.12 + 6.0 * t * t;
    let q = p + P.sun * (reach / horizontal);
    let uv = sw_uv(q.xz);
    if (any(uv < vec2f(0.0)) || any(uv > vec2f(1.0))) {
      break;
    }
    let over = sw_bed_at(uv) - (q.y + 0.02);
    lit = min(lit, 1.0 - smoothstep(0.0, 0.3 + 0.12 * reach, over));
  }
  return lit;
}

// --- Materials ------------------------------------------------------------------------------------

// Sand turning to rock where the ground is steep or high. `detail` 1 adds pebbles (the terrain pass),
// 0 is the cheap version for the bed seen through water.
fn sw_ground_albedo(xz: vec2f, bed: f32, slope: f32, detail: f32) -> vec3f {
  var sand: vec3f;
  if (detail > 0.5) {
    sand = backdrop_sand(xz * 1.25, 0.28) * vec3f(0.95, 0.9, 0.82);
  } else {
    sand = srgb_to_linear(vec3f(0.84, 0.74, 0.54)) * (0.88 + 0.16 * gnoise_2d(xz * 1.1 + 7.0));
  }
  let grain = gnoise_2d(xz * 2.3) * 0.5 + gnoise_2d(xz * 7.0 + 3.0) * 0.3;
  let rock = srgb_to_linear(vec3f(0.62, 0.55, 0.47) * mix(1.0, 0.62, P.palette.y)) * (0.75 + 0.35 * grain) * (0.92 + 0.08 * sin(bed * 38.0 + grain * 3.0));
  let rockiness = max(
    max(smoothstep(0.32, 0.75, slope), smoothstep(1.05, 1.5, bed)),
    P.palette.y * smoothstep(0.3, 0.55, bed + 0.15 * grain),
  );
  // Meadow on the gentle ground of the scenes that have one: scrub and grass in patches.
  let clump = smoothstep(-0.25, 0.3, gnoise_2d(xz * 0.33 + 9.0) + 0.3 * gnoise_2d(xz * 1.4));
  let grass = srgb_to_linear(vec3f(0.36, 0.45, 0.2)) * (0.65 + 0.5 * (0.5 + 0.5 * gnoise_2d(xz * 3.1)));
  let green = P.palette.x * clump * (1.0 - smoothstep(0.2, 0.5, slope)) * (1.0 - smoothstep(1.6, 2.4, bed));
  return mix(mix(sand, grass, green), rock, rockiness);
}

// Wet ground is darker and a little greyer. (wet_albedo from wetness.wgsl saturates the sand to
// orange at the shoreline, which reads as a painted line.)
fn sw_wet_albedo(albedo: vec3f, wetness: f32) -> vec3f {
  let w = clamp(wetness, 0.0, 1.0);
  let dark = albedo * mix(vec3f(1.0), vec3f(0.5, 0.47, 0.45), w);
  return dark;
}

// A thin ring where the pointer is, drawn on whatever surface is under it.
fn sw_brush_ring(xz: vec2f) -> f32 {
  if (P.brush.w <= 0.0) {
    return 0.0;
  }
  let d = length(xz - P.brush.xy);
  let w = 0.035 + 0.02 * P.brush.z;
  return P.brush.w * (1.0 - smoothstep(0.0, w, abs(d - P.brush.z)));
}

// --- Sky ------------------------------------------------------------------------------------------

@fragment fn sky_fs(in: FullscreenVarying) -> @location(0) vec4f {
  let dir = camera_world_ray(cam, in.uv);
  // Above the horizon the real sky. Below it, which is nearly all a camera looking down at a diorama
  // sees, a soft vignetted backdrop in the horizon's colour instead of a flat grey ground.
  var colour = sky_radiance(dir, P.sun);
  let r = length((in.uv - vec2f(0.5, 0.45)) * vec2f(1.1, 1.3));
  let backdrop = sky_horizon_color(P.sun, 0.3) * mix(0.85, 0.32, smoothstep(0.1, 0.85, r));
  colour = mix(backdrop, colour, smoothstep(-0.02, 0.1, dir.y));
  return vec4f(sw_output(colour, in.position.xy), 1.0);
}

// --- Terrain --------------------------------------------------------------------------------------

struct GridVarying {
  @builtin(position) position: vec4f,
  @location(0) world: vec3f,
  @location(1) uv: vec2f,
}

fn sw_grid_uv(index: u32) -> vec2f {
  let row = P.mesh.x + 1u;
  return vec2f(f32(index % row), f32(index / row)) / vec2f(f32(P.mesh.x), f32(P.mesh.y));
}

@vertex fn terrain_vs(@builtin(vertex_index) index: u32) -> GridVarying {
  let uv = sw_grid_uv(index);
  let bed = sw_bed_at(uv);
  let xz = sw_world_xz(uv);
  var out: GridVarying;
  out.world = vec3f(xz.x, bed, xz.y);
  out.uv = uv;
  out.position = cam.view_proj * vec4f(out.world, 1.0);
  return out;
}

@fragment fn terrain_fs(in: GridVarying) -> @location(0) vec4f {
  let g = textureSampleLevel(ground_tex, samp, in.uv, 0.0);
  let w = textureSampleLevel(water_tex, samp, in.uv, 0.0);
  // Under water that is fully opaque (the water pass replaces what is below it) there is nothing to
  // shade: skipping it halves the work wherever there is a lake. (Three edge widths of depth, not one,
  // so a glancing ray through the translucent shoreline never finds the black.)
  if (P.view == VIEW_BEAUTY && swe_depth(w) >= 3.0 * P.edge) {
    return vec4f(0.0, 0.0, 0.0, 1.0);
  }
  let n = swe_sample_bed_normal(ground_tex, samp, in.uv, P.texel, P.cell);
  let slope = length(n.xz) / max(n.y, 1e-3);
  let ndl = max(dot(n, P.sun), 0.0);
  let shadow = sw_sun_shadow(in.world + n * 0.03);

  var colour: vec3f;
  if (P.view != VIEW_BEAUTY) {
    // Debug: a quiet grey relief with contour lines, so the false-colour water reads on top.
    let contour = 1.0 - smoothstep(0.0, 0.06, abs(fract(in.world.y * 4.0 + 0.5) - 0.5)) * 0.6;
    colour = vec3f(0.16) * (0.35 + 0.65 * ndl) * contour;
  } else {
    var albedo = ground_albedo_wet(in.world, slope, g, w);
    // Curvature as cheap occlusion: hollows darker, ridges lighter, so relief reads in flat light.
    let r = 0.55;
    let avg = 0.25 * (
      sw_bed_at(in.uv + vec2f(r / P.domain.x, 0.0)) + sw_bed_at(in.uv - vec2f(r / P.domain.x, 0.0)) +
      sw_bed_at(in.uv + vec2f(0.0, r / P.domain.y)) + sw_bed_at(in.uv - vec2f(0.0, r / P.domain.y))
    );
    albedo *= clamp(0.85 + 0.75 * (in.world.y - avg), 0.55, 1.1);
    let bounce = sw_sun_irradiance() * vec3f(0.5, 0.42, 0.3) * 0.22 * (1.0 - n.y);
    colour = albedo * (sw_sun_irradiance() * ndl * shadow + sw_sky_ambient() * (0.55 + 0.45 * n.y) + bounce);
  }
  colour += vec3f(1.0, 1.0, 1.0) * sw_brush_ring(in.world.xz) * 0.35;
  return vec4f(sw_output(colour, in.position.xy), 1.0);
}

fn ground_albedo_wet(world: vec3f, slope: f32, g: vec4f, w: vec4f) -> vec3f {
  let dry = sw_ground_albedo(world.xz, world.y, slope, 1.0);
  let wetness = max(swe_ground_wetness(g), swe_wet_mask(swe_depth(w), 0.01));
  return sw_wet_albedo(dry, wetness);
}

// --- The skirt: the diorama's sides, a strip per edge from the bed down to a base ------------------

struct SkirtVarying {
  @builtin(position) position: vec4f,
  @location(0) world: vec3f,
  @location(1) normal: vec3f,
}

@vertex fn skirt_vs(
  @builtin(vertex_index) index: u32,
  @builtin(instance_index) side: u32,
) -> SkirtVarying {
  let sideways = (side & 1u) == 1u;
  let count = select(P.mesh.x, P.mesh.y, sideways);
  let along = min(index >> 1u, count);
  let t = f32(along) / f32(count);
  var uv = vec2f(t, 0.0);
  var normal = vec3f(0.0, 0.0, -1.0);
  if (side == 1u) {
    uv = vec2f(1.0, t);
    normal = vec3f(1.0, 0.0, 0.0);
  } else if (side == 2u) {
    uv = vec2f(1.0 - t, 1.0);
    normal = vec3f(0.0, 0.0, 1.0);
  } else if (side == 3u) {
    uv = vec2f(0.0, 1.0 - t);
    normal = vec3f(-1.0, 0.0, 0.0);
  }
  let top = (index & 1u) == 0u;
  let xz = sw_world_xz(uv);
  let y = select(-P.skirt, sw_bed_at(uv), top);
  var out: SkirtVarying;
  out.world = vec3f(xz.x, y, xz.y);
  out.normal = normal;
  out.position = cam.view_proj * vec4f(out.world, 1.0);
  return out;
}

@fragment fn skirt_fs(in: SkirtVarying) -> @location(0) vec4f {
  let n = normalize(in.normal);
  let along = in.world.x * abs(n.z) + in.world.z * abs(n.x);
  let layers = gnoise_2d(vec2f(along * 0.5, in.world.y * 3.2));
  let band = 0.5 + 0.5 * sin(in.world.y * 14.0 + layers * 2.5);
  let earth = mix(srgb_to_linear(vec3f(0.36, 0.29, 0.22)), srgb_to_linear(vec3f(0.24, 0.19, 0.15)), band);
  let depth_fade = 0.4 + 0.6 * smoothstep(-P.skirt, 0.0, in.world.y);
  let colour = earth * depth_fade * (sw_sun_irradiance() * max(dot(n, P.sun), 0.0) * 0.28 + sw_sky_ambient() * 0.6);
  return vec4f(sw_output(colour, in.position.xy), 1.0);
}

// --- The water ------------------------------------------------------------------------------------

@vertex fn water_vs(@builtin(vertex_index) index: u32) -> GridVarying {
  let uv = sw_grid_uv(index);
  let g = textureSampleLevel(ground_tex, samp, uv, 0.0);
  let w = textureSampleLevel(water_tex, samp, uv, 0.0);
  let xz = sw_world_xz(uv);
  var out: GridVarying;
  out.world = vec3f(xz.x, swe_surface(w, g), xz.y);
  out.uv = uv;
  out.position = cam.view_proj * vec4f(out.world, 1.0);
  return out;
}

// Ripples carried by the flow: a noise slope field, two copies on phases half a cycle apart.
fn sw_ripples(xz: vec2f, velocity: vec2f, speed: f32) -> vec2f {
  let scale = 1.7;
  let flow = velocity * scale;
  let cycle = 1.1;
  let offset = flow_phase_noise(xz * 0.35) * 0.9;
  let ph = flow_phases(xz * scale, flow, P.time, cycle, offset);
  let t = P.time * 0.25;
  let a = gnoise_3d_d(vec3f(ph.uv_a, t)).yz + 0.5 * gnoise_3d_d(vec3f(ph.uv_a * 2.3 + 5.0, t * 1.3)).yz;
  let b = gnoise_3d_d(vec3f(ph.uv_b, t)).yz + 0.5 * gnoise_3d_d(vec3f(ph.uv_b * 2.3 + 5.0, t * 1.3)).yz;
  let slope = flow_blend_slope(a, b, ph.weight_a, ph.weight_b);
  let fast = flow_strength(velocity * 0.3, 0.25, 1.0);
  return slope * P.ripple * 0.05 * fast;
}

@fragment fn water_fs(in: GridVarying) -> @location(0) vec4f {
  let w = textureSampleLevel(water_tex, samp, in.uv, 0.0);
  let g = textureSampleLevel(ground_tex, samp, in.uv, 0.0);
  let depth = swe_depth(w);
  let cover = swe_wet_mask(depth, P.edge);
  if (cover < 0.004) {
    discard;
  }
  let velocity = swe_velocity(w);
  let speed = swe_speed(w);
  let v = normalize(cam.position - in.world);

  // Surface normal: the solver's slope plus the flow-carried ripples (calm water is glassy).
  let slope = swe_surface_slope(g) + sw_ripples(in.world.xz, velocity, speed) * smoothstep(0.0, 0.05, depth);
  let n = normalize(vec3f(-slope.x, 1.0, -slope.y));
  let ndv = max(dot(n, v), 0.0);

  // Debug views: false colour on the water, the terrain left grey.
  if (P.view == VIEW_DEPTH) {
    let c = palette_cosine(clamp(depth / 1.3, 0.0, 1.0), vec3f(0.5), vec3f(0.5), vec3f(1.0), vec3f(0.55, 0.65, 0.75));
    return vec4f(sw_output(c * 1.4, in.position.xy), 1.0);
  }
  if (P.view == VIEW_SPEED) {
    let c = palette_cosine(clamp(speed / 5.0, 0.0, 1.0), vec3f(0.5), vec3f(0.5), vec3f(1.0), vec3f(0.0, 0.33, 0.67));
    return vec4f(sw_output(c * 1.4, in.position.xy), 1.0);
  }
  if (P.view == VIEW_SURFACE) {
    let c = palette_cosine(clamp(swe_surface(w, g) / 2.5, 0.0, 1.0), vec3f(0.5), vec3f(0.5), vec3f(1.0), vec3f(0.0, 0.33, 0.67));
    let lines = 1.0 - smoothstep(0.0, 0.05, abs(fract(swe_surface(w, g) * 5.0 + 0.5) - 0.5)) * 0.5;
    return vec4f(sw_output(c * 1.4 * lines, in.position.xy), 1.0);
  }
  if (P.view == VIEW_FOAM) {
    return vec4f(sw_output(vec3f(swe_foam(w)) * 2.0, in.position.xy), 1.0);
  }

  let shadow = sw_sun_shadow(in.world + vec3f(0.0, 0.02, 0.0));
  let kind = absorb_water(i32(P.water_kind));

  // The bed seen through the water: where the refracted ray lands, lit and darkened by the water.
  let refracted = refract(-v, n, 1.0 / IOR);
  let path_v = depth * P.density;
  let reach = depth / max(-refracted.y, 0.25);
  let bed_xz = in.world.xz + refracted.xz * reach;
  let bed_uv = clamp(sw_uv(bed_xz), vec2f(0.001), vec2f(0.999));
  let bed_g = textureSampleLevel(ground_tex, samp, bed_uv, 0.0);
  let bed_n = swe_sample_bed_normal(ground_tex, samp, bed_uv, P.texel, P.cell);
  let bed_slope = length(bed_n.xz) / max(bed_n.y, 1e-3);
  let bed_albedo = sw_wet_albedo(sw_ground_albedo(bed_xz, bed_g.x, bed_slope, 0.0), 1.0);
  let sun_in_water = absorb_transmittance(kind.sigma, path_v / max(P.sun.y, 0.25));
  let bed_lit = bed_albedo * (
    sw_sun_irradiance() * max(dot(bed_n, P.sun), 0.0) * shadow * sun_in_water + sw_sky_ambient() * 0.8
  );
  let path = absorb_path_length(path_v, max(ndv, 0.1), IOR);
  let scatter = kind.scatter * (sw_sky_ambient() * 0.9 + sw_sun_irradiance() * 0.12 * shadow);
  var colour = absorb_composite(bed_lit, kind.sigma, kind.turbidity, scatter, path);

  // Reflection of the sky and the sun's glint.
  let fresnel = fresnel_water(ndv);
  var reflected = reflect(-v, n);
  reflected.y = max(reflected.y, 0.02);
  let sky = sky_radiance_ex(normalize(reflected), P.sun, 0.3, 0.0);
  let rough = 0.04 + 0.1 * flow_strength(velocity * 0.3, 0.0, 1.0);
  let glint = spec_sun(n, v, P.sun, rough, SKY_SUN_ANGULAR_RADIUS) * sky_sun_color(P.sun) * SKY_SUN_RADIANCE * shadow;
  colour = colour * (1.0 - fresnel) + sky * fresnel + glint * P.shine;

  // Foam: the solver's (carried with the flow) plus a lace at the waterline.
  let lace_noise = 0.5 + 0.5 * gnoise_2d(in.world.xz * 3.0 + vec2f(0.0, P.time * 0.2));
  let shore = foam_mask(depth, 0.045, lace_noise, 0.7) * 0.5;
  let coverage = clamp(swe_foam(w) * P.foam_amount + shore * P.foam_amount, 0.0, 1.0);
  if (coverage > 0.001) {
    let thickness = foam_texture(in.world.xz * 8.0, P.time * 1.2);
    let foam = foam_dissolve(thickness, coverage, 0.5);
    let foam_colour = vec3f(0.9, 0.93, 0.95) * (sw_sky_ambient() * 0.8 + sw_sun_irradiance() * max(n.y * 0.7 + 0.3, 0.0) * shadow * 0.6);
    colour = mix(colour, foam_colour, foam * 0.92);
  }

  colour += vec3f(1.0) * sw_brush_ring(in.world.xz) * 0.5;
  let srgb = sw_output(colour, in.position.xy);
  return vec4f(srgb * cover, cover);
}

// --- The water's cross-section on the diorama's sides ------------------------------------------------
//
// Where water reaches the edge of the grid it would otherwise float above the skirt. A strip of
// translucent water from the bed up to the surface closes it, like the glass of a tank.

struct SectionVarying {
  @builtin(position) position: vec4f,
  @location(0) world: vec3f,
  @location(1) uv: vec2f,
  @location(2) bed: f32,
}

@vertex fn section_vs(
  @builtin(vertex_index) index: u32,
  @builtin(instance_index) side: u32,
) -> SectionVarying {
  let sideways = (side & 1u) == 1u;
  let count = select(P.mesh.x, P.mesh.y, sideways);
  let t = f32(min(index >> 1u, count)) / f32(count);
  var uv = vec2f(t, 0.0);
  if (side == 1u) {
    uv = vec2f(1.0, t);
  } else if (side == 2u) {
    uv = vec2f(1.0 - t, 1.0);
  } else if (side == 3u) {
    uv = vec2f(0.0, 1.0 - t);
  }
  let g = textureSampleLevel(ground_tex, samp, uv, 0.0);
  let w = textureSampleLevel(water_tex, samp, uv, 0.0);
  let top = (index & 1u) == 0u;
  let xz = sw_world_xz(uv);
  var out: SectionVarying;
  out.world = vec3f(xz.x, select(g.x, swe_surface(w, g), top), xz.y);
  out.uv = uv;
  out.bed = g.x;
  out.position = cam.view_proj * vec4f(out.world, 1.0);
  return out;
}

@fragment fn section_fs(in: SectionVarying) -> @location(0) vec4f {
  let w = textureSampleLevel(water_tex, samp, in.uv, 0.0);
  let depth = swe_depth(w);
  let cover = swe_wet_mask(depth, P.edge);
  if (cover < 0.01 || P.view != VIEW_BEAUTY) {
    discard;
  }
  let kind = absorb_water(i32(P.water_kind));
  let amb = sw_sky_ambient() * 0.9 + sw_sun_irradiance() * 0.08;
  let deep = absorb_deep_color(kind.sigma, kind.turbidity, kind.scatter * amb);
  // Clear near the surface, murky toward the bed, with a bright line along the top edge.
  let up = clamp((in.world.y - in.bed) / max(depth, 1e-3), 0.0, 1.0);
  let thick = clamp(depth * P.density * 0.35, 0.0, 1.0);
  var colour = deep * mix(0.55, 1.0, up) + vec3f(0.9, 0.95, 1.0) * smoothstep(0.93, 1.0, up) * 0.35;
  let alpha = cover * mix(0.35, 0.85, thick);
  return vec4f(sw_output(colour, in.position.xy) * alpha, alpha);
}

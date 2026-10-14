#include "fullscreen.wgsl"
#include "normals.wgsl"
#include "flow.wgsl"
#include "color.wgsl"
#include "math.wgsl"
#include "params.wgsl"

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var flow_map: texture_2d<f32>;
@group(0) @binding(2) var flow_samp: sampler;
@group(0) @binding(3) var normal_map: texture_2d<f32>;
@group(0) @binding(4) var normal_samp: sampler;

// The flow map is 2 world units wide and 1 tall, periodic in x (see createFlowMap).
const WORLD_W: f32 = 2.0;
// World units per second of a flow of 1 at speed 1.
const FLOW_UNITS: f32 = 0.3;
// Where the oblique viewer is, relative to the water (x right, y up, z toward the bottom of the screen).
const VIEW: vec3f = vec3f(0.0, 0.86, 0.5);

struct Layer {
  slope: vec2f,   // blended slope xy of the two phases (variance preserving)
  height: f32,    // the same blend of the height channel
  weight_a: f32,
}

// One scale of ripple: two phases of the normal map advected by the flow, per flow_phases.
fn flow_layer(world: vec2f, flow: vec2f, scale: f32, seed: f32) -> Layer {
  let q = world * scale + vec2f(seed, seed * 0.37);
  // flow is in world units per second; the lookup position q is in tiles, hence * scale.
  let velocity = flow * (params.speed * FLOW_UNITS * scale);
  let offset = params.phase_noise * flow_phase_noise(world * 5.0 + vec2f(seed * 3.1, 0.0));
  let ph = flow_phases(q, velocity, params.time, params.cycle, offset);
  // Gradients of the unadvected position: the advected uv jumps at each phase reset, which would
  // otherwise pick a blurry mip along thin seams (see flow.wgsl, "Mips").
  let ddx = dpdx(q);
  let ddy = dpdy(q);
  let a = textureSampleGrad(normal_map, normal_samp, ph.uv_a, ddx, ddy);
  let b = textureSampleGrad(normal_map, normal_samp, ph.uv_b, ddx, ddy);
  var layer: Layer;
  layer.slope = flow_blend_slope(a.xy, b.xy, ph.weight_a, ph.weight_b);
  layer.height = flow_blend_slope(vec2f(a.z, 0.0), vec2f(b.z, 0.0), ph.weight_a, ph.weight_b).x;
  layer.weight_a = ph.weight_a;
  return layer;
}

fn hsv(h: f32, s: f32, v: f32) -> vec3f {
  let k = fract(vec3f(h) + vec3f(0.0, 2.0 / 3.0, 1.0 / 3.0));
  return v * mix(vec3f(1.0), clamp(abs(k * 6.0 - 3.0) - 1.0, vec3f(0.0), vec3f(1.0)), s);
}

fn sky(r: vec3f, sun: vec3f) -> vec3f {
  let horizon = vec3f(0.75, 0.86, 0.98);
  let zenith = vec3f(0.03, 0.12, 0.40);
  var c = mix(horizon, zenith, pow(clamp(r.y, 0.0, 1.0), 1.3));
  let s = max(dot(r, sun), 0.0);
  c += vec3f(1.0, 0.82, 0.55) * (pow(s, 300.0) * 8.0 + pow(s, 40.0) * 0.2);
  return c;
}

@fragment fn fs(in: FullscreenVarying) -> @location(0) vec4f {
  // World position: the view is `1 / zoom` world units tall, centred on the middle of the river.
  let world = vec2f((in.uv.x - 0.5) * params.aspect / params.zoom + 1.0, (in.uv.y - 0.5) / params.zoom + 0.5);
  let map = textureSampleLevel(flow_map, flow_samp, vec2f(world.x / WORLD_W, world.y), 0.0);
  let flow = flow_decode(map);
  let foam_mask = map.b;
  let depth = map.a;

  // Two scales of ripple, each with its own phase offsets so their resets never coincide.
  let big = flow_layer(world, flow, params.tile, 0.0);
  let fine = flow_layer(world, flow, params.tile * 2.3, 11.7);
  let calm = flow_strength(flow, 0.45, 1.35) * params.strength;
  let n_big = normal_scale(normal_from_xy(big.slope), calm);
  let n_fine = normal_scale(normal_from_xy(fine.slope), calm * 0.7);
  let n = normal_blend_whiteout(n_big, n_fine);
  let nw = normal_to_world_up(n);

  // --- shading: a sky reflection and a body colour that follows the water depth -------------------
  let sun_elevation = radians(params.sun_height);
  let sun = normalize(vec3f(0.12, sin(sun_elevation), -cos(sun_elevation)));
  let v = normalize(VIEW);
  var r = reflect(-v, nw);
  r.y = abs(r.y);
  let fresnel = 0.02 + 0.98 * pow(1.0 - clamp(dot(nw, v), 0.0, 1.0), 5.0);
  let bed = mix(params.shallow, params.deep, smoothstep(0.0, 0.75, depth));
  // Seen from above the plain Fresnel reflection is only a few percent, which hides the ripples.
  // Real rivers show them through refracted light on the bed and through glints, so both are faked
  // here with the same normal: a wrapped N.L on the body colour and a broad Blinn-Phong highlight.
  let wrap = 0.55 + 0.9 * clamp(dot(nw, normalize(sun + vec3f(0.0, 0.6, 0.0))), 0.0, 1.0);
  let half_vector = normalize(sun + v);
  let glint = pow(clamp(dot(nw, half_vector), 0.0, 1.0), 220.0);
  var water = mix(bed * wrap, sky(r, sun), fresnel) + vec3f(1.0, 0.9, 0.75) * (glint * 1.6);

  // Foam: the baked mask says where, a thresholded height pattern (advected like the ripples)
  // says what it looks like.
  let pattern = smoothstep(0.1, 0.28, 0.5 * (big.height + fine.height) + 0.2);
  let foam = clamp(foam_mask * params.foam * (0.35 + 0.9 * pattern), 0.0, 1.0);
  let diffuse = 0.55 + 0.45 * clamp(dot(nw, sun), 0.0, 1.0);
  water = mix(water, vec3f(0.85, 0.9, 0.92) * diffuse, foam);

  // Land: banks and rocks, a static texture from the same map.
  let still = textureSampleLevel(normal_map, normal_samp, world * 1.7, 0.0);
  let fine_still = textureSampleLevel(normal_map, normal_samp, world * 6.0, 0.0);
  let grain = clamp(0.5 + 1.0 * still.z + 0.6 * fine_still.z, 0.0, 1.0);
  let ground = mix(vec3f(0.03, 0.026, 0.02), vec3f(0.1, 0.082, 0.062), grain);
  let wet = 1.0 - smoothstep(0.045, 0.2, depth);
  let land = 1.0 - smoothstep(0.035, 0.06, depth);
  var color = mix(water * (1.0 - 0.35 * wet), ground * (1.0 - 0.0 * wet), land);

  // --- debug views -----------------------------------------------------------------------------
  if (params.debug == 1u) {
    let speed = flow_speed(flow);
    color = select(// Hue is the direction, exaggerated 2.5x about "right" (cyan) so small bends are visible.
    hsv(atan2(flow.y, flow.x) * 2.5 / TAU + 0.5, 1.0, 0.2 + 0.8 * speed), vec3f(0.02), land > 0.5);
    return vec4f(color, 1.0);
  }
  if (params.debug == 2u) {
    return vec4f(vec3f(big.weight_a), 1.0);
  }
  return vec4f(linear_to_srgb(tonemap_aces(color)), 1.0);
}

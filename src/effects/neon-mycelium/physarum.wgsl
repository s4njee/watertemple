// Adapted from HoloGPU's Physarum: three trail sensors, atomic deposits, diffusion and decay.
#include "mycelium_params.wgsl"
#include "sdf.wgsl"
struct Agent { position: vec2f, heading: f32, seed: u32 };
struct Deposit { r: atomic<u32>, g: atomic<u32>, b: atomic<u32>, hits: atomic<u32> };
@group(0) @binding(0) var<uniform> P: MyceliumParams;
@group(0) @binding(1) var trail: texture_2d<f32>;
@group(0) @binding(2) var next: texture_storage_2d<rgba16float, write>;
@group(0) @binding(3) var<storage, read_write> agents: array<Agent>;
@group(0) @binding(4) var<storage, read_write> deposits: array<Deposit>;
@group(0) @binding(5) var walls: texture_2d<f32>;
@group(0) @binding(6) var next_walls: texture_storage_2d<rgba16float, write>;

fn hash(n: u32) -> u32 {
  var x = n;
  x = (x ^ (x >> 16u)) * 0x7feb352du;
  x = (x ^ (x >> 15u)) * 0x846ca68bu;
  return x ^ (x >> 16u);
}
fn random(n: u32) -> f32 { return f32(hash(n)) / 4294967296.0; }
fn cell(uv: vec2f) -> vec2i { return clamp(vec2i(uv * vec2f(P.grid)), vec2i(0), vec2i(P.grid) - 1); }
fn wrap_cell(p: vec2i) -> vec2i { let n = vec2i(P.grid); return ((p % n) + n) % n; }
fn load_trail(p: vec2i) -> vec4f { return textureLoad(trail, wrap_cell(p), 0); }
fn sample_trail(uv: vec2f) -> vec4f {
  let q = uv * vec2f(P.grid) - 0.5;
  let b = vec2i(floor(q)); let t = fract(q);
  return mix(mix(load_trail(b), load_trail(b + vec2i(1,0)), t.x),
    mix(load_trail(b + vec2i(0,1)), load_trail(b + vec2i(1,1)), t.x), t.y);
}
fn wall_distance(uv: vec2f) -> f32 { return textureLoad(walls, cell(uv), 0).r; }
fn wall_normal(uv: vec2f) -> vec2f {
  let p = cell(uv); let n = vec2i(P.grid) - 1;
  let x = textureLoad(walls, clamp(p + vec2i(1,0), vec2i(0), n), 0).r
    - textureLoad(walls, clamp(p - vec2i(1,0), vec2i(0), n), 0).r;
  let y = textureLoad(walls, clamp(p + vec2i(0,1), vec2i(0), n), 0).r
    - textureLoad(walls, clamp(p - vec2i(0,1), vec2i(0), n), 0).r;
  return normalize(vec2f(x,y) + vec2f(1e-7, 0.0));
}
fn safe_point(point: vec2f) -> vec2f {
  var p = clamp(point, vec2f(0.003), vec2f(0.997));
  let margin = 2.0 / f32(P.grid.y);
  for (var i = 0u; i < 4u; i++) {
    let d = wall_distance(p);
    if (d >= margin) { break; }
    p += wall_normal(p) * (margin - d) / vec2f(P.aspect, 1.0);
    p = clamp(p, vec2f(0.003), vec2f(0.997));
  }
  return p;
}
fn sensor(p: vec2f, angle: f32) -> f32 {
  let uv = fract(p + P.sensor_distance / vec2f(P.grid) * vec2f(cos(angle), sin(angle)) + 1.0);
  let distance = wall_distance(uv);
  if (distance < 1.5 / f32(P.grid.y)) { return -100.0; }
  let ink = sample_trail(uv);
  // A mild nutrient band guides growth near the content without imposing an orbit.
  let nutrient = exp(-pow((distance - P.shell) / 0.04, 2.0)) * P.pull;
  return dot(ink.rgb, vec3f(0.3333)) + nutrient * 0.012;
}

@compute @workgroup_size(8,8) fn mask(@builtin(global_invocation_id) id: vec3u) {
  if (any(id.xy >= P.grid)) { return; }
  let p = (vec2f(id.xy) + 0.5) / vec2f(P.grid) * vec2f(P.aspect, 1.0);
  var distance = 2.0;
  for (var i = 0u; i < P.rect_count; i++) {
    let box = P.boxes[i];
    distance = min(distance, sd_rounded_box_2d(p - box.xy, box.zw, P.extras[i].x));
  }
  textureStore(next_walls, vec2i(id.xy), vec4f(distance, 0, 0, 1));
}

@compute @workgroup_size(64) fn initialize(@builtin(global_invocation_id) id: vec3u) {
  if (id.x >= P.count) { return; }
  let seed = hash(id.x + 0x91c15u);
  let angle = random(seed + 19u) * 6.2831853;
  var p = vec2f(random(seed + 7u), random(seed + 47u));
  if (P.rect_count > 0u && id.x % 4u == 0u) {
    let index = hash(seed + 33u) % P.rect_count;
    if (P.extras[index].y > 0.5) {
      let box = P.boxes[index];
      let direction = vec2f(cos(angle), sin(angle));
      let shell = 0.018 + random(seed + 17u) * 0.09;
      let extent = box.zw + shell;
      let scale = min(extent.x / max(abs(direction.x), 1e-4), extent.y / max(abs(direction.y), 1e-4));
      p = (box.xy + direction * scale) / vec2f(P.aspect, 1.0);
    }
  }
  agents[id.x] = Agent(safe_point(p), angle, seed);
}

@compute @workgroup_size(64) fn advance(@builtin(global_invocation_id) id: vec3u) {
  if (id.x >= P.count) { return; }
  var a = agents[id.x];
  a.position = safe_point(a.position);
  a.seed = hash(a.seed + 13u);
  let forward = sensor(a.position, a.heading);
  let left = sensor(a.position, a.heading - P.sensor_angle);
  let right = sensor(a.position, a.heading + P.sensor_angle);
  if (forward < left || forward < right) {
    if (left > right) { a.heading -= P.turn; }
    else if (right > left) { a.heading += P.turn; }
    else { a.heading += select(-P.turn, P.turn, random(a.seed) > 0.5); }
  }
  // Very occasional exploration stops mature networks from locking into a single route.
  if (random(a.seed + 3u) < 0.025) { a.heading += (random(a.seed + 5u) - 0.5) * 1.5; }
  // The viewport is a window onto the colony, not a wall that collects a permanent border.
  var proposed = fract(a.position + P.speed / vec2f(P.grid) * vec2f(cos(a.heading), sin(a.heading)) + 1.0);
  if (wall_distance(proposed) < 1.5 / f32(P.grid.y)) {
    a.heading += select(-1.3, 1.3, random(a.seed + 31u) > 0.5);
    proposed = safe_point(proposed);
  }
  a.position = proposed;
  // Short exploratory lifetimes keep established routes from becoming a static outline.
  if (random(a.seed + 71u) < 0.0025) {
    a.position = safe_point(vec2f(random(a.seed + 91u), random(a.seed + 119u)));
    a.heading = random(a.seed + 139u) * 6.2831853;
  }
  if (wall_distance(a.position) > 1.0 / f32(P.grid.y)) {
    let c = cell(a.position);
    let index = u32(c.x) + u32(c.y) * P.grid.x;
    let tint = P.colours[id.x % 3u].rgb;
    atomicAdd(&deposits[index].r, u32(tint.r * 1024.0));
    atomicAdd(&deposits[index].g, u32(tint.g * 1024.0));
    atomicAdd(&deposits[index].b, u32(tint.b * 1024.0));
    atomicAdd(&deposits[index].hits, 1024u);
  }
  agents[id.x] = a;
}

@compute @workgroup_size(8,8) fn diffuse(@builtin(global_invocation_id) id: vec3u) {
  if (any(id.xy >= P.grid)) { return; }
  let p = vec2i(id.xy);
  if (wall_distance((vec2f(id.xy) + 0.5) / vec2f(P.grid)) < 1.0 / f32(P.grid.y)) {
    textureStore(next, p, vec4f(0)); return;
  }
  var sum = vec4f(0); var weight = 0.0;
  for (var y = -1; y <= 1; y++) { for (var x = -1; x <= 1; x++) {
    let q = wrap_cell(p + vec2i(x,y));
    if (textureLoad(walls, q, 0).r >= 1.0 / f32(P.grid.y)) { sum += load_trail(q); weight += 1.0; }
  } }
  let center = load_trail(p);
  let index = id.x + id.y * P.grid.x;
  let added = vec4f(f32(atomicLoad(&deposits[index].r)), f32(atomicLoad(&deposits[index].g)),
    f32(atomicLoad(&deposits[index].b)), f32(atomicLoad(&deposits[index].hits))) / 1024.0 * P.deposit;
  let retention = exp(-P.decay / 60.0);
  let value = min(vec4f(48.0), mix(center, sum / max(weight, 1.0), P.diffusion) * retention + added);
  // A press plants a small fresh patch on open ground.
  let uv = (vec2f(id.xy) + 0.5) / vec2f(P.grid);
  let brush = exp(-dot((uv - P.pointer.xy) * vec2f(P.aspect,1.0), (uv - P.pointer.xy) * vec2f(P.aspect,1.0)) / 0.00012) * P.pointer.z;
  textureStore(next, p, value + vec4f(P.colours[0].rgb, 1.0) * brush * 0.045);
}

#include "life_params.wgsl"
@group(0) @binding(0) var<uniform> P: LifeParams;
@group(0) @binding(1) var cells: texture_2d<u32>;
@group(0) @binding(2) var next: texture_storage_2d<r32uint, write>;
fn wrap(p: vec2i) -> vec2i { let n = vec2i(P.grid); return ((p % n) + n) % n; }
fn load(p: vec2i) -> u32 { return textureLoad(cells, wrap(p), 0).r; }
// bit 0: alive; bits 8–15: age; bits 16–23: visual echo. Only bit 0 affects the rules.
@compute @workgroup_size(8,8) fn advance(@builtin(global_invocation_id) id: vec3u) {
  if (any(id.xy >= P.grid)) { return; }
  let p = vec2i(id.xy); let old = load(p);
  var neighbours = 0u;
  for (var y = -1; y <= 1; y++) { for (var x = -1; x <= 1; x++) {
    if (x != 0 || y != 0) { neighbours += load(p + vec2i(x,y)) & 1u; }
  }}
  let alive = neighbours == 3u || ((old & 1u) != 0u && neighbours == 2u);
  let age = select(0u, min(255u, ((old >> 8u) & 255u) + 1u), alive);
  let previous_echo = (old >> 16u) & 255u;
  let echo = select(select(0u, previous_echo - min(previous_echo, 1u), previous_echo > 0u), 9u, (old & 1u) != 0u);
  textureStore(next,p,vec4u(select(0u,1u,alive) | (age << 8u) | (echo << 16u),0u,0u,0u));
}
@compute @workgroup_size(8,8) fn plant(@builtin(global_invocation_id) id: vec3u) {
  if (any(id.xy >= P.grid)) { return; }
  let p = vec2i(id.xy); let d = wrap(p - vec2i(P.plant.xy * vec2f(P.grid)));
  let glider = (d.y == 0 && d.x == 1) || (d.y == 1 && d.x == 2) || (d.y == 2 && d.x <= 2);
  textureStore(next,p,vec4u(select(load(p),257u,glider),0u,0u,0u));
}

fn hash(n: u32) -> u32 {
  var x = n;
  x = (x ^ (x >> 16u)) * 0x7feb352du;
  x = (x ^ (x >> 15u)) * 0x846ca68bu;
  return x ^ (x >> 16u);
}
// Add small pockets of random life without clearing the evolving world or its surviving cells.
@compute @workgroup_size(8,8) fn sprinkle(@builtin(global_invocation_id) id: vec3u) {
  if (any(id.xy >= P.grid)) { return; }
  let p = vec2i(id.xy);
  var value = load(p);
  for (var i = 0u; i < P.patches; i++) {
    let center = vec2i(vec2u(hash(P.seed + i * 17u) % P.grid.x, hash(P.seed + i * 17u + 7u) % P.grid.y));
    let delta = abs(p-center);
    let distance = vec2f(min(delta,vec2i(P.grid)-delta));
    let noise = hash(id.x + id.y * P.grid.x + P.seed + i * 131u);
    if (dot(distance,distance) <= 36.0 && noise % 100u < 28u && (value & 1u) == 0u) { value = 257u; }
  }
  textureStore(next,p,vec4u(value,0u,0u,0u));
}

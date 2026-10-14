// Synthetic particle sources for the fluid-surface demo. Each particle's position is a pure function
// of its index and the time, so one dispatch fills the whole buffer and the demo scales to a million
// particles without any CPU work or neighbour search. Points are spread with the R3 Kronecker
// sequence in exact integer arithmetic (an even, low-discrepancy fill of a volume, like the roughly
// regular spacing a real solver keeps) and then mapped into the shape of the source.
//
// World: +Y up, the floor at y = 0, a tank 4 wide (x, z in -2..2).
//   kind 0  ball    two blobs that wobble, bridge, split and merge
//   kind 1  wave    a slab of liquid whose surface carries waves and a travelling swell
//   kind 2  stream  a jet pouring from above into a pool, breaking into drops, splashing
// The w component is a foam amount in 0..1 (steep wave crests, the neck of a splitting blob, the
// spray and the churn at the impact point), which the renderer's foam hook can show.

#include "hash.wgsl"
#include "source-params.wgsl"

@group(0) @binding(0) var<uniform> params: SourceParams;
@group(0) @binding(1) var<storage, read_write> positions: array<vec4f>;

const TAU = 6.283185307;
const GRAVITY = 9.8;

// R3 sequence in fixed point: i * alpha mod 1 is exact for any i, where a float32 product would
// have lost all its low bits by i = 10^5. The sequence alone is too regular (its points line up in
// rows that show as stripes on a surface), so each point is jittered by about one spacing, which
// leaves the even coverage and the roughly regular spacing of a settled particle fluid.
fn kronecker(i: u32) -> vec3f {
  let h = vec3u(i * 3518319154u, i * 2882110344u, i * 2360945575u) + vec3u(2147483648u);
  let u = vec3f(h >> vec3u(8u)) * (1.0 / 16777216.0);
  let jitter = to_unorm_3d(pcg_3d(vec3u(i, 0x9e3779b9u, 77u))) - vec3f(0.5);
  return fract(u + jitter * (1.1 * pow(1.0 / f32(max(params.count, 1u)), 1.0 / 3.0)));
}

// An equal-volume map of the unit cube onto the unit ball.
fn ball_point(u: vec3f) -> vec3f {
  let r = pow(u.x, 1.0 / 3.0);
  let cz = 2.0 * u.y - 1.0;
  let sz = sqrt(max(1.0 - cz * cz, 0.0));
  let phi = TAU * u.z;
  return r * vec3f(sz * cos(phi), cz, sz * sin(phi));
}

fn disc_point(u: vec2f) -> vec2f {
  let r = sqrt(u.x);
  let phi = TAU * u.y;
  return r * vec2f(cos(phi), sin(phi));
}

// --- ball ----------------------------------------------------------------------------------------

fn source_ball(i: u32) -> vec4f {
  let t = params.time;
  let blob = i & 1u;
  let u = ball_point(kronecker((i >> 1u) + 11u));
  let side = select(-1.0, 1.0, blob == 1u);
  let blob_radius = 0.8;

  // Separation swings 0 to 2.4 and back every 9 s; the axis between the blobs turns slowly.
  let s = 0.5 - 0.5 * cos(TAU * t / 9.0);
  let d = 2.4 * s;
  let angle = 0.3 * t + 0.4;
  let axis = vec3f(cos(angle), 0.0, sin(angle));
  let toward_other = -side * axis;
  let centre = vec3f(0.0, 1.0 + 0.1 * sin(0.9 * t), 0.0) + axis * side * d * 0.5;

  // A smooth displacement field: the blob wobbles without its particles changing neighbours.
  let wobble = 0.16 * vec3f(
    sin(2.1 * u.y + 1.7 * t + side),
    sin(1.9 * u.z - 1.3 * t),
    sin(2.3 * u.x + 0.9 * t - side),
  ) * (0.4 + length(u));

  // While the blobs are close, the facing hemispheres reach for each other: a neck forms,
  // thins, and breaks as the blobs part.
  let bridge = smoothstep(1.2, 1.6, d) * (1.0 - smoothstep(1.8, 2.2, d));
  let facing = smoothstep(-0.1, 1.0, dot(u, toward_other));
  let pull = toward_other * blob_radius * 0.75 * bridge * facing * facing;

  let p = centre + (u + wobble) * blob_radius + pull;
  let foam = clamp(bridge * facing + 0.35 * pow(length(u), 3.0) * (0.3 + 0.7 * s), 0.0, 1.0);
  return vec4f(p, foam);
}

// --- wave ----------------------------------------------------------------------------------------

fn wave_height(x: f32, z: f32, t: f32) -> f32 {
  var h = 0.9 + 0.15 * sin(1.7 * x + 1.1 * t) + 0.13 * sin(2.0 * z - 0.8 * t)
    + 0.08 * sin(2.9 * (x + z) + 1.9 * t);
  // A swell running along z that sweeps back and forth across the tank.
  let c = 1.5 * sin(0.4 * t);
  h += 0.55 * exp(-1.6 * (x - c) * (x - c)) * (0.65 + 0.35 * sin(1.3 * z + t));
  return h;
}

fn source_wave(i: u32) -> vec4f {
  let t = params.time;
  let u = kronecker(i + 5u);
  let x = (2.0 * u.x - 1.0) * 1.95;
  let z = (2.0 * u.z - 1.0) * 1.95;
  let h = wave_height(x, z, t);
  let e = 0.05;
  let gx = (wave_height(x + e, z, t) - wave_height(x - e, z, t)) / (2.0 * e);
  let gz = (wave_height(x, z + e, t) - wave_height(x, z - e, t)) / (2.0 * e);
  let foam = smoothstep(0.45, 1.1, length(vec2f(gx, gz))) * smoothstep(0.75, 1.0, u.y);
  return vec4f(x, u.y * h, z, foam);
}

// --- stream --------------------------------------------------------------------------------------

const POOL_RADIUS = 1.35;
const POOL_LEVEL = 0.34;
const NOZZLE_HEIGHT = 2.9;
const NOZZLE_SPEED = 1.6;
const NOZZLE_RADIUS = 0.2;

// Time for a drop to fall from the nozzle to the pool: solves h = v t + g t^2 / 2.
fn fall_time() -> f32 {
  let h = NOZZLE_HEIGHT - POOL_LEVEL;
  return (-NOZZLE_SPEED + sqrt(NOZZLE_SPEED * NOZZLE_SPEED + 2.0 * GRAVITY * h)) / GRAVITY;
}

fn pool_height(x: f32, z: f32, t: f32) -> f32 {
  let rho = length(vec2f(x, z));
  // Rings run out from the impact point and fade; the middle churns.
  return POOL_LEVEL
    + 0.05 * sin(11.0 * rho - 7.0 * t) * exp(-1.1 * rho)
    + 0.03 * sin(5.0 * rho - 3.0 * t + 1.0) * exp(-0.4 * rho)
    + 0.05 * exp(-14.0 * rho * rho);
}

fn source_stream(i: u32) -> vec4f {
  let t = params.time;
  let n = params.count;
  let tf = fall_time();
  let pool_n = u32(f32(n) * 0.66);
  let stream_n = u32(f32(n) * 0.26);

  // The pool: a disc of liquid whose surface carries rings from the impact.
  if (i < pool_n) {
    let u = kronecker(i + 3u);
    let xy = disc_point(u.xz) * POOL_RADIUS;
    let h = pool_height(xy.x, xy.y, t);
    let rho = length(xy);
    let foam = smoothstep(0.5, 0.0, rho) * 0.8 + smoothstep(0.85, 1.0, u.y) * 0.06;
    return vec4f(xy.x, params.radius + u.y * (h - params.radius), xy.y, foam);
  }

  // The stream: clusters of particles on ballistic paths. Near the nozzle the clusters are close and
  // fuse into a column; speed grows, so lower down they are far apart and the jet breaks into drops.
  if (i < pool_n + stream_n) {
    let j = i - pool_n;
    let cluster = j / 6u;
    let member = j % 6u;
    let h = to_unorm_3d(pcg_3d(vec3u(cluster, 7u, 0u)));
    let age = fract(h.x + t / (tf * 1.15)) * tf * 1.15;
    if (age > tf) {
      // The last stretch of the cycle is the cluster in the pool, hidden: park it with the pool.
      let u = kronecker(i + 3u);
      let xy = disc_point(u.xz) * POOL_RADIUS * 0.6;
      return vec4f(xy.x, params.radius + u.y * 0.2, xy.y, 0.8);
    }
    let speed = NOZZLE_SPEED + GRAVITY * age;
    // The jet narrows as it speeds up (conservation of flow), and wobbles a little.
    let width = NOZZLE_RADIUS * sqrt(NOZZLE_SPEED / speed);
    let lateral = disc_point(h.yz) * width * 0.7;
    let sway = 0.05 * sin(7.0 * age + 31.0 * h.x);
    let centre = vec3f(
      lateral.x + sway,
      NOZZLE_HEIGHT - NOZZLE_SPEED * age - 0.5 * GRAVITY * age * age,
      lateral.y,
    );
    let local = ball_point(kronecker(member + 40u * (cluster % 7u + 1u))) * (2.0 * params.radius);
    // Stretch along the fall: faster drops are longer.
    let p = centre + vec3f(local.x, local.y * (1.0 + 0.15 * speed), local.z);
    return vec4f(p, 0.0);
  }

  // Splash: drops thrown out of the impact point on parabolas that end back in the pool.
  let k = i - pool_n - stream_n;
  let h = to_unorm_4d(pcg_4d(vec4u(k, 99u, 0u, 0u)));
  let flight_speed = 1.6 + 2.0 * h.y;
  let out_speed = 0.5 + 1.1 * h.z;
  let flight = 2.0 * flight_speed / GRAVITY;
  let phase = fract(h.x + t / (flight * 1.25 + 0.2));
  let tau = phase * (flight * 1.25 + 0.2);
  if (tau > flight) {
    let u = kronecker(i + 3u);
    let xy = disc_point(u.xz) * POOL_RADIUS * 0.4;
    return vec4f(xy.x, params.radius + u.y * 0.15, xy.y, 0.9);
  }
  let direction = TAU * h.w;
  let planar = out_speed * vec2f(cos(direction), sin(direction)) * tau;
  let y = POOL_LEVEL + 0.08 + flight_speed * tau - 0.5 * GRAVITY * tau * tau;
  return vec4f(planar.x, y, planar.y, 1.0);
}

@compute @workgroup_size(64)
fn cs(@builtin(global_invocation_id) id: vec3u) {
  let i = id.x;
  if (i >= params.count) {
    return;
  }
  var result = vec4f(0.0);
  if (params.kind == 0u) {
    result = source_ball(i);
  } else if (params.kind == 1u) {
    result = source_wave(i);
  } else {
    result = source_stream(i);
  }
  positions[i] = result;
}

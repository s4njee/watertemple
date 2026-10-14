// pigment-step.wgsl -- one step of the pigment layer (src/gpu/sim/pigment.ts) on the dye grid: the
// transport of the suspended pigment (diffusion and edge drift), its exchange with the paper
// (deposition and lifting), and the water that makes it all move (wetness). No advection: the flow
// carried the pigment already, this is what happens to it in the water and on the paper.
//
// Bindings (the layout is spelled out in pigment.ts; an entry point uses only some of them):
//   0 U         PigmentUniforms (the generated pigment_uniforms.wgsl)
//   1 dye_in    suspended pigment, rgba16float: rgb = optical density (absorbance) per channel
//   2 stain_in  deposited pigment, rgba16float: rgb = optical density, a = shaped wetness for shading
//   3 wet_in    wetness, r32float, read with textureLoad
//   4 paper     the paper tile, rgba8unorm: r = grain height, g = fibre (wash.wgsl wash_paper_tile),
//               one tile texel per dye texel, repeating
//   5 dye_out   rgba16float storage, the new suspended pigment (a = mean density, for buoyancy)
//   6 stain_out rgba16float storage (cs_step only)
//   7 wet_out   r32float storage (cs_step only)
//
// Entry points
//   cs_transport   diffusion and drift only: dye_in -> dye_out. Run n - 1 times when a step needs n
//                  sub-steps (see pigmentSubsteps in pigment.ts).
//   cs_step        the same transport, then the settle (deposit, lift, fade) and the wetness update:
//                  writes all three outputs. The last (or only) sub-step.
//
// THE MODEL, all of it local to a texel and its four neighbours, in texels and per-step fractions.
//
// 1. Wetness w in [0, 1] is the film of water on the paper. Every rule below looks at the SHAPED
//    wetness ws = smoothstep(wet_lo, wet_hi, w), which is 0 on dry paper, 1 once there is a film, and
//    changes over the thin fringe where a film runs out: the wet edge.
//
// 2. Diffusion. Between a texel and a neighbour the pigment exchanges k * (c_neighbour - c) with
//    k = min(w_diff * wetter * paper, w_cap):
//      w_diff   D dt / dx^2, the diffusivity D (image heights^2 / s) as a per-step fraction
//      wetter   mix(dry_diff, 1, the mean ws of the two texels): pigment barely spreads on dry paper
//      paper    1 + feather * (3 fibre + 1.6 (grain - 0.5)) of the mean of the two texels: streaks of
//               fibre carry pigment further, which is the feathering of a wet edge
//      w_cap    the stability cap: the sum of the four k and of the drift below stays <= 1, so no
//               texel can lose more than it has and nothing goes negative
//    k is symmetric (the same for a -> b and b -> a), so what one texel loses its neighbour gains:
//    diffusion conserves the total pigment exactly, up to the rounding of the 16-bit store. The
//    variance of a spreading point grows by 2 k per step and axis, linearly in time.
//
// 3. Edge drift. Where the film thins the water is drawn out towards the drier paper and carries
//    pigment with it: the bloom of a drop on wet paper, and, where a flat-topped film ends, the
//    "coffee ring" of edge darkening. Across a face from a to b the pigment moves
//    s = edge_v * smoothstep(0.3 edge_g0, edge_g0, |grad w|) * (w_a - w_b) / |grad w| texels per
//    step, from the wetter texel to the drier one, as an UPWIND flux (the donor texel's value
//    leaves): conservative, positive, and it stops at once where the film is uniform. |grad w| is the
//    gradient of the raw wetness at the face (normal difference and mean tangential central
//    difference), so the drift is isotropic: with a per-axis sign it drew diamonds and crosses round
//    every drop. The smoothstep is a dead zone: a film that is only nearly flat
//    (the ripple of a stroke laid in capsules is a percent or two) must not pull pigment into bars,
//    which it does for any gradient however small if the drift is saturating (seen: stripes across
//    every stroke, one per frame of the drag).
//
// 4. Settling. Suspended pigment c moves into the paper (the stain S) at the rate
//      kd = absorb * mix(1, wet_stick, ws) * gran      deposited: 1 - exp(-kd dt) of c per step
//    fast on dry paper, slow under water, and faster in the valleys of the paper than on its
//    peaks (gran = 1 + granulation * 1.4 * (0.5 - grain)): granulation. Re-wetted stain dissolves
//    again at the rate lift * ws^2 (back-runs). What leaves c enters S and the other way round, so
//    c + S is conserved by this pass too. A residue below dep_floor deposits in one go (half
//    floats cannot add a tiny amount to a big stain, and it would otherwise hang about for ever).
//    Both then fade by exp(-fade dt): washing out, 0 keeps everything.
//
// 5. Wetness. It spreads by capillary action (wet_spread, a plain diffusion), then dries: w * dry_decay
//    (exponential: the edge of a wash stays where it is for a long time and thins) minus dry_lin
//    (so it does reach zero), and never falls under wet_floor (1 = the whole sheet is under water).

#include "pigment_uniforms.wgsl"

@group(0) @binding(0) var<uniform> U: PigmentUniforms;
@group(0) @binding(1) var dye_in: texture_2d<f32>;
@group(0) @binding(2) var stain_in: texture_2d<f32>;
@group(0) @binding(3) var wet_in: texture_2d<f32>;
@group(0) @binding(4) var paper: texture_2d<f32>;
@group(0) @binding(5) var dye_out: texture_storage_2d<rgba16float, write>;
@group(0) @binding(6) var stain_out: texture_storage_2d<rgba16float, write>;
@group(0) @binding(7) var wet_out: texture_storage_2d<r32float, write>;

// The texel itself for a neighbour outside the grid: no flux through the edge (a closed box).
fn texel(p: vec2i) -> vec2i {
  return clamp(p, vec2i(0), vec2i(U.size) - vec2i(1));
}

fn shaped(w: f32) -> f32 {
  return smoothstep(U.wet_lo, U.wet_hi, w);
}

fn paper_at(p: vec2i) -> vec2f {
  return textureLoad(paper, p % vec2i(textureDimensions(paper)), 0).xy;
}

// 0 for NaN and +-Inf, x otherwise (tests the exponent bits: `x != x` may be folded away).
fn finite_or_zero(x: f32) -> f32 {
  return select(x, 0.0, (bitcast<u32>(x) & 0x7f800000u) == 0x7f800000u);
}

fn finite3(v: vec3f) -> vec3f {
  return vec3f(finite_or_zero(v.x), finite_or_zero(v.y), finite_or_zero(v.z));
}

// Raw wetness at an offset from t.
fn wet_at(t: vec2i, d: vec2i) -> f32 {
  return textureLoad(wet_in, texel(t + d), 0).x;
}

// What this texel gains across the face to its neighbour at offset `d` (a unit step along x or y):
// diffusion plus upwind drift. The neighbour computes the negative of the same number for the same face.
fn exchange(t: vec2i, d: vec2i, c: vec3f, w: f32, p: vec2f) -> vec3f {
  let n = texel(t + d);
  let cn = textureLoad(dye_in, n, 0).rgb;
  let wr = wet_at(t, d);
  let ws = shaped(w);
  let wn = shaped(wr);
  let pn = paper_at(n);
  let wetter = mix(U.dry_diff, 1.0, 0.5 * (ws + wn));
  let fibre = 0.5 * (p.y + pn.y);
  let grain = 0.5 * (p.x + pn.x);
  let rough = max(1.0 + U.feather * (3.0 * fibre + 1.6 * (grain - 0.5)), 0.2);
  let k = min(U.w_diff * wetter * rough, U.w_cap);

  // Drift: along the face normal, with the speed set by the size of the whole gradient of the RAW
  // wetness at the face (the normal difference, and the mean of the two texels' central differences
  // along the face), so a film that thins on a slant drives pigment the same fast as one that thins
  // along an axis, instead of moving it along x and y separately (which draws diamonds). s > 0 moves
  // pigment from this texel to the neighbour (this one is the wetter).
  let side = vec2i(d.y, d.x);
  let along = 0.25 * (wet_at(t, side) - wet_at(t, -side) + wet_at(t + d, side) - wet_at(t + d, -side));
  let normal = w - wr;
  let slope = sqrt(normal * normal + along * along);
  var s = 0.0;
  if (slope > 1e-9) {
    s = U.edge_v * smoothstep(0.3 * U.edge_g0, U.edge_g0, slope) * normal / slope;
  }
  return k * (cn - c) + max(-s, 0.0) * cn - max(s, 0.0) * c;
}

fn transport(t: vec2i) -> vec3f {
  let c = finite3(textureLoad(dye_in, t, 0).rgb);
  let w = textureLoad(wet_in, t, 0).x;
  let p = paper_at(t);
  var out = c;
  out += exchange(t, vec2i(1, 0), c, w, p);
  out += exchange(t, vec2i(-1, 0), c, w, p);
  out += exchange(t, vec2i(0, 1), c, w, p);
  out += exchange(t, vec2i(0, -1), c, w, p);
  return max(out, vec3f(0.0));
}

fn mean3(v: vec3f) -> f32 {
  return (v.x + v.y + v.z) / 3.0;
}

@compute @workgroup_size(16, 16)
fn cs_transport(@builtin(global_invocation_id) id: vec3u) {
  if (any(id.xy >= U.size)) {
    return;
  }
  let t = vec2i(id.xy);
  let c = transport(t);
  textureStore(dye_out, t, vec4f(c, mean3(c)));
}

@compute @workgroup_size(16, 16)
fn cs_step(@builtin(global_invocation_id) id: vec3u) {
  if (any(id.xy >= U.size)) {
    return;
  }
  let t = vec2i(id.xy);
  let c1 = transport(t);

  // Settling.
  let w = textureLoad(wet_in, t, 0).x;
  let ws = shaped(w);
  let p = paper_at(t);
  let s = finite3(textureLoad(stain_in, t, 0).rgb);
  let gran = clamp(1.0 + U.gran * 1.4 * (0.5 - p.x), 0.1, 3.5);
  let kd = U.absorb * mix(1.0, U.wet_stick, ws) * gran;
  var dep = c1 * (1.0 - exp(-kd * U.dt));
  if (kd > 0.0) {
    dep = select(dep, c1, c1 < vec3f(U.dep_floor));
  }
  let lifted = s * (1.0 - exp(-U.lift * ws * ws * U.dt));
  let c2 = min(max(c1 - dep + lifted, vec3f(0.0)) * U.keep, vec3f(U.dye_max));
  let s2 = min(max(s + dep - lifted, vec3f(0.0)) * U.keep, vec3f(U.dye_max));

  // Wetness.
  var lap = textureLoad(wet_in, texel(t + vec2i(1, 0)), 0).x - w;
  lap += textureLoad(wet_in, texel(t + vec2i(-1, 0)), 0).x - w;
  lap += textureLoad(wet_in, texel(t + vec2i(0, 1)), 0).x - w;
  lap += textureLoad(wet_in, texel(t + vec2i(0, -1)), 0).x - w;
  var w2 = (w + U.wet_spread * lap) * U.dry_decay - U.dry_lin;
  w2 = clamp(max(finite_or_zero(w2), U.wet_floor), 0.0, 1.0);

  textureStore(dye_out, t, vec4f(c2, mean3(c2)));
  textureStore(stain_out, t, vec4f(s2, shaped(w2)));
  textureStore(wet_out, t, vec4f(w2, 0.0, 0.0, 0.0));
}

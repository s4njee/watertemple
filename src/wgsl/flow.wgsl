// flow.wgsl -- flow-map helpers: the two-phase advection of Vlachos, "Water Flow in Portal 2".
//
// The problem: to make a texture (normals, foam, dye) travel along a flow field you cannot just
// offset the uv by flow * time, because neighbouring pixels have different flows and the texture
// shears into streaks without bound. The fix is to advect for a LIMITED time, snap back, and hide
// the snap: two copies of the texture run on phases half a cycle apart, each fading to zero weight
// exactly when it resets, so the picture is always made of at least one copy that is young and
// undistorted, and the reset is never seen.
//
// What the caller does (texture sampling stays in the caller, so any texture or layer count works):
//   let flow = flow_decode(textureSampleLevel(flow_map, flow_samp, flow_uv, 0.0)) * speed;
//   let ph   = flow_phases(uv, flow, time, cycle, flow_phase_noise(uv * 6.0) * noise_amount);
//   let a    = textureSampleGrad(tex, samp, ph.uv_a, dpdx(uv), dpdy(uv));    // see "Mips" below
//   let b    = textureSampleGrad(tex, samp, ph.uv_b, dpdx(uv), dpdy(uv));
//   let s    = flow_blend_slope(a.xy, b.xy, ph.weight_a, ph.weight_b);        // normals, see below
//
// Flow encoding (the 8-bit "flow map" texture, e.g. from createFlowMap in src/gpu/bake.ts)
//   rg = velocity in texture axes (x right, y down, the screen's orientation), one unit = the
//   fastest current in the map. Byte 127 is EXACT zero, bytes 0 and 254 are -1 and +1 (255 clamps
//   to +1). The common 0.5 + 0.5 * v encoding cannot store zero in 8 bits (0.5 is not a multiple
//   of 1/255), so still water would drift at 1/255 of full speed and, advected for a whole cycle,
//   creep visibly: keep the byte-127 convention for any 8-bit map you author.
//   flow_encode(v: vec2f) -> vec2f      velocity (|components| <= 1, clamped) to rg in [0, 1]
//   flow_decode(texel: vec4f) -> vec2f  texel rg back to velocity. Filtering the encoded bytes
//                                       bilinearly is fine: decoding is affine, so it commutes
//
// Phases
//   struct FlowPhases { uv_a, uv_b: vec2f; weight_a, weight_b: f32 }
//       uv_a / uv_b   where to sample the texture for the two copies
//       weight_*      triangle-wave fade in [0, 1]; weight_a + weight_b == 1 always
//   flow_phases(uv, flow, time, cycle, phase_offset) -> FlowPhases
//       uv            texture coordinate of the pixel (the same units as `flow` times seconds)
//       flow          velocity in uv units per SECOND (already multiplied by speed and tiling), so
//                     a feature drifts by flow * cycle uv units before its copy resets
//       time          seconds, accumulated by the caller (so changing speed never jumps)
//       cycle         seconds per full phase, > 0 (clamped to >= 1e-3). 1-2 s is typical. Each copy
//                     moves for one cycle then resets; the pair is refreshed every cycle / 2
//       phase_offset  added to the phase, in cycles; use flow_phase_noise() * amount per pixel
//     phase_a = fract(time / cycle + phase_offset), phase_b = fract(phase_a + 0.5),
//     uv_x = uv - flow * cycle * phase_x, weight_x = 1 - |2 * phase_x - 1|.
//     Properties (tested): with flow == 0 both uv equal the input; an offset is exactly zero at the
//     phase's reset, which is also where its weight is zero; the weights are continuous in time.
//   flow_phase_noise(p: vec2f) -> f32
//       smooth per-pixel phase offset in [0, 1] (gradient noise, one octave, `p` in lattice units:
//       uv * 4 .. 12). Multiply by an amount in [0, 1] cycles. WITHOUT it every pixel resets at the
//       same instant and the whole surface pulses at 2 / cycle Hz (the texture contrast dips
//       when the two copies are mixed 50/50 and recovers as one copy takes over). With full-range
//       noise the resets are spread over the whole cycle and the pulse disappears into a
//       steady shimmer. Too-fast noise (high `p` frequency) shows as sparkle, too slow as bands.
//
// Blending the two copies
//   flow_blend_slope(a, b, weight_a, weight_b) -> vec2f
//       for mean-zero data (tangent-space normal xy, height deltas). A plain w_a * a + w_b * b halves
//       the variance at the 50/50 point (two independent copies average out), which shows as the
//       contrast "breathing" with the cycle. This divides by sqrt(w_a^2 + w_b^2) instead (the
//       variance-preserving blend of Heitz and Neyret's histogram-preserving work), keeping the
//       roughness constant. Do NOT use it for colours or masks that have a nonzero mean: blend those
//       with weight_a * a + weight_b * b.
//   flow_strength(flow: vec2f, calm: f32, fast: f32) -> f32
//       normal-strength multiplier from the decoded flow speed: `calm` where the water is still,
//       `fast` where |flow| >= 1 (mix in between). Fast water is rougher, backwaters glassy.
//   flow_speed(flow: vec2f) -> f32       |flow|, clamped to [0, 1] (decoded, before your speed)
//
// Mips: the two uv sets jump by flow * cycle when a phase resets, so hardware derivatives from
// textureSample see a huge uv gradient there and select a blurry mip, drawing a thin seam
// exactly where one copy has weight ~0. Pass the derivatives of the UNOFFSET uv (dpdx(uv), dpdy(uv))
// to textureSampleGrad instead; they are continuous. They must be computed in uniform control flow
// (before any non-uniform `if`, discard or early return).
//
// Pitfalls: a long `cycle` lets each copy stretch along the flow before it resets (the shear grows
// with cycle * |grad flow|); a short one makes the texture visibly refresh or "boil". Rocks and
// banks where the flow turns sharply stretch first. See the flow-map effect notes.
//
// Cost: flow_phases is about 15 ALU ops, flow_phase_noise one gradient noise. The caller pays two
// texture fetches per layer instead of one.

#include "noise.wgsl"

struct FlowPhases {
  uv_a: vec2f,
  uv_b: vec2f,
  weight_a: f32,
  weight_b: f32,
}

// Byte 127 = 0. Byte b maps to (b - 127) / 127, so 254 is +1; the one asymmetric byte (0) is -1.
const _FLOW_ZERO_BYTE: f32 = 127.0;
const _FLOW_SCALE: f32 = 127.0;

fn flow_encode(v: vec2f) -> vec2f {
  return (_FLOW_ZERO_BYTE + _FLOW_SCALE * clamp(v, vec2f(-1.0), vec2f(1.0))) / 255.0;
}

fn flow_decode(texel: vec4f) -> vec2f {
  return clamp((texel.rg * 255.0 - _FLOW_ZERO_BYTE) / _FLOW_SCALE, vec2f(-1.0), vec2f(1.0));
}

fn flow_phases(uv: vec2f, flow: vec2f, time: f32, cycle: f32, phase_offset: f32) -> FlowPhases {
  let c = max(cycle, 1e-3);
  let phase_a = fract(time / c + phase_offset);
  let phase_b = fract(phase_a + 0.5);
  var r: FlowPhases;
  r.uv_a = uv - flow * (c * phase_a);
  r.uv_b = uv - flow * (c * phase_b);
  r.weight_a = 1.0 - abs(2.0 * phase_a - 1.0);
  r.weight_b = 1.0 - abs(2.0 * phase_b - 1.0);
  return r;
}

fn flow_phase_noise(p: vec2f) -> f32 {
  // gnoise_2d reaches about +-0.93, so the stretch fills [0, 1] without clipping often.
  return clamp(0.5 + 0.5 * gnoise_2d(p) * 1.07, 0.0, 1.0);
}

fn flow_blend_slope(a: vec2f, b: vec2f, weight_a: f32, weight_b: f32) -> vec2f {
  let norm = max(sqrt(weight_a * weight_a + weight_b * weight_b), 1e-4);
  return (weight_a * a + weight_b * b) / norm;
}

fn flow_speed(flow: vec2f) -> f32 {
  return min(length(flow), 1.0);
}

fn flow_strength(flow: vec2f, calm: f32, fast: f32) -> f32 {
  return mix(calm, fast, flow_speed(flow));
}

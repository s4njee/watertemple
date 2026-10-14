// shallow-water-core.wgsl -- the maths of the shallow-water solver, shared by the render-pass and the
// compute-pass variant so that both run the SAME code and differ only in how they are launched.
//
// Included (as an effect-local chunk) by shallow-water-render.wgsl and shallow-water-compute.wgsl,
// after the generated `sw_params.wgsl` (struct SimParams). Bindings, group 0:
//   0  P         uniform SimParams
//   1  state_in  rgba32float  the previous state, one texel per cell:
//                             x h   water depth (m, >= 0)
//                             y hu  momentum along +x  (m^2 / s)
//                             z hv  momentum along +y  (m^2 / s); +y is DOWN the texture, like uv
//                             w b   bed height (m)
//   2  aux_in    the upload texture (rgba32float: x bed, y depth) in the upload pass, the history
//                texture (rgba16float: x foam, y wetness memory) in the resolve pass
//
// SCHEME. The 2D shallow-water equations in conservative form,
//     h_t + (hu)_x + (hv)_y = 0
//     (hu)_t + (hu^2 + g h^2 / 2)_x + (huv)_y = -g h b_x - friction
//     (hv)_t + (huv)_x + (hv^2 + g h^2 / 2)_y = -g h b_y - friction
// solved by a first-order finite-volume scheme with the HLL approximate Riemann solver (Einfeldt wave
// speeds, Toro's dry-bed speeds) and Audusse's hydrostatic reconstruction of the interface states:
//   * well balanced: a lake at rest over any terrain stays at rest to rounding error, because the
//     pressure of the reconstructed states cancels the bed slope term exactly;
//   * positivity preserving: h never goes negative while the timestep obeys the CFL rule below, and a
//     cell next to a dry one sees a wall or a dam break, whichever physics says;
//   * conservative: the mass flux through a face is computed from the same two states for the cell on
//     either side, so water only moves, it is never created or lost (apart from sources, the depth
//     cap and open edges);
//   * one pass per step, no flux textures: each cell recomputes the four fluxes around it.
// A solid obstacle is just a bed higher than the water next to it.
//
// STABILITY. The timestep is fixed per solver: dt = cfl dx / (2 lambda_cap), lambda_cap =
// 2 sqrt(g h_max). Every step clamps h <= h_max and then |velocity| <= lambda_cap - sqrt(g h), so no
// cell can have |u| + sqrt(g h) above lambda_cap, whatever happened. (Ritter's dry dam-break solution
// has |u| + c <= 2 sqrt(g h0), which is exactly what lambda_cap allows.) With cfl <= 0.5 this is the
// positivity condition of the HLL flux in two dimensions.

#include "sw_params.wgsl"

@group(0) @binding(0) var<uniform> P: SimParams;
@group(0) @binding(1) var state_in: texture_2d<f32>;
@group(0) @binding(2) var aux_in: texture_2d<f32>;

// Below this depth a cell is dry: no momentum, no wave speed.
const DRY: f32 = 1e-5;
// Velocity desingularisation length: u = m h / (h^2 + EPS^2), which is m / h for h >> EPS and goes
// smoothly to zero as the film vanishes, so a sliver of water cannot carry a huge velocity.
const EPS_U: f32 = 1e-3;
// Visible water: the depth from which a cell counts as wet for shading, foam and the wetness memory.
const WET: f32 = 1e-4;
// The depth from which the ground counts as soaked: a film thinner than a millimetre is not a wet stain.
const WET_GROUND: f32 = 1.5e-3;

fn sw_velocity(h: f32, m: f32) -> f32 {
  return m * h / (h * h + EPS_U * EPS_U);
}

fn sw_in_grid(c: vec2i) -> bool {
  return all(c >= vec2i(0)) && all(c < vec2i(P.size));
}

fn sw_load(c: vec2i) -> vec4f {
  return textureLoad(state_in, clamp(c, vec2i(0), vec2i(P.size) - vec2i(1)), 0);
}

// The state of the neighbour of cell `c` (state `s`) in the axis direction `d`; beyond the grid a
// ghost cell. Wall: the same water with the momentum along the normal reversed (a mirror).
// Open: dry ground at the same height, so water can run off the edge like a free outfall.
fn sw_neighbour(c: vec2i, d: vec2i, s: vec4f) -> vec4f {
  let n = c + d;
  if (sw_in_grid(n)) {
    return textureLoad(state_in, n, 0);
  }
  if (P.edge_open != 0u) {
    return vec4f(0.0, 0.0, 0.0, s.w);
  }
  var g = s;
  if (d.x != 0) {
    g.y = -s.y;
  } else {
    g.z = -s.z;
  }
  return g;
}

// One interface between a cell A (the lower-index side) and a cell B, along the face normal.
// Components are (mass, normal momentum, tangential momentum); `pa` and `pb` are the hydrostatic
// pressure corrections to add to the normal-momentum flux seen by A and by B.
struct Face {
  f: vec3f,
  pa: f32,
  pb: f32,
}

fn sw_face(
  ha: f32, mna: f32, mta: f32, ba: f32,
  hb: f32, mnb: f32, mtb: f32, bb: f32,
) -> Face {
  let g = P.g;
  // Hydrostatic reconstruction: the water on each side above the higher of the two beds.
  let z = max(ba, bb);
  var hl = max(ha + ba - z, 0.0);
  var hr = max(hb + bb - z, 0.0);
  if (hl < DRY) {
    hl = 0.0;
  }
  if (hr < DRY) {
    hr = 0.0;
  }
  var out: Face;
  out.pa = 0.5 * g * (ha * ha - hl * hl);
  out.pb = 0.5 * g * (hb * hb - hr * hr);
  out.f = vec3f(0.0);
  if (hl <= 0.0 && hr <= 0.0) {
    return out;
  }

  // The velocities do not change in the reconstruction.
  let ul = sw_velocity(ha, mna);
  let vl = sw_velocity(ha, mta);
  let ur = sw_velocity(hb, mnb);
  let vr = sw_velocity(hb, mtb);
  let cl = sqrt(g * hl);
  let cr = sqrt(g * hr);

  var sl: f32;
  var sr: f32;
  if (hr <= 0.0) {
    sl = ul - cl;
    sr = ul + 2.0 * cl;
  } else if (hl <= 0.0) {
    sl = ur - 2.0 * cr;
    sr = ur + cr;
  } else {
    let u_star = 0.5 * (ul + ur) + cl - cr;
    let c_star = 0.5 * (cl + cr) + 0.25 * (ul - ur);
    sl = min(ul - cl, u_star - c_star);
    sr = max(ur + cr, u_star + c_star);
  }

  let fl = vec3f(hl * ul, hl * ul * ul + 0.5 * g * hl * hl, hl * ul * vl);
  let fr = vec3f(hr * ur, hr * ur * ur + 0.5 * g * hr * hr, hr * ur * vr);
  if (sl >= 0.0) {
    out.f = fl;
  } else if (sr <= 0.0) {
    out.f = fr;
  } else {
    let ql = vec3f(hl, hl * ul, hl * vl);
    let qr = vec3f(hr, hr * ur, hr * vr);
    out.f = (sr * fl - sl * fr + sl * sr * (qr - ql)) / (sr - sl);
  }
  return out;
}

fn sw_bump(q: f32) -> f32 {
  let t = max(1.0 - q * q, 0.0);
  return t * t;
}

// Sources, pushes, sculpting and level zones, applied once per step. Entry i is two vec4s:
//   kind 0 (a disc):   a = (x, y, radius, depth rate), b = (accel x, accel y, bed rate, 0)
//                      x, y, radius in cells; depth rate = peak m / s of water (negative removes);
//                      accel in m / s^2; bed rate in m / s (positive raises).
//   kind 1 (a level zone): a = (x0, y0, x1, y1) in cells, b = (level, rate, 0, 1)
//                      nudges the surface b + h in the rectangle toward `level` at `rate` per second.
fn sw_apply_sources(c: vec2i, s: vec4f) -> vec4f {
  var h = s.x;
  var hu = s.y;
  var hv = s.z;
  var bed = s.w;
  let p = vec2f(c) + vec2f(0.5);
  for (var i = 0u; i < P.source_count; i++) {
    let a = P.sources[2u * i];
    let e = P.sources[2u * i + 1u];
    if (e.w > 0.5) {
      if (p.x >= a.x && p.x < a.z && p.y >= a.y && p.y < a.w) {
        let k = clamp(e.y * P.dt, 0.0, 1.0);
        h = max(h + k * (e.x - (bed + h)), 0.0);
      }
    } else {
      let w = sw_bump(length(p - a.xy) / max(a.z, 1e-3));
      if (w > 0.0) {
        h = max(h + a.w * P.dt * w, 0.0);
        hu += h * e.x * P.dt * w;
        hv += h * e.y * P.dt * w;
        bed = clamp(bed + e.z * P.dt * w, P.bed_min, P.bed_max);
      }
    }
  }
  return vec4f(h, hu, hv, bed);
}

fn sw_step(c: vec2i) -> vec4f {
  let s = textureLoad(state_in, c, 0);
  let sw = sw_neighbour(c, vec2i(-1, 0), s);
  let se = sw_neighbour(c, vec2i(1, 0), s);
  let sn = sw_neighbour(c, vec2i(0, -1), s);
  let ss = sw_neighbour(c, vec2i(0, 1), s);

  // x faces: normal = x, tangential = y. The cell is B of its west face and A of its east face.
  let fw = sw_face(sw.x, sw.y, sw.z, sw.w, s.x, s.y, s.z, s.w);
  let fe = sw_face(s.x, s.y, s.z, s.w, se.x, se.y, se.z, se.w);
  // y faces: normal = y, tangential = x.
  let fn_ = sw_face(sn.x, sn.z, sn.y, sn.w, s.x, s.z, s.y, s.w);
  let fs = sw_face(s.x, s.z, s.y, s.w, ss.x, ss.z, ss.y, ss.w);

  let r = P.dt / P.dx;
  var h = s.x - r * ((fe.f.x - fw.f.x) + (fs.f.x - fn_.f.x));
  var hu = s.y - r * ((fe.f.y + fe.pa) - (fw.f.y + fw.pb) + (fs.f.z - fn_.f.z));
  var hv = s.z - r * ((fe.f.z - fw.f.z) + (fs.f.y + fs.pa) - (fn_.f.y + fn_.pb));

  h = max(h, 0.0);
  var next = vec4f(h, hu, hv, s.w);
  next.x += P.rain * P.dt;
  if (P.source_count > 0u) {
    next = sw_apply_sources(c, next);
  }
  h = min(next.x, P.h_max);
  hu = next.y;
  hv = next.z;

  if (h < DRY) {
    return vec4f(h, 0.0, 0.0, next.w);
  }

  // Bed friction, treated implicitly so a thin film comes to rest instead of flipping sign:
  // Manning's law, tau = g n^2 |u| u / h^(1/3), plus a linear drag.
  let speed = length(vec2f(hu, hv)) * h / (h * h + EPS_U * EPS_U);
  let manning = P.g * P.manning * P.manning * speed / pow(max(h, EPS_U), 4.0 / 3.0);
  let damp = 1.0 / (1.0 + P.dt * (manning + P.drag));
  hu *= damp;
  hv *= damp;

  // The stability clamp: |u| + sqrt(g h) <= lambda_cap.
  let cap = max(P.lambda_cap - sqrt(P.g * h), 0.0);
  let spd = length(vec2f(sw_velocity(h, hu), sw_velocity(h, hv)));
  if (spd > cap) {
    let k = cap / spd;
    hu *= k;
    hv *= k;
  }
  return vec4f(h, hu, hv, next.w);
}

// Copies the uploaded bed and/or depth (mode 0 both, 1 bed only, 2 depth only) into the state.
fn sw_upload(c: vec2i) -> vec4f {
  let s = textureLoad(state_in, c, 0);
  let u = textureLoad(aux_in, c, 0);
  if (P.upload_mode == 0u) {
    return vec4f(clamp(u.y, 0.0, P.h_max), 0.0, 0.0, clamp(u.x, P.bed_min, P.bed_max));
  }
  if (P.upload_mode == 1u) {
    return vec4f(s.xyz, clamp(u.x, P.bed_min, P.bed_max));
  }
  return vec4f(clamp(u.y, 0.0, P.h_max), 0.0, 0.0, s.w);
}

// --- Resolve: the consumer-facing textures ------------------------------------------------------

struct Resolved {
  water: vec4f,
  ground: vec4f,
  history: vec4f,
}

fn sw_eta_eff(n: vec4f, centre_wet: bool, eta_c: f32) -> f32 {
  if (!centre_wet) {
    return n.w;
  }
  if (n.x >= WET) {
    return n.w + n.x;
  }
  // A dry neighbour presents the centre's own level: the surface is flat up to the shore.
  return eta_c;
}

fn sw_history(p: vec2f) -> vec2f {
  // Bilinear fetch of the history texture at cell coordinates p (cell centres at +0.5).
  let q = p - vec2f(0.5);
  let i = vec2i(floor(q));
  let f = q - floor(q);
  let hi = vec2i(P.size) - vec2i(1);
  let a = textureLoad(aux_in, clamp(i, vec2i(0), hi), 0).xy;
  let b = textureLoad(aux_in, clamp(i + vec2i(1, 0), vec2i(0), hi), 0).xy;
  let cc = textureLoad(aux_in, clamp(i + vec2i(0, 1), vec2i(0), hi), 0).xy;
  let d = textureLoad(aux_in, clamp(i + vec2i(1, 1), vec2i(0), hi), 0).xy;
  return mix(mix(a, b, f.x), mix(cc, d, f.x), f.y);
}

fn sw_resolve(c: vec2i) -> Resolved {
  // The 3x3 neighbourhood, edge cells replicated.
  let s = sw_load(c);
  let nw = sw_load(c + vec2i(-1, -1));
  let nn = sw_load(c + vec2i(0, -1));
  let ne = sw_load(c + vec2i(1, -1));
  let ww = sw_load(c + vec2i(-1, 0));
  let ee = sw_load(c + vec2i(1, 0));
  let sw_ = sw_load(c + vec2i(-1, 1));
  let ss = sw_load(c + vec2i(0, 1));
  let se = sw_load(c + vec2i(1, 1));

  let h = s.x;
  let wet = h >= WET;
  let eta_c = s.w + h;
  let u = sw_velocity(h, s.y);
  let v = sw_velocity(h, s.z);

  // Surface slope: Sobel gradient of the water level where there is water, of the bed where not.
  let e_nw = sw_eta_eff(nw, wet, eta_c);
  let e_nn = sw_eta_eff(nn, wet, eta_c);
  let e_ne = sw_eta_eff(ne, wet, eta_c);
  let e_ww = sw_eta_eff(ww, wet, eta_c);
  let e_ee = sw_eta_eff(ee, wet, eta_c);
  let e_sw = sw_eta_eff(sw_, wet, eta_c);
  let e_ss = sw_eta_eff(ss, wet, eta_c);
  let e_se = sw_eta_eff(se, wet, eta_c);
  let k = 1.0 / (8.0 * P.dx);
  let slope = vec2f(
    ((e_ne + 2.0 * e_ee + e_se) - (e_nw + 2.0 * e_ww + e_sw)) * k,
    ((e_sw + 2.0 * e_ss + e_se) - (e_nw + 2.0 * e_nn + e_ne)) * k,
  );

  // Foam. Where it is born: a bore (the flow converging, depth jumping), thin fast supercritical
  // sheets, and an advancing front (fast water next to dry ground).
  let spd = length(vec2f(u, v));
  let cw = sqrt(P.g * max(h, WET));
  let froude = spd / cw;
  let uw = sw_velocity(ww.x, ww.y);
  let ue = sw_velocity(ee.x, ee.y);
  let vn = sw_velocity(nn.x, nn.z);
  let vs = sw_velocity(ss.x, ss.z);
  // How much the velocity drops across one cell, in units of the wave speed.
  let converge = max(-((ue - uw) + (vs - vn)) * 0.5 / max(cw, 0.05), 0.0);
  let bore = smoothstep(0.08, 0.35, converge) * smoothstep(0.015, 0.08, h);
  // Thin sheets that are fast enough to be supercritical, but not the film that merely creeps.
  let fast = smoothstep(1.0, 2.4, froude) * (1.0 - smoothstep(0.04, 0.3, h)) * smoothstep(0.01, 0.03, h);
  let dry_next = min(min(ww.x, ee.x), min(nn.x, ss.x)) < WET;
  let front = select(0.0, smoothstep(0.6, 2.2, spd) * (1.0 - smoothstep(0.04, 0.3, h)) * smoothstep(0.006, 0.02, h), dry_next);
  let born = clamp(0.9 * bore + 0.5 * fast + 0.55 * front, 0.0, 1.0);

  // Foam drifts with the flow and fades; the wet ground dries out slowly.
  let back = vec2f(c) + vec2f(0.5) - vec2f(u, v) * (P.resolve_dt / P.dx);
  let before = sw_history(back);
  let foam = select(0.0, max(before.x * exp(-P.resolve_dt / max(P.foam_life, 1e-3)), born), wet);
  let damp = max(before.y * exp(-P.resolve_dt / max(P.wet_life, 1e-3)), select(0.0, 1.0, h >= WET_GROUND));

  var r: Resolved;
  r.water = vec4f(h, u, v, foam);
  r.ground = vec4f(s.w, slope.x, slope.y, damp);
  r.history = vec4f(foam, damp, 0.0, 0.0);
  return r;
}

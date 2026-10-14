// obstacle-drive.wgsl -- what the shapes and the tunnel do to the VELOCITY, as a custom pass for the
// solver's `forces` hook (it runs after the built-in forces and before the pressure projection, once per
// step). A fullscreen fragment pass over the staggered velocity texture ((W+1) x (H+1) texels, u in .x
// on the left face of a cell, v in .y on its top face; layout in fluid-core.wgsl) that writes every
// texel.
//
// MOVING SOLIDS. The solver's solids are static: a solid face has velocity zero and the projection
// keeps it so. A solid that MOVES therefore pushes nothing by itself: cells swept by its front are
// simply overwritten (their fluid and dye vanish) and the cells it leaves are born at rest. This pass
// supplies the missing boundary condition the standard way: in a thin band of fluid round each shape
// the NORMAL component of the velocity is relaxed to the normal component of the solid's own velocity
// (translation plus spin), the tangential part is left alone (the fluid slides along the surface).
// The projection that follows turns the discrepancy into a pressure, so the fluid ahead of a moving body
// is pushed round it and the fluid behind is drawn in after it. A static shape gets only "no flow into
// me" in the band, which the projection would enforce anyway.
//
// INFLOW. An optional band at the left edge pulls the velocity toward a target (the wind of the
// tunnel). With the solver's `open` boundary the cells outside are air at pressure 0, which cannot
// carry a velocity, so the inflow is a body-force band instead of a boundary condition: inside the band
// the velocity is blended toward the target by `inlet.w` every step, which the projection then makes
// consistent with the solids downstream.
//
// Bindings: 0 the shape table (ObstacleTable), 1 the velocity to read, 2 the parameters.

#include "fullscreen.wgsl"
#include "obstacle-shapes.wgsl"
#include "obstacle_table.wgsl"
#include "drive_params.wgsl"

@group(0) @binding(0) var<uniform> T: ObstacleTable;
@group(0) @binding(1) var vel: texture_2d<f32>;
@group(0) @binding(2) var<uniform> D: DriveParams;

// The nearest shape within the band of the point: (weight 0..1, shape index). Weight 1 is inside.
fn nearest(p: vec2f, band: f32) -> vec2f {
  var best = vec2f(0.0, -1.0);
  for (var i = 0u; i < T.count; i++) {
    let d = obstacle_sd(T.data[3u * i], T.data[3u * i + 1u], p);
    if (d < band) {
      let w = 1.0 - smoothstep(0.0, band, d);
      if (w > best.x) {
        best = vec2f(w, f32(i));
      }
    }
  }
  return best;
}

fn load_vel(t: vec2i) -> vec2f {
  let size = vec2i(D.grid) + vec2i(1);
  return textureLoad(vel, clamp(t, vec2i(0), size - vec2i(1)), 0).xy;
}

// The fluid velocity vector at a u face (axis 0) or v face (axis 1): the stored component and the
// average of the four surrounding faces of the other one.
fn fluid_at(axis: u32, t: vec2i) -> vec2f {
  if (axis == 0u) {
    let v = 0.25 * (load_vel(t + vec2i(-1, 0)).y + load_vel(t).y
                  + load_vel(t + vec2i(-1, 1)).y + load_vel(t + vec2i(0, 1)).y);
    return vec2f(load_vel(t).x, v);
  }
  let u = 0.25 * (load_vel(t + vec2i(0, -1)).x + load_vel(t + vec2i(1, -1)).x
                + load_vel(t).x + load_vel(t + vec2i(1, 0)).x);
  return vec2f(u, load_vel(t).y);
}

// The new value of one face component.
fn drive(axis: u32, t: vec2i) -> f32 {
  let g = vec2i(D.grid);
  // Position of the face in cells, then in image heights.
  var cells = vec2f(f32(t.x), f32(min(t.y, g.y - 1)) + 0.5);
  if (axis == 1u) {
    cells = vec2f(f32(min(t.x, g.x - 1)) + 0.5, f32(t.y));
  }
  let p = cells / D.grid.y;
  let own = fluid_at(axis, t);
  var value = select(own.y, own.x, axis == 0u);

  if (T.count > 0u && D.coupling > 0.0) {
    let hit = nearest(p, D.band / D.grid.y);
    if (hit.x > 0.0) {
      let i = u32(hit.y);
      let geo = T.data[3u * i];
      let size = T.data[3u * i + 1u];
      let motion = T.data[3u * i + 2u];
      let n = obstacle_normal(geo, size, p);
      let body = clamp_speed(obstacle_velocity(geo, motion, p));
      let along = dot(body, n) - dot(own, n);
      let slip_update = select(n.y, n.x, axis == 0u) * along;
      let stick_update = select(body.y, body.x, axis == 0u) - value;
      // A body that is standing still only gets free slip (the tunnel's pegs must not be sticky); one
      // that moves drags the fluid in the band along.
      let stick = D.stick * smoothstep(0.02, 0.2, length(body));
      value += D.coupling * hit.x * mix(slip_update, stick_update, stick);
    }
  }

  // The inlet: a smooth band at the left edge, in image heights.
  if (D.inlet.w > 0.0) {
    let w = D.inlet.w * (1.0 - smoothstep(D.inlet.z * 0.5, D.inlet.z, p.x));
    value = mix(value, select(D.inlet.y, D.inlet.x, axis == 0u), w);
  }
  return clamp(value, -D.vmax, D.vmax);
}

fn clamp_speed(v: vec2f) -> vec2f {
  let l = length(v);
  return select(v, v * (D.vmax / l), l > D.vmax);
}

@fragment fn fs(in: FullscreenVarying) -> @location(0) vec4f {
  let t = vec2i(in.position.xy);
  return vec4f(drive(0u, t), drive(1u, t), 0.0, 0.0);
}

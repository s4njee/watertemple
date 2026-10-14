import { createFullscreenPipeline, renderFullscreen } from '../fullscreen.ts';
import { noProfiler, type PassProfiler } from '../profiler.ts';
import { createUniformBuffer, defineUniformStruct, uniformArray } from '../uniforms.ts';
import type { Fluid, FluidForceContext } from './fluid.ts';
import driveSource from './obstacle-drive.wgsl';
import rasterSource from './obstacle-raster.wgsl';
import shapesSource from './obstacle-shapes.wgsl';
import { OBSTACLE_FORMAT, type ObstacleRect } from './obstacles.ts';

// Solid shapes for the stable-fluids solver (story 6.5): analytic shapes (a circle, a box, a rounded
// box, a capsule, a wing) with a position, a rotation and a size are drawn into the solver's obstacle
// mask with a render pass, so flow goes AROUND them, and the same table lets a shading pass draw them
// crisply, anti-aliased from the signed distance rather than from the blocky mask.
//
//   const shapes = createObstacleShapes({ device, fluid });
//   shapes.set([{ kind: 'wing', x: 0.4, y: 0.5, size: [0.2, 0.05], rotation: 0.15 }]);
//   // each frame, before fluid.update:
//   shapes.rasterize(encoder);                 // only does work when something changed
//   // and once, if the shapes move and should push the fluid:
//   shapes.attach();                           // installs the moving-boundary pass in fluid.hooks.forces
//   // shading: bind shapes.table (an ObstacleTable uniform) and call obstacle_sd per shape; see
//   // obstacle-shapes.wgsl, and src/effects/fluid-obstacles/ for the pattern.
//
// UNITS. Positions are canvas uv (0..1, origin top-left); lengths (size, radius, velocity) are image
// heights, like the solver's. Rotation is in radians and positive turns the shape CLOCKWISE on the
// screen. Velocities are image heights per second.
//
// THE MASK. One r8unorm texel per solver cell; a cell is solid when its CENTRE is inside a shape. That
// is a staircase, one cell wide, so a circle of N cells across has an outline off by up to half a cell
// (a few percent of its diameter at 50 cells) and the flow past it sees the steps; the shading pass draws the
// exact outline over it. A shape thinner than a cell can fall between cell centres and vanish: keep
// plates at least two cells thick (or pass `grow`, which counts cells whose centre is within `grow`
// image heights of the shape).
//
// MOVING SHAPES. The solver's solid faces are static. A shape that moves just changes the mask:
// the fluid and dye in the cells it covers are removed (the solver zeroes dye and velocity in solid
// cells every step, so nothing is ever left inside a shape), and cells it leaves are born clear and
// at rest. To make a moving shape PUSH the fluid, `attach()` installs a pass in `fluid.hooks.forces` that
// relaxes the normal velocity of the fluid in a thin band round each shape to the shape's own normal
// velocity (translation plus spin), `shape.velocity` and `shape.spin`; the projection that follows does
// the rest. Ahead of the body the fluid is pushed round it; behind it, drawn in. What is not done:
// a true moving-boundary condition inside the pressure solve (that would be a change to the solver),
// so a fast body still sweeps through cells in a single step; keep it under about one cell per step
// (the effect limits the drag speed) and it is stable at any speed, only less exact.

export type ObstacleKind = 'circle' | 'box' | 'rounded-box' | 'capsule' | 'wing';

const KIND_INDEX: Record<ObstacleKind, number> = {
  circle: 0,
  box: 1,
  'rounded-box': 2,
  capsule: 3,
  wing: 4,
};

/** Most shapes in one table. */
export const OBSTACLE_MAX_SHAPES = 64;

export interface ObstacleShape {
  kind: ObstacleKind;
  /** Centre in canvas uv. */
  x: number;
  y: number;
  /**
   * Half extents in image heights, in the shape's own frame (x along its long axis). `circle` uses
   * `size[0]` as the radius; `capsule` has end caps of radius `size[1]`; `wing` is a teardrop with its round
   * leading edge (radius `size[1]`) at the shape's -x and a sharp tail at +x.
   */
  size: readonly [number, number];
  /** Corner radius for `rounded-box`, image heights. Default a quarter of the shorter half extent. */
  radius?: number;
  /** Radians, positive clockwise on the screen. Default 0. */
  rotation?: number;
  /** Velocity of the centre, image heights per second. Default 0. */
  velocity?: readonly [number, number];
  /** Angular velocity, radians per second, positive clockwise on the screen. Default 0. */
  spin?: number;
}

// --- The table --------------------------------------------------------------------------------------

/** The shape table as a uniform struct: three vec4f per shape (see obstacle-shapes.wgsl). */
export const ObstacleTable = defineUniformStruct('ObstacleTable', {
  count: 'u32',
  aspect: 'f32',
  grow: 'f32',
  pad: 'f32',
  data: uniformArray('vec4f', OBSTACLE_MAX_SHAPES * 3),
});

/** The WGSL text of the chunk: `#include "obstacle-shapes.wgsl"` resolves to this when passed in `chunks`. */
export const OBSTACLE_SHAPES_WGSL = shapesSource;
/** The generated `ObstacleTable` struct: pass as the chunk `obstacle_table.wgsl`. */
export const OBSTACLE_TABLE_WGSL = ObstacleTable.wgsl;

/** The chunks any shader including obstacle-shapes.wgsl or the table needs. */
export const OBSTACLE_CHUNKS: Readonly<Record<string, string>> = {
  'obstacle-shapes.wgsl': shapesSource.code,
  ...shapesSource.chunks,
  'obstacle_table.wgsl': ObstacleTable.wgsl,
};

/** Packs shapes into the table's `data` array (three vec4f each). `aspect` is canvas width / height. */
export function packObstacleShapes(
  shapes: readonly ObstacleShape[],
  aspect: number,
): { data: Float32Array; count: number } {
  const count = Math.min(shapes.length, OBSTACLE_MAX_SHAPES);
  const data = new Float32Array(OBSTACLE_MAX_SHAPES * 12);
  for (let i = 0; i < count; i++) {
    const s = shapes[i]!;
    const angle = s.rotation ?? 0;
    const hx = Math.max(Math.abs(s.size[0]), 1e-5);
    const hy = Math.max(Math.abs(s.size[1]), 1e-5);
    const radius = s.radius ?? 0.25 * Math.min(hx, hy);
    const o = i * 12;
    data[o] = s.x * aspect;
    data[o + 1] = s.y;
    data[o + 2] = Math.cos(angle);
    data[o + 3] = Math.sin(angle);
    data[o + 4] = hx;
    data[o + 5] = hy;
    data[o + 6] = Math.max(0, radius);
    data[o + 7] = KIND_INDEX[s.kind];
    data[o + 8] = s.velocity?.[0] ?? 0;
    data[o + 9] = s.velocity?.[1] ?? 0;
    data[o + 10] = s.spin ?? 0;
    for (let k = 0; k < 12; k++) if (!Number.isFinite(data[o + k])) data[o + k] = 0;
  }
  return { data, count };
}

// --- CPU reference ----------------------------------------------------------------------------------

/** The wing's signed distance in its own frame; the same maths as `obstacle_sd_wing`. */
function wingDistance(qx: number, qy: number, halfChord: number, halfThickness: number): number {
  const r1 = halfThickness;
  const r2 = Math.max(halfThickness * 0.05, 1e-4);
  const c = Math.max(halfChord, r1 * 1.5);
  const x1 = -c + r1;
  const x2 = c - r2;
  const h = x2 - x1;
  const px = Math.abs(qy);
  const py = qx - x1;
  const b = (r1 - r2) / h;
  const a = Math.sqrt(Math.max(1 - b * b, 0));
  const k = px * -b + py * a;
  if (k < 0) return Math.hypot(px, py) - r1;
  if (k > a * h) return Math.hypot(px, py - h) - r2;
  return px * a + py * b - r1;
}

/**
 * Signed distance, in image heights, from the canvas point (x, y) in uv to a shape. Negative inside.
 * The same function the GPU evaluates (the tests compare the two); also what picking uses.
 */
export function obstacleDistance(
  shape: ObstacleShape,
  x: number,
  y: number,
  aspect: number,
): number {
  const dx = x * aspect - shape.x * aspect;
  const dy = y - shape.y;
  const angle = shape.rotation ?? 0;
  const c = Math.cos(angle);
  const s = Math.sin(angle);
  const qx = dx * c + dy * s;
  const qy = -dx * s + dy * c;
  const hx = Math.max(Math.abs(shape.size[0]), 1e-5);
  const hy = Math.max(Math.abs(shape.size[1]), 1e-5);
  switch (shape.kind) {
    case 'circle':
      return Math.hypot(qx, qy) - hx;
    case 'box':
      return boxDistance(qx, qy, hx, hy);
    case 'rounded-box': {
      const r = Math.min(Math.max(shape.radius ?? 0.25 * Math.min(hx, hy), 0), hx, hy);
      return boxDistance(qx, qy, hx - r, hy - r) - r;
    }
    case 'capsule': {
      const r = Math.min(hx, hy);
      const tail = Math.max(hx - r, 0);
      const t = Math.min(Math.max(qx, -tail), tail);
      return Math.hypot(qx - t, qy) - r;
    }
    case 'wing':
      return wingDistance(qx, qy, hx, hy);
  }
}

function boxDistance(qx: number, qy: number, hx: number, hy: number): number {
  const ax = Math.abs(qx) - hx;
  const ay = Math.abs(qy) - hy;
  return Math.hypot(Math.max(ax, 0), Math.max(ay, 0)) + Math.min(Math.max(ax, ay), 0);
}

/** The mask the GPU pass draws, on the CPU: 255 where the cell centre is inside a shape. Row 0 at the top. */
export function rasterizeObstacleShapes(
  shapes: readonly ObstacleShape[],
  width: number,
  height: number,
  grow = 0,
): Uint8Array {
  const data = new Uint8Array(width * height);
  const aspect = width / height;
  const list = shapes.slice(0, OBSTACLE_MAX_SHAPES);
  for (let j = 0; j < height; j++) {
    for (let i = 0; i < width; i++) {
      const x = (i + 0.5) / width;
      const y = (j + 0.5) / height;
      for (const shape of list) {
        if (obstacleDistance(shape, x, y, aspect) <= grow) {
          data[j * width + i] = 255;
          break;
        }
      }
    }
  }
  return data;
}

/**
 * The index of the shape under the canvas point (x, y), or -1: the last (topmost) shape whose
 * distance is at most `margin` image heights, so a thin shape can be grabbed a little outside its outline.
 */
export function pickObstacle(
  shapes: readonly ObstacleShape[],
  x: number,
  y: number,
  aspect: number,
  margin = 0,
): number {
  for (let i = shapes.length - 1; i >= 0; i--) {
    if (obstacleDistance(shapes[i]!, x, y, aspect) <= margin) return i;
  }
  return -1;
}

/**
 * Turns the rectangles that `trackDomRects` measures into rounded-box shapes, so page elements and
 * analytic shapes can share one table. `aspect` is canvas width / height. `motion` is the optional
 * edge speed of the element in uv per second (`Obstacle.motion` from dom-rects.ts): the shape gets the
 * velocity of its centre.
 */
export function shapeFromRect(
  rect: ObstacleRect,
  aspect: number,
  motion?: { left: number; right: number; top: number; bottom: number },
): ObstacleShape {
  const hx = (rect.width * aspect) / 2;
  const hy = rect.height / 2;
  return {
    kind: 'rounded-box',
    x: rect.x + rect.width / 2,
    y: rect.y + rect.height / 2,
    size: [hx, hy],
    radius: Math.min(Math.max(rect.radius ?? 0, 0), hx, hy),
    velocity: motion
      ? [((motion.left + motion.right) / 2) * aspect, (motion.top + motion.bottom) / 2]
      : undefined,
  };
}

// --- The drive pass options -----------------------------------------------------------------------------

export interface ObstacleDriveOptions {
  /**
   * How strongly the fluid near a shape takes the shape's normal velocity, 0 (static solids only) to
   * 1 (the full relaxation, every step). Default 1.
   */
  coupling: number;
  /** Width of the band round each shape, in solver cells. Default 3. */
  band: number;
  /**
   * An inflow band at the left edge: the velocity is blended toward `velocity` (image heights per second)
   * inside the first `width` image heights, by `strength` (0..1) per step. Off when strength is 0.
   */
  inlet: { velocity: readonly [number, number]; width: number; strength: number };
  /**
   * 0 relaxes only the NORMAL velocity of the fluid in the band to the solid's (free slip: the fluid
   * slides along the surface), 1 relaxes the whole velocity (no slip: the fluid in the band is carried
   * along). Applies only to a shape that is moving (more than about 0.1 image heights per second); a
   * shape at rest always gets free slip. Default 1. Free slip alone pushes very little: the projection
   * that follows takes most of a normal-only velocity back out (measured in tests/fluid-obstacles.test.ts).
   */
  stick: number;
  /** Cap on any velocity this pass writes, image heights per second. Default 6. */
  maxSpeed: number;
}

export const DEFAULT_OBSTACLE_DRIVE: Readonly<ObstacleDriveOptions> = {
  coupling: 1,
  band: 3,
  inlet: { velocity: [0, 0], width: 0.06, strength: 0 },
  stick: 1,
  maxSpeed: 6,
};

export interface ObstacleShapesOptions {
  device: GPUDevice;
  fluid: Fluid;
  /** Counts cells whose centre is within this many image heights of a shape as solid. Default 0. */
  grow?: number;
  profiler?: PassProfiler;
  label?: string;
}

export interface ObstacleShapes {
  /** The shapes last given to `set` (read only; call `set` to change them). */
  readonly shapes: readonly ObstacleShape[];
  readonly count: number;
  /** The shape table, an `ObstacleTable` uniform buffer for shading passes. Kept current by `set` and `rasterize`. */
  readonly table: GPUBuffer;
  /** Live options of the moving-boundary / inflow pass. */
  readonly drive: ObstacleDriveOptions;
  /** Replaces the shapes (copies them; more than `OBSTACLE_MAX_SHAPES` are dropped) and marks the mask stale. */
  set(shapes: readonly ObstacleShape[]): void;
  /** Counts cells within `grow` image heights of a shape as solid (default 0). Marks the mask stale. */
  setGrow(grow: number): void;
  /**
   * Draws the shapes into `fluid.obstacles` if they changed, or if the solver re-created its grids
   * (it clears the mask then), or `force`; sets `fluid.obstacles.enabled`. Call before `fluid.update`
   * in the same encoder. Returns whether a pass was encoded.
   */
  rasterize(encoder: GPUCommandEncoder, force?: boolean): boolean;
  /**
   * Installs the moving-boundary and inflow pass as `fluid.hooks.forces` (there is one slot, so a hook
   * that was there is put back by `detach` and does not run meanwhile).
   */
  attach(): void;
  detach(): void;
  destroy(): void;
}

const DriveParams = defineUniformStruct('DriveParams', {
  grid: 'vec2f',
  band: 'f32',
  coupling: 'f32',
  inlet: 'vec4f',
  vmax: 'f32',
  stick: 'f32',
});

export function createObstacleShapes(options: ObstacleShapesOptions): ObstacleShapes {
  const { device, fluid, profiler = noProfiler, label = 'obstacles' } = options;
  const drive: ObstacleDriveOptions = {
    ...DEFAULT_OBSTACLE_DRIVE,
    inlet: { ...DEFAULT_OBSTACLE_DRIVE.inlet },
  };
  let grow = options.grow ?? 0;
  let shapes: ObstacleShape[] = [];
  let count = 0;
  let dirty = true;
  let uploadedAspect = 0;
  let drawnGeneration = -1;
  let destroyed = false;

  const table = createUniformBuffer(device, ObstacleTable, `${label} shape table`);
  const driveUniforms = createUniformBuffer(device, DriveParams, `${label} drive params`);
  const rasterPipeline = createFullscreenPipeline(device, {
    fragment: {
      label: 'sim/obstacle-raster.wgsl',
      code: rasterSource.code,
      chunks: OBSTACLE_CHUNKS,
    },
    format: OBSTACLE_FORMAT,
  });
  const drivePipeline = createFullscreenPipeline(device, {
    fragment: {
      label: 'sim/obstacle-drive.wgsl',
      code: driveSource.code,
      chunks: { ...driveSource.chunks, ...OBSTACLE_CHUNKS, 'drive_params.wgsl': DriveParams.wgsl },
    },
    format: 'rgba16float',
  });
  const rasterGroup = device.createBindGroup({
    label: `${label} raster`,
    layout: rasterPipeline.getBindGroupLayout(0),
    entries: [{ binding: 0, resource: { buffer: table.buffer } }],
  });
  const driveGroups = new Map<GPUTextureView, GPUBindGroup>();
  let hook: Fluid['hooks']['forces'];
  let previousHook: Fluid['hooks']['forces'];

  function upload() {
    const aspect = fluid.grid.aspect;
    const packed = packObstacleShapes(shapes, aspect);
    count = packed.count;
    table.set({ count, aspect, grow, pad: 0, data: packed.data });
    table.upload();
    uploadedAspect = aspect;
  }
  upload();

  function runDrive(context: FluidForceContext): boolean {
    const g = context.grid;
    driveUniforms.set({
      grid: [g.width, g.height],
      band: Math.max(0.5, drive.band),
      coupling: Math.min(1, Math.max(0, drive.coupling)),
      inlet: [
        drive.inlet.velocity[0],
        drive.inlet.velocity[1],
        Math.max(1e-3, drive.inlet.width),
        Math.min(1, Math.max(0, drive.inlet.strength)),
      ],
      vmax: Math.max(0.1, drive.maxSpeed),
      stick: Math.min(1, Math.max(0, drive.stick)),
    });
    driveUniforms.upload();
    const active = count > 0 && drive.coupling > 0;
    if (!active && !(drive.inlet.strength > 0)) return false;
    let group = driveGroups.get(context.read);
    if (!group) {
      if (driveGroups.size > 16) driveGroups.clear();
      group = device.createBindGroup({
        label: `${label} drive`,
        layout: drivePipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: table.buffer } },
          { binding: 1, resource: context.read },
          { binding: 2, resource: { buffer: driveUniforms.buffer } },
        ],
      });
      driveGroups.set(context.read, group);
    }
    renderFullscreen(context.encoder, context.write, drivePipeline, [group], {
      label: `${label} drive`,
      timestampWrites: profiler.pass('obstacle drive'),
    });
    return true;
  }

  const api: ObstacleShapes = {
    get shapes() {
      return shapes;
    },
    get count() {
      return count;
    },
    get table() {
      return table.buffer;
    },
    drive,
    set(next) {
      shapes = next.slice(0, OBSTACLE_MAX_SHAPES).map((shape) => ({ ...shape }));
      dirty = true;
      upload();
    },
    setGrow(next) {
      grow = Number.isFinite(next) ? next : 0;
      dirty = true;
      upload();
    },
    rasterize(encoder, force = false) {
      if (destroyed) return false;
      if (fluid.grid.aspect !== uploadedAspect) {
        upload();
        dirty = true;
      }
      if (!dirty && !force && drawnGeneration === fluid.generation) return false;
      const mask = fluid.obstacles;
      renderFullscreen(encoder, mask.view, rasterPipeline, [rasterGroup], {
        label: `${label} mask`,
        timestampWrites: profiler.pass('obstacle mask'),
      });
      mask.enabled = count > 0;
      dirty = false;
      drawnGeneration = fluid.generation;
      return true;
    },
    attach() {
      if (hook) return;
      previousHook = fluid.hooks.forces;
      hook = runDrive;
      fluid.hooks.forces = hook;
    },
    detach() {
      if (!hook) return;
      if (fluid.hooks.forces === hook) fluid.hooks.forces = previousHook;
      hook = undefined;
      previousHook = undefined;
    },
    destroy() {
      if (destroyed) return;
      api.detach();
      destroyed = true;
      table.destroy();
      driveUniforms.destroy();
    },
  };
  return api;
}

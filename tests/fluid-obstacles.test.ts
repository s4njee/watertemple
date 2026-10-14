import { describe, expect, it, vi } from 'vitest';
import { createFluid, halfToFloat, type Fluid, type FluidParams } from '../src/gpu/sim/fluid.ts';
import {
  OBSTACLE_MAX_SHAPES,
  createObstacleShapes,
  obstacleDistance,
  packObstacleShapes,
  pickObstacle,
  rasterizeObstacleShapes,
  shapeFromRect,
  type ObstacleShape,
  type ObstacleShapes,
} from '../src/gpu/sim/obstacle-shapes.ts';
import { expectValid, getTestDevice, readTexture } from './gpu.ts';

vi.setConfig({ testTimeout: 120_000 });

// --- Helpers -------------------------------------------------------------------------------------

async function submit(device: GPUDevice, record: (encoder: GPUCommandEncoder) => void) {
  const encoder = device.createCommandEncoder();
  record(encoder);
  device.queue.submit([encoder.finish()]);
  await device.queue.onSubmittedWorkDone();
}

function createSolver(
  device: GPUDevice,
  params: Partial<FluidParams> = {},
  resolution = 96,
  aspect = 1,
): Fluid {
  return createFluid({
    device,
    resolution,
    dyeResolution: resolution * 2,
    width: aspect,
    height: 1,
    params: { velocityDissipation: 0, dyeDissipation: 0, ...params },
  });
}

async function readMask(device: GPUDevice, fluid: Fluid): Promise<Uint8Array> {
  return readTexture(device, fluid.obstacles.texture, 1);
}

async function readHalf(device: GPUDevice, texture: GPUTexture): Promise<Float32Array> {
  const bytes = await readTexture(device, texture, 8);
  const half = new Uint16Array(bytes.buffer, bytes.byteOffset, texture.width * texture.height * 4);
  const out = new Float32Array(half.length);
  for (let i = 0; i < half.length; i++) out[i] = halfToFloat(half[i]!);
  return out;
}

interface Velocity {
  width: number;
  height: number;
  data: Float32Array;
  u(i: number, j: number): number;
  v(i: number, j: number): number;
}

async function readVelocity(device: GPUDevice, fluid: Fluid): Promise<Velocity> {
  const { width, height, velocityWidth } = fluid.grid;
  const data = await readHalf(device, fluid.velocity.texture);
  return {
    width,
    height,
    data,
    u: (i, j) => data[(j * velocityWidth + i) * 4]!,
    v: (i, j) => data[(j * velocityWidth + i) * 4 + 1]!,
  };
}

async function readDye(device: GPUDevice, fluid: Fluid) {
  const { dyeWidth: width, dyeHeight: height } = fluid.grid;
  const data = await readHalf(device, fluid.dye.texture);
  return {
    width,
    height,
    data,
    at: (x: number, y: number, c = 3) => data[(y * width + x) * 4 + c]!,
  };
}

const allFinite = (data: ArrayLike<number>) => {
  for (let i = 0; i < data.length; i++) if (!Number.isFinite(data[i])) return false;
  return true;
};

/** Area, centroid (uv) and orientation (radians, y down) of the solid cells of a mask. */
function moments(mask: Uint8Array, width: number, height: number) {
  const aspect = width / height;
  let n = 0;
  let sx = 0;
  let sy = 0;
  for (let j = 0; j < height; j++) {
    for (let i = 0; i < width; i++) {
      if (mask[j * width + i]! < 128) continue;
      n++;
      sx += ((i + 0.5) / width) * aspect;
      sy += (j + 0.5) / height;
    }
  }
  const cx = sx / n;
  const cy = sy / n;
  let xx = 0;
  let yy = 0;
  let xy = 0;
  for (let j = 0; j < height; j++) {
    for (let i = 0; i < width; i++) {
      if (mask[j * width + i]! < 128) continue;
      const dx = ((i + 0.5) / width) * aspect - cx;
      const dy = (j + 0.5) / height - cy;
      xx += dx * dx;
      yy += dy * dy;
      xy += dx * dy;
    }
  }
  return {
    cells: n,
    area: n / (height * height),
    centre: [cx / aspect, cy] as [number, number],
    angle: 0.5 * Math.atan2(2 * xy, xx - yy),
    elongation: Math.sqrt(Math.max(xx, yy) / Math.max(Math.min(xx, yy), 1e-12)),
  };
}

const solidAt = (mask: Uint8Array, width: number, height: number) => (i: number, j: number) =>
  i >= 0 && j >= 0 && i < width && j < height && mask[j * width + i]! >= 128;

// --- The shapes ------------------------------------------------------------------------------------

describe('the signed distance', () => {
  const aspect = 1.5;
  const at = (s: ObstacleShape, x: number, y: number) => obstacleDistance(s, x, y, aspect);

  it('is exact for the primitives', () => {
    const circle: ObstacleShape = { kind: 'circle', x: 0.5, y: 0.5, size: [0.1, 0.1] };
    expect(at(circle, 0.5, 0.5)).toBeCloseTo(-0.1, 6);
    expect(at(circle, 0.5 + 0.3 / aspect, 0.5)).toBeCloseTo(0.2, 6);
    const box: ObstacleShape = { kind: 'box', x: 0.5, y: 0.5, size: [0.2, 0.1] };
    expect(at(box, 0.5 + 0.2 / aspect, 0.5)).toBeCloseTo(0, 6);
    expect(at(box, 0.5, 0.5 + 0.1 + 0.05)).toBeCloseTo(0.05, 6);
    // Outside a corner the distance is Euclidean.
    expect(at(box, 0.5 + 0.23 / aspect, 0.5 + 0.14)).toBeCloseTo(0.05, 6);
    const rounded: ObstacleShape = {
      kind: 'rounded-box',
      x: 0.5,
      y: 0.5,
      size: [0.2, 0.1],
      radius: 0.05,
    };
    // The corner (0.2, 0.1) is cut: its distance is sqrt(2) * 0.05 - ... (0.15, 0.05) is the arc centre.
    expect(at(rounded, 0.5 + 0.2 / aspect, 0.5 + 0.1)).toBeCloseTo(
      Math.hypot(0.05, 0.05) - 0.05,
      6,
    );
    const capsule: ObstacleShape = { kind: 'capsule', x: 0.5, y: 0.5, size: [0.2, 0.05] };
    expect(at(capsule, 0.5 + 0.2 / aspect, 0.5)).toBeCloseTo(0, 6);
    expect(at(capsule, 0.5, 0.5 + 0.05)).toBeCloseTo(0, 6);
  });

  it('turns clockwise on the screen with a positive rotation', () => {
    const bar: ObstacleShape = {
      kind: 'box',
      x: 0.5,
      y: 0.5,
      size: [0.2, 0.02],
      rotation: Math.PI / 2,
    };
    // Turned a quarter clockwise: the bar's +x end points DOWN the screen.
    expect(at(bar, 0.5, 0.5 + 0.15)).toBeLessThan(0);
    expect(at(bar, 0.5 + 0.15 / aspect, 0.5)).toBeGreaterThan(0.1);
    const tilted: ObstacleShape = { ...bar, rotation: Math.PI / 4 };
    // Eighth of a turn: the end towards lower right is inside, upper right is outside.
    expect(at(tilted, 0.5 + 0.1 / aspect, 0.5 + 0.1)).toBeLessThan(0);
    expect(at(tilted, 0.5 + 0.1 / aspect, 0.5 - 0.1)).toBeGreaterThan(0.05);
  });

  it('describes a wing: round at the front, sharp at the back, symmetric', () => {
    const wing: ObstacleShape = { kind: 'wing', x: 0.5, y: 0.5, size: [0.2, 0.05] };
    // Leading edge at -x: reaches 0.2 from the centre.
    expect(at(wing, 0.5 - 0.2 / aspect, 0.5)).toBeCloseTo(0, 2);
    expect(at(wing, 0.5 - 0.2 / aspect - 0.02, 0.5)).toBeGreaterThan(0.015);
    // The tail is thin: 0.05 above the axis at the tail end is far outside, at the leading
    // circle's centre it is inside.
    expect(at(wing, 0.5 + 0.18 / aspect, 0.5 + 0.02)).toBeGreaterThan(0.01);
    expect(at(wing, 0.5 - 0.15 / aspect, 0.5 + 0.03)).toBeLessThan(0);
    expect(at(wing, 0.5, 0.5 + 0.03)).toBeCloseTo(at(wing, 0.5, 0.5 - 0.03), 8);
    // Thickest at the leading circle: the half thickness there is 0.05.
    expect(at(wing, 0.5 - 0.15 / aspect, 0.5 + 0.05)).toBeCloseTo(0, 2);
  });

  it('picks the topmost shape under a point, with a margin', () => {
    const shapes: ObstacleShape[] = [
      { kind: 'circle', x: 0.3, y: 0.5, size: [0.1, 0.1] },
      { kind: 'box', x: 0.3, y: 0.5, size: [0.05, 0.05] },
      { kind: 'circle', x: 0.7, y: 0.5, size: [0.05, 0.05] },
    ];
    expect(pickObstacle(shapes, 0.3, 0.5, 1)).toBe(1);
    expect(pickObstacle(shapes, 0.3 + 0.08, 0.5, 1)).toBe(0);
    expect(pickObstacle(shapes, 0.5, 0.5, 1)).toBe(-1);
    expect(pickObstacle(shapes, 0.7 + 0.07, 0.5, 1, 0.03)).toBe(2);
  });

  it('packs shapes into the table and ignores more than fit', () => {
    const shapes: ObstacleShape[] = Array.from({ length: OBSTACLE_MAX_SHAPES + 5 }, (_, i) => ({
      kind: 'circle' as const,
      x: 0.5,
      y: 0.5,
      size: [0.01 * (i + 1), 0] as [number, number],
    }));
    const packed = packObstacleShapes(shapes, 2);
    expect(packed.count).toBe(OBSTACLE_MAX_SHAPES);
    expect(packed.data[0]).toBeCloseTo(1, 6); // centre x in image heights
    expect(packed.data[4]).toBeCloseTo(0.01, 6);
    expect(packed.data[7]).toBe(0);
    const bad = packObstacleShapes(
      [
        {
          kind: 'wing',
          x: Number.NaN,
          y: 0.5,
          size: [0.1, 0.02],
          rotation: Number.POSITIVE_INFINITY,
        },
      ],
      1,
    );
    expect(Array.from(bad.data.slice(0, 12)).every(Number.isFinite)).toBe(true);
  });

  it("turns a measured rectangle into a rounded box with the element's own velocity", () => {
    const shape = shapeFromRect({ x: 0.2, y: 0.3, width: 0.4, height: 0.2, radius: 0.05 }, 2, {
      left: 0.1,
      right: 0.3,
      top: -0.2,
      bottom: -0.2,
    });
    expect(shape.x).toBeCloseTo(0.4, 6);
    expect(shape.y).toBeCloseTo(0.4, 6);
    expect(shape.size[0]).toBeCloseTo(0.4, 6);
    expect(shape.size[1]).toBeCloseTo(0.1, 6);
    expect(shape.velocity?.[0]).toBeCloseTo(0.4, 6);
    expect(shape.velocity?.[1]).toBeCloseTo(-0.2, 6);
  });
});

describe('rasterising into the mask', () => {
  const cases: { name: string; shape: ObstacleShape; area: number; aspect: number }[] = [
    {
      name: 'circle',
      shape: { kind: 'circle', x: 0.4, y: 0.45, size: [0.2, 0.2] },
      area: Math.PI * 0.2 ** 2,
      aspect: 1,
    },
    {
      name: 'circle on a wide canvas (round in image heights)',
      shape: { kind: 'circle', x: 0.4, y: 0.45, size: [0.2, 0.2] },
      area: Math.PI * 0.2 ** 2,
      aspect: 2,
    },
    {
      name: 'rotated box',
      shape: { kind: 'box', x: 0.55, y: 0.5, size: [0.25, 0.08], rotation: 0.6 },
      area: 4 * 0.25 * 0.08,
      aspect: 1.5,
    },
    {
      name: 'rounded box',
      shape: {
        kind: 'rounded-box',
        x: 0.5,
        y: 0.5,
        size: [0.25, 0.12],
        radius: 0.06,
        rotation: -0.4,
      },
      area: 4 * 0.25 * 0.12 - (4 - Math.PI) * 0.06 ** 2,
      aspect: 1,
    },
    {
      name: 'capsule',
      shape: { kind: 'capsule', x: 0.5, y: 0.5, size: [0.3, 0.06], rotation: 1.1 },
      area: 4 * (0.3 - 0.06) * 0.06 + Math.PI * 0.06 ** 2,
      aspect: 1.2,
    },
  ];

  it.each(cases)(
    'draws a $name with the right area, place and orientation',
    async ({ shape, area, aspect }) => {
      const device = await getTestDevice();
      const fluid = createSolver(device, {}, 128, aspect);
      const shapes = createObstacleShapes({ device, fluid });
      shapes.set([shape]);
      expect(fluid.obstacles.enabled).toBe(false);
      await submit(device, (e) => expect(shapes.rasterize(e)).toBe(true));
      expect(fluid.obstacles.enabled).toBe(true);
      const { width, height } = fluid.grid;
      const mask = await readMask(device, fluid);
      const m = moments(mask, width, height);
      expect(m.area / area).toBeGreaterThan(0.97);
      expect(m.area / area).toBeLessThan(1.03);
      expect(m.centre[0]).toBeCloseTo(shape.x, 2);
      expect(m.centre[1]).toBeCloseTo(shape.y, 2);
      if (shape.kind !== 'circle') {
        // The long axis lies at the rotation (mod pi).
        const wanted = shape.rotation ?? 0;
        const diff = Math.abs(((m.angle - wanted + Math.PI / 2) % Math.PI) - Math.PI / 2);
        expect(diff).toBeLessThan(0.03);
      }
      // The GPU and the CPU reference agree cell for cell.
      const cpu = rasterizeObstacleShapes([shape], width, height);
      let different = 0;
      for (let i = 0; i < cpu.length; i++) if (cpu[i] !== mask[i]) different++;
      expect(different).toBeLessThanOrEqual(Math.ceil(0.002 * m.cells));
      shapes.destroy();
      fluid.destroy();
    },
  );

  it('draws a wing: sharp tail at +x, round nose at -x, turned by its angle of attack', async () => {
    const device = await getTestDevice();
    const fluid = createSolver(device, {}, 192);
    const shapes = createObstacleShapes({ device, fluid });
    const wing: ObstacleShape = { kind: 'wing', x: 0.5, y: 0.5, size: [0.3, 0.07] };
    shapes.set([wing]);
    await submit(device, (e) => shapes.rasterize(e));
    const { width, height } = fluid.grid;
    const flat = await readMask(device, fluid);
    // Thickness (cells) of the wing near the nose and near the tail: the tail is far thinner.
    const thickness = (mask: Uint8Array, column: number) => {
      let n = 0;
      for (let j = 0; j < height; j++) if (mask[j * width + column]! >= 128) n++;
      return n;
    };
    const nose = Math.round((0.5 - 0.3 + 0.07) * width);
    const tail = Math.round((0.5 + 0.3 - 0.04) * width);
    expect(thickness(flat, nose)).toBeGreaterThan(0.12 * height);
    expect(thickness(flat, tail)).toBeLessThan(0.06 * height);
    expect(thickness(flat, tail)).toBeGreaterThan(0);
    // Area between that of the leading circle and the bounding box; compare with the CPU reference.
    const m = moments(flat, width, height);
    const reference =
      rasterizeObstacleShapes([wing], 1024, 1024).reduce((n, v) => n + (v >= 128 ? 1 : 0), 0) /
      1024 ** 2;
    expect(m.area / reference).toBeGreaterThan(0.97);
    expect(m.area / reference).toBeLessThan(1.03);

    // Angle of attack 0.3 rad clockwise: the nose goes UP the screen relative to the tail... the tail
    // (+x end) rotates down: its centroid row is lower than the nose's.
    shapes.set([{ ...wing, rotation: 0.3 }]);
    await submit(device, (e) => shapes.rasterize(e));
    const tilted = await readMask(device, fluid);
    const rows = (x0: number, x1: number) => {
      let sum = 0;
      let n = 0;
      for (let j = 0; j < height; j++) {
        for (let i = x0; i < x1; i++) {
          if (tilted[j * width + i]! >= 128) {
            sum += j;
            n++;
          }
        }
      }
      return sum / n;
    };
    expect(rows(Math.round(0.6 * width), Math.round(0.8 * width))).toBeGreaterThan(
      rows(Math.round(0.2 * width), Math.round(0.4 * width)) + 0.05 * height,
    );
    shapes.destroy();
    fluid.destroy();
  });

  it('draws only when something changed, and again after the solver re-creates its grids', async () => {
    const device = await getTestDevice();
    await expectValid(device, async () => {
      const fluid = createSolver(device, {}, 64);
      const shapes = createObstacleShapes({ device, fluid });
      shapes.set([{ kind: 'circle', x: 0.5, y: 0.5, size: [0.15, 0.15] }]);
      const drew: boolean[] = [];
      for (let k = 0; k < 3; k++) await submit(device, (e) => drew.push(shapes.rasterize(e)));
      expect(drew).toEqual([true, false, false]);
      // A new aspect: the mask is a new texture; the shapes are drawn into it.
      expect(fluid.resize(2, 1)).toBe(true);
      await submit(device, (e) => expect(shapes.rasterize(e)).toBe(true));
      const { width, height } = fluid.grid;
      const m = moments(await readMask(device, fluid), width, height);
      expect(m.area / (Math.PI * 0.15 ** 2)).toBeGreaterThan(0.95);
      expect(m.area / (Math.PI * 0.15 ** 2)).toBeLessThan(1.05);
      // No shapes: the mask is cleared and the solver is told.
      shapes.set([]);
      await submit(device, (e) => shapes.rasterize(e));
      expect(fluid.obstacles.enabled).toBe(false);
      expect((await readMask(device, fluid)).every((v) => v === 0)).toBe(true);
      shapes.destroy();
      fluid.destroy();
    });
  });
});

// --- Flow around them ------------------------------------------------------------------------------------

/** Faces whose value the solver must hold at zero: next to at least one solid cell. */
function solidFaces(v: Velocity, solid: (i: number, j: number) => boolean) {
  const faces: { axis: 0 | 1; i: number; j: number }[] = [];
  for (let j = 0; j < v.height; j++) {
    for (let i = 0; i <= v.width; i++) {
      if (solid(i - 1, j) || solid(i, j)) faces.push({ axis: 0, i, j });
    }
  }
  for (let j = 0; j <= v.height; j++) {
    for (let i = 0; i < v.width; i++) {
      if (solid(i, j - 1) || solid(i, j)) faces.push({ axis: 1, i, j });
    }
  }
  return faces;
}

function tunnel(shapes: ObstacleShapes, speed = 1) {
  shapes.drive.inlet = { velocity: [speed, 0], width: 0.08, strength: 0.5 };
  shapes.attach();
}

/** Dye inside the cells of the mask (should be none), and elsewhere. */
function dyeInside(
  dye: Awaited<ReturnType<typeof readDye>>,
  solid: (i: number, j: number) => boolean,
  grid: { width: number; height: number },
) {
  let inside = 0;
  let outside = 0;
  for (let y = 0; y < dye.height; y++) {
    for (let x = 0; x < dye.width; x++) {
      const cell = solid(
        Math.min(grid.width - 1, Math.floor(((x + 0.5) * grid.width) / dye.width)),
        Math.min(grid.height - 1, Math.floor(((y + 0.5) * grid.height) / dye.height)),
      );
      if (cell) inside += Math.abs(dye.at(x, y)) + Math.abs(dye.at(x, y, 0));
      else outside += dye.at(x, y);
    }
  }
  return { inside, outside };
}

describe('a steady wind round solid shapes', () => {
  it('holds the velocity at solid faces to zero, keeps dye out of the shapes, and the flow is real', async () => {
    const device = await getTestDevice();
    const fluid = createSolver(device, { boundary: 'open', vorticity: 8, viscosity: 0.0002 });
    const shapes = createObstacleShapes({ device, fluid });
    shapes.set([
      { kind: 'circle', x: 0.28, y: 0.5, size: [0.09, 0.09] },
      { kind: 'box', x: 0.55, y: 0.3, size: [0.012, 0.12], rotation: 0.4 },
      { kind: 'wing', x: 0.75, y: 0.62, size: [0.18, 0.05], rotation: 0.2 },
      { kind: 'rounded-box', x: 0.5, y: 0.8, size: [0.07, 0.04], radius: 0.02 },
    ]);
    tunnel(shapes, 1);
    // Dye released upstream, in rows, every few steps.
    for (let k = 0; k < 40; k++) {
      for (let r = 0; r < 8; r++) {
        const y = 0.1 + 0.1 * r;
        fluid.addDye([0.02, y], [0.05, y], 0.012, [1, 1, 1, 1]);
      }
      await submit(device, (e) => {
        shapes.rasterize(e);
        fluid.advance(e, 3);
      });
    }
    const { width, height } = fluid.grid;
    const mask = await readMask(device, fluid);
    const solid = solidAt(mask, width, height);
    const v = await readVelocity(device, fluid);
    expect(allFinite(v.data)).toBe(true);

    const faces = solidFaces(v, solid);
    expect(faces.length).toBeGreaterThan(300);
    let worst = 0;
    for (const f of faces) {
      worst = Math.max(worst, Math.abs(f.axis === 0 ? v.u(f.i, f.j) : v.v(f.i, f.j)));
    }
    expect(worst).toBe(0);

    // The wind is blowing: through a clear column upstream, and downstream of everything.
    const column = (x: number, y0: number, y1: number) => {
      let sum = 0;
      let n = 0;
      for (let j = Math.floor(y0 * height); j < Math.floor(y1 * height); j++) {
        sum += v.u(Math.round(x * width), j);
        n++;
      }
      return sum / n;
    };
    expect(column(0.12, 0.05, 0.95)).toBeGreaterThan(0.5);
    expect(column(0.97, 0.05, 0.95)).toBeGreaterThan(0.1);
    // ... and it goes round the cylinder: above and below it the flow is faster than the wind.
    expect(column(0.28, 0.33, 0.37)).toBeGreaterThan(1.0);

    // No dye in any shape, and dye did reach the shapes' neighbourhood and beyond.
    const dye = await readDye(device, fluid);
    expect(allFinite(dye.data)).toBe(true);
    const where = dyeInside(dye, solid, fluid.grid);
    expect(where.inside).toBe(0);
    expect(where.outside).toBeGreaterThan(20);
    shapes.destroy();
    fluid.destroy();
  });

  it('does not leak through a one-cell wall, or round the corners of a closed box of thin walls', async () => {
    const device = await getTestDevice();
    const fluid = createSolver(device, { boundary: 'open' }, 64);
    const shapes = createObstacleShapes({ device, fluid });
    const { width, height } = fluid.grid;
    const cell = 1 / height;
    // A hollow square, walls exactly one cell thick (centred on cell centres), with closed corners.
    const edge = (n: number) => (n + 0.5) / width;
    const lo = 20;
    const hi = 43;
    const half = 0.5 * cell;
    const wall = (kind: 'h' | 'v', a: number, b: number, at: number): ObstacleShape =>
      kind === 'h'
        ? { kind: 'box', x: edge((a + b) / 2), y: edge(at), size: [((b - a + 1) / 2) * cell, half] }
        : {
            kind: 'box',
            x: edge(at),
            y: edge((a + b) / 2),
            size: [half, ((b - a + 1) / 2) * cell],
          };
    shapes.set([
      wall('h', lo, hi, lo),
      wall('h', lo, hi, hi),
      wall('v', lo, hi, lo),
      wall('v', lo, hi, hi),
    ]);
    await submit(device, (e) => shapes.rasterize(e));
    const mask = await readMask(device, fluid);
    const solid = solidAt(mask, width, height);
    // Exactly one cell thick.
    for (let i = lo + 2; i < hi - 1; i++) {
      expect(solid(i, lo)).toBe(true);
      expect(solid(i, lo + 1)).toBe(false);
      expect(solid(i, lo - 1)).toBe(false);
    }
    // Dye inside, and shoves from inside towards each wall and every corner.
    fluid.setDye((x, y) => (Math.hypot(x - 0.5, y - 0.5) < 0.12 ? [1, 1, 1, 1] : [0, 0, 0, 0]));
    for (let k = 0; k < 24; k++) {
      for (const [dx, dy] of [
        [1, 0],
        [-1, 0],
        [0, 1],
        [0, -1],
        [1, 1],
        [-1, 1],
        [1, -1],
        [-1, -1],
      ] as [number, number][]) {
        fluid.addForce([0.5, 0.5], [0.5 + 0.17 * dx, 0.5 + 0.17 * dy], 0.07, [3 * dx, 3 * dy]);
      }
      await submit(device, (e) => fluid.advance(e, 5));
    }
    const v = await readVelocity(device, fluid);
    for (const f of solidFaces(v, solid)) {
      expect(f.axis === 0 ? v.u(f.i, f.j) : v.v(f.i, f.j)).toBe(0);
    }
    // The dye is still inside the box, and none got out.
    const dye = await readDye(device, fluid);
    let inside = 0;
    let outside = 0;
    for (let y = 0; y < dye.height; y++) {
      for (let x = 0; x < dye.width; x++) {
        const cx = Math.floor((x * width) / dye.width);
        const cy = Math.floor((y * height) / dye.height);
        const within = cx > lo && cx < hi && cy > lo && cy < hi;
        if (within) inside += dye.at(x, y);
        else outside += Math.abs(dye.at(x, y));
      }
    }
    expect(inside).toBeGreaterThan(50);
    expect(outside).toBe(0);
    shapes.destroy();
    fluid.destroy();
  });

  it('stops dye at a thin diagonal wall two cells thick', async () => {
    const device = await getTestDevice();
    const fluid = createSolver(device, { boundary: 'wall' }, 64);
    const shapes = createObstacleShapes({ device, fluid });
    // A long diagonal bar across the whole canvas, 2 cells thick.
    shapes.set([{ kind: 'box', x: 0.5, y: 0.5, size: [0.9, 2 / 64], rotation: Math.PI / 4 }]);
    await submit(device, (e) => shapes.rasterize(e));
    const { width, height } = fluid.grid;
    const solid = solidAt(await readMask(device, fluid), width, height);
    // Dye in the upper-right triangle only, pushed hard towards the wall.
    fluid.setDye((x, y) => (x - 0.5 > y - 0.5 + 0.12 ? [1, 1, 1, 1] : [0, 0, 0, 0]));
    for (let k = 0; k < 20; k++) {
      fluid.addForce([0.8, 0.2], [0.35, 0.65], 0.1, [-2.5, 2.5]);
      await submit(device, (e) => fluid.advance(e, 6));
    }
    const dye = await readDye(device, fluid);
    let other = 0;
    let own = 0;
    for (let y = 0; y < dye.height; y++) {
      for (let x = 0; x < dye.width; x++) {
        const u = (x + 0.5) / dye.width;
        const w = (y + 0.5) / dye.height;
        if (u - 0.5 < w - 0.5 - 0.06) other += Math.abs(dye.at(x, y));
        else if (u - 0.5 > w - 0.5 + 0.06) own += dye.at(x, y);
      }
    }
    expect(own).toBeGreaterThan(100);
    expect(other).toBe(0);
    expect(dyeInside(dye, solid, fluid.grid).inside).toBe(0);
    shapes.destroy();
    fluid.destroy();
  });

  it('does not let a wall of one cell pass a wind with dye (a full-height plate in the tunnel)', async () => {
    const device = await getTestDevice();
    const fluid = createSolver(device, { boundary: 'open' }, 64);
    const shapes = createObstacleShapes({ device, fluid });
    const { width, height } = fluid.grid;
    shapes.set([{ kind: 'box', x: (32 + 0.5) / width, y: 0.5, size: [0.5 / height, 0.6] }]);
    tunnel(shapes, 1);
    for (let k = 0; k < 30; k++) {
      for (let r = 0; r < 6; r++) {
        const y = 0.12 + 0.15 * r;
        fluid.addDye([0.02, y], [0.06, y], 0.015, [1, 1, 1, 1]);
      }
      await submit(device, (e) => {
        shapes.rasterize(e);
        fluid.advance(e, 4);
      });
    }
    const dye = await readDye(device, fluid);
    let upstream = 0;
    let behind = 0;
    for (let y = 0; y < dye.height; y++) {
      for (let x = 0; x < dye.width; x++) {
        if (x < dye.width / 2 - 3) upstream += dye.at(x, y);
        else if (x > dye.width / 2 + 3) behind += Math.abs(dye.at(x, y));
      }
    }
    expect(upstream).toBeGreaterThan(50);
    expect(behind).toBe(0);
    shapes.destroy();
    fluid.destroy();
  });
});

// --- Moving shapes -----------------------------------------------------------------------------------------

describe('a shape that moves', () => {
  async function drag(device: GPUDevice, coupling: number) {
    const fluid = createSolver(device, { boundary: 'wall' }, 64);
    const shapes = createObstacleShapes({ device, fluid });
    shapes.drive.coupling = coupling;
    shapes.attach();
    fluid.setDye(() => [1, 1, 1, 1]);
    const speed = 0.8;
    let x = 0.2;
    for (let frame = 0; frame < 24; frame++) {
      x += (speed / 60) * 2;
      shapes.set([{ kind: 'circle', x, y: 0.5, size: [0.07, 0.07], velocity: [speed, 0] }]);
      await submit(device, (e) => {
        shapes.rasterize(e);
        fluid.advance(e, 2);
      });
    }
    const v = await readVelocity(device, fluid);
    const { width, height } = fluid.grid;
    // Mean u in a patch just ahead of the body and one just behind it.
    const patch = (cx: number) => {
      let sum = 0;
      let n = 0;
      for (let j = Math.floor(0.46 * height); j < Math.floor(0.54 * height); j++) {
        for (let i = Math.floor((cx - 0.02) * width); i < Math.floor((cx + 0.02) * width); i++) {
          sum += v.u(i, j);
          n++;
        }
      }
      return sum / n;
    };
    const ahead = patch(x + 0.07 + 0.04);
    const behind = patch(x - 0.07 - 0.04);
    const dye = await readDye(device, fluid);
    const solid = solidAt(await readMask(device, fluid), width, height);
    const where = dyeInside(dye, solid, fluid.grid);
    const finite = allFinite(v.data) && allFinite(dye.data);
    shapes.destroy();
    fluid.destroy();
    return { ahead, behind, inside: where.inside, finite };
  }

  it('pushes the fluid ahead of it and draws the fluid in behind, where a static mask does nothing', async () => {
    const device = await getTestDevice();
    const pushed = await drag(device, 1);
    const plain = await drag(device, 0);
    expect(pushed.finite && plain.finite).toBe(true);
    expect(pushed.inside).toBe(0);
    expect(plain.inside).toBe(0);
    // Ahead of the body the fluid is shoved along with it.
    expect(pushed.ahead).toBeGreaterThan(0.1);
    expect(pushed.ahead).toBeGreaterThan(4 * Math.abs(plain.ahead));
    // Behind it the fluid follows the body in.
    expect(pushed.behind).toBeGreaterThan(0.05);
  });

  it('leaves no dye in its new footprint when dragged fast through dyed fluid, and stays finite', async () => {
    const device = await getTestDevice();
    const fluid = createSolver(device, { boundary: 'open', vorticity: 8 }, 64);
    const shapes = createObstacleShapes({ device, fluid });
    shapes.attach();
    fluid.setDye(() => [1, 1, 1, 1]);
    const { width, height } = fluid.grid;
    const path: [number, number][] = [
      [0.15, 0.2],
      [0.85, 0.3], // 0.7 of the canvas in one frame
      [0.2, 0.8],
      [0.8, 0.75],
      [0.5, 0.15],
      [0.5, 0.9],
      [0.1, 0.1],
    ];
    let previous = path[0]!;
    for (const [x, y] of path) {
      const dt = 1 / 60;
      shapes.set([
        {
          kind: 'wing',
          x,
          y,
          size: [0.14, 0.04],
          rotation: Math.atan2(y - previous[1], x - previous[0]),
          velocity: [(x - previous[0]) / dt, (y - previous[1]) / dt],
        },
        { kind: 'rounded-box', x: 1 - x, y: 1 - y, size: [0.05, 0.04], radius: 0.015 },
      ]);
      previous = [x, y];
      await submit(device, (e) => {
        shapes.rasterize(e);
        fluid.advance(e, 3);
      });
      const solid = solidAt(await readMask(device, fluid), width, height);
      const dye = await readDye(device, fluid);
      expect(allFinite(dye.data)).toBe(true);
      expect(dyeInside(dye, solid, fluid.grid).inside).toBe(0);
      const v = await readVelocity(device, fluid);
      expect(allFinite(v.data)).toBe(true);
      let peak = 0;
      for (let i = 0; i < v.data.length; i += 4) {
        peak = Math.max(peak, Math.abs(v.data[i]!), Math.abs(v.data[i + 1]!));
      }
      // Bounded by the solver's own cap (6 cells per step), whatever the body did.
      expect(peak).toBeLessThanOrEqual((fluid.params.maxCfl / fluid.stepSeconds / height) * 1.01);
      for (const f of solidFaces(v, solid)) {
        expect(f.axis === 0 ? v.u(f.i, f.j) : v.v(f.i, f.j)).toBe(0);
      }
    }
    shapes.destroy();
    fluid.destroy();
  });

  it('spins: a rotating bar sets the fluid round it turning the same way', async () => {
    const device = await getTestDevice();
    const fluid = createSolver(device, { boundary: 'wall' }, 64);
    const shapes = createObstacleShapes({ device, fluid });
    shapes.attach();
    let angle = 0;
    const spin = 4; // rad/s, clockwise on the screen
    for (let frame = 0; frame < 30; frame++) {
      angle += (spin * 2) / 60;
      shapes.set([{ kind: 'box', x: 0.5, y: 0.5, size: [0.2, 0.025], rotation: angle, spin }]);
      await submit(device, (e) => {
        shapes.rasterize(e);
        fluid.advance(e, 2);
      });
    }
    const v = await readVelocity(device, fluid);
    const { width, height } = fluid.grid;
    // Clockwise on the screen with y down: the fluid to the right of the centre moves DOWN, below it LEFT.
    let right = 0;
    let below = 0;
    let n = 0;
    for (let k = 0; k < 8; k++) {
      const i = Math.floor((0.5 + 0.26 + 0.005 * k) * width);
      const j = Math.floor(0.5 * height);
      right += v.v(i, j);
      const i2 = Math.floor(0.5 * width);
      const j2 = Math.floor((0.5 + 0.26 + 0.005 * k) * height);
      below += v.u(i2, j2);
      n++;
    }
    expect(right / n).toBeGreaterThan(0.02);
    expect(below / n).toBeLessThan(-0.02);
    shapes.destroy();
    fluid.destroy();
  });
});

describe('robustness', () => {
  it('survives absurd shapes: NaN, zero size, off the canvas, more than fit', async () => {
    const device = await getTestDevice();
    await expectValid(device, async () => {
      const fluid = createSolver(device, { boundary: 'open' }, 48);
      const shapes = createObstacleShapes({ device, fluid });
      shapes.drive.inlet = { velocity: [1e30, -1e30], width: 5, strength: 5 };
      shapes.attach();
      const list: ObstacleShape[] = [
        { kind: 'circle', x: Number.NaN, y: 0.5, size: [0.1, 0.1] },
        { kind: 'wing', x: 0.5, y: 0.5, size: [0, 0] },
        { kind: 'box', x: -3, y: 9, size: [0.1, 0.1], velocity: [1e30, 1e30] },
        { kind: 'capsule', x: 0.5, y: 0.5, size: [1e9, 1e-9], rotation: 1e9, spin: 1e30 },
        ...Array.from({ length: 100 }, (_, i) => ({
          kind: 'rounded-box' as const,
          x: (i % 10) / 10,
          y: Math.floor(i / 10) / 10,
          size: [0.02, 0.02] as [number, number],
        })),
      ];
      shapes.set(list);
      expect(shapes.count).toBe(OBSTACLE_MAX_SHAPES);
      for (let k = 0; k < 3; k++) {
        await submit(device, (e) => {
          shapes.rasterize(e);
          fluid.addForce([0.5, 0.5], [0.9, 0.9], 0.2, [3e38, 3e38]);
          fluid.advance(e, 4);
        });
      }
      const v = await readVelocity(device, fluid);
      expect(allFinite(v.data)).toBe(true);
      expect(allFinite((await readDye(device, fluid)).data)).toBe(true);
      shapes.destroy();
      fluid.destroy();
    });
  });

  it('rasterises nothing and writes nothing after destroy', async () => {
    const device = await getTestDevice();
    const fluid = createSolver(device, {}, 32);
    const shapes = createObstacleShapes({ device, fluid });
    shapes.set([{ kind: 'circle', x: 0.5, y: 0.5, size: [0.1, 0.1] }]);
    shapes.destroy();
    await submit(device, (e) => expect(shapes.rasterize(e)).toBe(false));
    fluid.destroy();
  });
});

import { describe, expect, it, vi } from 'vitest';
import { loadEffect } from '../src/effects/registry.ts';
import {
  CAMERA_FOV_Y,
  INITIAL_BALL,
  INITIAL_ORBIT,
  MIN_HORIZONTAL_FOV,
  POOL_HALF,
  POOL_RIM,
  ballWaterInfluence,
  beginDrag,
  clampBall,
  dragOnPlane,
  flatSunRefraction,
  intersectPlane,
  pointerRay,
  rayBoxExit,
  raySphere,
  stepBall,
  submergedFraction,
  sunDirection,
  toCausticsFrame,
  worldToPoolUv,
  type BallBody,
  type Vec3,
} from '../src/effects/pool-3d/maths.ts';
import poolRays from '../src/effects/pool-3d/pool-rays.wgsl?raw';
import { defaultParams } from '../src/effects/types.ts';
import { noProfiler } from '../src/gpu/profiler.ts';
import {
  computeCamera,
  createCameraMatrices,
  fitFovY,
  projectPoint,
  unprojectPoint,
} from '../src/gpu/scene/camera.ts';
import { orbitEye } from '../src/gpu/scene/orbit.ts';
import { expectValid, getTestDevice, readTexture } from './gpu.ts';
import { gpuF32 } from './wgsl-eval.ts';

vi.setConfig({ testTimeout: 60_000 });

const close = (a: ArrayLike<number>, b: ArrayLike<number>, digits = 5) => {
  expect(a.length).toBe(b.length);
  for (let i = 0; i < a.length; i++) expect(a[i]).toBeCloseTo(b[i]!, digits);
};

describe('raySphere', () => {
  const center: Vec3 = [0, 0, 0];
  it('finds where a ray enters a sphere', () => {
    expect(raySphere([0, 0, -5], [0, 0, 1], center, 1)).toBeCloseTo(4, 6);
    expect(raySphere([3, 0, 0], [-1, 0, 0], center, 2)).toBeCloseTo(1, 6);
    const d = Math.SQRT1_2;
    expect(raySphere([-5, 5, 0], [d, -d, 0], center, 1)).toBeCloseTo(Math.hypot(5, 5) - 1, 5);
  });
  it('misses, grazes and looks the other way', () => {
    expect(raySphere([0, 2, -5], [0, 0, 1], center, 1)).toBeNull();
    expect(raySphere([0, 0, 5], [0, 0, 1], center, 1)).toBeNull(); // sphere behind the origin
    expect(raySphere([0, 1.0001, -5], [0, 0, 1], center, 1)).toBeNull();
    expect(raySphere([0, 0.9, -5], [0, 0, 1], center, 1)).not.toBeNull();
  });
  it('does not hit a sphere the ray starts inside of', () => {
    expect(raySphere([0, 0, 0], [0, 0, 1], center, 1)).toBeNull();
  });
});

describe('rayBoxExit', () => {
  it('leaves through the wall it points at', () => {
    const east = rayBoxExit([0, -0.3, 0], [1, 0, 0], -1, POOL_RIM);
    expect(east.t).toBeCloseTo(POOL_HALF, 6);
    expect(east.normal).toEqual([-1, 0, 0]);
    const north = rayBoxExit([0.5, -0.3, 0], [0, 0, -1], -1, POOL_RIM);
    expect(north.t).toBeCloseTo(POOL_HALF, 6);
    expect(north.normal).toEqual([0, 0, 1]);
    const west = rayBoxExit([-0.25, -0.3, 0.2], [-1, 0, 0], -1, POOL_RIM);
    expect(west.t).toBeCloseTo(0.75, 6);
    expect(west.normal).toEqual([1, 0, 0]);
  });
  it('finds the floor and the open top', () => {
    const down = rayBoxExit([0, 0, 0], [0, -1, 0], -0.8, 0);
    expect(down.t).toBeCloseTo(0.8, 6);
    expect(down.normal).toEqual([0, 1, 0]);
    const up = rayBoxExit([0.2, 0, 0.1], [0, 1, 0], 0, POOL_RIM);
    expect(up.t).toBeCloseTo(POOL_RIM, 6);
    expect(up.normal).toEqual([0, -1, 0]);
  });
  it('picks the nearest face on a diagonal', () => {
    // Down and toward +x: steeper than the corner, so it hits the floor before the wall.
    const s = Math.hypot(0.5, 1);
    const steep = rayBoxExit([0, 0, 0], [0.5 / s, -1 / s, 0], -0.5, 0);
    expect(steep.normal).toEqual([0, 1, 0]);
    expect(steep.t).toBeCloseTo(0.5 * s, 6);
    // Shallow: reaches the wall x = 1 first.
    const flat = rayBoxExit(
      [0, 0, 0],
      [1 / Math.hypot(1, 0.2), -0.2 / Math.hypot(1, 0.2), 0],
      -1,
      0,
    );
    expect(flat.normal).toEqual([-1, 0, 0]);
    expect(flat.t).toBeCloseTo(Math.hypot(1, 0.2), 6);
  });
  it('never leaves through a face it runs parallel to', () => {
    const r = rayBoxExit([0, -0.5, 0], [0, 0, 1], -1, 0);
    expect(r.normal).toEqual([0, 0, -1]);
    expect(Number.isFinite(r.t)).toBe(true);
  });
});

describe('pointer picking and dragging', () => {
  it('intersects planes', () => {
    expect(intersectPlane([0, 5, 0], [0, -1, 0], [0, 0, 0], [0, 1, 0])).toBeCloseTo(5, 6);
    expect(intersectPlane([0, 5, 0], [1, 0, 0], [0, 0, 0], [0, 1, 0])).toBeNull(); // parallel
    expect(intersectPlane([0, 5, 0], [0, 1, 0], [0, 0, 0], [0, 1, 0])).toBeNull(); // behind
  });

  it('keeps the grab offset: pressing off-centre does not make the ball jump', () => {
    const center: Vec3 = [0.2, 0.1, -0.1];
    const origin: Vec3 = [0, 2, 4];
    const toward = (p: Vec3): Vec3 => {
      const d: Vec3 = [p[0] - origin[0], p[1] - origin[1], p[2] - origin[2]];
      const l = Math.hypot(...d);
      return [d[0] / l, d[1] / l, d[2] / l];
    };
    const dir = toward([0.25, 0.15, 0.05]);
    const t = raySphere(origin, dir, center, 0.24)!;
    expect(t).not.toBeNull();
    const hit: Vec3 = [origin[0] + dir[0] * t, origin[1] + dir[1] * t, origin[2] + dir[2] * t];
    const forward = toward(center);
    const drag = beginDrag(hit, center, forward);
    // The same ray gives the same centre.
    close(dragOnPlane(origin, dir, drag)!, center, 5);
    // A ray 0.3 to the right on the drag plane moves the centre by the same amount on that plane.
    const target: Vec3 = [hit[0] + 0.3, hit[1], hit[2]];
    const moved = dragOnPlane(origin, toward(target), drag)!;
    expect(moved[0] - center[0]).toBeGreaterThan(0.2);
    // The plane faces the camera, so the ball stays on it: its distance along the view direction is unchanged.
    const along = (p: Vec3) => p[0] * forward[0] + p[1] * forward[1] + p[2] * forward[2];
    expect(along(moved)).toBeCloseTo(along(center), 5);
    // A ray that never meets the plane gives nothing.
    expect(dragOnPlane(origin, [-forward[0], -forward[1], -forward[2]], drag)).toBeNull();
  });

  it('turns a pointer into the world ray through the right pixel', () => {
    const camera = createCameraMatrices();
    computeCamera(
      {
        position: orbitEye({ ...INITIAL_ORBIT, yawVelocity: 0, pitchVelocity: 0, idleTime: 0 }),
        target: INITIAL_ORBIT.target,
        fovY: fitFovY(CAMERA_FOV_Y, 1.6, MIN_HORIZONTAL_FOV),
        aspect: 1.6,
        near: 0.05,
        far: 200,
      },
      camera,
    );
    const unproject = (u: number, v: number, depth: number) =>
      unprojectPoint(camera.invViewProj, u, v, depth);
    // Project the ball to the screen, cast a ray back through that pixel, and hit the ball.
    const p = projectPoint(camera.viewProj, INITIAL_BALL);
    expect(p.u).toBeGreaterThan(0);
    expect(p.u).toBeLessThan(1);
    expect(p.v).toBeGreaterThan(0);
    expect(p.v).toBeLessThan(1);
    const ray = pointerRay(unproject, p.u, p.v);
    const t = raySphere(ray.origin, ray.dir, INITIAL_BALL, 0.24)!;
    expect(t).not.toBeNull();
    expect(t).toBeCloseTo(Math.hypot(...ray.origin.map((o, i) => INITIAL_BALL[i]! - o)) - 0.24, 1);
    // The ray starts on the near plane, a hair in front of the eye, and points where the eye looks through that pixel.
    const eye: Vec3 = [camera.position[0]!, camera.position[1]!, camera.position[2]!];
    expect(Math.hypot(...ray.origin.map((o, i) => o - eye[i]!))).toBeLessThan(0.2);
    const throughBall = ray.origin.map((o, i) => INITIAL_BALL[i]! - o);
    const lengthBall = Math.hypot(...throughBall);
    close(
      ray.dir,
      throughBall.map((x) => x / lengthBall),
      1,
    );
    // A pixel far from the ball misses it.
    const away = pointerRay(unproject, Math.max(0.02, p.u - 0.3), p.v);
    expect(raySphere(away.origin, away.dir, INITIAL_BALL, 0.24)).toBeNull();
    // The centre of the screen looks at the camera target.
    const centre = pointerRay(unproject, 0.5, 0.5);
    const toTarget = INITIAL_ORBIT.target.map((x, i) => x - centre.origin[i]!);
    const l = Math.hypot(...toTarget);
    close(
      centre.dir,
      toTarget.map((x) => x / l),
      4,
    );
  });

  it('keeps the ball inside the pool', () => {
    expect(clampBall([5, -9, -5], 0.25, -1, 1.4)).toEqual([0.75, -0.75, -0.75]);
    expect(clampBall([0, 9, 0], 0.25, -1, 1.4)).toEqual([0, 1.4, 0]);
    expect(clampBall([0.1, 0, 0.2], 0.25, -1, 1.4)).toEqual([0.1, 0, 0.2]);
  });

  it('maps world positions to heightfield uv and the light into the caustics frame', () => {
    expect(worldToPoolUv(-1, -1)).toEqual([0, 0]);
    expect(worldToPoolUv(1, 1)).toEqual([1, 1]);
    expect(worldToPoolUv(0, 0.5)).toEqual([0.5, 0.75]);
    expect(toCausticsFrame([0.1, 0.9, 0.3])).toEqual([0.1, 0.3, 0.9]);
  });
});

describe('the ball in the water', () => {
  it('knows how much of it is under water', () => {
    expect(submergedFraction(0, 0.25)).toBeCloseTo(0.5, 6);
    expect(submergedFraction(0.25, 0.25)).toBeCloseTo(0, 6);
    expect(submergedFraction(1, 0.25)).toBe(0);
    expect(submergedFraction(-0.25, 0.25)).toBeCloseTo(1, 6);
    expect(submergedFraction(-2, 0.25)).toBeCloseTo(1, 6);
    // A cap of a sphere: a quarter of the radius under water is 5/32 of the volume.
    expect(submergedFraction(0.25 * 0.5, 0.25)).toBeCloseTo(0.15625, 4);
  });

  const options = (density: number, held = false) => ({
    radius: 0.25,
    density,
    floorY: -1,
    maxY: 1.4,
    held,
  });
  const run = (body: BallBody, density: number, seconds: number, held = false) => {
    for (let i = 0; i < seconds * 60; i++) stepBall(body, options(density, held), 1 / 60);
    return body;
  };

  it('floats with the submerged fraction equal to its density', () => {
    for (const density of [0.3, 0.5, 0.8]) {
      const body = run({ center: [0, 0.6, 0], vy: 0 }, density, 12);
      expect(submergedFraction(body.center[1], 0.25)).toBeCloseTo(density, 1);
      expect(Math.abs(body.vy)).toBeLessThan(0.01);
    }
  });
  it('sinks to the floor and stays', () => {
    const body = run({ center: [0, 0.3, 0], vy: 0 }, 2, 6);
    expect(body.center[1]).toBeCloseTo(-1 + 0.25, 3);
    expect(Math.abs(body.vy)).toBeLessThan(0.01);
  });
  it('holds still while held and never goes NaN on a huge step', () => {
    const body = run({ center: [0, 0.5, 0], vy: 3 }, 0.5, 1, true);
    expect(body.center[1]).toBe(0.5);
    expect(body.vy).toBe(0);
    stepBall(body, options(0.5), 5);
    expect(Number.isFinite(body.center[1])).toBe(true);
    expect(body.center[1]).toBeLessThanOrEqual(1.4);
  });
  it('couples to the surface only while it is touching or under it', () => {
    expect(ballWaterInfluence(0.5, 0.25).weight).toBe(0);
    expect(ballWaterInfluence(0, 0.25)).toEqual({ weight: 1, radius: 0.25 });
    const cut = ballWaterInfluence(0.2, 0.25);
    expect(cut.weight).toBe(1);
    expect(cut.radius).toBeCloseTo(Math.max(Math.sqrt(0.25 ** 2 - 0.2 ** 2), 0.1), 6);
    const deep = ballWaterInfluence(-0.8, 0.25);
    expect(deep.weight).toBeGreaterThan(0);
    expect(deep.weight).toBeLessThan(0.1);
  });
});

describe('the sun', () => {
  it('is a unit vector and refracts by Snell', () => {
    for (const [e, a] of [
      [1, 0],
      [0.4, 2],
      [Math.PI / 2, 1],
    ] as const) {
      expect(Math.hypot(...sunDirection(e, a))).toBeCloseTo(1, 6);
    }
    const { dir, shift } = flatSunRefraction(sunDirection(Math.PI / 4, 0), 1.333);
    expect(Math.hypot(...dir)).toBeCloseTo(1, 6);
    // sin(theta_water) = sin(45 degrees) / 1.333, and the beam heads away from the sun.
    expect(Math.hypot(dir[0], dir[2])).toBeCloseTo(Math.SQRT1_2 / 1.333, 5);
    expect(dir[0]).toBeLessThan(0);
    expect(dir[1]).toBeLessThan(0);
    expect(shift[0]).toBeCloseTo(dir[0] / -dir[1], 6);
    // Overhead sun: no sideways travel.
    const overhead = flatSunRefraction([0, 1, 0], 1.333);
    close(overhead.shift, [0, 0], 6);
  });
});

describe('the WGSL ray helpers match the TypeScript ones', () => {
  it('pl_ray_sphere agrees on a table of rays', async () => {
    const cases: [Vec3, Vec3, Vec3, number][] = [
      [[0, 0, -5], [0, 0, 1], [0, 0, 0], 1],
      [[0, 2, -5], [0, 0, 1], [0, 0, 0], 1],
      [[3, 0, 0], [-1, 0, 0], [0, 0, 0], 2],
      [[0, 0, 0], [0, 0, 1], [0, 0, 0], 1],
      [[0, 0, 5], [0, 0, 1], [0, 0, 0], 1],
      [[0.5, 1, 0.2], [0.3, -0.9, 0.1], [0.6, -0.1, 0.25], 0.3],
    ];
    const unit = (v: Vec3): Vec3 => {
      const l = Math.hypot(...v);
      return [v[0] / l, v[1] / l, v[2] / l];
    };
    // vec4f in, vec4f out: pack one case over several calls (origin and direction, then centre and radius).
    const input = new Float32Array(cases.length * 4);
    const input2 = new Float32Array(cases.length * 4);
    cases.forEach(([o, d, c, r], i) => {
      input.set([...o, 0], i * 4);
      input2.set([...unit(d), 0], i * 4);
      void c;
      void r;
    });
    // The harness takes one vec4 per row, so encode the case index and rebuild the table in WGSL.
    const table = cases
      .map(([o, d, c, r]) => {
        const u = unit(d);
        return `PlCase(vec3f(${o.join(',')}), vec3f(${u.join(',')}), vec3f(${c.join(',')}), ${r})`;
      })
      .join(',\n');
    const code = `${poolRays}
struct PlCase { o: vec3f, d: vec3f, c: vec3f, r: f32 }
const CASES = array<PlCase, ${cases.length}>(${table});
fn f(x: vec4f) -> vec4f {
  let k = CASES[u32(x.x)];
  return vec4f(pl_ray_sphere(k.o, k.d, k.c, k.r), 0.0, 0.0, 0.0);
}`;
    const rows = new Float32Array(cases.length * 4);
    cases.forEach((_, i) => (rows[i * 4] = i));
    const out = await gpuF32(code, rows);
    cases.forEach(([o, d, c, r], i) => {
      const expected = raySphere(o, unit(d), c, r);
      expect(out[i * 4]).toBeCloseTo(expected ?? -1, 4);
    });
  });

  it('pl_exit agrees on rays from inside the pool', async () => {
    type Case = { o: Vec3; d: Vec3; yLo: number; yHi: number };
    const unit = (v: Vec3): Vec3 => {
      const l = Math.hypot(...v);
      return [v[0] / l, v[1] / l, v[2] / l];
    };
    const cases: Case[] = [
      { o: [0, -0.3, 0], d: [1, 0, 0], yLo: -1, yHi: 0.22 },
      { o: [0.2, 0, 0.1], d: unit([0.3, -1, 0.2]), yLo: -1, yHi: 0.22 },
      { o: [-0.4, 0, 0.7], d: unit([-0.5, -0.2, 0.9]), yLo: -0.6, yHi: 0 },
      { o: [0.1, 0, -0.8], d: unit([0.2, 0.7, -0.4]), yLo: 0, yHi: 0.22 },
      { o: [0, 0, 0], d: [0, 0, -1], yLo: -1, yHi: 0.22 },
      { o: [0.9, -0.5, 0.9], d: unit([1, -0.1, 1]), yLo: -1, yHi: 0.22 },
    ];
    const table = cases
      .map(({ o, d }) => `PlCase(vec3f(${o.join(',')}), vec3f(${d.join(',')}))`)
      .join(',\n');
    const ys = cases.map(({ yLo, yHi }) => `vec2f(${yLo}, ${yHi})`).join(',');
    const code = `${poolRays}
struct PlCase { o: vec3f, d: vec3f }
const CASES = array<PlCase, ${cases.length}>(${table});
const YS = array<vec2f, ${cases.length}>(${ys});
fn f(x: vec4f) -> vec4f {
  let i = u32(x.x);
  let e = pl_exit(CASES[i].o, CASES[i].d, YS[i].x, YS[i].y);
  // distance, then the normal's dominant axis as a signed index (x 1, y 2, z 3).
  var axis = 0.0;
  if (abs(e.n.x) > 0.5) { axis = 1.0 * sign(e.n.x); }
  if (abs(e.n.y) > 0.5) { axis = 2.0 * sign(e.n.y); }
  if (abs(e.n.z) > 0.5) { axis = 3.0 * sign(e.n.z); }
  return vec4f(e.t, axis, 0.0, 0.0);
}`;
    const rows = new Float32Array(cases.length * 4);
    cases.forEach((_, i) => (rows[i * 4] = i));
    const out = await gpuF32(code, rows);
    cases.forEach(({ o, d, yLo, yHi }, i) => {
      const expected = rayBoxExit(o, d, yLo, yHi);
      expect(out[i * 4]).toBeCloseTo(expected.t, 4);
      const axis = expected.normal[0]
        ? expected.normal[0]
        : expected.normal[1]
          ? 2 * expected.normal[1]
          : 3 * expected.normal[2];
      expect(out[i * 4 + 1]).toBe(axis);
    });
  });
});

// --- The effect, headless ---------------------------------------------------------------------------

const WIDTH = 160;
const HEIGHT = 120;

function fakeCanvas(width = WIDTH, height = HEIGHT) {
  const canvas = Object.assign(new EventTarget(), {
    width,
    height,
    style: {} as Record<string, string>,
    getBoundingClientRect: () => ({ left: 0, top: 0, width, height }),
    setPointerCapture() {},
    releasePointerCapture() {},
  });
  const fire = (type: string, u: number, v: number, id = 1) => {
    const event = new Event(type);
    const props = {
      clientX: u * width,
      clientY: v * height,
      pointerId: id,
      pointerType: 'mouse',
      pressure: 0.5,
      timeStamp: performance.now(),
    };
    for (const [key, value] of Object.entries(props)) {
      Object.defineProperty(event, key, { value });
    }
    canvas.dispatchEvent(event);
  };
  return { canvas: canvas as unknown as HTMLCanvasElement, style: canvas.style, fire };
}

function halfToFloat(bits: number): number {
  const sign = bits & 0x8000 ? -1 : 1;
  const exponent = (bits >> 10) & 0x1f;
  const fraction = bits & 0x3ff;
  if (exponent === 0) return sign * 2 ** -14 * (fraction / 1024);
  if (exponent === 31) return fraction ? Number.NaN : sign * Infinity;
  return sign * 2 ** (exponent - 15) * (1 + fraction / 1024);
}

async function renderPool(
  overrides: Record<string, unknown>,
  options: {
    frames?: number;
    width?: number;
    height?: number;
    script?: (frame: number, fire: ReturnType<typeof fakeCanvas>['fire']) => void;
  } = {},
): Promise<{ values: Float32Array; width: number; height: number; touchAction: unknown }> {
  const { frames = 6, width = WIDTH, height = HEIGHT, script } = options;
  const device = await getTestDevice();
  const target = device.createTexture({
    size: [width, height],
    format: 'rgba16float',
    usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC,
  });
  const { canvas, style, fire } = fakeCanvas(width, height);
  style.touchAction = 'auto';
  let touchAction: unknown;
  await expectValid(device, async () => {
    const effect = await loadEffect('pool-3d');
    const params = { ...defaultParams(effect.params), ...overrides } as ReturnType<
      typeof defaultParams<typeof effect.params>
    >;
    const instance = await effect.create({
      device,
      format: 'rgba16float',
      canvas,
      params,
      profiler: noProfiler,
      hdr: false,
    });
    instance.resize(width, height, 1);
    touchAction = style.touchAction;
    for (let frame = 0; frame < frames; frame++) {
      script?.(frame, fire);
      const encoder = device.createCommandEncoder();
      instance.frame({ time: frame / 60, delta: 1 / 60, frame }, encoder, target.createView());
      device.queue.submit([encoder.finish()]);
    }
    instance.destroy();
    expect(style.touchAction).toBe('auto');
  });
  const bytes = await readTexture(device, target, 8);
  const half = new Uint16Array(bytes.buffer, bytes.byteOffset, width * height * 4);
  const values = new Float32Array(half.length);
  for (let i = 0; i < half.length; i++) values[i] = halfToFloat(half[i]!);
  target.destroy();
  return { values, width, height, touchAction };
}

function expectFiniteAndDrawn({ values }: { values: Float32Array }) {
  let min = Infinity;
  let max = -Infinity;
  const colours = new Set<number>();
  for (let i = 0; i < values.length; i++) {
    const v = values[i]!;
    expect(Number.isFinite(v)).toBe(true);
    if (i % 4 === 3) continue;
    min = Math.min(min, v);
    max = Math.max(max, v);
    colours.add(Math.round(v * 255));
  }
  expect(min).toBeGreaterThanOrEqual(-1e-3);
  expect(max).toBeLessThanOrEqual(1.002);
  expect(max - min).toBeGreaterThan(0.2);
  expect(colours.size).toBeGreaterThan(40);
}

const difference = (a: Float32Array, b: Float32Array, region?: (i: number) => boolean) => {
  let sum = 0;
  for (let i = 0; i < a.length; i += 4) {
    if (region && !region(i / 4)) continue;
    sum +=
      Math.abs(a[i]! - b[i]!) + Math.abs(a[i + 1]! - b[i + 1]!) + Math.abs(a[i + 2]! - b[i + 2]!);
  }
  return sum;
};

describe('the pool-3d effect, headless', () => {
  it('captures the pointer for drags and gives the touch action back', async () => {
    const device = await getTestDevice();
    const { canvas, style } = fakeCanvas();
    style.touchAction = 'pan-y';
    const effect = await loadEffect('pool-3d');
    const instance = await effect.create({
      device,
      format: 'rgba8unorm',
      canvas,
      params: defaultParams(effect.params),
      profiler: noProfiler,
      hdr: false,
    });
    expect(style.touchAction).toBe('none');
    instance.destroy();
    expect(style.touchAction).toBe('pan-y');
  });

  it('draws every preset with finite colour', async () => {
    const effect = await loadEffect('pool-3d');
    expect(Object.keys(effect.presets ?? {}).length).toBeGreaterThanOrEqual(2);
    for (const [name, preset] of Object.entries(effect.presets ?? {})) {
      const image = await renderPool(preset as Record<string, unknown>, { frames: 5 });
      expectFiniteAndDrawn(image);
      void name;
    }
  });

  it('survives the extremes of its parameters', async () => {
    const effect = await loadEffect('pool-3d');
    const range = (key: string) => {
      const def = effect.params[key]!;
      if (def.type !== 'number') throw new Error(`${key} is not a number`);
      return def;
    };
    const keys = Object.entries(effect.params)
      .filter(([, def]) => def.type === 'number')
      .map(([key]) => key);
    const low: Record<string, unknown> = {};
    const high: Record<string, unknown> = {};
    for (const key of keys) {
      low[key] = range(key).min;
      high[key] = range(key).max;
    }
    for (const overrides of [
      low,
      high,
      { ...high, gridResolution: '384', meshResolution: '192', causticsMethod: 'jacobian' },
      { ...low, gridResolution: '128', meshResolution: '64', edges: 'absorb', caustics: false },
      {
        ...high,
        quality: effect.params.quality && 'cheap (sky reflection, no ball in refraction)',
      },
      { sunElevation: 90, sunAzimuth: 0, waterDepth: 0.45 },
      { sunElevation: 15, waterDepth: 2.2, causticsResolution: '1024' },
      { view: 'caustics debug' },
    ]) {
      const image = await renderPool(overrides, { frames: 8 });
      expectFiniteAndDrawn(image);
    }
  });

  it('runs on a phone-shaped and a wide canvas', async () => {
    for (const [width, height] of [
      [90, 200],
      [240, 80],
      [1, 1],
    ] as const) {
      const { values } = await renderPool({}, { width, height, frames: 3 });
      for (const v of values) expect(Number.isFinite(v)).toBe(true);
    }
  });

  // Where the ball starts on this canvas, in uv, through the same camera maths the effect uses.
  function ballUv(aspect: number): [number, number] {
    const camera = createCameraMatrices();
    computeCamera(
      {
        position: orbitEye({ ...INITIAL_ORBIT, yawVelocity: 0, pitchVelocity: 0, idleTime: 0 }),
        target: INITIAL_ORBIT.target,
        fovY: fitFovY(CAMERA_FOV_Y, aspect, MIN_HORIZONTAL_FOV),
        aspect,
        near: 0.05,
        far: 200,
      },
      camera,
    );
    const p = projectPoint(camera.viewProj, INITIAL_BALL);
    return [p.u, p.v];
  }

  it('drags the ball when a drag starts on it and orbits the camera when it does not', async () => {
    const quiet = { rainRate: 0, autoRotate: false };
    const [bu, bv] = ballUv(WIDTH / HEIGHT);
    const drag =
      (from: [number, number], to: [number, number]) =>
      (frame: number, fire: ReturnType<typeof fakeCanvas>['fire']) => {
        if (frame === 1) fire('pointerdown', from[0], from[1]);
        if (frame >= 2 && frame <= 9) {
          const k = (frame - 1) / 8;
          fire('pointermove', from[0] + (to[0] - from[0]) * k, from[1] + (to[1] - from[1]) * k);
        }
        if (frame === 10) fire('pointerup', to[0], to[1]);
      };
    const still = await renderPool(quiet, { frames: 14 });
    const ballDrag = await renderPool(quiet, {
      frames: 14,
      script: drag([bu, bv], [Math.max(0.05, bu - 0.25), bv + 0.15]),
    });
    const orbitDrag = await renderPool(quiet, {
      frames: 14,
      script: drag([0.15, 0.15], [0.4, 0.2]),
    });
    expectFiniteAndDrawn(ballDrag);
    expectFiniteAndDrawn(orbitDrag);
    const ballChange = difference(still.values, ballDrag.values);
    const orbitChange = difference(still.values, orbitDrag.values);
    expect(ballChange).toBeGreaterThan(5);
    expect(orbitChange).toBeGreaterThan(5);
    // Orbiting turns the whole view; dragging the ball changes a lot less of it.
    expect(orbitChange).toBeGreaterThan(ballChange * 1.5);
    // A drag that started on the ball did not orbit: the far corner (deck, far from the ball and
    // from the ripples) is untouched, while the orbit has moved it.
    const corner = (i: number) => {
      const x = i % WIDTH;
      const y = Math.floor(i / WIDTH);
      return x < 12 && y < 12;
    };
    expect(difference(still.values, ballDrag.values, corner)).toBeLessThan(
      difference(still.values, orbitDrag.values, corner) * 0.5 + 0.5,
    );
  });

  it('makes a drop where a click lands on the water', async () => {
    const quiet = { rainRate: 0, autoRotate: false };
    const [bu, bv] = ballUv(WIDTH / HEIGHT);
    const centre = [0.5, 0.58] as const;
    const click = (frame: number, fire: ReturnType<typeof fakeCanvas>['fire']) => {
      if (frame === 1) fire('pointerdown', centre[0], centre[1]);
      if (frame === 2) fire('pointerup', centre[0], centre[1]);
    };
    // Frames: enough for the ring to spread but not to leave the pool.
    const before = await renderPool(quiet, { frames: 40 });
    const after = await renderPool(quiet, { frames: 40, script: click });
    void bu;
    void bv;
    expectFiniteAndDrawn(after);
    expect(difference(before.values, after.values)).toBeGreaterThan(1);
  });
});

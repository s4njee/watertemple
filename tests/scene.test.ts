import { mat4 } from 'wgpu-matrix';
import { describe, expect, it } from 'vitest';
import {
  CameraUniform,
  computeCamera,
  fitFovY,
  horizontalPlane,
  linearizeDepth,
  mirrorCamera,
  projectPoint,
  reflectionMatrix,
  signedDistance,
  unprojectPoint,
  type CameraDescription,
  type Plane,
} from '../src/gpu/scene/camera.ts';
import { createDepthTarget } from '../src/gpu/scene/depth.ts';
import { createForwardRenderer, type SceneLight } from '../src/gpu/scene/forward.ts';
import {
  boxGeometry,
  cylinderGeometry,
  planeGeometry,
  sphereGeometry,
  VERTEX_FLOATS,
  type Geometry,
} from '../src/gpu/scene/geometry.ts';
import { createMesh } from '../src/gpu/scene/mesh.ts';
import {
  createOrbitState,
  DEFAULT_ORBIT_OPTIONS,
  orbitEye,
  stepOrbit,
  type OrbitInput,
} from '../src/gpu/scene/orbit.ts';
import { getChunk } from '../src/wgsl/index.ts';
import { expectValid, getTestDevice, readTexture } from './gpu.ts';

const near = (actual: number, expected: number, tolerance = 1e-4) =>
  expect(Math.abs(actual - expected)).toBeLessThanOrEqual(tolerance);

const DESCRIPTION: CameraDescription = {
  position: [0, 2, 5],
  target: [0, 0, 0],
  fovY: Math.PI / 3,
  aspect: 2,
  near: 0.1,
  far: 100,
};

describe('camera maths', () => {
  const camera = computeCamera(DESCRIPTION);

  it('projects the look-at target to the screen centre at the expected depth', () => {
    const p = projectPoint(camera.viewProj, [0, 0, 0]);
    near(p.u, 0.5);
    near(p.v, 0.5);
    const distance = Math.hypot(0, 2, 5);
    near(p.w, distance);
    near(p.depth, (100 * (distance - 0.1)) / (distance * (100 - 0.1)), 1e-5);
    near(linearizeDepth(0.1, 100, p.depth), distance, 1e-3);
  });

  it('puts +X on the right, +Y up, with v growing downwards', () => {
    const centre = projectPoint(camera.viewProj, [0, 0, 0]);
    const right = projectPoint(camera.viewProj, [1, 0, 0]);
    const up = projectPoint(camera.viewProj, [0, 1, 0]);
    expect(right.u).toBeGreaterThan(centre.u);
    expect(up.v).toBeLessThan(centre.v);
    // A point 1 unit right of the target: x_ndc = x / (z * tan(fov/2) * aspect).
    const z = Math.hypot(2, 5);
    near(right.u, 0.5 + 0.5 * (1 / (z * Math.tan(Math.PI / 6) * 2)), 1e-4);
  });

  it('maps the frustum corners to the depth range ends', () => {
    // On the view axis, `near` and `far` away.
    const dir = [0, -2, -5].map((c) => c / Math.hypot(2, 5));
    const at = (d: number) => [0, 2 + dir[1]! * d, 5 + dir[2]! * d] as const;
    near(projectPoint(camera.viewProj, at(0.1)).depth, 0, 1e-5);
    near(projectPoint(camera.viewProj, at(100)).depth, 1, 1e-5);
  });

  it('round-trips through the inverse matrices', () => {
    const points: [number, number, number][] = [
      [0, 0, 0],
      [3, 1, -4],
      [-2.5, 0.2, 2],
      [10, 5, -30],
    ];
    for (const point of points) {
      const p = projectPoint(camera.viewProj, point);
      const back = unprojectPoint(camera.invViewProj, p.u, p.v, p.depth);
      for (let k = 0; k < 3; k++) near(back[k]!, point[k]!, 2e-3);
    }
    const closeToIdentity = (a: Float32Array, b: Float32Array) => {
      const product = mat4.multiply(a, b);
      const identity = mat4.identity();
      for (let i = 0; i < 16; i++) near(product[i]!, identity[i]!, 2e-3);
    };
    closeToIdentity(camera.view, camera.invView);
    closeToIdentity(camera.proj, camera.invProj);
    closeToIdentity(camera.viewProj, camera.invViewProj);
    expect(Array.from(camera.position)).toEqual([0, 2, 5]);
  });

  it('widens the field of view on tall canvases only', () => {
    const base = Math.PI / 4;
    near(fitFovY(base, 2, 1), base);
    const tall = fitFovY(base, 0.5, 1.0);
    expect(tall).toBeGreaterThan(base);
    // The horizontal field of view of the fitted camera is the requested minimum.
    near(2 * Math.atan(Math.tan(tall / 2) * 0.5), 1.0, 1e-6);
  });
});

describe('mirror matrix', () => {
  const planes: [string, Plane][] = [
    ['y = 0.5', horizontalPlane(0.5)],
    ['tilted', { normal: [1 / 3, 2 / 3, 2 / 3], offset: -0.7 }],
  ];

  it.each(planes)('maps a point above the plane (%s) to its mirror image', (_name, plane) => {
    const m = reflectionMatrix(plane);
    const above: [number, number, number] = [1.5, 3.25, -0.75];
    const d = signedDistance(plane, above);
    expect(d).toBeGreaterThan(0);
    const mirrored = [0, 1, 2].map(
      (r) => m[r]! * above[0] + m[4 + r]! * above[1] + m[8 + r]! * above[2] + m[12 + r]!,
    ) as [number, number, number];
    near(signedDistance(plane, mirrored), -d);
    // The offset between the two points is along the normal.
    const delta = [0, 1, 2].map((k) => above[k]! - mirrored[k]!);
    for (let k = 0; k < 3; k++) near(delta[k]!, 2 * d * plane.normal[k]!);
  });

  it.each(planes)('is an involution and fixes the plane (%s)', (_name, plane) => {
    const m = reflectionMatrix(plane);
    expect(mat4.equalsApproximately(mat4.multiply(m, m), mat4.identity())).toBe(true);
    // A point on the plane: p - d n.
    const p = [0.3, -1, 2].map(
      (c, k) => c - signedDistance(plane, [0.3, -1, 2]) * plane.normal[k]!,
    );
    near(signedDistance(plane, p as [number, number, number]), 0, 1e-6);
    const image = [0, 1, 2].map(
      (r) => m[r]! * p[0]! + m[4 + r]! * p[1]! + m[8 + r]! * p[2]! + m[12 + r]!,
    );
    for (let k = 0; k < 3; k++) near(image[k]!, p[k]!);
    near(mat4.determinant(m), -1);
  });
});

describe('mirrored camera', () => {
  const plane = horizontalPlane(0);
  const camera = computeCamera({ ...DESCRIPTION, position: [1, 3, 8], aspect: 1.5 });
  const mirror = mirrorCamera(camera, plane);

  it('puts the eye at the mirror image of the eye', () => {
    expect(Array.from(mirror.position)).toEqual([1, -3, 8]);
    // The inverse view moves the origin of view space to that eye.
    const eye = [12, 13, 14].map((i) => mirror.invView[i]!);
    for (let k = 0; k < 3; k++) near(eye[k]!, [1, -3, 8][k]!, 1e-4);
  });

  it('flips handedness, so the pass needs clockwise front faces', () => {
    expect(mat4.determinant(camera.view)).toBeGreaterThan(0);
    expect(mat4.determinant(mirror.view)).toBeLessThan(0);
  });

  it('sees a point on the plane exactly where the real camera does', () => {
    for (const point of [
      [0, 0, 0],
      [-2, 0, 1],
      [3, 0, -4],
    ] as const) {
      const a = projectPoint(camera.viewProj, point);
      const b = projectPoint(mirror.viewProj, point);
      near(a.u, b.u, 1e-4);
      near(a.v, b.v, 1e-4);
    }
  });

  it('sees a point above the plane where the real camera sees its mirror image', () => {
    const above = [0.5, 1.2, -1] as const;
    const image = [0.5, -1.2, -1] as const;
    const a = projectPoint(mirror.viewProj, above);
    const b = projectPoint(camera.viewProj, image);
    near(a.u, b.u, 1e-4);
    near(a.v, b.v, 1e-4);
  });

  it('clips everything below the plane with an oblique near plane (depth 0 on the plane)', () => {
    const onPlane = projectPoint(mirror.viewProj, [1.5, 0, -2]);
    near(onPlane.depth, 0, 1e-4);
    const below = projectPoint(mirror.viewProj, [1.5, -0.3, -2]);
    expect(below.depth).toBeLessThan(-1e-3);
    const above = projectPoint(mirror.viewProj, [1.5, 0.3, -2]);
    expect(above.depth).toBeGreaterThan(1e-3);
    expect(above.depth).toBeLessThan(1);
    // Far geometry above the plane is still inside the depth range.
    const far = projectPoint(mirror.viewProj, [0, 4, -60]);
    expect(far.depth).toBeGreaterThan(0);
    expect(far.depth).toBeLessThan(1);
  });

  it('leaves the projection alone when the eye is below the plane', () => {
    const under = computeCamera({ ...DESCRIPTION, position: [0, -2, 5] });
    const again = mirrorCamera(under, plane);
    expect(Array.from(again.proj)).toEqual(Array.from(under.proj));
  });
});

describe('camera uniform and chunk', () => {
  it('has the layout WGSL expects (six matrices, then eye, near, far, viewport)', () => {
    const offsets = Object.fromEntries(CameraUniform.layout.map((f) => [f.name, f.offset]));
    expect(offsets).toEqual({
      view: 0,
      proj: 64,
      view_proj: 128,
      inv_view: 192,
      inv_proj: 256,
      inv_view_proj: 320,
      position: 384,
      near: 396,
      far: 400,
      viewport: 408,
    });
    expect(CameraUniform.size).toBe(416);
  });

  it('matches the struct in camera.wgsl', () => {
    const chunk = getChunk('camera.wgsl') ?? '';
    expect(chunk).toContain(CameraUniform.wgsl.trim().replace(/\n/g, '\n'));
  });
});

// --- Orbit ---------------------------------------------------------------------------------------

describe('orbit control', () => {
  const idle: OrbitInput = { yaw: 0, pitch: 0, dragging: false };
  const options = { ...DEFAULT_ORBIT_OPTIONS, autoSpeed: 0 };

  it('follows a drag directly and ignores input when not dragging', () => {
    const state = createOrbitState({ yaw: 0, pitch: 0.4 });
    stepOrbit(state, { yaw: 0.25, pitch: 0.1, dragging: true }, 1 / 60, options);
    near(state.yaw, 0.25, 1e-9);
    near(state.pitch, 0.5, 1e-9);
    const still = createOrbitState({ yaw: 1 });
    stepOrbit(still, { yaw: 5, pitch: 5, dragging: false }, 1 / 60, options);
    expect(still.yaw).toBe(1);
  });

  it('keeps coasting after release, then damps to a stop', () => {
    const state = createOrbitState();
    for (let i = 0; i < 10; i++) {
      stepOrbit(state, { yaw: 0.02, pitch: 0, dragging: true }, 1 / 60, options);
    }
    const released = state.yaw;
    expect(state.yawVelocity).toBeGreaterThan(0.3);
    stepOrbit(state, idle, 1 / 60, options);
    expect(state.yaw).toBeGreaterThan(released);
    for (let i = 0; i < 600; i++) stepOrbit(state, idle, 1 / 60, options);
    expect(state.yawVelocity).toBe(0);
    const settled = state.yaw;
    stepOrbit(state, idle, 1 / 60, options);
    expect(state.yaw).toBe(settled);
  });

  it('holding still before release leaves no inertia', () => {
    const state = createOrbitState();
    for (let i = 0; i < 6; i++) {
      stepOrbit(state, { yaw: 0.02, pitch: 0, dragging: true }, 1 / 60, options);
    }
    for (let i = 0; i < 30; i++) {
      stepOrbit(state, { yaw: 0, pitch: 0, dragging: true }, 1 / 60, options);
    }
    expect(Math.abs(state.yawVelocity)).toBeLessThan(1e-3);
  });

  it('clamps pitch to the limits and drops the velocity at the stop', () => {
    const state = createOrbitState({ pitch: 1.2 });
    stepOrbit(state, { yaw: 0, pitch: 1, dragging: true }, 1 / 60, options);
    expect(state.pitch).toBe(options.maxPitch);
    stepOrbit(state, { yaw: 0, pitch: 0, dragging: false }, 1 / 60, options);
    expect(state.pitchVelocity).toBe(0);
    stepOrbit(state, { yaw: 0, pitch: -5, dragging: true }, 1 / 60, options);
    expect(state.pitch).toBe(options.minPitch);
  });

  it('starts drifting only after the idle delay, easing in', () => {
    const auto = { ...DEFAULT_ORBIT_OPTIONS, autoSpeed: 0.3, idleDelay: 2, idleRamp: 2 };
    const state = createOrbitState();
    for (let i = 0; i < 60 * 2; i++) stepOrbit(state, idle, 1 / 60, auto);
    near(state.yaw, 0, 1e-9); // still inside the delay
    let previous = state.yaw;
    const speeds: number[] = [];
    for (let i = 0; i < 60 * 3; i++) {
      stepOrbit(state, idle, 1 / 60, auto);
      speeds.push((state.yaw - previous) * 60);
      previous = state.yaw;
    }
    expect(speeds[0]).toBeLessThan(0.01);
    expect(speeds[speeds.length - 1]!).toBeCloseTo(0.3, 2);
    for (let i = 1; i < speeds.length; i++)
      expect(speeds[i]!).toBeGreaterThanOrEqual(speeds[i - 1]! - 1e-9);
    // Touching the camera resets the clock.
    stepOrbit(state, { yaw: 0, pitch: 0, dragging: true }, 1 / 60, auto);
    expect(state.idleTime).toBe(0);
  });

  it('is stable for a zero or huge time step and bounds the yaw', () => {
    const state = createOrbitState({ yaw: 3 });
    stepOrbit(state, idle, 0, options);
    stepOrbit(state, { yaw: 100, pitch: 0, dragging: true }, 0, options);
    expect(Number.isFinite(state.yaw)).toBe(true);
    expect(Math.abs(state.yaw)).toBeLessThanOrEqual(Math.PI);
  });

  it('places the eye on a sphere around the target, above it for positive pitch', () => {
    const state = createOrbitState({ yaw: 0, pitch: 0.5, distance: 10, target: [1, 2, 3] });
    const eye = orbitEye(state);
    near(Math.hypot(eye[0] - 1, eye[1] - 2, eye[2] - 3), 10);
    near(eye[1], 2 + 10 * Math.sin(0.5));
    near(eye[0], 1); // yaw 0 is on the +Z side
    expect(eye[2]).toBeGreaterThan(3);
    state.yaw = Math.PI / 2;
    expect(orbitEye(state)[0]).toBeGreaterThan(1);
  });
});

// --- Geometry ------------------------------------------------------------------------------------

const triangles = (g: Geometry) =>
  Array.from({ length: g.indexCount / 3 }, (_, t) => [
    g.indices[t * 3]!,
    g.indices[t * 3 + 1]!,
    g.indices[t * 3 + 2]!,
  ]);
const position = (g: Geometry, i: number) => Array.from(g.vertices.subarray(i * 6, i * 6 + 3));
const normal = (g: Geometry, i: number) => Array.from(g.vertices.subarray(i * 6 + 3, i * 6 + 6));
const cross = (a: number[], b: number[]) => [
  a[1]! * b[2]! - a[2]! * b[1]!,
  a[2]! * b[0]! - a[0]! * b[2]!,
  a[0]! * b[1]! - a[1]! * b[0]!,
];
const sub = (a: number[], b: number[]) => a.map((x, k) => x - b[k]!);
const dot = (a: number[], b: number[]) => a.reduce((s, x, k) => s + x * b[k]!, 0);

describe('mesh generation', () => {
  const geometries: [string, Geometry, { vertices: number; indices: number }][] = [
    [
      'plane',
      planeGeometry({ width: 4, depth: 2, segmentsX: 3, segmentsZ: 2 }),
      { vertices: 12, indices: 36 },
    ],
    ['box', boxGeometry({ width: 2, height: 1, depth: 3 }), { vertices: 24, indices: 36 }],
    [
      'sphere',
      sphereGeometry({ radius: 2, widthSegments: 12, heightSegments: 8 }),
      { vertices: 13 * 9, indices: 6 * 12 * 7 },
    ],
    [
      'cylinder',
      cylinderGeometry({ radiusTop: 0.4, radiusBottom: 0.6, height: 2, radialSegments: 10 }),
      { vertices: 2 * 11 + 2 * 12, indices: 6 * 10 + 2 * 3 * 10 },
    ],
    [
      'cone',
      cylinderGeometry({ radiusTop: 0, radiusBottom: 1, height: 2, radialSegments: 8 }),
      { vertices: 2 * 9 + 10, indices: 3 * 8 + 3 * 8 },
    ],
  ];

  it.each(geometries)('%s has the expected counts and valid indices', (_name, g, expected) => {
    expect(g.vertexCount).toBe(expected.vertices);
    expect(g.indexCount).toBe(expected.indices);
    expect(g.vertices.length).toBe(g.vertexCount * VERTEX_FLOATS);
    expect(g.indexCount % 3).toBe(0);
    for (const index of g.indices) {
      expect(index).toBeGreaterThanOrEqual(0);
      expect(index).toBeLessThan(g.vertexCount);
    }
  });

  it.each(geometries)('%s has unit normals and finite positions', (_name, g) => {
    for (let i = 0; i < g.vertexCount; i++) {
      near(Math.hypot(...normal(g, i)), 1, 1e-5);
      for (const c of position(g, i)) expect(Number.isFinite(c)).toBe(true);
    }
  });

  it.each(geometries)('%s is wound counter-clockwise seen from outside', (_name, g) => {
    for (const [a, b, c] of triangles(g)) {
      const pa = position(g, a!);
      const faceNormal = cross(sub(position(g, b!), pa), sub(position(g, c!), pa));
      const area = Math.hypot(...faceNormal);
      expect(area).toBeGreaterThan(1e-9); // no degenerate triangles
      const vertexNormal = normal(g, a!).map((x, k) => x + normal(g, b!)[k]! + normal(g, c!)[k]!);
      expect(dot(faceNormal, vertexNormal)).toBeGreaterThan(0);
    }
  });

  it('puts closed solids around the origin with outward normals', () => {
    for (const [name, g] of geometries.filter(([n]) => ['box', 'sphere'].includes(n))) {
      for (let i = 0; i < g.vertexCount; i++) {
        expect(dot(position(g, i), normal(g, i)), name).toBeGreaterThan(0);
      }
    }
  });

  it('scales the primitives as asked', () => {
    const box = boxGeometry({ width: 2, height: 4, depth: 6 });
    const extent = [0, 1, 2].map((k) => {
      const values = Array.from({ length: box.vertexCount }, (_, i) => position(box, i)[k]!);
      return Math.max(...values) - Math.min(...values);
    });
    expect(extent).toEqual([2, 4, 6]);
    const sphere = sphereGeometry({ radius: 2 });
    for (let i = 0; i < sphere.vertexCount; i++) near(Math.hypot(...position(sphere, i)), 2, 1e-5);
    const cylinder = cylinderGeometry({ height: 3 });
    const ys = Array.from({ length: cylinder.vertexCount }, (_, i) => position(cylinder, i)[1]!);
    expect([Math.min(...ys), Math.max(...ys)]).toEqual([-1.5, 1.5]);
  });

  it('uses 32-bit indices only when 16 bits do not fit', () => {
    expect(sphereGeometry().indices).toBeInstanceOf(Uint16Array);
    expect(planeGeometry({ segmentsX: 300, segmentsZ: 300 }).indices).toBeInstanceOf(Uint32Array);
  });
});

// --- Forward pass on the GPU ---------------------------------------------------------------------

const LIGHT: SceneLight = {
  sunDirection: [0.3, 0.8, 0.5],
  sunColor: [1, 1, 1],
  sunIntensity: 1,
  skyColor: [0.4, 0.5, 0.8],
  groundColor: [0.2, 0.2, 0.2],
  horizonColor: [0.9, 0.9, 0.9],
  ambient: 0.5,
  fogDensity: 0,
};

const SIZE = 64;

async function setup() {
  const device = await getTestDevice();
  const renderer = createForwardRenderer(device, {
    colorFormat: 'rgba8unorm',
    depthFormat: 'depth32float',
  });
  const color = device.createTexture({
    size: [SIZE, SIZE],
    format: 'rgba8unorm',
    usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC,
  });
  const depth = createDepthTarget(device, { label: 'test depth', width: SIZE, height: SIZE });
  const box = createMesh(device, boxGeometry(), 'box');
  const pixel = (pixels: Uint8Array, u: number, v: number) => {
    const x = Math.min(SIZE - 1, Math.floor(u * SIZE));
    const y = Math.min(SIZE - 1, Math.floor(v * SIZE));
    return Array.from(pixels.subarray((y * SIZE + x) * 4, (y * SIZE + x) * 4 + 4));
  };
  return { device, renderer, color, depth, box, pixel };
}

const flat = (r: number, g: number, b: number) => [r, g, b] as const;

describe('forward renderer', () => {
  it('draws a box where the camera maths says it should, with the clear colour elsewhere', async () => {
    const { device, renderer, color, depth, box, pixel } = await setup();
    const camera = computeCamera({
      position: [0, 0, 10],
      target: [0, 0, 0],
      fovY: Math.PI / 3,
      aspect: 1,
      near: 0.1,
      far: 50,
    });
    await expectValid(device, async () => {
      const view = renderer.createView('test');
      view.update(camera, SIZE, SIZE, LIGHT);
      const batch = renderer.upload([
        { mesh: box, model: mat4.scaling([2, 2, 2]), color: flat(1, 0, 0), emissive: 1 },
      ]);
      const encoder = device.createCommandEncoder();
      renderer.draw(encoder, batch, {
        view,
        color: color.createView(),
        depth: depth.view,
        clearColor: { r: 0, g: 1, b: 0, a: 1 },
      });
      device.queue.submit([encoder.finish()]);
    });
    const pixels = await readTexture(device, color);
    // A 2x2x2 box at the origin seen from z = 10 spans x in [-1, 1] at z = 1 (distance 9).
    const half = projectPoint(camera.viewProj, [1, 0, 1]).u;
    expect(pixel(pixels, 0.5, 0.5)).toEqual([255, 0, 0, 255]);
    expect(pixel(pixels, half - 0.03, 0.5)).toEqual([255, 0, 0, 255]);
    expect(pixel(pixels, half + 0.03, 0.5)).toEqual([0, 255, 0, 255]);
    expect(pixel(pixels, 0.02, 0.02)).toEqual([0, 255, 0, 255]);
    // Covered area matches the projected size within a pixel on each side.
    let red = 0;
    for (let i = 0; i < pixels.length; i += 4) if (pixels[i] === 255 && pixels[i + 1] === 0) red++;
    const expectedSide = (half - (1 - half)) * SIZE;
    expect(Math.abs(Math.sqrt(red) - expectedSide)).toBeLessThan(2);
  });

  it.each([
    ['near first', 0],
    ['far first', 1],
  ])('orders overlapping objects by depth, not draw order (%s)', async (_name, farFirst) => {
    const { device, renderer, color, depth, box, pixel } = await setup();
    const camera = computeCamera({
      position: [0, 0, 10],
      target: [0, 0, 0],
      fovY: Math.PI / 3,
      aspect: 1,
      near: 0.1,
      far: 50,
    });
    const nearBox = {
      mesh: box,
      model: mat4.scaling([2, 2, 2]),
      color: flat(1, 0, 0),
      emissive: 1,
    };
    const farBox = {
      mesh: box,
      model: mat4.multiply(mat4.translation([0, 0, -3]), mat4.scaling([8, 8, 2])),
      color: flat(0, 0, 1),
      emissive: 1,
    };
    await expectValid(device, async () => {
      const view = renderer.createView('test');
      view.update(camera, SIZE, SIZE, LIGHT);
      const batch = renderer.upload(farFirst ? [farBox, nearBox] : [nearBox, farBox]);
      const encoder = device.createCommandEncoder();
      renderer.draw(encoder, batch, { view, color: color.createView(), depth: depth.view });
      device.queue.submit([encoder.finish()]);
    });
    const pixels = await readTexture(device, color);
    expect(pixel(pixels, 0.5, 0.5)).toEqual([255, 0, 0, 255]);
    // Off to the side the far box is all there is.
    const sideU = projectPoint(camera.viewProj, [2.5, 0, -2]).u;
    expect(pixel(pixels, sideU, 0.5)).toEqual([0, 0, 255, 255]);
  });

  it('writes depth that matches the camera maths', async () => {
    const { device, renderer, box } = await setup();
    const depth = createDepthTarget(device, {
      label: 'readback depth',
      width: SIZE,
      height: SIZE,
      readback: true,
    });
    const color = device.createTexture({
      size: [SIZE, SIZE],
      format: 'rgba8unorm',
      usage: GPUTextureUsage.RENDER_ATTACHMENT,
    });
    const camera = computeCamera({
      position: [0, 0, 10],
      target: [0, 0, 0],
      fovY: Math.PI / 3,
      aspect: 1,
      near: 0.1,
      far: 50,
    });
    const view = renderer.createView('depth');
    view.update(camera, SIZE, SIZE, LIGHT);
    const batch = renderer.upload([
      { mesh: box, model: mat4.scaling([2, 2, 2]), color: flat(1, 1, 1) },
    ]);
    const buffer = device.createBuffer({
      size: SIZE * 4 * SIZE,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });
    const encoder = device.createCommandEncoder();
    renderer.draw(encoder, batch, { view, color: color.createView(), depth: depth.view });
    encoder.copyTextureToBuffer(
      { texture: depth.texture, aspect: 'depth-only' },
      { buffer, bytesPerRow: SIZE * 4 },
      { width: SIZE, height: SIZE },
    );
    device.queue.submit([encoder.finish()]);
    await buffer.mapAsync(GPUMapMode.READ);
    const values = new Float32Array(buffer.getMappedRange().slice(0));
    buffer.unmap();
    const centre = values[(SIZE / 2) * SIZE + SIZE / 2]!;
    // The front face of the box is at z = 1, i.e. 9 in front of the eye.
    near(linearizeDepth(0.1, 50, centre), 9, 0.01);
    expect(values[0]).toBe(1); // the corner is empty
  });

  it('renders a mirrored pass: the reflection lines up and nothing below the plane leaks in', async () => {
    const { device, renderer, color, depth, box, pixel } = await setup();
    const camera = computeCamera({
      position: [0, 3, 8],
      target: [0, 0, 0],
      fovY: Math.PI / 3,
      aspect: 1,
      near: 0.1,
      far: 60,
    });
    const mirror = mirrorCamera(camera, horizontalPlane(0));
    const objects = [
      // The thing to reflect: a red box floating above the plane.
      { mesh: box, model: mat4.translation([0, 1, 0]), color: flat(1, 0, 0), emissive: 1 },
      // A large blue slab entirely below the plane, between the mirrored eye and the plane.
      {
        mesh: box,
        model: mat4.multiply(mat4.translation([0, -2, 0]), mat4.scaling([30, 1, 30])),
        color: flat(0, 0, 1),
        emissive: 1,
      },
    ];
    await expectValid(device, async () => {
      const view = renderer.createView('mirror');
      view.update(mirror, SIZE, SIZE, LIGHT);
      const batch = renderer.upload(objects);
      const encoder = device.createCommandEncoder();
      renderer.draw(encoder, batch, {
        view,
        color: color.createView(),
        depth: depth.view,
        mirrored: true,
        clearColor: { r: 0, g: 1, b: 0, a: 1 },
      });
      device.queue.submit([encoder.finish()]);
    });
    const pixels = await readTexture(device, color);
    // The camera sees the reflection of the box's underside at plane points in front of it.
    // Ray from the mirrored eye (0,-3,8) through (0,0,0.9) reaches y = 0.5 at z = -0.28: inside the box.
    const hit = projectPoint(mirror.viewProj, [0, 0, 0.9]);
    expect(pixel(pixels, hit.u, hit.v)).toEqual([255, 0, 0, 255]);
    // Through (0,0,3) the ray passes in front of the box (z = 2.2 at y = 0.5): only empty space.
    const miss = projectPoint(mirror.viewProj, [0, 0, 3]);
    expect(pixel(pixels, miss.u, miss.v)).toEqual([0, 255, 0, 255]);
    // No blue anywhere: the slab below the plane was clipped.
    let blue = 0;
    for (let i = 0; i < pixels.length; i += 4)
      if (pixels[i + 2] === 255 && pixels[i] === 0 && pixels[i + 1] === 0) blue++;
    expect(blue).toBe(0);
  });

  it('shades with the sun, ambient and a correct normal matrix under non-uniform scale', async () => {
    const { device, renderer, color, depth, box, pixel } = await setup();
    const camera = computeCamera({
      position: [0, 6, 0.001],
      target: [0, 0, 0],
      up: [0, 0, -1],
      fovY: Math.PI / 4,
      aspect: 1,
      near: 0.1,
      far: 50,
    });
    // Seen from straight above, the top face of a heavily squashed box has normal +Y.
    const light: SceneLight = { ...LIGHT, sunDirection: [0, 1, 0], ambient: 0 };
    const view = renderer.createView('shade');
    view.update(camera, SIZE, SIZE, light);
    const batch = renderer.upload([
      { mesh: box, model: mat4.scaling([4, 0.1, 4]), color: flat(1, 0.5, 0.25) },
    ]);
    const encoder = device.createCommandEncoder();
    renderer.draw(encoder, batch, { view, color: color.createView(), depth: depth.view });
    device.queue.submit([encoder.finish()]);
    const pixels = await readTexture(device, color);
    // n . l = 1, sun colour 1, intensity 1: output is the albedo itself (rounded to 8 bits).
    const [r, g, b] = pixel(pixels, 0.5, 0.5);
    near(r! / 255, 1, 0.01);
    near(g! / 255, 0.5, 0.01);
    near(b! / 255, 0.25, 0.01);
  });

  it('rejects more objects than its buffer holds', async () => {
    const { device, box } = await setup();
    const small = createForwardRenderer(device, {
      colorFormat: 'rgba8unorm',
      depthFormat: 'depth32float',
      maxObjects: 2,
    });
    const object = { mesh: box, model: mat4.identity(), color: flat(1, 1, 1) };
    expect(() => small.upload([object, object, object])).toThrow(RangeError);
    small.destroy();
  });
});

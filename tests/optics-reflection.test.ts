import { mat4, vec3 } from 'wgpu-matrix';
import { describe, expect, it } from 'vitest';
import {
  CameraUniform,
  computeCamera,
  linearizeDepth,
  projectPoint,
  setCameraUniform,
  unprojectPoint,
  type CameraMatrices,
} from '../src/gpu/scene/camera.ts';
import { createDepthTarget } from '../src/gpu/scene/depth.ts';
import {
  createForwardRenderer,
  type SceneLight,
  type SceneObject,
} from '../src/gpu/scene/forward.ts';
import { boxGeometry, cylinderGeometry } from '../src/gpu/scene/geometry.ts';
import { createMesh } from '../src/gpu/scene/mesh.ts';
import { checkShader, formatDiagnostic } from '../src/gpu/shader.ts';
import { createUniformBuffer } from '../src/gpu/uniforms.ts';
import { expectValid, getTestDevice } from './gpu.ts';

// --- Harness: run `fn f(...) -> vec4f` per row with a Camera uniform and a depth texture ----------

let counter = 0;

interface Harness {
  /** Runs `code` (which defines the function below and may #include) for every case. */
  run(code: string, cases: number[][]): Promise<Float32Array>;
  destroy(): void;
}

/**
 * Cases are rows of three vec4f (12 numbers). The shader code defines
 * `fn f(depth: texture_depth_2d, cam: Camera, a: vec4f, b: vec4f, c: vec4f) -> vec4f`.
 */
async function createHarness(
  camera: CameraMatrices,
  width: number,
  height: number,
  depthView: GPUTextureView,
): Promise<Harness> {
  const device = await getTestDevice();
  const uniform = createUniformBuffer(device, CameraUniform, 'test camera');
  setCameraUniform(uniform, camera, width, height);
  uniform.upload();

  return {
    async run(code, cases) {
      const source = `#include "camera.wgsl"
${code}
@group(0) @binding(0) var<uniform> test_cam: Camera;
@group(0) @binding(1) var<storage, read> test_in: array<vec4f>;
@group(0) @binding(2) var<storage, read_write> test_out: array<vec4f>;
@group(0) @binding(3) var test_depth: texture_depth_2d;
@compute @workgroup_size(32) fn test_main(@builtin(global_invocation_id) id: vec3u) {
  if (id.x < ${cases.length}u) {
    test_out[id.x] = f(test_depth, test_cam, test_in[id.x * 3u], test_in[id.x * 3u + 1u], test_in[id.x * 3u + 2u]);
  }
}`;
      const { module, diagnostics } = await checkShader(device, {
        label: `reflection-test-${counter++}.wgsl`,
        code: source,
      });
      const errors = diagnostics.filter((d) => d.severity === 'error');
      if (errors.length > 0) throw new Error(errors.map(formatDiagnostic).join('\n\n'));

      const input = new Float32Array(cases.length * 12);
      cases.forEach((row, i) => input.set(row, i * 12));
      const inputBuffer = device.createBuffer({
        size: input.byteLength,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      });
      device.queue.writeBuffer(inputBuffer, 0, input);
      const outputBytes = cases.length * 16;
      const outputBuffer = device.createBuffer({
        size: outputBytes,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
      });
      const readBuffer = device.createBuffer({
        size: outputBytes,
        usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
      });

      const result = await expectValid(device, async () => {
        const pipeline = device.createComputePipeline({
          layout: 'auto',
          compute: { module, entryPoint: 'test_main' },
        });
        const bindGroup = device.createBindGroup({
          layout: pipeline.getBindGroupLayout(0),
          entries: [
            { binding: 0, resource: { buffer: uniform.buffer } },
            { binding: 1, resource: { buffer: inputBuffer } },
            { binding: 2, resource: { buffer: outputBuffer } },
            { binding: 3, resource: depthView },
          ],
        });
        const encoder = device.createCommandEncoder();
        const pass = encoder.beginComputePass();
        pass.setPipeline(pipeline);
        pass.setBindGroup(0, bindGroup);
        pass.dispatchWorkgroups(Math.ceil(cases.length / 32));
        pass.end();
        encoder.copyBufferToBuffer(outputBuffer, 0, readBuffer, 0, outputBytes);
        device.queue.submit([encoder.finish()]);
        await readBuffer.mapAsync(GPUMapMode.READ);
        const copy = new Float32Array(readBuffer.getMappedRange().slice(0));
        readBuffer.unmap();
        return copy;
      });
      for (const buffer of [inputBuffer, outputBuffer, readBuffer]) buffer.destroy();
      return result;
    },
    destroy() {
      uniform.destroy();
    },
  };
}

/** Draws objects with the forward renderer and keeps the depth buffer for sampling. */
async function renderDepth(camera: CameraMatrices, size: number, objects: SceneObject[]) {
  const device = await getTestDevice();
  const renderer = createForwardRenderer(device, {
    colorFormat: 'rgba8unorm',
    depthFormat: 'depth32float',
  });
  const color = device.createTexture({
    size: [size, size],
    format: 'rgba8unorm',
    usage: GPUTextureUsage.RENDER_ATTACHMENT,
  });
  const depth = createDepthTarget(device, { label: 'ssr test depth', width: size, height: size });
  const light: SceneLight = {
    sunDirection: [0, 1, 0],
    sunColor: [1, 1, 1],
    sunIntensity: 1,
    skyColor: [1, 1, 1],
    groundColor: [1, 1, 1],
    horizonColor: [1, 1, 1],
    ambient: 1,
    fogDensity: 0,
  };
  const view = renderer.createView('ssr test');
  view.update(camera, size, size, light);
  const batch = renderer.upload(objects);
  const encoder = device.createCommandEncoder();
  renderer.draw(encoder, batch, { view, color: color.createView(), depth: depth.view });
  device.queue.submit([encoder.finish()]);
  return { depth, device };
}

// --- camera.wgsl ---------------------------------------------------------------------------------

const SIZE = 128;

const camera = computeCamera({
  position: [0, 2, 8],
  target: [0, 1, 0],
  fovY: Math.PI / 3,
  aspect: 1,
  near: 0.1,
  far: 60,
});

const normalized = (v: number[]) => {
  const out = vec3.normalize(v);
  return Array.from(out);
};

describe('camera.wgsl against the TypeScript camera', () => {
  it('linearises depth, reconstructs positions and projects back, matching the CPU', async () => {
    const device = await getTestDevice();
    const dummy = createDepthTarget(device, { label: 'dummy depth', width: 1, height: 1 });
    const harness = await createHarness(camera, SIZE, SIZE, dummy.view);

    // Points in front of the camera: uv and depth come from the CPU projection.
    const points: [number, number, number][] = [
      [0, 1, 0],
      [2, 0.5, -3],
      [-3, 3, 2],
      [5, 0, -20],
      [-0.5, 1.2, 6],
    ];
    const projected = points.map((p) => ({ p, s: projectPoint(camera.viewProj, p) }));
    const out = await harness.run(
      `
fn f(depth: texture_depth_2d, cam: Camera, a: vec4f, b: vec4f, c: vec4f) -> vec4f {
  // a = (u, v, depth), b = world point
  let z = camera_linear_depth(cam, a.z);
  let world = camera_world_pos(cam, a.xy, a.z);
  let uvd = camera_world_to_uv(cam, b.xyz);
  return vec4f(z, length(world - b.xyz), length(uvd - a.xyz), camera_depth_from_linear(cam, z) - a.z);
}`,
      projected.map(({ p, s }) => [s.u, s.v, s.depth, 0, p[0], p[1], p[2], 0, 0, 0, 0, 0]),
    );
    projected.forEach(({ s }, i) => {
      const [z, worldError, projectionError, depthRoundTrip] = out.subarray(i * 4, i * 4 + 4);
      expect(Math.abs(z! - linearizeDepth(0.1, 60, s.depth))).toBeLessThan(1e-3 * s.w);
      expect(Math.abs(z! - s.w)).toBeLessThan(2e-3 * s.w); // view-space distance == clip w
      expect(worldError).toBeLessThan(5e-3 * s.w);
      expect(projectionError).toBeLessThan(2e-3);
      expect(Math.abs(depthRoundTrip!)).toBeLessThan(1e-5);
    });
    harness.destroy();
    dummy.destroy();
  });

  it('gives world rays that match the CPU and a view-space z of -linear depth', async () => {
    const device = await getTestDevice();
    const dummy = createDepthTarget(device, { label: 'dummy depth', width: 1, height: 1 });
    const harness = await createHarness(camera, SIZE, SIZE, dummy.view);
    const uvs = [
      [0.5, 0.5],
      [0.1, 0.9],
      [0.93, 0.07],
      [0.3, 0.6],
    ];
    const out = await harness.run(
      `
fn f(depth: texture_depth_2d, cam: Camera, a: vec4f, b: vec4f, c: vec4f) -> vec4f {
  let ray = camera_world_ray(cam, a.xy);
  let view = camera_view_pos(cam, a.xy, a.z);
  return vec4f(ray, view.z + camera_linear_depth(cam, a.z));
}`,
      uvs.map(([u, v]) => [u!, v!, 0.9, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
    );
    uvs.forEach(([u, v], i) => {
      const world = unprojectPoint(camera.invViewProj, u!, v!, 0.9);
      const expected = normalized([world[0]! - 0, world[1]! - 2, world[2]! - 8]);
      for (let k = 0; k < 3; k++) {
        expect(Math.abs(out[i * 4 + k]! - expected[k]!)).toBeLessThan(1e-3);
      }
      expect(Math.abs(out[i * 4 + 3]!)).toBeLessThan(2e-3 * 60);
    });
    // The centre ray is the view direction, towards (0, 1, 0) from (0, 2, 8).
    const centre = normalized([0, -1, -8]);
    for (let k = 0; k < 3; k++) expect(Math.abs(out[k]! - centre[k]!)).toBeLessThan(1e-3);
    harness.destroy();
    dummy.destroy();
  });

  it('flags points behind the eye', async () => {
    const device = await getTestDevice();
    const dummy = createDepthTarget(device, { label: 'dummy depth', width: 1, height: 1 });
    const harness = await createHarness(camera, SIZE, SIZE, dummy.view);
    const out = await harness.run(
      `fn f(depth: texture_depth_2d, cam: Camera, a: vec4f, b: vec4f, c: vec4f) -> vec4f {
  return vec4f(camera_world_to_uv(cam, a.xyz), 0.0);
}`,
      [[0, 2, 12, 0, 0, 0, 0, 0, 0, 0, 0, 0]],
    );
    expect(out[2]).toBe(-1);
    harness.destroy();
    dummy.destroy();
  });
});

// --- ssr.wgsl ------------------------------------------------------------------------------------

const TRACE = `
#include "ssr.wgsl"
// a = origin, steps; b = dir, max distance; c = thickness, jitter
fn f(depth: texture_depth_2d, cam: Camera, a: vec4f, b: vec4f, c: vec4f) -> vec4f {
  let hit = ssr_trace(depth, cam, a.xyz, normalize(b.xyz), u32(a.w), b.w, c.x, c.y);
  return vec4f(hit.uv, hit.confidence, hit.distance);
}`;

const WHITE: [number, number, number] = [1, 1, 1];

async function sceneWithWall(
  extra: (
    box: ReturnType<typeof createMesh>,
    cylinder: ReturnType<typeof createMesh>,
  ) => SceneObject[] = () => [],
) {
  const device = await getTestDevice();
  const box = createMesh(device, boxGeometry(), 'wall box');
  const cylinder = createMesh(device, cylinderGeometry({ radialSegments: 32 }), 'pillar');
  // A wall 24 wide and 8 high whose front face is the plane z = -3.9.
  const wall: SceneObject = {
    mesh: box,
    model: mat4.multiply(mat4.translation([0, 3, -4]), mat4.scaling([24, 8, 0.2])),
    color: WHITE,
  };
  const { depth } = await renderDepth(camera, SIZE, [wall, ...extra(box, cylinder)]);
  const harness = await createHarness(camera, SIZE, SIZE, depth.view);
  return { harness, depth };
}

const hitUv = (world: [number, number, number]) => {
  const p = projectPoint(camera.viewProj, world);
  return [p.u, p.v] as const;
};

const PIXEL = 1 / SIZE;

describe('ssr_trace', () => {
  it('hits a wall in a synthetic depth buffer at the expected uv', async () => {
    const { harness, depth } = await sceneWithWall();
    // From the floor in front of the wall, heading up and away from the camera.
    const dir = normalized([0.1, 0.3, -1]);
    const origin: [number, number, number] = [0, 0.5, 4];
    const out = await harness.run(TRACE, [
      [...origin, 64, ...dir, 40, 0.3, 0, 0, 0],
      [...origin, 64, ...dir, 40, 0.3, 0.7, 0, 0], // jittered start
      [...origin, 16, ...dir, 40, 0.3, 0, 0, 0], // coarse
    ]);
    // The ray reaches z = -3.9 after travelling 7.9 / |dir.z| along it.
    const s = 7.9 / Math.abs(dir[2]!);
    const world: [number, number, number] = [
      origin[0] + dir[0]! * s,
      origin[1] + dir[1]! * s,
      -3.9,
    ];
    const [u, v] = hitUv(world);
    for (let i = 0; i < 3; i++) {
      const [hu, hv, confidence, distance] = out.subarray(i * 4, i * 4 + 4);
      expect(Math.abs(hu! - u), `case ${i} u`).toBeLessThan(2 * PIXEL);
      expect(Math.abs(hv! - v), `case ${i} v`).toBeLessThan(2 * PIXEL);
      expect(confidence, `case ${i} confidence`).toBeGreaterThan(0.9);
      expect(Math.abs(distance! - s), `case ${i} distance`).toBeLessThan(0.1);
    }
    harness.destroy();
    depth.destroy();
  });

  it('reports a miss for a ray that leaves the screen', async () => {
    const { harness, depth } = await sceneWithWall();
    const out = await harness.run(TRACE, [
      // Sideways out of the right edge long before any wall.
      [3, 1.5, 2, 64, 1, 0, 0, 40, 0.3, 0, 0, 0],
      // Straight up past the top of the frame (the wall's top is at y = 7, above the view).
      [0, 1, 0, 64, 0, 1, -0.05, 40, 0.3, 0, 0, 0],
      // Away over the camera's shoulder: behind the near plane.
      [0, 1, 10, 64, 0, 0, 1, 40, 0.3, 0, 0, 0],
    ]);
    for (let i = 0; i < 3; i++) expect(out[i * 4 + 2], `case ${i}`).toBe(0);
    harness.destroy();
    depth.destroy();
  });

  it('misses when the wall is farther than max distance, and when it lies behind empty sky', async () => {
    const { harness, depth } = await sceneWithWall();
    const dir = normalized([0, 0.1, -1]);
    const out = await harness.run(TRACE, [
      [0, 1, 4, 64, ...dir, 3, 0.3, 0, 0, 0],
      [0, 1, 4, 64, ...dir, 40, 0.3, 0, 0, 0],
    ]);
    expect(out[2]).toBe(0);
    expect(out[6]).toBeGreaterThan(0.9);
    harness.destroy();
    depth.destroy();
  });

  it('keeps marching past an occluder it is far behind, and ghosts onto it with a huge thickness', async () => {
    const { harness, depth } = await sceneWithWall((_box, cylinder) => [
      {
        // A thin pillar at the origin, 4 high.
        mesh: cylinder,
        model: mat4.multiply(mat4.translation([0, 2, 0]), mat4.scaling([0.6, 4, 0.6])),
        color: WHITE,
      },
    ]);
    // Starts well behind the pillar (z = -2.5), passes behind it on screen, ends on the wall.
    const origin = [3, 1, -2.5];
    const dir = normalized([-6, 0.5, -1.4]);
    const s = (3.9 - 2.5) / Math.abs(dir[2]!);
    const wallHit: [number, number, number] = [
      origin[0]! + dir[0]! * s,
      origin[1]! + dir[1]! * s,
      -3.9,
    ];
    const out = await harness.run(TRACE, [
      [...origin, 128, ...dir, 40, 0.3, 0, 0, 0],
      [...origin, 128, ...dir, 40, 50, 0, 0, 0],
    ]);
    const [u, v] = hitUv(wallHit);
    const sensible = out.subarray(0, 4);
    expect(Math.abs(sensible[0]! - u)).toBeLessThan(2.5 * PIXEL);
    expect(Math.abs(sensible[1]! - v)).toBeLessThan(2.5 * PIXEL);
    expect(sensible[2]).toBeGreaterThan(0.5);
    // With a thickness larger than the whole scene the pillar's silhouette counts as a hit.
    const ghost = out.subarray(4, 8);
    expect(ghost[2]).toBeGreaterThan(0);
    expect(ghost[3]!).toBeLessThan(sensible[3]! - 0.5);
    harness.destroy();
    depth.destroy();
  });

  it('fades the confidence towards the screen edge, matching ssr_edge_fade', async () => {
    const { harness, depth } = await sceneWithWall();
    const origin = [0, 1, 4];
    const slopes = [0.2, 0.5, 0.65, 0.75, 0.8];
    const out = await harness.run(
      TRACE,
      slopes.map((x) => [...origin, 64, ...normalized([x, 0.1, -1]), 40, 0.3, 0, 0, 0]),
    );
    const smoothstep = (t: number) => {
      const x = Math.min(1, Math.max(0, t / 0.1));
      return x * x * (3 - 2 * x);
    };
    const confidences: number[] = [];
    slopes.forEach((_, i) => {
      const [u, v, confidence] = out.subarray(i * 4, i * 4 + 4);
      const expected = smoothstep(Math.min(u!, 1 - u!, v!, 1 - v!));
      expect(Math.abs(confidence! - expected), `ray ${i} at u=${u}`).toBeLessThan(0.03);
      confidences.push(confidence!);
    });
    expect(confidences[0]).toBeGreaterThan(0.97);
    // The last rays land inside the 10% border: strictly between a miss and a solid hit.
    expect(confidences.some((c) => c > 0.02 && c < 0.9)).toBe(true);
    harness.destroy();
    depth.destroy();
  });
});

describe('ssr_edge_fade', () => {
  it('is 1 inside and 0 on the border, monotonic in between', async () => {
    const device = await getTestDevice();
    const dummy = createDepthTarget(device, { label: 'dummy depth', width: 1, height: 1 });
    const harness = await createHarness(camera, SIZE, SIZE, dummy.view);
    const out = await harness.run(
      `#include "ssr.wgsl"
fn f(depth: texture_depth_2d, cam: Camera, a: vec4f, b: vec4f, c: vec4f) -> vec4f {
  return vec4f(ssr_edge_fade(a.xy, 0.1), ssr_edge_fade(b.xy, 0.1), ssr_edge_fade(c.xy, 0.1), 0.0);
}`,
      [[0.5, 0.5, 0, 0, 0, 0.5, 0, 0, 0.05, 0.5, 0, 0]],
    );
    expect(out[0]).toBe(1);
    expect(out[1]).toBe(0);
    expect(out[2]).toBeGreaterThan(0.3);
    expect(out[2]).toBeLessThan(0.7);
    harness.destroy();
    dummy.destroy();
  });
});

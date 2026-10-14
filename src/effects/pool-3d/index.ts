import { createUniformBuffer, defineUniformStruct } from '../../gpu/uniforms.ts';
import { trackPointers } from '../../gpu/pointer.ts';
import { createShader } from '../../gpu/shader.ts';
import {
  CameraUniform,
  computeCamera,
  createCameraMatrices,
  fitFovY,
  unprojectPoint,
} from '../../gpu/scene/camera.ts';
import { boxGeometry, planeGeometry, sphereGeometry } from '../../gpu/scene/geometry.ts';
import { createMesh, MESH_VERTEX_LAYOUT, type Mesh } from '../../gpu/scene/mesh.ts';
import {
  createOrbitState,
  DEFAULT_ORBIT_OPTIONS,
  orbitEye,
  stepOrbit,
} from '../../gpu/scene/orbit.ts';
import { setCameraUniform } from '../../gpu/scene/camera.ts';
import {
  createHeightfieldCaustics,
  gridIndices,
  type CausticsMethod,
} from '../../gpu/sim/caustics.ts';
import {
  createHeightfield,
  type BoundaryMode,
  type Heightfield,
} from '../../gpu/sim/heightfield.ts';
import { defineEffect } from '../types.ts';
import {
  CAMERA_FOV_Y,
  INITIAL_BALL,
  INITIAL_ORBIT,
  MIN_HORIZONTAL_FOV,
  POOL_HALF,
  POOL_RIM,
  ballWaterInfluence,
  beginDrag,
  clamp,
  clampBall,
  dragOnPlane,
  flatSunRefraction,
  intersectPlane,
  pointerRay,
  raySphere,
  stepBall,
  submergedFraction,
  sunDirection,
  toCausticsFrame,
  worldToPoolUv,
  type BallBody,
  type Drag,
  type Vec3,
} from './maths.ts';
import { meta } from './meta.ts';
import poolShader from './pool.wgsl?raw';
import poolRays from './pool-rays.wgsl?raw';

const PoolParams = defineUniformStruct('PoolParams', {
  sun: 'vec3f',
  sun_power: 'f32',
  ball: 'vec4f',
  ball_color: 'vec3f',
  wave_scale: 'f32',
  sigma: 'vec3f',
  turbidity: 'f32',
  scatter: 'vec3f',
  ior: 'f32',
  sun_water: 'vec3f',
  depth: 'f32',
  sun_shift: 'vec2f',
  rim: 'f32',
  tile_scale: 'f32',
  height_scale: 'f32',
  exposure: 'f32',
  ambient: 'f32',
  caustics_on: 'f32',
  mesh_cells: 'u32',
  quality: 'u32',
  view: 'u32',
});

const GRIDS = ['128', '192', '256', '384'] as const;
const MESHES = ['64', '128', '192'] as const;
const CAUSTIC_RESOLUTIONS = ['256', '512', '768', '1024'] as const;
const METHODS = ['jacobian', 'mesh'] as const;
const EDGES = ['reflect', 'absorb'] as const;
const QUALITY = ['full (ray traced)', 'cheap (sky reflection, no ball in refraction)'] as const;
const VIEWS = ['pool', 'caustics debug'] as const;

const DEPTH_FORMAT: GPUTextureFormat = 'depth24plus';
const DEG = Math.PI / 180;
const DECK_SIZE = 160;
const MAX_BALL_HEIGHT = 1.4;

/** mulberry32: a small seeded generator, so the idle drops of a given preset repeat. */
function createRandom(seed: number): () => number {
  let state = seed >>> 0;
  return () => {
    state = (state + 0x6d2b79f5) >>> 0;
    let t = state;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

export const effect = defineEffect({
  meta,
  params: {
    gridResolution: {
      type: 'select',
      default: '192',
      options: GRIDS,
      label: 'Surface grid cells (re-creates the water)',
    },
    meshResolution: {
      type: 'select',
      default: '128',
      options: MESHES,
      label: 'Surface mesh cells (the geometry; shading reads the grid per pixel)',
    },
    caustics: { type: 'boolean', default: true, label: 'Caustics' },
    causticsMethod: {
      type: 'select',
      default: 'mesh',
      options: METHODS,
      label: 'Caustics method',
    },
    causticsResolution: {
      type: 'select',
      default: '512',
      options: CAUSTIC_RESOLUTIONS,
      label: 'Caustics texels per side',
    },
    quality: { type: 'select', default: QUALITY[0], options: QUALITY, label: 'Ray tracing' },
    ballSize: {
      type: 'number',
      default: 0.24,
      min: 0.1,
      max: 0.45,
      step: 0.005,
      label: 'Ball radius (pool half-width = 1)',
    },
    ballDensity: {
      type: 'number',
      default: 0.65,
      min: 0.15,
      max: 2.5,
      step: 0.01,
      label: 'Ball density (below 1 floats, above 1 sinks)',
    },
    ballPush: {
      type: 'number',
      default: 3,
      min: 0,
      max: 8,
      step: 0.1,
      label: 'How hard the ball pushes the water',
    },
    waterColor: {
      type: 'color',
      default: [0.42, 0.85, 0.92],
      label: 'Water colour after one pool depth',
    },
    murk: {
      type: 'number',
      default: 0.25,
      min: 0,
      max: 1.5,
      step: 0.01,
      label: 'Murk (scattering)',
    },
    waterDepth: {
      type: 'number',
      default: 1,
      min: 0.45,
      max: 2.2,
      step: 0.01,
      label: 'Water depth (pool half-width = 1)',
    },
    sunElevation: {
      type: 'number',
      default: 66,
      min: 15,
      max: 90,
      step: 0.5,
      label: 'Sun elevation (degrees)',
    },
    sunAzimuth: {
      type: 'number',
      default: 210,
      min: 0,
      max: 360,
      step: 1,
      label: 'Sun direction around the pool (degrees)',
    },
    waveHeight: {
      type: 'number',
      default: 1.2,
      min: 0.2,
      max: 3,
      step: 0.01,
      label: 'Wave steepness (slope gain)',
    },
    waveSpeed: {
      type: 'number',
      default: 0.5,
      min: 0.15,
      max: 1.5,
      step: 0.01,
      label: 'Wave speed (pool widths / s)',
    },
    damping: {
      type: 'number',
      default: 0.35,
      min: 0,
      max: 3,
      step: 0.01,
      label: 'Damping (1 / s)',
    },
    edges: { type: 'select', default: 'reflect', options: EDGES, label: 'Pool walls' },
    dropSize: {
      type: 'number',
      default: 0.1,
      min: 0.03,
      max: 0.3,
      step: 0.005,
      label: 'Drop radius (pool half-width = 1)',
    },
    dropStrength: {
      type: 'number',
      default: 1.1,
      min: 0,
      max: 4,
      step: 0.01,
      label: 'Drop strength',
    },
    rainRate: {
      type: 'number',
      default: 0.5,
      min: 0,
      max: 12,
      step: 0.1,
      label: 'Idle drops (per second)',
    },
    autoRotate: { type: 'boolean', default: true, label: 'Camera drifts when left alone' },
    exposure: { type: 'number', default: 0.85, min: 0.3, max: 2, step: 0.01 },
    view: { type: 'select', default: 'pool', options: VIEWS, label: 'View' },
  },
  presets: {
    'afternoon pool': {
      sunElevation: 66,
      sunAzimuth: 210,
      waveHeight: 1.2,
      rainRate: 0.5,
      waterDepth: 1,
      caustics: true,
      ballDensity: 0.65,
    },
    'high noon': {
      sunElevation: 84,
      sunAzimuth: 120,
      waveHeight: 1.4,
      rainRate: 1.2,
      waterDepth: 0.8,
      murk: 0.15,
      caustics: true,
    },
    'low sun': {
      sunElevation: 32,
      sunAzimuth: 200,
      waveHeight: 1.1,
      damping: 0.25,
      rainRate: 0.4,
      murk: 0.35,
    },
    'rain storm': {
      sunElevation: 70,
      sunAzimuth: 40,
      waveHeight: 1.6,
      waveSpeed: 0.7,
      damping: 0.6,
      rainRate: 9,
      dropSize: 0.07,
      dropStrength: 1.6,
      edges: 'absorb',
    },
    'heavy ball': {
      ballSize: 0.3,
      ballDensity: 2,
      ballPush: 5,
      waveSpeed: 0.6,
      damping: 0.25,
      sunElevation: 62,
    },
    'cheap cut': {
      quality: QUALITY[1],
      causticsMethod: 'jacobian',
      causticsResolution: '256',
      gridResolution: '128',
      meshResolution: '64',
    },
  },

  async create({ device, format, canvas, params, profiler }) {
    const random = createRandom(0x9001);
    const pointers = trackPointers(canvas, { capture: true });

    // --- Simulation and caustics ---------------------------------------------------------------
    const solver: Heightfield = createHeightfield({
      device,
      kind: 'render',
      resolution: Number(params.gridResolution),
      width: 1,
      height: 1,
      params: { boundary: 'reflect', leak: 0.08 },
      profiler,
      label: 'pool',
    });
    const caustics = createHeightfieldCaustics({
      device,
      heightfield: solver,
      resolution: Number(params.causticsResolution),
      method: params.causticsMethod as CausticsMethod,
      profiler,
      label: 'caustics',
    });

    // --- Geometry ------------------------------------------------------------------------------
    // The pool's inside: a unit box without its top face (indices 12..17 are +Y), culled from the front.
    const box = boxGeometry();
    const boxNoTop = {
      ...box,
      indices: Uint16Array.from([...box.indices.slice(0, 12), ...box.indices.slice(18)]),
      indexCount: box.indexCount - 6,
    };
    const boxMesh: Mesh = createMesh(device, boxNoTop, 'pool box');
    const deckMesh: Mesh = createMesh(
      device,
      planeGeometry({ width: DECK_SIZE, depth: DECK_SIZE }),
      'pool deck',
    );
    const ballMesh: Mesh = createMesh(
      device,
      sphereGeometry({ radius: 1, widthSegments: 48, heightSegments: 24 }),
      'pool ball',
    );

    let meshCells = 0;
    let waterIndices: GPUBuffer | undefined;
    let waterIndexCount = 0;
    function allocateWaterMesh(cells: number) {
      if (cells === meshCells && waterIndices) return;
      meshCells = cells;
      waterIndices?.destroy();
      const data = gridIndices(cells, cells);
      waterIndexCount = data.length;
      waterIndices = device.createBuffer({
        label: 'pool water indices',
        size: data.byteLength,
        usage: GPUBufferUsage.INDEX | GPUBufferUsage.COPY_DST,
      });
      device.queue.writeBuffer(waterIndices, 0, data);
    }
    allocateWaterMesh(Number(params.meshResolution));

    // --- Pipelines -----------------------------------------------------------------------------
    const both = GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT;
    const layout = device.createBindGroupLayout({
      label: 'pool layout',
      entries: [
        { binding: 0, visibility: both, buffer: { type: 'uniform' } },
        { binding: 1, visibility: both, buffer: { type: 'uniform' } },
        { binding: 2, visibility: both, texture: { sampleType: 'float' } },
        { binding: 3, visibility: both, texture: { sampleType: 'float' } },
        { binding: 4, visibility: both, sampler: { type: 'filtering' } },
      ],
    });
    const module = createShader(device, {
      label: 'pool-3d/pool.wgsl',
      code: poolShader,
      chunks: { 'pool_params.wgsl': PoolParams.wgsl, 'pool_rays.wgsl': poolRays },
    });
    const pipelineLayout = device.createPipelineLayout({
      label: 'pool pipeline layout',
      bindGroupLayouts: [layout],
    });
    const makePipeline = (
      name: string,
      vertex: string,
      fragment: string,
      options: {
        buffers?: GPUVertexBufferLayout[];
        cull: GPUCullMode;
        depthCompare?: GPUCompareFunction;
        depthWrite?: boolean;
      },
    ) =>
      device.createRenderPipeline({
        label: `pool ${name}`,
        layout: pipelineLayout,
        vertex: { module, entryPoint: vertex, buffers: options.buffers ?? [] },
        fragment: { module, entryPoint: fragment, targets: [{ format }] },
        primitive: { topology: 'triangle-list', cullMode: options.cull },
        depthStencil: {
          format: DEPTH_FORMAT,
          depthWriteEnabled: options.depthWrite ?? true,
          depthCompare: options.depthCompare ?? 'less',
        },
      });
    const meshBuffers = [MESH_VERTEX_LAYOUT];
    const skyPipeline = makePipeline('sky', 'fullscreen_vs', 'sky_fs', {
      cull: 'none',
      depthCompare: 'always',
      depthWrite: false,
    });
    const deckPipeline = makePipeline('deck', 'deck_vs', 'deck_fs', {
      buffers: meshBuffers,
      cull: 'back',
    });
    const boxPipeline = makePipeline('walls', 'box_vs', 'box_fs', {
      buffers: meshBuffers,
      cull: 'front',
    });
    const ballPipeline = makePipeline('ball', 'ball_vs', 'ball_fs', {
      buffers: meshBuffers,
      cull: 'back',
    });
    const waterPipeline = makePipeline('water', 'water_vs', 'water_fs', { cull: 'none' });

    // --- Uniforms, bindings, targets ------------------------------------------------------------
    const cameraBuffer = createUniformBuffer(device, CameraUniform, 'pool camera');
    const paramsBuffer = createUniformBuffer(device, PoolParams, 'pool params');
    paramsBuffer.set({ rim: POOL_RIM });
    const sampler = device.createSampler({
      label: 'pool sampler',
      magFilter: 'linear',
      minFilter: 'linear',
      addressModeU: 'clamp-to-edge',
      addressModeV: 'clamp-to-edge',
    });
    let bindGroup: GPUBindGroup | undefined;
    let boundKey = '';
    let depthTexture: GPUTexture | undefined;
    let depthView: GPUTextureView | undefined;
    let width = Math.max(1, canvas.width || 1);
    let height = Math.max(1, canvas.height || 1);

    // --- Interaction state ----------------------------------------------------------------------
    const orbit = createOrbitState({ ...INITIAL_ORBIT });
    const orbitOptions = { ...DEFAULT_ORBIT_OPTIONS, minPitch: 0.28, maxPitch: 1.45 };
    const camera = createCameraMatrices();
    const ball: BallBody = { center: [...INITIAL_BALL], vy: 0 };
    let previousCenter: Vec3 = [...ball.center];
    let previousSubmerged = submergedFraction(ball.center[1], Number(params.ballSize));
    let drag: Drag | undefined;
    type Gesture = { id: number; mode: 'ball' | 'orbit'; travel: number; start: number };
    let gesture: Gesture | undefined;
    let clock = 0;
    let rainDebt = 0;
    const clickDropStrength = 1.3;

    const unproject = (u: number, v: number, depth: number) =>
      unprojectPoint(camera.invViewProj, u, v, depth);
    const pointerWorldRay = (u: number, v: number) => pointerRay(unproject, u, v);

    function dropAt(x: number, z: number, strength: number, scale = 1) {
      const [u, v] = worldToPoolUv(x, z);
      if (u < 0 || u > 1 || v < 0 || v > 1) return;
      solver.addDrop(u, v, (params.dropSize * scale) / (2 * POOL_HALF), -strength);
    }

    // Two drops to start with, so the first frame is not a flat pool.
    dropAt(-0.45, 0.3, 1.4);
    dropAt(0.2, -0.55, 1.1);

    function handlePointers(delta: number) {
      const frame = pointers.poll();
      let orbitDx = 0;
      let orbitDy = 0;
      let dragging = false;
      const radius = params.ballSize;

      for (const pointer of frame) {
        if (!gesture && pointer.justPressed) {
          const ray = pointerWorldRay(pointer.x, pointer.y);
          const t = raySphere(ray.origin, ray.dir, ball.center, radius * 1.12);
          if (t !== null) {
            const hit: Vec3 = [
              ray.origin[0] + ray.dir[0] * t,
              ray.origin[1] + ray.dir[1] * t,
              ray.origin[2] + ray.dir[2] * t,
            ];
            const forward: Vec3 = [
              orbit.target[0] - camera.position[0]!,
              orbit.target[1] - camera.position[1]!,
              orbit.target[2] - camera.position[2]!,
            ];
            const length = Math.hypot(...forward) || 1;
            drag = beginDrag(hit, ball.center, [
              forward[0] / length,
              forward[1] / length,
              forward[2] / length,
            ]);
            gesture = { id: pointer.id, mode: 'ball', travel: 0, start: clock };
          } else {
            gesture = { id: pointer.id, mode: 'orbit', travel: 0, start: clock };
          }
        }
        if (!gesture || pointer.id !== gesture.id) continue;

        if (gesture.mode === 'ball' && drag && pointer.down) {
          const ray = pointerWorldRay(pointer.x, pointer.y);
          const target = dragOnPlane(ray.origin, ray.dir, drag);
          if (target) {
            ball.center = clampBall(target, radius, -params.waterDepth, MAX_BALL_HEIGHT);
          }
        } else if (gesture.mode === 'orbit' && pointer.down) {
          orbitDx += pointer.dx;
          orbitDy += pointer.dy;
          dragging = true;
          gesture.travel += Math.abs(pointer.dx) + Math.abs(pointer.dy);
        }

        if (pointer.justReleased || !pointer.down) {
          const wasClick =
            gesture.mode === 'orbit' &&
            !pointer.cancelled &&
            gesture.travel < 0.012 &&
            clock - gesture.start < 0.5;
          if (wasClick) {
            // A click on the water: a drop where the ray meets the surface.
            const ray = pointerWorldRay(pointer.x, pointer.y);
            const t = intersectPlane(ray.origin, ray.dir, [0, 0, 0], [0, 1, 0]);
            if (t !== null) {
              dropAt(
                ray.origin[0] + ray.dir[0] * t,
                ray.origin[2] + ray.dir[2] * t,
                clickDropStrength * params.dropStrength,
              );
            }
          }
          gesture = undefined;
          drag = undefined;
        }
      }
      if (gesture && !frame.some((p) => p.id === gesture!.id)) {
        gesture = undefined;
        drag = undefined;
      }

      const options = params.autoRotate ? orbitOptions : { ...orbitOptions, autoSpeed: 0 };
      stepOrbit(
        orbit,
        { yaw: dragging ? -orbitDx * 3 : 0, pitch: dragging ? orbitDy * 2 : 0, dragging },
        delta,
        options,
      );
      return gesture?.mode === 'ball';
    }

    function stirFromBall(held: boolean, delta: number) {
      const r = params.ballSize;
      const body = ball;
      stepBall(
        body,
        {
          radius: r,
          density: params.ballDensity,
          floorY: -params.waterDepth,
          maxY: MAX_BALL_HEIGHT,
          held,
        },
        delta,
      );
      body.center = clampBall(body.center, r, -params.waterDepth, MAX_BALL_HEIGHT);

      // Horizontal and vertical motion both move water; the displaced volume is conserved.
      const influence = ballWaterInfluence(body.center[1], r);
      const [fromU, fromV] = worldToPoolUv(previousCenter[0], previousCenter[2]);
      const [toU, toV] = worldToPoolUv(body.center[0], body.center[2]);
      const radius = influence.radius / (2 * POOL_HALF);
      if (influence.weight > 0.01 && params.ballPush > 0) {
        solver.displace([fromU, fromV], [toU, toV], radius, params.ballPush * influence.weight);
      }
      const submerged = submergedFraction(body.center[1], r);
      const dVolume = (submerged - previousSubmerged) * ((4 / 3) * Math.PI * r ** 3);
      if (Math.abs(dVolume) > 1e-7 && params.ballPush > 0) {
        // Volume as a dent over the footprint, in the solver's units (see Heightfield.addDrop).
        const footprint = Math.max(influence.radius, 0.5 * r) / (2 * POOL_HALF);
        const dent = dVolume / (2 * POOL_HALF) ** 3 / (Math.PI * footprint * footprint);
        solver.addDrop(
          toU,
          toV,
          footprint,
          clamp((-dent / (0.2 * footprint)) * 0.5 * params.ballPush, -6, 6),
        );
      }
      previousCenter = [...body.center];
      previousSubmerged = submerged;
    }

    function idleRain(delta: number, busy: boolean) {
      rainDebt += delta * params.rainRate * (busy ? 0.3 : 1);
      let count = 0;
      while (rainDebt >= 1 && count < 6) {
        rainDebt -= 1;
        count++;
        const size = 0.7 + 0.6 * random();
        dropAt(
          (random() * 2 - 1) * 0.92 * POOL_HALF,
          (random() * 2 - 1) * 0.92 * POOL_HALF,
          params.dropStrength * (0.7 + 0.6 * random()) * 1.3,
          size,
        );
      }
      rainDebt = Math.min(rainDebt, 1);
    }

    return {
      resize(w, h) {
        width = Math.max(1, Math.round(w));
        height = Math.max(1, Math.round(h));
        depthTexture?.destroy();
        depthTexture = device.createTexture({
          label: 'pool depth',
          size: [width, height],
          format: DEPTH_FORMAT,
          usage: GPUTextureUsage.RENDER_ATTACHMENT,
        });
        depthView = depthTexture.createView();
      },

      frame({ delta }, encoder, target) {
        if (!depthView) return;
        clock += delta;
        const aspect = width / height;

        // Parameters that re-create things.
        solver.setResolution(Number(params.gridResolution));
        const wave = solver.params;
        wave.waveSpeed = params.waveSpeed;
        wave.damping = params.damping;
        wave.boundary = params.edges as BoundaryMode;
        allocateWaterMesh(Number(params.meshResolution));
        caustics.method = params.causticsMethod as CausticsMethod;
        caustics.setResolution(Number(params.causticsResolution));

        // The camera comes first: picking needs this frame's matrices. (The ones from the last
        // frame are what the visitor was looking at when they pressed.)
        const holdingBall = handlePointers(delta);
        computeCamera(
          {
            position: orbitEye(orbit),
            target: orbit.target,
            fovY: fitFovY(CAMERA_FOV_Y, aspect, MIN_HORIZONTAL_FOV),
            aspect,
            near: 0.05,
            far: 200,
          },
          camera,
        );
        setCameraUniform(cameraBuffer, camera, width, height);
        cameraBuffer.upload();

        stirFromBall(holdingBall, delta);
        idleRain(delta, gesture !== undefined);
        solver.update(encoder, delta);

        // Light, in the water and above it.
        const sun = sunDirection(params.sunElevation * DEG, params.sunAzimuth * DEG);
        const flat = flatSunRefraction(sun, 1.333);
        const radius = params.ballSize;
        const depth = params.waterDepth;
        if (params.caustics) {
          const c = ball.center;
          caustics.update(encoder, {
            lightDir: toCausticsFrame(sun),
            depth: depth / (2 * POOL_HALF),
            ior: 1.333,
            strength: params.waveHeight,
            extend: false,
            smooth: params.causticsMethod === 'mesh' ? 0.9 : 0.6,
            useHeight: false,
            occluder: {
              center: [
                c[0] / (2 * POOL_HALF) + 0.5,
                c[2] / (2 * POOL_HALF) + 0.5,
                c[1] / (2 * POOL_HALF),
              ],
              radius: radius / (2 * POOL_HALF),
            },
          });
        }

        const key = `${solver.generation}:${caustics.generation}`;
        if (!bindGroup || key !== boundKey) {
          boundKey = key;
          bindGroup = device.createBindGroup({
            label: 'pool bind group',
            layout,
            entries: [
              { binding: 0, resource: { buffer: cameraBuffer.buffer } },
              { binding: 1, resource: { buffer: paramsBuffer.buffer } },
              { binding: 2, resource: solver.view },
              { binding: 3, resource: caustics.view },
              { binding: 4, resource: sampler },
            ],
          });
        }

        const sigma = params.waterColor.map((c) => -Math.log(clamp(c, 1e-3, 1))) as Vec3;
        paramsBuffer.set({
          sun,
          sun_power: 2.3,
          ball: [ball.center[0], ball.center[1], ball.center[2], radius],
          ball_color: [0.8, 0.1, 0.04],
          wave_scale: params.waveHeight,
          sigma,
          turbidity: params.murk,
          scatter: [0.03, 0.16, 0.24],
          ior: 1.333,
          sun_water: flat.dir,
          depth,
          sun_shift: flat.shift,
          rim: POOL_RIM,
          tile_scale: 6,
          height_scale: params.waveHeight * ((2 * POOL_HALF) / solver.grid.width),
          exposure: params.exposure,
          ambient: 1.3,
          caustics_on: params.caustics ? 1 : 0,
          mesh_cells: meshCells,
          quality: params.quality === QUALITY[0] ? 0 : 1,
          view: params.view === 'caustics debug' ? 1 : 0,
        });
        paramsBuffer.upload();

        const pass = encoder.beginRenderPass({
          label: 'pool scene',
          colorAttachments: [
            { view: target, loadOp: 'clear', storeOp: 'store', clearValue: [0, 0, 0, 1] },
          ],
          depthStencilAttachment: {
            view: depthView,
            depthLoadOp: 'clear',
            depthStoreOp: 'discard',
            depthClearValue: 1,
          },
          timestampWrites: profiler.pass('scene'),
        });
        pass.setBindGroup(0, bindGroup);

        pass.setPipeline(skyPipeline);
        pass.draw(3);

        const drawMesh = (pipeline: GPURenderPipeline, mesh: Mesh) => {
          pass.setPipeline(pipeline);
          pass.setVertexBuffer(0, mesh.vertexBuffer);
          pass.setIndexBuffer(mesh.indexBuffer, mesh.indexFormat);
          pass.drawIndexed(mesh.indexCount);
        };
        drawMesh(deckPipeline, deckMesh);
        drawMesh(boxPipeline, boxMesh);
        drawMesh(ballPipeline, ballMesh);

        pass.setPipeline(waterPipeline);
        pass.setIndexBuffer(waterIndices!, 'uint32');
        pass.drawIndexed(waterIndexCount);
        pass.end();
      },

      destroy() {
        pointers.destroy();
        caustics.destroy();
        solver.destroy();
        boxMesh.destroy();
        deckMesh.destroy();
        ballMesh.destroy();
        waterIndices?.destroy();
        depthTexture?.destroy();
        cameraBuffer.destroy();
        paramsBuffer.destroy();
      },
    };
  },
});

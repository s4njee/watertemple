import { mat4 } from 'wgpu-matrix';
import {
  createFluidSurface,
  FILTERS,
  type FluidFilter,
  type FluidSurface,
  type FluidSurfaceParams,
} from '../../gpu/fluid-surface/index.ts';
import { createOutputPass } from '../../gpu/post/index.ts';
import { createBillboardRenderer, createComputePass, type Rgb } from '../../gpu/particles/index.ts';
import { trackPointers, type TrackedPointer } from '../../gpu/pointer.ts';
import {
  computeCamera,
  createCameraMatrices,
  fitFovY,
  unprojectPoint,
} from '../../gpu/scene/camera.ts';
import { createDepthTarget } from '../../gpu/scene/depth.ts';
import {
  createForwardRenderer,
  SCENE_STRUCT_CHUNKS,
  type SceneLight,
  type SceneObject,
} from '../../gpu/scene/forward.ts';
import { boxGeometry, cylinderGeometry } from '../../gpu/scene/geometry.ts';
import { createMesh, type Mesh } from '../../gpu/scene/mesh.ts';
import {
  createOrbitState,
  DEFAULT_ORBIT_OPTIONS,
  orbitEye,
  stepOrbit,
} from '../../gpu/scene/orbit.ts';
import { createShader } from '../../gpu/shader.ts';
import { createRenderTarget } from '../../gpu/targets.ts';
import {
  createPbf,
  type Pbf,
  type PbfBlock,
  type PbfCollider,
  type PbfEmitter,
} from '../../gpu/sim/pbf.ts';
import { createWhitewater } from '../../gpu/particles/whitewater.ts';
import { createContainerController } from '../../gpu/particles/container.ts';
import { defineEffect } from '../types.ts';
import { meta } from './meta.ts';
import sceneShader from './scene.wgsl';
import speedShader from './speed.wgsl';

// The 3D position-based-fluid tank: the PBF solver (src/gpu/sim/pbf.ts) feeding the screen-space fluid surface
// (src/gpu/fluid-surface/), in a small scene drawn with the forward renderer (sky, tiled floor, the tank's frame,
// pillars, a wave-maker paddle, a nozzle). Everything physical happens in metres: the tank is 1.6 x 0.8 x 0.6 m.

const COLOR_FORMAT: GPUTextureFormat = 'rgba16float';
const DEPTH_FORMAT = 'depth32float';

const SCENES = ['dam break', 'pour', 'wave tank'] as const;
type SceneName = (typeof SCENES)[number];

const VIEWS = ['liquid', 'spheres', 'speed', 'foam'] as const;
type View = (typeof VIEWS)[number];

/** Water types of absorption.wgsl, by the name the control shows. `custom` uses the colour. */
const WATER = ['clear', 'coastal', 'murky', 'tea', 'milk', 'pool', 'ocean', 'custom'] as const;
const SCALES: Record<string, number> = { '1': 1, '1/2': 0.5, '1/4': 0.25 };

/** The inside of the tank. */
const TANK = {
  min: [-0.8, 0, -0.3] as [number, number, number],
  max: [0.8, 0.8, 0.3] as [number, number, number],
};
const TANK_AREA = (TANK.max[0] - TANK.min[0]) * (TANK.max[2] - TANK.min[2]);

const SUN_DIRECTION: readonly [number, number, number] = [0.5, 0.7, 0.35];
const LIGHT: SceneLight = {
  sunDirection: SUN_DIRECTION,
  sunColor: [1, 0.95, 0.85],
  sunIntensity: 1.5,
  skyColor: [0.5, 0.65, 0.9],
  groundColor: [0.3, 0.28, 0.25],
  horizonColor: [0.6, 0.72, 0.85],
  ambient: 0.5,
  fogDensity: 0.012,
};

const FOV_Y = (40 * Math.PI) / 180;
const MIN_HORIZONTAL_FOV = (64 * Math.PI) / 180;

/** Pillars of the dam break: collider boxes, also drawn. */
const PILLARS: readonly PbfCollider[] = [
  { kind: 'box', min: [0.0, 0, -0.2], max: [0.12, 0.6, -0.08] },
  { kind: 'box', min: [0.32, 0, 0.06], max: [0.44, 0.6, 0.18] },
];

/** The nozzle of the pour. */
const NOZZLE_X = -0.3;
const NOZZLE: PbfEmitter = { position: [NOZZLE_X, 0.74, 0], velocity: [0, -2.2, 0], radius: 0.06 };

interface SceneSpec {
  /** Liquid volume once the scene is full (m^3): the particle spacing follows from it and the count. */
  volume: number;
  blocks(): PbfBlock[];
  colliders: readonly PbfCollider[];
  emitter?: PbfEmitter;
}

const SPECS: Record<SceneName, SceneSpec> = {
  'dam break': {
    volume: 0.6 * 0.6 * 0.6,
    blocks: () => [{ min: [-0.8, 0, -0.3], max: [-0.2, 0.6, 0.3], jitter: 0.1 }],
    colliders: PILLARS,
  },
  pour: {
    volume: TANK_AREA * 0.3,
    blocks: () => [{ min: [-0.8, 0, -0.3], max: [0.8, 0.04, 0.3], jitter: 0.1 }],
    colliders: [],
    emitter: NOZZLE,
  },
  'wave tank': {
    volume: TANK_AREA * 0.3,
    blocks: () => [{ min: [-0.8, 0, -0.3], max: [0.8, 0.3, 0.3], jitter: 0.1 }],
    colliders: [],
  },
};

const clamp = (x: number, lo: number, hi: number) => Math.min(hi, Math.max(lo, x));
const indexOf = <T extends string>(list: readonly T[], value: string): T =>
  list[Math.max(0, list.indexOf(value as T))]!;

type V3 = [number, number, number];

// --- Rays ----------------------------------------------------------------------------------------------------------

interface Ray {
  origin: V3;
  dir: V3;
}

function normalise(v: V3): V3 {
  const l = Math.hypot(v[0], v[1], v[2]) || 1;
  return [v[0] / l, v[1] / l, v[2] / l];
}

/** Entry and exit distances of a ray through an axis-aligned box, or undefined if it misses. */
function rayBox(ray: Ray, min: V3, max: V3): [number, number] | undefined {
  let t0 = 0;
  let t1 = Infinity;
  for (let a = 0; a < 3; a++) {
    const d = ray.dir[a]!;
    if (Math.abs(d) < 1e-9) {
      if (ray.origin[a]! < min[a]! || ray.origin[a]! > max[a]!) return undefined;
      continue;
    }
    let near = (min[a]! - ray.origin[a]!) / d;
    let far = (max[a]! - ray.origin[a]!) / d;
    if (near > far) [near, far] = [far, near];
    t0 = Math.max(t0, near);
    t1 = Math.min(t1, far);
    if (t0 > t1) return undefined;
  }
  return [t0, t1];
}

/** Where a ray meets the plane through `point` with normal `normal`, or undefined. */
function rayPlane(ray: Ray, point: V3, normal: V3): V3 | undefined {
  const denominator = ray.dir[0] * normal[0] + ray.dir[1] * normal[1] + ray.dir[2] * normal[2];
  if (Math.abs(denominator) < 1e-6) return undefined;
  const t =
    ((point[0] - ray.origin[0]) * normal[0] +
      (point[1] - ray.origin[1]) * normal[1] +
      (point[2] - ray.origin[2]) * normal[2]) /
    denominator;
  if (t < 0) return undefined;
  return [
    ray.origin[0] + ray.dir[0] * t,
    ray.origin[1] + ray.dir[1] * t,
    ray.origin[2] + ray.dir[2] * t,
  ];
}

// --- Palettes of the particle views ----------------------------------------------------------------------------------

const SPEED_PALETTE: Rgb[] = [
  [0.01, 0.07, 0.25],
  [0.03, 0.35, 0.72],
  [0.35, 0.85, 1],
  [1, 1, 1],
];
const FOAM_PALETTE: Rgb[] = [
  [0.02, 0.12, 0.35],
  [0.06, 0.4, 0.75],
  [0.55, 0.9, 1],
  [1, 1, 1],
];

// --- The world: a solver, and the numbers its scene needs ----------------------------------------------------------

interface World {
  scene: SceneName;
  /** Particles asked for (the scene's final count). */
  count: number;
  spacing: number;
  pbf: Pbf;
  whitewater: ReturnType<typeof createWhitewater>;
  /** Nominal liquid depth if the liquid lay flat: where to aim the pointer. */
  level(): number;
}

export const effect = defineEffect({
  meta,
  params: {
    scene: {
      type: 'select',
      default: SCENES[0],
      options: SCENES,
      label: 'Scene (restarts)',
    },
    restart: {
      type: 'boolean',
      default: false,
      label: 'Restart (flip to restart the scene)',
    },
    particles: {
      type: 'number',
      default: 10000,
      min: 5000,
      max: 500000,
      step: 1000,
      label: 'Particles (restarts)',
    },
    view: {
      type: 'select',
      default: VIEWS[0],
      options: VIEWS,
      label: 'View (liquid surface, plain spheres, speed, foam)',
    },
    iterations: {
      type: 'number',
      default: 2,
      min: 1,
      max: 8,
      step: 1,
      label: 'Solver iterations per sub-step',
    },
    substeps: {
      type: 'number',
      default: 3,
      min: 1,
      max: 6,
      step: 1,
      label: 'Sub-steps per 1/60 s (each rebuilds the neighbour grid)',
    },
    viscosity: {
      type: 'number',
      default: 0.01,
      min: 0,
      max: 0.4,
      step: 0.005,
      label: 'Viscosity (XSPH)',
    },
    vorticity: {
      type: 'number',
      default: 0,
      min: 0,
      max: 0.5,
      step: 0.01,
      label: 'Vorticity confinement',
    },
    artificialPressure: {
      type: 'number',
      default: 0.06,
      min: 0,
      max: 0.3,
      step: 0.005,
      label: 'Artificial pressure (anti-clumping)',
    },
    gravity: {
      type: 'number',
      default: 9.81,
      min: 0,
      max: 25,
      step: 0.1,
      label: 'Gravity (m/s^2)',
    },
    waveHeight: {
      type: 'number',
      default: 0.1,
      min: 0,
      max: 0.25,
      step: 0.005,
      label: 'Wave tank: paddle stroke (m)',
    },
    waveFrequency: {
      type: 'number',
      default: 0.9,
      min: 0.2,
      max: 2,
      step: 0.01,
      label: 'Wave tank: paddle frequency (Hz)',
    },
    water: { type: 'select', default: 'clear', options: WATER, label: 'Water type' },
    tint: {
      type: 'color',
      default: [0.25, 0.6, 0.85],
      label: 'Custom colour (after one unit of thickness)',
    },
    density: {
      type: 'number',
      default: 12,
      min: 0.25,
      max: 40,
      step: 0.05,
      label: 'Density (optical thickness multiplier)',
    },
    refraction: {
      type: 'number',
      default: 1,
      min: 0,
      max: 3,
      step: 0.01,
      label: 'Refraction strength',
    },
    foam: { type: 'number', default: 0.5, min: 0, max: 1, step: 0.01, label: 'Foam (particle w)' },
    renderScale: {
      type: 'select',
      default: '1/2',
      options: ['1', '1/2', '1/4'],
      label: 'Surface resolution (scale)',
    },
    filterRadius: {
      type: 'number',
      default: 2.5,
      min: 0.5,
      max: 5,
      step: 0.05,
      label: 'Surface smoothing radius (particle radii)',
    },
    filterIterations: {
      type: 'number',
      default: 3,
      min: 1,
      max: 5,
      step: 1,
      label: 'Surface smoothing iterations',
    },
    thicknessStride: {
      type: 'number',
      default: 1,
      min: 1,
      max: 6,
      step: 1,
      label: 'Thickness: draw every n-th particle',
    },
    exposure: { type: 'number', default: 1, min: 0.3, max: 2.5, step: 0.01 },
    speed: { type: 'number', default: 1, min: 0, max: 2, step: 0.01, label: 'Simulation speed' },
    ambient: {
      type: 'boolean',
      default: true,
      label: 'Ambient splashes when untouched',
    },
    interaction: {
      type: 'select',
      default: 'stir',
      options: ['stir', 'container'],
      label: 'Drag action',
    },
    shake: { type: 'number', default: 0, min: 0, max: 1, step: 0.01, label: 'Shake container' },
    deviceTilt: { type: 'boolean', default: false, label: 'Device tilt (enable, then tap tank)' },
    whitewater: { type: 'boolean', default: true, label: 'Secondary spray, foam and bubbles' },
    autoRotate: { type: 'boolean', default: true, label: 'Idle camera rotation' },
  },
  presets: {
    'shake and spray': {
      scene: SCENES[2],
      interaction: 'container',
      shake: 0.6,
      whitewater: true,
      autoRotate: false,
    },
    'calm vessel': {
      scene: SCENES[2],
      interaction: 'container',
      shake: 0,
      ambient: false,
      waveHeight: 0,
      whitewater: false,
      autoRotate: false,
    },
    'dam break': { scene: SCENES[0], particles: 10000 },
    pour: { scene: SCENES[1], particles: 10000 },
    'wave tank': { scene: SCENES[2], particles: 10000 },
    'deep blue': { water: 'ocean', density: 16, refraction: 0.8, foam: 0.8 },
    'bag of marbles': { view: VIEWS[1] },
    'speed view': { view: VIEWS[2] },
    'foam view': { view: VIEWS[3] },
    'more particles': { particles: 100000, filterRadius: 2.5 },
    'tier B cut': {
      particles: 15000,
      substeps: 2,
      iterations: 2,
      renderScale: '1/4',
      filterIterations: 1,
      filterRadius: 1.5,
      thicknessStride: 2,
    },
  },

  async create({ device, format, canvas, params, profiler }) {
    const renderer = createForwardRenderer(device, {
      colorFormat: COLOR_FORMAT,
      depthFormat: DEPTH_FORMAT,
    });
    const meshes = {
      box: createMesh(device, boxGeometry(), 'box'),
      cylinder: createMesh(device, cylinderGeometry({ radialSegments: 28 }), 'cylinder'),
    };
    const view = renderer.createView('main');
    const sceneColor = createRenderTarget(device, {
      label: 'pbf-3d scene colour',
      format: COLOR_FORMAT,
      width: 1,
      height: 1,
    });
    const sceneDepth = createDepthTarget(device, {
      label: 'pbf-3d scene depth',
      width: 1,
      height: 1,
      format: DEPTH_FORMAT,
    });
    const liquid = createRenderTarget(device, {
      label: 'pbf-3d liquid',
      format: COLOR_FORMAT,
      width: 1,
      height: 1,
    });

    // The sky and the floor are drawn inside the forward pass, before the objects.
    const sceneModule = createShader(device, {
      label: 'pbf-3d/scene.wgsl',
      code: sceneShader.code,
      chunks: SCENE_STRUCT_CHUNKS,
    });
    const layout = device.createPipelineLayout({ bindGroupLayouts: [renderer.viewLayout] });
    const skyPipeline = device.createRenderPipeline({
      label: 'pbf-3d sky',
      layout,
      vertex: { module: sceneModule, entryPoint: 'fullscreen_vs' },
      fragment: { module: sceneModule, entryPoint: 'fs_sky', targets: [{ format: COLOR_FORMAT }] },
      depthStencil: { format: DEPTH_FORMAT, depthWriteEnabled: false, depthCompare: 'always' },
    });
    const floorPipeline = device.createRenderPipeline({
      label: 'pbf-3d floor',
      layout,
      vertex: { module: sceneModule, entryPoint: 'vs_floor' },
      fragment: {
        module: sceneModule,
        entryPoint: 'fs_floor',
        targets: [{ format: COLOR_FORMAT }],
      },
      depthStencil: { format: DEPTH_FORMAT, depthWriteEnabled: true, depthCompare: 'less' },
    });

    const output = createOutputPass(device, { format });
    const fluid: FluidSurface = createFluidSurface({
      device,
      format: COLOR_FORMAT,
      width: 1,
      height: 1,
    });
    // The 'spheres', 'speed' and 'foam' views: the framework's billboard renderer, opaque sphere impostors.
    const billboards = createBillboardRenderer({
      device,
      format,
      depthFormat: DEPTH_FORMAT,
      profiler,
      label: 'pbf-3d',
      style: { mode: 'opaque', softness: 0, shading: 1, minPixels: 0.8, scalarRange: [0, 1] },
    });
    const secondaryRenderer = createBillboardRenderer({
      device,
      format,
      depthFormat: DEPTH_FORMAT,
      profiler,
      label: 'whitewater',
      style: {
        mode: 'alpha',
        intensity: 0.55,
        softness: 0.6,
        minPixels: 0.6,
        scalarRange: [1, 3],
        palette: [
          [0.85, 0.94, 1],
          [1, 1, 1],
          [0.35, 0.65, 0.85],
        ],
      },
    });
    const container = createContainerController(canvas);
    const speedPass = createComputePass(device, {
      label: 'pbf-3d/speed.wgsl',
      code: speedShader.code,
      profiler,
      timestamp: 'speed',
    });
    let speed: GPUBuffer | undefined;
    let speedCapacity = 0;

    // --- Interaction -----------------------------------------------------------------------------------------
    const pointers = trackPointers(canvas, { capture: true });
    const orbit = createOrbitState({
      yaw: 0.55,
      pitch: 0.34,
      distance: 2.1,
      target: [0, 0.25, 0],
    });
    const orbitOptions = {
      ...DEFAULT_ORBIT_OPTIONS,
      minPitch: 0.04,
      maxPitch: 1.4,
      autoSpeed: 0.1,
    };
    const camera = createCameraMatrices();
    type Gesture = {
      id: number;
      mode: 'push' | 'orbit';
      travel: number;
      start: number;
      last?: V3;
    };
    let gesture: Gesture | undefined;
    let clock = 0;
    let lastTouch = -10;
    let nextAmbient = 4;
    let ambientSeed = 7;

    // --- The world -------------------------------------------------------------------------------------------
    let world: World | undefined;
    let appliedRestart = params.restart;
    let pendingCount = { value: params.particles, since: 0 };

    function makeWorld(scene: SceneName, count: number): World {
      const spec = SPECS[scene];
      const spacing = Math.cbrt(spec.volume / count);
      const pbf = createPbf({
        device,
        dimension: 3,
        capacity: Math.ceil(count * 1.2) + 512,
        spacing,
        bounds: { min: TANK.min, max: TANK.max },
        profiler,
        timedPasses: 18,
        label: 'pbf',
      });
      pbf.reset({
        blocks: spec.blocks(),
        colliders: spec.colliders,
        emitter: spec.emitter ?? null,
      });
      return {
        scene,
        count,
        spacing,
        pbf,
        whitewater: createWhitewater({
          device,
          capacity: 4096,
          radius: spacing * 0.2,
          searchRadius: spacing * 2,
          bounds: TANK,
          profiler,
        }),
        level: () => Math.max(0.05, Math.min(0.7, (pbf.count * spacing ** 3) / TANK_AREA)),
      };
    }

    function destroyWorld(w: World | undefined): void {
      w?.pbf.destroy();
      w?.whitewater.destroy();
      container.reset();
    }

    function ensureWorld(): World {
      const scene = indexOf(SCENES, params.scene);
      const asked = Math.round(clamp(params.particles, 1000, 600000));
      // A slider drag produces a value every frame; rebuild once it has been still for a moment.
      if (asked !== pendingCount.value) pendingCount = { value: asked, since: clock };
      const settled = clock - pendingCount.since > 0.25;
      const restartFlipped = params.restart !== appliedRestart;
      if (
        !world ||
        world.scene !== scene ||
        (settled && world.count !== pendingCount.value) ||
        restartFlipped
      ) {
        appliedRestart = params.restart;
        const count = world && !settled ? world.count : pendingCount.value;
        destroyWorld(world);
        world = makeWorld(scene, count);
        nextAmbient = clock + 4;
        gesture = undefined;
      }
      return world;
    }

    function applyParams(w: World): void {
      const p = w.pbf.params;
      p.iterations = Math.round(params.iterations);
      p.substeps = Math.round(params.substeps);
      p.viscosity = params.viscosity;
      p.vorticity = params.vorticity;
      p.artificialPressure = params.artificialPressure;
      p.gravity = [0, -params.gravity, 0];
      // Spray starts to read as foam at splash speeds of this tank (a metre or two a second), not at a walking pace.
      p.foamSpeed = 2.4;
    }

    // --- Pointer -------------------------------------------------------------------------------------------
    const ndcRay = (u: number, v: number): Ray => {
      const near = unprojectPoint(camera.invViewProj, u, v, 0);
      const far = unprojectPoint(camera.invViewProj, u, v, 1);
      return {
        origin: [near[0]!, near[1]!, near[2]!],
        dir: normalise([far[0]! - near[0]!, far[1]! - near[1]!, far[2]! - near[2]!]),
      };
    };

    /** The box the liquid can be in, for deciding whether a press is on the water. */
    const liquidBox = (w: World): [V3, V3] => [
      [TANK.min[0] - 0.02, 0, TANK.min[2] - 0.02],
      [TANK.max[0] + 0.02, Math.max(0.2, w.level() + 0.12), TANK.max[2] + 0.02],
    ];

    function dropBlob(w: World, ray: Ray): void {
      // A cube of liquid released above the point of the water's surface under the pointer.
      const level = w.level();
      const point = rayPlane(ray, [0, level, 0], [0, 1, 0]);
      if (!point) return;
      const half = 3 * w.spacing;
      const margin = half + 0.03;
      if (
        point[0] < TANK.min[0] - 0.05 ||
        point[0] > TANK.max[0] + 0.05 ||
        point[2] < TANK.min[2] - 0.05 ||
        point[2] > TANK.max[2] + 0.05
      ) {
        return;
      }
      const x = clamp(point[0], TANK.min[0] + margin, TANK.max[0] - margin);
      const z = clamp(point[2], TANK.min[2] + margin, TANK.max[2] - margin);
      const y = Math.min(0.72, level + 0.25);
      // Not inside a pillar.
      for (const c of SPECS[w.scene].colliders) {
        if (
          c.kind === 'box' &&
          x > c.min[0] - margin &&
          x < c.max[0] + margin &&
          z > c.min[2] - margin &&
          z < c.max[2] + margin
        ) {
          return;
        }
      }
      w.pbf.addBlock({
        min: [x - half, y - half, z - half],
        max: [x + half, y + half, z + half],
        velocity: [0, -1, 0],
        jitter: 0.1,
      });
    }

    function handlePointers(
      w: World,
      delta: number,
    ): { dragging: boolean; dx: number; dy: number } {
      const frame: TrackedPointer[] = pointers.poll();
      if (params.interaction === 'container') {
        gesture = undefined;
        const down = frame.filter((p) => p.down);
        return {
          dragging: down.length > 0,
          dx: down.reduce((n, p) => n + p.dx, 0),
          dy: down.reduce((n, p) => n + p.dy, 0),
        };
      }
      let dx = 0;
      let dy = 0;
      let dragging = false;
      const forward = normalise([
        orbit.target[0] - camera.position[0]!,
        orbit.target[1] - camera.position[1]!,
        orbit.target[2] - camera.position[2]!,
      ]);
      const [boxMin, boxMax] = liquidBox(w);
      const center: V3 = [0, w.level() * 0.5, 0];

      for (const pointer of frame) {
        if (!gesture && pointer.justPressed) {
          const hit = rayBox(ndcRay(pointer.x, pointer.y), boxMin, boxMax);
          gesture = {
            id: pointer.id,
            mode: hit ? 'push' : 'orbit',
            travel: 0,
            start: clock,
          };
        }
        if (!gesture || pointer.id !== gesture.id) continue;
        lastTouch = clock;

        if (pointer.down) {
          gesture.travel += Math.abs(pointer.dx) + Math.abs(pointer.dy);
          if (gesture.mode === 'orbit') {
            dx += pointer.dx;
            dy += pointer.dy;
            dragging = true;
          } else if (gesture.travel > 0.004) {
            // Push: the point where the pointer's ray meets the plane through the liquid facing the camera, and the
            // velocity it moved at since the last frame, handed to the water inside a radius.
            const ray = ndcRay(pointer.x, pointer.y);
            const point = rayPlane(ray, center, forward);
            if (
              point &&
              point[0] > TANK.min[0] - 0.05 &&
              point[0] < TANK.max[0] + 0.05 &&
              point[2] > TANK.min[2] - 0.05 &&
              point[2] < TANK.max[2] + 0.05
            ) {
              if (gesture.last && delta > 0) {
                const inv = 1 / Math.max(delta, 1 / 120);
                const v: V3 = [
                  (point[0] - gesture.last[0]) * inv,
                  (point[1] - gesture.last[1]) * inv,
                  (point[2] - gesture.last[2]) * inv,
                ];
                const length = Math.hypot(v[0], v[1], v[2]);
                const limit = 5;
                const k = length > limit ? limit / length : 1;
                w.pbf.impulse({
                  position: point,
                  radius: Math.max(0.12, 8 * w.spacing),
                  velocity: [v[0] * k, v[1] * k, v[2] * k],
                  strength: 0.6,
                });
              }
              gesture.last = point;
            } else {
              gesture.last = undefined;
            }
          }
        }

        if (pointer.justReleased || !pointer.down) {
          const tap = !pointer.cancelled && gesture.travel < 0.012 && clock - gesture.start < 0.5;
          if (tap) dropBlob(w, ndcRay(pointer.x, pointer.y));
          gesture = undefined;
        }
      }
      if (gesture && !frame.some((p) => p.id === gesture!.id)) gesture = undefined;
      return { dragging, dx, dy };
    }

    /** Something periodic, so the tank is alive untouched: a mass-free pulse that lifts the surface somewhere. */
    function ambient(w: World): void {
      if (!params.ambient || clock < nextAmbient || clock - lastTouch < 4) return;
      nextAmbient = clock + 3.5 + (ambientSeed % 3);
      ambientSeed = (ambientSeed * 1103515245 + 12345) >>> 0;
      const r1 = ((ambientSeed >>> 8) % 1000) / 1000;
      ambientSeed = (ambientSeed * 1103515245 + 12345) >>> 0;
      const r2 = ((ambientSeed >>> 8) % 1000) / 1000;
      const level = w.level();
      w.pbf.impulse({
        position: [
          TANK.min[0] + 0.15 + r1 * (TANK.max[0] - TANK.min[0] - 0.3),
          Math.max(0.03, level * 0.6),
          TANK.min[2] + 0.1 + r2 * (TANK.max[2] - TANK.min[2] - 0.2),
        ],
        radius: 0.12,
        outward: 1.6,
        strength: 0,
      });
    }

    // --- The scene objects ------------------------------------------------------------------------------------
    function box(out: SceneObject[], min: V3, max: V3, color: V3): void {
      const model = mat4.translation([
        (min[0] + max[0]) / 2,
        (min[1] + max[1]) / 2,
        (min[2] + max[2]) / 2,
      ]);
      mat4.scale(model, [max[0] - min[0], max[1] - min[1], max[2] - min[2]], model);
      out.push({ mesh: meshes.box, model, color });
    }

    const metal: V3 = [0.55, 0.58, 0.62];
    const staticObjects: SceneObject[] = [];
    {
      // The frame of the tank: twelve thin bars on the edges of the inside.
      const bar = 0.025;
      const [x0, y0, z0] = TANK.min;
      const [x1, y1, z1] = TANK.max;
      for (const y of [y0, y1]) {
        for (const z of [z0, z1]) {
          box(staticObjects, [x0 - bar, y - bar, z - bar], [x1 + bar, y + bar, z + bar], metal);
        }
        for (const x of [x0, x1]) {
          box(staticObjects, [x - bar, y - bar, z0 - bar], [x + bar, y + bar, z1 + bar], metal);
        }
      }
      for (const x of [x0, x1]) {
        for (const z of [z0, z1]) {
          box(staticObjects, [x - bar, y0 - bar, z - bar], [x + bar, y1 + bar, z + bar], metal);
        }
      }
      // A striped board behind the tank: its straight lines show what the water does to the view.
      const stripe = 0.12;
      for (let i = 0; i < 16; i++) {
        const x = -0.96 + i * stripe;
        box(
          staticObjects,
          [x, 0, z0 - 0.09],
          [x + stripe, 0.85, z0 - 0.07],
          i % 2 === 0 ? [0.9, 0.38, 0.06] : [0.04, 0.14, 0.32],
        );
      }
    }

    function sceneObjects(w: World): SceneObject[] {
      const objects: SceneObject[] = [...staticObjects];
      for (const collider of SPECS[w.scene].colliders) {
        if (collider.kind === 'box') {
          box(objects, collider.min as V3, collider.max as V3, [0.78, 0.2, 0.12]);
        }
      }
      if (w.scene === 'wave tank') {
        // The paddle: the moving wall, drawn just behind the water's edge.
        const x = w.pbf.bounds.min[0]!;
        box(objects, [x - 0.05, 0, TANK.min[2]], [x, 0.78, TANK.max[2]], [0.36, 0.42, 0.52]);
      }
      if (w.scene === 'pour') {
        const model = mat4.translation([NOZZLE.position[0], 0.9, NOZZLE.position[2]]);
        mat4.scale(model, [NOZZLE.radius * 2.6, 0.32, NOZZLE.radius * 2.6], model);
        objects.push({ mesh: meshes.cylinder, model, color: [0.5, 0.52, 0.56] });
      }
      return objects;
    }

    // --- Frame -----------------------------------------------------------------------------------------------
    let width = 1;
    let height = 1;
    let frameCount = 0;

    function ensureSpeed(w: World): GPUBuffer {
      const capacity = w.pbf.capacity;
      if (!speed || speedCapacity !== capacity) {
        speed?.destroy();
        speedCapacity = capacity;
        speed = device.createBuffer({
          label: 'pbf-3d speed',
          size: capacity * 4,
          usage: GPUBufferUsage.STORAGE | GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
        });
      }
      return speed;
    }

    return {
      resize(w, h) {
        width = Math.max(1, Math.round(w));
        height = Math.max(1, Math.round(h));
        sceneColor.resize(width, height);
        sceneDepth.resize(width, height);
        liquid.resize(width, height);
        fluid.resize(width, height);
        billboards.resize(width, height);
        secondaryRenderer.resize(width, height);
      },

      frame({ delta }, encoder, target) {
        const dt = Math.min(delta, 0.1);
        clock += dt;
        frameCount++;
        const w = ensureWorld();
        const { pbf } = w;
        applyParams(w);

        // 1. Input and the camera.
        const input = handlePointers(w, dt);
        ambient(w);
        container.setTiltEnabled(params.deviceTilt);
        pbf.params.gravity = container.gravity(dt, params.gravity, {
          ...input,
          dragging: params.interaction === 'container' && input.dragging,
          shake: params.shake,
        });
        stepOrbit(
          orbit,
          {
            yaw: input.dragging && params.interaction !== 'container' ? -input.dx * 3 : 0,
            pitch: input.dragging && params.interaction !== 'container' ? input.dy * 2 : 0,
            dragging: input.dragging,
          },
          dt,
          params.autoRotate ? orbitOptions : { ...orbitOptions, autoSpeed: 0 },
        );
        const aspect = width / height;
        computeCamera(
          {
            position: orbitEye(orbit),
            target: orbit.target,
            fovY: fitFovY(FOV_Y, aspect, MIN_HORIZONTAL_FOV),
            aspect,
            near: 0.05,
            far: 60,
          },
          camera,
        );

        // 2. The solver. The wave-maker is the left wall of the tank moving in and out.
        if (w.scene === 'wave tank') {
          const phase = 2 * Math.PI * params.waveFrequency * pbf.time;
          const stroke = params.waveHeight * 0.5 * (1 - Math.cos(phase));
          pbf.setBounds([TANK.min[0] + stroke, TANK.min[1], TANK.min[2]], TANK.max);
        }
        pbf.update(encoder, dt * params.speed);
        if (params.whitewater)
          w.whitewater.update(encoder, pbf.particles, dt * params.speed, pbf.params.gravity);

        // 3. The scene the liquid is drawn into.
        view.update(camera, width, height, LIGHT);
        const batch = renderer.upload(sceneObjects(w));
        renderer.draw(encoder, batch, {
          view,
          color: sceneColor.view,
          depth: sceneDepth.view,
          background: (pass) => {
            pass.setPipeline(skyPipeline);
            pass.setBindGroup(0, view.bindGroup);
            pass.draw(3);
            pass.setPipeline(floorPipeline);
            pass.draw(6);
          },
          label: 'pbf-3d scene pass',
          timestampWrites: profiler.pass('scene'),
        });

        // 4. The liquid, or the particles themselves.
        const mode = indexOf(VIEWS, params.view) as View;
        const particles = pbf.particles;
        const tonemap = { exposure: params.exposure, tonemap: 'aces' as const, frame: frameCount };
        if (mode === 'liquid') {
          fluid.setQuality({
            scale: SCALES[params.renderScale] ?? 0.5,
            thicknessScale: 0.25,
            thicknessStride: Math.round(params.thicknessStride),
            filter: FILTERS[1] as FluidFilter,
            filterRadius: params.filterRadius,
            iterations: Math.round(params.filterIterations),
          });
          const water = indexOf(WATER, params.water);
          const custom = water === 'custom';
          const tint = params.tint.map((c) => Math.min(1, Math.max(1e-3, c)));
          const radius = particles.radius;
          const material: FluidSurfaceParams = {
            waterType: custom ? -1 : WATER.indexOf(water),
            absorption: [-Math.log(tint[0]!), -Math.log(tint[1]!), -Math.log(tint[2]!)],
            scatter: [tint[0]! * 0.6, tint[1]! * 0.6, tint[2]! * 0.6],
            density: params.density,
            refraction: params.refraction,
            sunDirection: SUN_DIRECTION,
            // The volume a particle stands for over the volume of the sphere that is drawn for it.
            thicknessGain: (particles.restVolume ?? 1) / ((4 / 3) * Math.PI * radius ** 3),
            foamAmount: params.foam,
            maxRefractionThickness: 0.45,
          };
          fluid.render(encoder, {
            particles,
            camera,
            sceneColor: sceneColor.view,
            sceneDepth: sceneDepth.view,
            target: liquid.view,
            params: material,
            profiler,
          });
          output.encode(encoder, target, liquid.view, undefined, tonemap, profiler.pass('output'));
        } else {
          // The scene, tone-mapped, then the particles as sphere impostors over it, depth-tested against it.
          output.encode(
            encoder,
            target,
            sceneColor.view,
            undefined,
            tonemap,
            profiler.pass('output'),
          );
          let scalar: GPUBuffer | undefined;
          const style = billboards.style;
          style.minPixels = 0.8;
          if (mode === 'speed') {
            scalar = ensureSpeed(w);
            speedPass.runCount(encoder, [pbf.system.velocities!, scalar], particles.count);
            style.palette = SPEED_PALETTE;
            style.scalarRange = [0, 3];
          } else if (mode === 'foam') {
            style.palette = FOAM_PALETTE;
            style.scalarRange = [0, 1];
          } else {
            style.palette = [
              [0.03, 0.22, 0.6],
              [0.05, 0.45, 0.85],
              [0.35, 0.75, 1],
              [1, 1, 1],
            ];
            style.scalarRange = [0, 1.5];
          }
          billboards.draw(encoder, particles, camera, target, {
            width,
            height,
            depth: sceneDepth.view,
            clearDepth: false,
            clear: null,
            scalar,
          });
        }
        if (params.whitewater)
          secondaryRenderer.draw(encoder, w.whitewater.particles, camera, target, {
            width,
            height,
            depth: sceneDepth.view,
            clearDepth: false,
            clear: null,
            timestamp: 'whitewater draw',
          });
      },

      destroy() {
        destroyWorld(world);
        world = undefined;
        pointers.destroy();
        container.destroy();
        secondaryRenderer.destroy();
        fluid.destroy();
        billboards.destroy();
        speedPass.destroy();
        speed?.destroy();
        output.destroy();
        sceneColor.destroy();
        sceneDepth.destroy();
        liquid.destroy();
        view.destroy();
        renderer.destroy();
        for (const mesh of Object.values(meshes) as Mesh[]) mesh.destroy();
      },
    };
  },
});

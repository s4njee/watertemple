import { mat4 } from 'wgpu-matrix';
import {
  createFluidSurface,
  FILTERS,
  type FluidFilter,
  type FluidSurface,
  type FluidSurfaceParams,
} from '../../gpu/fluid-surface/index.ts';
import { createBillboardRenderer } from '../../gpu/particles/billboards.ts';
import { createOutputPass } from '../../gpu/post/index.ts';
import { trackPointers } from '../../gpu/pointer.ts';
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
import {
  boxGeometry,
  sphereGeometry,
  VERTEX_FLOATS,
  type Geometry,
} from '../../gpu/scene/geometry.ts';
import { createMesh, type Mesh } from '../../gpu/scene/mesh.ts';
import {
  createOrbitState,
  DEFAULT_ORBIT_OPTIONS,
  orbitEye,
  stepOrbit,
} from '../../gpu/scene/orbit.ts';
import { createShader } from '../../gpu/shader.ts';
import {
  blockLattice,
  createMpm,
  MPM_EDGE_CELLS,
  type MpmCollider,
  type MpmSolver,
  type Vec3,
} from '../../gpu/sim/mpm.ts';
import { createRenderTarget } from '../../gpu/targets.ts';
import { createUniformBuffer, defineUniformStruct } from '../../gpu/uniforms.ts';
import { defineEffect } from '../types.ts';
import { meta } from './meta.ts';
import sceneShader from './scene.wgsl?raw';

const COLOR_FORMAT: GPUTextureFormat = 'rgba16float';
const DEPTH_FORMAT = 'depth32float';

const SCENES = ['wave tank', 'dam break', 'pour'] as const;
type SceneName = (typeof SCENES)[number];
const VIEWS = ['liquid', 'spheres', 'speed', 'density'] as const;
type View = (typeof VIEWS)[number];
const GRIDS = ['48', '64', '96', '128'] as const;
/** Water types of absorption.wgsl, by the name the control shows. `custom` uses the colour. */
const WATER = ['clear', 'coastal', 'murky', 'tea', 'milk', 'pool', 'ocean', 'custom'] as const;
const SCALES: Record<string, number> = { '1': 1, '1/2': 0.5, '1/4': 0.25 };
const THICKNESS_SCALES: Record<string, number> = { '1/2': 0.5, '1/4': 0.25, '1/8': 0.125 };

/** The tank's inside, in metres, whatever the grid: a finer grid has smaller cells, not a smaller tank. */
const TANK_LENGTH = 3.2;
const TANK_WIDTH = 1.4;
const TANK_HEIGHT = 1.4;
const SUN_DIRECTION: Vec3 = [0.5, 0.7, 0.35];
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
const MIN_HORIZONTAL_FOV = (60 * Math.PI) / 180;
const DEG = Math.PI / 180;

/** The speed limit of the solver's nodes and particles in this effect, m/s. */
const MAX_SPEED = 7;
/** The step is this fraction of a cell per (sound speed + speed limit); the measured stability limit is about 3. */
const COURANT = 1.25;
/** Frames the simulation is given per second of display time: sub-steps are counted against this. */
const FRAME_RATE = 60;

const RoomUniform = defineUniformStruct('MpmScene', {
  tank_min: 'vec2f',
  tank_max: 'vec2f',
  floor_y: 'f32',
  tile: 'f32',
});

const clamp = (x: number, lo: number, hi: number) => Math.min(hi, Math.max(lo, x));
const smoothstep = (x: number) => {
  const t = clamp(x, 0, 1);
  return t * t * (3 - 2 * t);
};

// --- Geometry and picking --------------------------------------------------------------------------------------------

/** A right-triangle prism: the toe at the origin, rising to `height` at x = `length`, `width` across z. */
function wedgeGeometry(length: number, height: number, width: number): Geometry {
  const w = width / 2;
  const slope = Math.hypot(height, length);
  const nx = -height / slope;
  const ny = length / slope;
  const vertices: number[] = [];
  const indices: number[] = [];
  const quad = (points: number[][], n: number[]) => {
    const first = vertices.length / VERTEX_FLOATS;
    for (const p of points) vertices.push(p[0]!, p[1]!, p[2]!, n[0]!, n[1]!, n[2]!);
    indices.push(first, first + 1, first + 2);
    if (points.length === 4) indices.push(first, first + 2, first + 3);
  };
  // Counter-clockwise seen from outside.
  quad(
    [
      [0, 0, -w],
      [0, 0, w],
      [length, height, w],
      [length, height, -w],
    ],
    [nx, ny, 0],
  );
  quad(
    [
      [length, 0, -w],
      [length, height, -w],
      [length, height, w],
      [length, 0, w],
    ],
    [1, 0, 0],
  );
  quad(
    [
      [0, 0, -w],
      [length, height, -w],
      [length, 0, -w],
    ],
    [0, 0, -1],
  );
  quad(
    [
      [0, 0, w],
      [length, 0, w],
      [length, height, w],
    ],
    [0, 0, 1],
  );
  return {
    vertices: new Float32Array(vertices),
    indices: new Uint16Array(indices),
    vertexCount: vertices.length / VERTEX_FLOATS,
    indexCount: indices.length,
  };
}

interface Ray {
  origin: Vec3;
  dir: Vec3;
}

function pointerRay(
  invViewProj: Parameters<typeof unprojectPoint>[0],
  position: ArrayLike<number>,
  u: number,
  v: number,
): Ray {
  const far = unprojectPoint(invViewProj, u, v, 1);
  const o: Vec3 = [position[0]!, position[1]!, position[2]!];
  const d: Vec3 = [far[0]! - o[0], far[1]! - o[1], far[2]! - o[2]];
  const length = Math.hypot(...d) || 1;
  return { origin: o, dir: [d[0] / length, d[1] / length, d[2] / length] };
}

/** Distance along the ray to the plane y = height, or null when parallel or behind. */
function rayPlaneY(ray: Ray, height: number): number | null {
  if (Math.abs(ray.dir[1]) < 0.04) return null;
  const t = (height - ray.origin[1]) / ray.dir[1];
  return t > 0 ? t : null;
}

/** Slab test: does the ray hit the box? */
function rayBox(ray: Ray, min: Vec3, max: Vec3): boolean {
  let near = 0;
  let far = Infinity;
  for (let a = 0; a < 3; a++) {
    const o = ray.origin[a]!;
    const d = ray.dir[a]!;
    if (Math.abs(d) < 1e-9) {
      if (o < min[a]! || o > max[a]!) return false;
      continue;
    }
    let t0 = (min[a]! - o) / d;
    let t1 = (max[a]! - o) / d;
    if (t0 > t1) [t0, t1] = [t1, t0];
    near = Math.max(near, t0);
    far = Math.min(far, t1);
    if (near > far) return false;
  }
  return true;
}

// --- Scenes ----------------------------------------------------------------------------------------------------------

/** What a scene needs besides its particles: solids, things to draw, where to drag, how it loops. */
interface SceneState {
  name: SceneName;
  colliders: MpmCollider[];
  /** Static objects: everything but the paddle and the pointer's sphere. */
  objects: SceneObject[];
  /** Height above the floor of the plane the pointer drags the push sphere on, and tap blobs fall from. */
  level: number;
  /** Seconds before the scene restarts itself when left alone; 0 never. */
  loop: number;
  /** The piston wall of the wave tank: x of the wall at rest. */
  pistonRest: number | undefined;
  /** The pour stops adding when the solver holds this many particles. */
  pourLimit: number;
  /** World height of the top of the tank. */
  top: number;
  /** Rest wall limits (the piston at its rest position). */
  min: Vec3;
  max: Vec3;
}

interface ScenePieces {
  box: Mesh;
  sphere: Mesh;
  wedge: Mesh;
}

const COLORS = {
  frame: [0.72, 0.75, 0.78] as Vec3,
  paddle: [0.3, 0.36, 0.44] as Vec3,
  sand: [0.8, 0.68, 0.44] as Vec3,
  pillar: [0.78, 0.2, 0.12] as Vec3,
  wall: [0.2, 0.5, 0.62] as Vec3,
  blob: [0.95, 0.75, 0.2] as Vec3,
};

function boxObject(
  mesh: Mesh,
  center: Vec3,
  size: Vec3,
  color: Vec3,
  emissive?: number,
): SceneObject {
  const model = mat4.translation(center);
  mat4.scale(model, size, model);
  return { mesh, model, color, emissive };
}

/** The twelve bars of the tank's outline. */
function frameObjects(box: Mesh, min: Vec3, max: Vec3, floor: number, top: number): SceneObject[] {
  const bar = 0.045;
  const objects: SceneObject[] = [];
  const cx = (min[0] + max[0]) / 2;
  const cz = (min[2] + max[2]) / 2;
  const sx = max[0] - min[0] + bar;
  const sz = max[2] - min[2] + bar;
  for (const y of [floor, top]) {
    for (const z of [min[2], max[2]])
      objects.push(boxObject(box, [cx, y, z], [sx, bar, bar], COLORS.frame));
    for (const x of [min[0], max[0]])
      objects.push(boxObject(box, [x, y, cz], [bar, bar, sz], COLORS.frame));
  }
  for (const x of [min[0], max[0]]) {
    for (const z of [min[2], max[2]]) {
      objects.push(
        boxObject(box, [x, (floor + top) / 2, z], [bar, top - floor, bar], COLORS.frame),
      );
    }
  }
  return objects;
}

export const effect = defineEffect({
  meta,
  params: {
    scene: {
      type: 'select',
      default: 'wave tank',
      options: SCENES,
      label: 'Scene (wave tank: a wall sends waves to a beach)',
    },
    restart: { type: 'boolean', default: false, label: 'Restart (flip to run the scene again)' },
    autoRestart: {
      type: 'boolean',
      default: true,
      label: 'Run the dam break and the pour again when they are done',
    },
    view: {
      type: 'select',
      default: 'liquid',
      options: VIEWS,
      label: 'View (liquid, or the particles: foam, speed, density)',
    },
    count: {
      type: 'number',
      default: 100000,
      min: 10000,
      max: 800000,
      step: 5000,
      label: 'Particles (the more per cell, the smoother; restarts the scene)',
    },
    grid: {
      type: 'select',
      default: '48',
      options: GRIDS,
      label: 'Grid cells along the tank (restarts the scene)',
    },
    stiffness: {
      type: 'number',
      default: 10,
      min: 3,
      max: 30,
      step: 0.5,
      label: 'Stiffness (sound speed, m/s; limited by the step)',
    },
    substeps: {
      type: 'number',
      default: 0,
      min: 0,
      max: 16,
      step: 1,
      label: 'Sub-steps per 1/60 s (0: the largest stable step; fewer reduce the stiffness)',
    },
    viscosity: {
      type: 'number',
      default: 0,
      min: 0,
      max: 0.2,
      step: 0.001,
      label: 'Viscosity (m²/s; water is 1e-6, honey is ~0.1)',
    },
    damping: {
      type: 'number',
      default: 0.5,
      min: 0,
      max: 3,
      step: 0.01,
      label: 'Damping (1/s): calms the slosh and the rest noise',
    },
    gravity: { type: 'number', default: 9.8, min: 0, max: 20, step: 0.1, label: 'Gravity (m/s²)' },
    waveMaker: {
      type: 'number',
      default: 1,
      min: 0,
      max: 2,
      step: 0.01,
      label: 'Wave maker stroke (wave tank)',
    },
    wavePeriod: {
      type: 'number',
      default: 1.9,
      min: 0.8,
      max: 5,
      step: 0.05,
      label: 'Wave maker period (s)',
    },
    sorted: { type: 'boolean', default: true, label: 'Sort particles by cell (fewer atomics)' },
    particleRadius: {
      type: 'number',
      default: 0.75,
      min: 0.4,
      max: 2,
      step: 0.01,
      label: 'Drawn particle radius (x particle spacing)',
    },
    distance: {
      type: 'number',
      default: 5.2,
      min: 3,
      max: 14,
      step: 0.1,
      label: 'Camera distance',
    },
    smoothScale: {
      type: 'select',
      default: '1/2',
      options: ['1', '1/2', '1/4'],
      label: 'Surface smoothing resolution',
    },
    thicknessScale: {
      type: 'select',
      default: '1/4',
      options: ['1/2', '1/4', '1/8'],
      label: 'Thickness resolution',
    },
    thicknessStride: {
      type: 'number',
      default: 1,
      min: 1,
      max: 8,
      step: 1,
      label: 'Thickness: draw every n-th particle',
    },
    filter: { type: 'select', default: 'narrow-range', options: FILTERS, label: 'Depth filter' },
    filterRadius: {
      type: 'number',
      default: 2.2,
      min: 0.25,
      max: 6,
      step: 0.05,
      label: 'Filter radius (particle radii)',
    },
    iterations: { type: 'number', default: 3, min: 1, max: 6, step: 1, label: 'Filter iterations' },
    water: { type: 'select', default: 'coastal', options: WATER, label: 'Water type' },
    tint: {
      type: 'color',
      default: [0.25, 0.62, 0.85],
      label: 'Custom colour (after one unit of thickness)',
    },
    murk: { type: 'number', default: 0.1, min: 0, max: 3, step: 0.01, label: 'Custom turbidity' },
    density: { type: 'number', default: 3, min: 0.25, max: 24, step: 0.05, label: 'Density' },
    refraction: { type: 'number', default: 1, min: 0, max: 3, step: 0.01, label: 'Refraction' },
    foam: { type: 'number', default: 0.75, min: 0, max: 1, step: 0.01, label: 'Foam' },
    exposure: { type: 'number', default: 1, min: 0.3, max: 2.5, step: 0.01 },
    autoRotate: { type: 'boolean', default: true, label: 'Camera drifts when left alone' },
  },
  presets: {
    'wave tank': { scene: 'wave tank', view: 'liquid', gravity: 9.8, viscosity: 0 },
    'dam break': { scene: 'dam break', view: 'liquid', gravity: 9.8, viscosity: 0 },
    pour: { scene: 'pour', view: 'liquid', gravity: 9.8, viscosity: 0 },
    'marbles (the spheres)': { view: 'spheres', particleRadius: 0.5 },
    speed: { view: 'speed' },
    density: { view: 'density' },
    honey: {
      water: 'custom',
      tint: [0.75, 0.4, 0.07],
      murk: 0.5,
      density: 1.4,
      viscosity: 0.12,
      gravity: 9.8,
      foam: 0,
    },
    'slow motion': { gravity: 3, wavePeriod: 3.2 },
    '300 000 particles': { count: 300000, grid: '64' },
    'cheap cut': {
      count: 40000,
      grid: '48',
      damping: 1,
      smoothScale: '1/4',
      thicknessScale: '1/8',
      thicknessStride: 2,
      iterations: 1,
      filterRadius: 1.4,
    },
  },

  async create({ device, format, canvas, params, profiler }) {
    const pointers = trackPointers(canvas, { capture: true });

    // --- Scene resources ---------------------------------------------------------------------------------------
    const renderer = createForwardRenderer(device, {
      colorFormat: COLOR_FORMAT,
      depthFormat: DEPTH_FORMAT,
    });
    const meshes: ScenePieces = {
      box: createMesh(device, boxGeometry(), 'mpm box'),
      sphere: createMesh(
        device,
        sphereGeometry({ radius: 1, widthSegments: 32, heightSegments: 16 }),
        'mpm sphere',
      ),
      wedge: createMesh(device, wedgeGeometry(1, 1, 1), 'mpm wedge'),
    };
    const view = renderer.createView('mpm main');
    const sceneColor = createRenderTarget(device, {
      label: 'mpm scene colour',
      format: COLOR_FORMAT,
      width: 1,
      height: 1,
    });
    const sceneDepth = createDepthTarget(device, {
      label: 'mpm scene depth',
      width: 1,
      height: 1,
      format: DEPTH_FORMAT,
    });
    const liquid = createRenderTarget(device, {
      label: 'mpm liquid',
      format: COLOR_FORMAT,
      width: 1,
      height: 1,
    });

    const roomBuffer = createUniformBuffer(device, RoomUniform, 'mpm room');
    const roomModule = createShader(device, {
      label: 'mpm-3d/scene.wgsl',
      code: sceneShader,
      chunks: { ...SCENE_STRUCT_CHUNKS, 'mpm_scene.wgsl': RoomUniform.wgsl },
    });
    const roomLayout = device.createBindGroupLayout({
      label: 'mpm room layout',
      entries: [
        {
          binding: 0,
          visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT,
          buffer: { type: 'uniform' },
        },
      ],
    });
    const roomGroup = device.createBindGroup({
      label: 'mpm room',
      layout: roomLayout,
      entries: [{ binding: 0, resource: { buffer: roomBuffer.buffer } }],
    });
    const roomPipelineLayout = device.createPipelineLayout({
      bindGroupLayouts: [renderer.viewLayout, roomLayout],
    });
    const skyPipeline = device.createRenderPipeline({
      label: 'mpm sky',
      layout: roomPipelineLayout,
      vertex: { module: roomModule, entryPoint: 'fullscreen_vs' },
      fragment: { module: roomModule, entryPoint: 'fs_sky', targets: [{ format: COLOR_FORMAT }] },
      depthStencil: { format: DEPTH_FORMAT, depthWriteEnabled: false, depthCompare: 'always' },
    });
    const floorPipeline = device.createRenderPipeline({
      label: 'mpm floor',
      layout: roomPipelineLayout,
      vertex: { module: roomModule, entryPoint: 'vs_floor' },
      fragment: { module: roomModule, entryPoint: 'fs_floor', targets: [{ format: COLOR_FORMAT }] },
      depthStencil: { format: DEPTH_FORMAT, depthWriteEnabled: true, depthCompare: 'less' },
    });

    const output = createOutputPass(device, { format });
    const fluid: FluidSurface = createFluidSurface({
      device,
      format: COLOR_FORMAT,
      width: 1,
      height: 1,
    });
    const billboards = createBillboardRenderer({
      device,
      format,
      depthFormat: DEPTH_FORMAT,
      profiler,
      style: {
        mode: 'opaque',
        minPixels: 1,
        shading: 1,
        scalarRange: [0, 1],
        palette: [
          [0.015, 0.14, 0.4],
          [0.06, 0.4, 0.7],
          [0.55, 0.85, 1],
          [1, 1, 1],
        ],
      },
    });

    // --- The solver and the scene in it -----------------------------------------------------------------------
    let solver: MpmSolver | undefined;
    let state: SceneState | undefined;
    let builtKey = '';
    let gridKey = '';
    let lastRestart = params.restart;
    let sceneClock = 0;
    let idle = 0;

    function gridShape(n: number): [number, number, number] {
      const h = TANK_LENGTH / (n - 2 * MPM_EDGE_CELLS);
      return [
        n,
        Math.ceil(TANK_HEIGHT / h) + 2 * MPM_EDGE_CELLS,
        Math.ceil(TANK_WIDTH / h) + 2 * MPM_EDGE_CELLS,
      ];
    }

    function ensureSolver(): MpmSolver {
      const n = Number(params.grid);
      const capacity = Math.ceil((params.count * 1.3 + 6000) / 1024) * 1024;
      if (solver && gridKey === `${n}` && solver.capacity >= capacity) return solver;
      if (solver && gridKey === `${n}`) {
        solver.setCapacity(capacity);
        return solver;
      }
      solver?.destroy();
      const cellSize = TANK_LENGTH / (n - 2 * MPM_EDGE_CELLS);
      solver = createMpm({
        device,
        grid: gridShape(n),
        cellSize,
        capacity,
        profiler,
        radiusScale: 0.75,
        params: { courant: COURANT, maxSpeed: MAX_SPEED },
      });
      gridKey = `${n}`;
      builtKey = '';
      return solver;
    }

    /** Fills the tank for the scene, with the particles per cell that make `count`. */
    function buildScene(name: SceneName, mpm: MpmSolver, count: number): SceneState {
      const h = mpm.cellSize;
      const edge = MPM_EDGE_CELLS * h;
      const min: Vec3 = [edge, edge, edge];
      const max: Vec3 = [mpm.size[0] - edge, mpm.size[1] - edge, mpm.size[2] - edge];
      const floor = min[1];
      const top = max[1];
      const length = max[0] - min[0];
      const depthZ = max[2] - min[2];
      const zc = (min[2] + max[2]) / 2;
      const colliders: MpmCollider[] = [];
      const objects: SceneObject[] = [...frameObjects(meshes.box, min, max, floor, top)];
      let volume: number;
      let inside: (x: number, y: number, z: number) => boolean = () => true;
      const boxMin: Vec3 = min;
      let boxMax: Vec3;
      let level: number;
      let loop = 0;
      let pistonRest: number | undefined;
      let pourLimit = 0;
      let pour: Parameters<MpmSolver['setPour']>[0] = null;

      if (name === 'wave tank') {
        const depth = 0.55;
        const angle = 24 * DEG;
        const toe = min[0] + 0.56 * length;
        const rise = Math.tan(angle) * (max[0] - toe);
        // A beach: a plane collider and the wedge that draws it.
        colliders.push({
          shape: 'plane',
          center: [toe, floor, zc],
          normal: [-Math.sin(angle), Math.cos(angle), 0],
          friction: 0.3,
        });
        const wedgeModel = mat4.translation([toe, floor, zc]);
        mat4.scale(wedgeModel, [max[0] - toe, rise, depthZ], wedgeModel);
        objects.push({ mesh: meshes.wedge, model: wedgeModel, color: COLORS.sand });
        boxMax = [max[0], floor + depth, max[2]];
        inside = (x, y) => y - floor >= (x - toe) * Math.tan(angle);
        const wedgeVolume = (0.5 * depth * depth) / Math.tan(angle);
        volume = (length * depth - Math.min(wedgeVolume, depth * (max[0] - toe))) * depthZ;
        level = depth;
        pistonRest = min[0];
        // The paddle is dynamic; see the frame.
      } else if (name === 'dam break') {
        const depth = clamp(0.7 * (top - floor), 0.8, 1.1);
        boxMax = [min[0] + 0.28 * length, floor + depth, max[2]];
        volume = (boxMax[0] - min[0]) * depth * depthZ;
        const pillars: [number, number, number, number, number][] = [
          [0.52, 0.42, -0.22, 0.17, 0.42],
          [0.66, 0.5, 0.2, 0.17, 0.5],
          [0.84, 0.13, 0, 0.1, depthZ * 0.34],
        ];
        for (const [fx, hy, fz, hx, hz] of pillars) {
          const half: Vec3 = [hx, hy, fx === 0.84 ? hz : hx];
          const center: Vec3 = [min[0] + fx * length, floor + hy, zc + fz * depthZ];
          colliders.push({ shape: 'box', center, halfExtent: half, friction: 0.2 });
          objects.push(
            boxObject(
              meshes.box,
              center,
              [half[0] * 2, half[1] * 2, half[2] * 2],
              fx === 0.84 ? COLORS.wall : COLORS.pillar,
            ),
          );
        }
        level = 0.35;
        loop = 22;
      } else {
        const poolDepth = 0.3;
        boxMax = [max[0], floor + poolDepth, max[2]];
        const nozzleRadius = 0.085;
        const speed = 3.2;
        const seconds = 8;
        const streamVolume = Math.PI * nozzleRadius * nozzleRadius * speed * seconds;
        volume = length * poolDepth * depthZ + streamVolume;
        pour = {
          position: [min[0] + 0.3 * length, Math.min(floor + 1.4, top - 0.3), zc],
          direction: [0, -1, 0],
          speed,
          radius: nozzleRadius,
          active: true,
        };
        pourLimit = count;
        level = poolDepth;
        loop = 20;
      }

      const ppc = clamp((count * h * h * h) / Math.max(volume, 1e-3), 2, 48);
      mpm.params.particlesPerCell = ppc;
      mpm.reset({ bounds: { min, max }, colliders, pour });
      const rows = blockLattice(
        { shape: 'box', min: boxMin, max: boxMax, jitter: 0.15 },
        mpm.spacing,
      );
      const kept = new Float32Array(rows.length);
      let n = 0;
      for (let i = 0; i < rows.length; i += 4) {
        if (!inside(rows[i]!, rows[i + 1]!, rows[i + 2]!)) continue;
        kept.set(rows.subarray(i, i + 4), n * 4);
        n++;
      }
      mpm.addParticles(kept.subarray(0, n * 4));
      mpm.system.radius = mpm.spacing * params.particleRadius;
      return {
        name,
        colliders,
        objects,
        level,
        loop,
        pistonRest,
        pourLimit,
        top,
        min,
        max,
      };
    }

    // --- Interaction -----------------------------------------------------------------------------------------
    const orbit = createOrbitState({
      yaw: 0.62,
      pitch: 0.4,
      distance: 5.2,
      target: [1.7, 0.4, 0.7],
    });
    const orbitOptions = {
      ...DEFAULT_ORBIT_OPTIONS,
      minPitch: 0.22,
      maxPitch: 1.35,
      autoSpeed: 0.1,
    };
    const camera = createCameraMatrices();
    interface Gesture {
      id: number;
      mode: 'orbit' | 'water';
      travel: number;
      start: number;
      lastX: number;
      lastY: number;
    }
    let gesture: Gesture | undefined;
    let pusher: { position: Vec3; velocity: Vec3; active: boolean } | undefined;
    let clock = 0;
    const PUSH_RADIUS = 0.2;

    function pointOnLevel(u: number, v: number): Vec3 | undefined {
      if (!solver || !state) return undefined;
      const ray = pointerRay(camera.invViewProj, camera.position, u, v);
      const y = solver.bounds.min[1] + state.level;
      const t = rayPlaneY(ray, y);
      if (t === null) return undefined;
      const b = solver.bounds;
      const margin = 0.12;
      return [
        clamp(ray.origin[0] + ray.dir[0] * t, b.min[0] + margin, b.max[0] - margin),
        y,
        clamp(ray.origin[2] + ray.dir[2] * t, b.min[2] + margin, b.max[2] - margin),
      ];
    }

    function handlePointers(delta: number): void {
      const frame = pointers.poll();
      let dx = 0;
      let dy = 0;
      let dragging = false;
      for (const pointer of frame) {
        if (!gesture && pointer.justPressed && solver && state) {
          const ray = pointerRay(camera.invViewProj, camera.position, pointer.x, pointer.y);
          const b = solver.bounds;
          const hit = rayBox(
            ray,
            [b.min[0] - 0.1, b.min[1] - 0.01, b.min[2]],
            [b.max[0], b.min[1] + state.level + 0.45, b.max[2]],
          );
          gesture = {
            id: pointer.id,
            mode: hit ? 'water' : 'orbit',
            travel: 0,
            start: clock,
            lastX: pointer.x,
            lastY: pointer.y,
          };
        }
        if (!gesture || pointer.id !== gesture.id) continue;
        gesture.travel += Math.abs(pointer.dx) + Math.abs(pointer.dy);
        idle = 0;
        if (gesture.mode === 'orbit' && pointer.down) {
          dx += pointer.dx;
          dy += pointer.dy;
          dragging = true;
        } else if (gesture.mode === 'water' && pointer.down && gesture.travel > 0.012) {
          const point = pointOnLevel(pointer.x, pointer.y);
          if (point) {
            const previous = pusher?.position ?? point;
            const dt = Math.max(delta, 1 / 240);
            const measured: Vec3 = [
              (point[0] - previous[0]) / dt,
              (point[1] - previous[1]) / dt,
              (point[2] - previous[2]) / dt,
            ];
            const old = pusher?.velocity ?? [0, 0, 0];
            const mixed: Vec3 = [0, 1, 2].map(
              (a) => old[a]! + (measured[a]! - old[a]!) * 0.5,
            ) as unknown as Vec3;
            const speed = Math.hypot(...mixed);
            const limit = 4;
            const scale = speed > limit ? limit / speed : 1;
            pusher = {
              position: point,
              velocity: [mixed[0] * scale, mixed[1] * scale, mixed[2] * scale],
              active: true,
            };
          }
        }
        if (pointer.justReleased || !pointer.down) {
          const tap =
            gesture.mode === 'water' &&
            !pointer.cancelled &&
            gesture.travel < 0.012 &&
            clock - gesture.start < 0.5;
          if (tap && solver && state) {
            const point = pointOnLevel(pointer.x, pointer.y);
            if (point) {
              solver.addBlock({
                shape: 'sphere',
                center: [point[0], point[1] + 0.32, point[2]],
                radius: 0.13,
                velocity: [0, -1.5, 0],
              });
            }
          }
          gesture = undefined;
          pusher = undefined;
        }
      }
      if (gesture && !frame.some((p) => p.id === gesture!.id)) {
        gesture = undefined;
        pusher = undefined;
      }
      const options = params.autoRotate ? orbitOptions : { ...orbitOptions, autoSpeed: 0 };
      stepOrbit(
        orbit,
        { yaw: dragging ? -dx * 3 : 0, pitch: dragging ? dy * 2 : 0, dragging },
        delta,
        options,
      );
    }

    let width = 1;
    let height = 1;
    let frameCount = 0;
    let phase = 0;

    function materialParams(mpm: MpmSolver): FluidSurfaceParams {
      const water = WATER.includes(params.water as (typeof WATER)[number])
        ? params.water
        : 'coastal';
      const custom = water === 'custom';
      const tint = params.tint.map((c) => clamp(c, 1e-3, 1));
      const rest = mpm.particles.restVolume ?? 1;
      const r = mpm.particles.radius;
      return {
        surface: 'smooth',
        waterType: custom ? -1 : WATER.indexOf(water as (typeof WATER)[number]),
        absorption: [-Math.log(tint[0]!), -Math.log(tint[1]!), -Math.log(tint[2]!)],
        scatter: [tint[0]! * 0.6, tint[1]! * 0.6, tint[2]! * 0.6],
        turbidity: params.murk,
        density: params.density,
        ior: 1.333,
        refraction: params.refraction,
        roughness: 0.04,
        sunDirection: SUN_DIRECTION,
        thicknessGain: rest / ((4 / 3) * Math.PI * r * r * r),
        foamAmount: params.foam,
        maxRefractionThickness: 1.2,
      };
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
      },

      frame({ delta }, encoder, target) {
        clock += delta;
        phase += delta;
        idle += delta;
        frameCount++;

        // --- The solver and the scene ---------------------------------------------------------------------
        const mpm = ensureSolver();
        const sceneName = (SCENES as readonly string[]).includes(params.scene)
          ? (params.scene as SceneName)
          : 'wave tank';
        const key = `${sceneName}|${params.grid}|${Math.round(params.count)}`;
        const restarted = params.restart !== lastRestart;
        lastRestart = params.restart;
        const looped =
          state !== undefined &&
          state.loop > 0 &&
          params.autoRestart &&
          sceneClock > state.loop &&
          idle > 5;
        if (key !== builtKey || restarted || looped || !state) {
          builtKey = key;
          sceneClock = 0;
          state = buildScene(sceneName, mpm, Math.round(params.count));
          orbit.target = [
            (state.min[0] + state.max[0]) / 2,
            state.min[1] + 0.4,
            (state.min[2] + state.max[2]) / 2,
          ];
        }
        sceneClock += delta;

        handlePointers(delta);
        orbit.distance = params.distance;

        // Live parameters.
        const p = mpm.params;
        p.gravity = [0, -params.gravity, 0];
        p.soundSpeed = params.stiffness;
        p.viscosity = params.viscosity;
        p.damping = params.damping;
        p.dt = params.substeps >= 1 ? 1 / (FRAME_RATE * Math.round(params.substeps)) : 0;
        p.sorted = params.sorted;
        const mode = (VIEWS as readonly string[]).includes(params.view)
          ? (params.view as View)
          : 'liquid';
        p.scalar = mode === 'speed' ? 'speed' : mode === 'density' ? 'density' : 'foam';
        mpm.system.radius = mpm.spacing * params.particleRadius;

        // The piston wall: a smooth stroke, eased in so the first wave is not a slap.
        if (state.pistonRest !== undefined) {
          const ramp = smoothstep(sceneClock / 2.5);
          const stroke = 0.4 * params.waveMaker * ramp;
          const x =
            state.pistonRest +
            stroke * 0.5 * (1 - Math.cos((2 * Math.PI * phase) / params.wavePeriod));
          mpm.setBounds({ min: [x, state.min[1], state.min[2]], max: state.max });
        }
        // The pour stops adding once the particle budget is reached.
        if (state.name === 'pour') {
          mpm.setPour({
            position: [
              state.min[0] + 0.3 * (state.max[0] - state.min[0]),
              Math.min(state.min[1] + 1.4, state.top - 0.3),
              (state.min[2] + state.max[2]) / 2,
            ],
            direction: [0, -1, 0],
            speed: 3.2,
            radius: 0.085,
            active: mpm.count < state.pourLimit,
          });
        }
        const colliders = [...state.colliders];
        if (pusher?.active) {
          colliders.push({
            shape: 'sphere',
            center: pusher.position,
            radius: PUSH_RADIUS,
            velocity: pusher.velocity,
            friction: 0,
          });
        }
        mpm.setColliders(colliders);
        mpm.update(encoder, delta);

        // --- The camera --------------------------------------------------------------------------------------
        const aspect = width / height;
        computeCamera(
          {
            position: orbitEye(orbit),
            target: orbit.target,
            fovY: fitFovY(FOV_Y, aspect, MIN_HORIZONTAL_FOV),
            aspect,
            near: 0.1,
            far: 120,
          },
          camera,
        );

        // --- The scene ---------------------------------------------------------------------------------------
        const b = mpm.bounds;
        const objects: SceneObject[] = [...state.objects];
        // The paddle: the moving wall, as a slab behind it.
        const paddleThickness = 0.14;
        objects.push(
          boxObject(
            meshes.box,
            [b.min[0] - paddleThickness / 2, (b.min[1] + state.top) / 2, (b.min[2] + b.max[2]) / 2],
            [paddleThickness, state.top - b.min[1] + 0.02, b.max[2] - b.min[2]],
            COLORS.paddle,
          ),
        );
        if (pusher?.active) {
          const model = mat4.translation(pusher.position);
          mat4.scale(model, [PUSH_RADIUS, PUSH_RADIUS, PUSH_RADIUS], model);
          objects.push({ mesh: meshes.sphere, model, color: COLORS.blob });
        }
        roomBuffer.set({
          tank_min: [state.min[0], state.min[2]],
          tank_max: [state.max[0], state.max[2]],
          floor_y: state.min[1],
          tile: 0.25,
        });
        roomBuffer.upload();
        view.update(camera, width, height, LIGHT);
        const batch = renderer.upload(objects);
        renderer.draw(encoder, batch, {
          view,
          color: sceneColor.view,
          depth: sceneDepth.view,
          background: (pass) => {
            pass.setBindGroup(1, roomGroup);
            pass.setPipeline(skyPipeline);
            pass.setBindGroup(0, view.bindGroup);
            pass.draw(3);
            pass.setPipeline(floorPipeline);
            pass.draw(6);
          },
          label: 'mpm scene pass',
          timestampWrites: profiler.pass('scene'),
        });

        // --- The liquid ---------------------------------------------------------------------------------------
        if (mode === 'liquid') {
          fluid.setQuality({
            scale: SCALES[params.smoothScale] ?? 0.5,
            thicknessScale: THICKNESS_SCALES[params.thicknessScale] ?? 0.25,
            thicknessStride: Math.round(params.thicknessStride),
            filter: (FILTERS as readonly string[]).includes(params.filter)
              ? (params.filter as FluidFilter)
              : 'narrow-range',
            filterRadius: params.filterRadius,
            iterations: Math.round(params.iterations),
          });
          fluid.render(encoder, {
            particles: mpm.particles,
            camera,
            sceneColor: sceneColor.view,
            sceneDepth: sceneDepth.view,
            target: liquid.view,
            params: materialParams(mpm),
            profiler,
          });
          output.encode(
            encoder,
            target,
            liquid.view,
            undefined,
            { exposure: params.exposure, tonemap: 'aces', frame: frameCount },
            profiler.pass('output'),
          );
        } else {
          // The particles themselves, as opaque spheres coloured by foam, speed or density, over the tonemapped scene.
          output.encode(
            encoder,
            target,
            sceneColor.view,
            undefined,
            { exposure: params.exposure, tonemap: 'aces', frame: frameCount },
            profiler.pass('output'),
          );
          billboards.style.sizeScale = 1;
          billboards.style.scalarRange =
            mode === 'speed' ? [0, 5] : mode === 'density' ? [0.3, 1.4] : [0, 1];
          billboards.draw(encoder, mpm.particles, camera, target, {
            width,
            height,
            depth: sceneDepth.view,
            clear: null,
            clearDepth: false,
            scalar:
              mode === 'speed' || mode === 'density' ? mpm.system.attribute('aux') : undefined,
            timestamp: 'particles draw',
          });
        }
      },

      destroy() {
        pointers.destroy();
        solver?.destroy();
        fluid.destroy();
        billboards.destroy();
        output.destroy();
        sceneColor.destroy();
        sceneDepth.destroy();
        liquid.destroy();
        roomBuffer.destroy();
        view.destroy();
        renderer.destroy();
        for (const mesh of Object.values(meshes)) mesh.destroy();
      },
    };
  },
});

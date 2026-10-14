import { trackPointers } from '../../gpu/pointer.ts';
import { createShader } from '../../gpu/shader.ts';
import {
  CameraUniform,
  computeCamera,
  createCameraMatrices,
  fitFovY,
  setCameraUniform,
  unprojectPoint,
} from '../../gpu/scene/camera.ts';
import {
  createOrbitState,
  DEFAULT_ORBIT_OPTIONS,
  orbitEye,
  stepOrbit,
} from '../../gpu/scene/orbit.ts';
import { gridIndices } from '../../gpu/sim/caustics.ts';
import {
  createShallowWater,
  type ShallowWater,
  type ShallowWaterKind,
} from '../../gpu/sim/shallow-water.ts';
import { createUniformBuffer, defineUniformStruct } from '../../gpu/uniforms.ts';
import { defineEffect } from '../types.ts';
import { meta } from './meta.ts';
import {
  SCENARIO_IDS,
  SCENARIOS,
  WORLD_ASPECT,
  WORLD_DEPTH,
  WORLD_WIDTH,
  sampleBed,
  type ScenarioId,
  type ScenarioInfo,
} from './scenarios.ts';
import sceneShader from './shallow.wgsl';

const SceneParams = defineUniformStruct('SceneParams', {
  sun: 'vec3f',
  sun_power: 'f32',
  brush: 'vec4f',
  palette: 'vec2f',
  domain: 'vec2f',
  texel: 'vec2f',
  mesh: 'vec2u',
  cell: 'f32',
  time: 'f32',
  exposure: 'f32',
  ripple: 'f32',
  edge: 'f32',
  gravity: 'f32',
  shine: 'f32',
  view: 'u32',
  skirt: 'f32',
  density: 'f32',
  foam_amount: 'f32',
  water_kind: 'u32',
});

const SOLVERS = ['compute', 'render'] as const;
const RESOLUTIONS = ['128', '192', '256', '384'] as const;
const TOOLS = [
  'add water',
  'remove water',
  'push water',
  'raise ground',
  'lower ground',
  'orbit camera',
] as const;
const TOUCH_MODES = ['scrolls the page', 'plays with the water'] as const;
const WATER_TYPES = [
  'clear tropical',
  'coastal',
  'murky green',
  'tea lake',
  'glacial milk',
  'swimming pool',
  'open ocean',
] as const;
const VIEWS = ['beauty', 'depth', 'speed', 'surface height', 'foam'] as const;

const DEPTH_FORMAT: GPUTextureFormat = 'depth24plus';
const DEG = Math.PI / 180;
const CAMERA_FOV_Y = 45 * DEG;
const MIN_HORIZONTAL_FOV = 66 * DEG;
const SKIRT_DEPTH = 1.1;
const SCULPT_RATE = 0.9;
/** A tap acts for at least this long, so it leaves something to see. */
const TAP_SECONDS = 0.28;

const clamp = (x: number, lo: number, hi: number) => Math.min(hi, Math.max(lo, x));

interface Hit {
  /** World position and domain uv. */
  x: number;
  y: number;
  z: number;
  u: number;
  v: number;
}

interface Gesture {
  until: number;
  down: boolean;
  invert: boolean;
  lastHit?: Hit;
  /** Smoothed hit velocity in metres per second (for pushing). */
  vx: number;
  vz: number;
}

export const effect = defineEffect({
  meta,
  params: {
    scenario: {
      type: 'select',
      default: 'dam break',
      options: SCENARIO_IDS,
      label: 'Scenario (restarts)',
    },
    restart: {
      type: 'boolean',
      default: false,
      label: 'Restart (flip to run the scenario again)',
    },
    autoRestart: {
      type: 'number',
      default: 0,
      min: 0,
      max: 120,
      step: 1,
      label: 'Restart by itself after the dam goes (s, 0 = never)',
    },
    speed: {
      type: 'number',
      default: 0.9,
      min: 0.1,
      max: 2.5,
      step: 0.01,
      label: 'Time scale (1 = real time)',
    },
    gravity: {
      type: 'number',
      default: 9.81,
      min: 2,
      max: 25,
      step: 0.01,
      label: 'Gravity (m / s²)',
    },
    friction: {
      type: 'number',
      default: 0.025,
      min: 0,
      max: 0.15,
      step: 0.001,
      label: 'Ground friction (Manning n)',
    },
    rain: { type: 'number', default: 0, min: 0, max: 30, step: 0.1, label: 'Rain (mm / s)' },
    inflow: {
      type: 'number',
      default: 1,
      min: 0,
      max: 3,
      step: 0.01,
      label: 'Spring flow / tide strength',
    },
    gridResolution: {
      type: 'select',
      default: '256',
      options: RESOLUTIONS,
      label: 'Grid cells on the long side (restarts)',
    },
    solver: {
      type: 'select',
      default: 'compute',
      options: SOLVERS,
      label: 'Solver passes (restarts)',
    },
    tool: { type: 'select', default: 'add water', options: TOOLS, label: 'Pointer tool' },
    brushSize: {
      type: 'number',
      default: 0.5,
      min: 0.15,
      max: 1.6,
      step: 0.01,
      label: 'Brush radius (m)',
    },
    brushStrength: {
      type: 'number',
      default: 1.2,
      min: 0.1,
      max: 5,
      step: 0.01,
      label: 'Pour rate (m³ / s)',
    },
    touch: {
      type: 'select',
      default: 'scrolls the page',
      options: TOUCH_MODES,
      label: 'Touch drags',
    },
    waterType: { type: 'select', default: 'coastal', options: WATER_TYPES, label: 'Water' },
    waterDensity: {
      type: 'number',
      default: 5,
      min: 1,
      max: 14,
      step: 0.1,
      label: 'Water density (visual depth multiplier)',
    },
    foamAmount: { type: 'number', default: 0.85, min: 0, max: 1.6, step: 0.01, label: 'Foam' },
    ripples: { type: 'number', default: 1, min: 0, max: 3, step: 0.01, label: 'Surface ripples' },
    shine: { type: 'number', default: 1, min: 0, max: 2, step: 0.01, label: 'Sun glint' },
    sunElevation: {
      type: 'number',
      default: 34,
      min: 12,
      max: 85,
      step: 0.5,
      label: 'Sun elevation (degrees)',
    },
    sunAzimuth: {
      type: 'number',
      default: 112,
      min: 0,
      max: 360,
      step: 1,
      label: 'Sun direction (degrees)',
    },
    exposure: { type: 'number', default: 0.9, min: 0.4, max: 1.8, step: 0.01 },
    autoRotate: { type: 'boolean', default: true, label: 'Camera sways gently' },
    view: { type: 'select', default: 'beauty', options: VIEWS, label: 'View' },
    stepsPerFrame: {
      type: 'number',
      default: 0,
      min: 0,
      max: 200,
      step: 1,
      label: 'Solver steps per frame (0 = by clock; for benchmarks)',
    },
    sceneRepeats: {
      type: 'number',
      default: 1,
      min: 1,
      max: 16,
      step: 1,
      label: 'Scene draws per frame (for benchmarks)',
    },
  },
  presets: {
    'dam break': {
      scenario: 'dam break',
      speed: 0.9,
      friction: 0.025,
      gravity: 9.81,
      rain: 0,
      waterType: 'coastal',
      autoRestart: 0,
      view: 'beauty',
    },
    'dam break, slow motion': {
      scenario: 'dam break',
      speed: 0.28,
      friction: 0.015,
      gravity: 9.81,
      rain: 0,
      waterType: 'swimming pool',
      view: 'beauty',
    },
    'dam break, on a loop': {
      scenario: 'dam break',
      speed: 1,
      friction: 0.025,
      gravity: 9.81,
      rain: 0,
      waterType: 'coastal',
      autoRestart: 16,
      view: 'beauty',
    },
    'muddy flood': {
      scenario: 'dam break',
      speed: 1,
      friction: 0.06,
      gravity: 9.81,
      rain: 0,
      waterType: 'tea lake',
      waterDensity: 8,
      foamAmount: 0.7,
      view: 'beauty',
    },
    'hillside spring': {
      scenario: 'hillside spring',
      speed: 1.5,
      friction: 0.03,
      gravity: 9.81,
      rain: 0,
      inflow: 1,
      waterType: 'clear tropical',
      view: 'beauty',
    },
    'rain on the hills': {
      scenario: 'hillside spring',
      speed: 1.6,
      friction: 0.035,
      gravity: 9.81,
      rain: 4,
      inflow: 0.4,
      waterType: 'murky green',
      view: 'beauty',
    },
    'tide pool': {
      scenario: 'tide pool',
      speed: 1.6,
      friction: 0.03,
      gravity: 9.81,
      rain: 0,
      inflow: 1,
      waterType: 'coastal',
      view: 'beauty',
    },
    'flow speed (debug)': {
      scenario: 'dam break',
      view: 'speed',
    },
  },

  async create({ device, format, canvas, params, profiler }) {
    // --- Pointer input -------------------------------------------------------------------------
    // A primary-button press is a tool use; a touch drag scrolls the page unless the visitor asked
    // for the water to take it (the same policy as `ripples`): the canvas of a hero must not trap a
    // scroll. Shift (or a secondary button) inverts the tool: pour <-> drain, raise <-> dig.
    const pointers = trackPointers(canvas, { capture: params.touch === TOUCH_MODES[1] });
    const listeners = new AbortController();
    const inverted = new Map<number, boolean>();
    canvas.addEventListener(
      'pointerdown',
      (event) => {
        const e = event as PointerEvent;
        inverted.set(e.pointerId, Boolean(e.shiftKey) || e.button === 2);
      },
      { signal: listeners.signal },
    );

    // --- The solver ----------------------------------------------------------------------------
    const makeSim = (): ShallowWater =>
      createShallowWater({
        device,
        kind: params.solver as ShallowWaterKind,
        resolution: Number(params.gridResolution),
        aspect: WORLD_ASPECT,
        size: WORLD_WIDTH,
        profiler,
        label: 'shallow water',
      });
    let sim = makeSim();

    let scenarioId = params.scenario as ScenarioId;
    let scenario: ScenarioInfo = SCENARIOS[scenarioId];
    let bedCpu = new Float32Array(0);
    let releasedBed: Float32Array | undefined;
    let released = false;
    let releasedAt = 0;
    let needsStart = true;
    let lastRestart = params.restart;
    let cameraReset = true;

    // --- Pipelines -----------------------------------------------------------------------------
    const layout = device.createBindGroupLayout({
      label: 'shallow water layout',
      entries: [
        {
          binding: 0,
          visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT,
          buffer: { type: 'uniform' },
        },
        {
          binding: 1,
          visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT,
          buffer: { type: 'uniform' },
        },
        {
          binding: 2,
          visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT,
          texture: { sampleType: 'float' },
        },
        {
          binding: 3,
          visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT,
          texture: { sampleType: 'float' },
        },
        {
          binding: 4,
          visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT,
          sampler: { type: 'filtering' },
        },
      ],
    });
    const module = createShader(device, {
      label: 'shallow-water/shallow.wgsl',
      code: sceneShader.code,
      chunks: { ...sceneShader.chunks, 'scene_params.wgsl': SceneParams.wgsl },
    });
    const pipelineLayout = device.createPipelineLayout({
      label: 'shallow water pipeline layout',
      bindGroupLayouts: [layout],
    });
    const makePipeline = (
      name: string,
      vertex: string,
      fragment: string,
      options: {
        topology?: GPUPrimitiveTopology;
        depthCompare?: GPUCompareFunction;
        depthWrite?: boolean;
        blend?: boolean;
        bias?: boolean;
      } = {},
    ) =>
      device.createRenderPipeline({
        label: `shallow water ${name}`,
        layout: pipelineLayout,
        vertex: { module, entryPoint: vertex },
        fragment: {
          module,
          entryPoint: fragment,
          targets: [
            {
              format,
              blend: options.blend
                ? {
                    color: { srcFactor: 'one', dstFactor: 'one-minus-src-alpha' },
                    alpha: { srcFactor: 'one', dstFactor: 'one-minus-src-alpha' },
                  }
                : undefined,
            },
          ],
        },
        primitive: { topology: options.topology ?? 'triangle-list', cullMode: 'none' },
        depthStencil: {
          format: DEPTH_FORMAT,
          depthWriteEnabled: options.depthWrite ?? true,
          depthCompare: options.depthCompare ?? 'less',
          depthBias: options.bias ? -2 : 0,
          depthBiasSlopeScale: options.bias ? -1 : 0,
        },
      });
    const skyPipeline = makePipeline('sky', 'fullscreen_vs', 'sky_fs', {
      depthCompare: 'always',
      depthWrite: false,
    });
    const terrainPipeline = makePipeline('terrain', 'terrain_vs', 'terrain_fs');
    const skirtPipeline = makePipeline('skirt', 'skirt_vs', 'skirt_fs', {
      topology: 'triangle-strip',
    });
    const sectionPipeline = makePipeline('section', 'section_vs', 'section_fs', {
      topology: 'triangle-strip',
      blend: true,
      depthCompare: 'less-equal',
      depthWrite: false,
    });
    const waterPipeline = makePipeline('water', 'water_vs', 'water_fs', {
      blend: true,
      depthCompare: 'less-equal',
      bias: true,
    });

    // --- Uniforms, bindings, targets -----------------------------------------------------------
    const cameraBuffer = createUniformBuffer(device, CameraUniform, 'shallow water camera');
    const paramsBuffer = createUniformBuffer(device, SceneParams, 'shallow water params');
    const sampler = device.createSampler({
      label: 'shallow water sampler',
      magFilter: 'linear',
      minFilter: 'linear',
      addressModeU: 'clamp-to-edge',
      addressModeV: 'clamp-to-edge',
    });
    let bindGroup: GPUBindGroup | undefined;
    let boundGeneration = -1;
    let boundSim: ShallowWater | undefined;
    let indexBuffer: GPUBuffer | undefined;
    let indexCount = 0;
    let indexKey = '';
    let depthTexture: GPUTexture | undefined;
    let depthView: GPUTextureView | undefined;
    let width = Math.max(1, canvas.width || 1);
    let height = Math.max(1, canvas.height || 1);

    function allocateMesh() {
      const { width: cx, height: cy } = sim.grid;
      const key = `${cx}x${cy}`;
      if (key === indexKey && indexBuffer) return;
      indexKey = key;
      indexBuffer?.destroy();
      const data = gridIndices(cx, cy);
      indexCount = data.length;
      indexBuffer = device.createBuffer({
        label: 'shallow water mesh indices',
        size: data.byteLength,
        usage: GPUBufferUsage.INDEX | GPUBufferUsage.COPY_DST,
      });
      device.queue.writeBuffer(indexBuffer, 0, data);
    }

    // --- Camera and picking --------------------------------------------------------------------
    const orbit = createOrbitState({ yaw: 0.5, pitch: 0.6, distance: 15, target: [0, 0.3, 0] });
    const orbitOptions = { ...DEFAULT_ORBIT_OPTIONS, minPitch: 0.22, maxPitch: 1.42, autoSpeed: 0 };
    const camera = createCameraMatrices();
    let clock = 0;

    function pick(px: number, py: number): Hit | undefined {
      const near = unprojectPoint(camera.invViewProj, px, py, 0);
      const far = unprojectPoint(camera.invViewProj, px, py, 1);
      const dx = far[0]! - near[0]!;
      const dy = far[1]! - near[1]!;
      const dz = far[2]! - near[2]!;
      const length = Math.hypot(dx, dy, dz) || 1;
      const dir = [dx / length, dy / length, dz / length] as const;
      const { width: gw, height: gh } = sim.grid;
      // Height above the bed of the point at distance t, or +Infinity outside the domain.
      const above = (t: number) => {
        const x = near[0]! + dir[0] * t;
        const y = near[1]! + dir[1] * t;
        const z = near[2]! + dir[2] * t;
        const u = x / WORLD_WIDTH + 0.5;
        const v = z / WORLD_DEPTH + 0.5;
        if (u < 0 || u > 1 || v < 0 || v > 1) return Number.POSITIVE_INFINITY;
        return y - sampleBed(bedCpu, gw, gh, u, v);
      };
      let previous = 0;
      for (let t = 0; t < 90; t += 0.08) {
        if (above(t) <= 0 && Number.isFinite(above(previous))) {
          let lo = previous;
          let hi = t;
          for (let i = 0; i < 12; i++) {
            const mid = (lo + hi) / 2;
            if (above(mid) <= 0) hi = mid;
            else lo = mid;
          }
          const x = near[0]! + dir[0] * hi;
          const y = near[1]! + dir[1] * hi;
          const z = near[2]! + dir[2] * hi;
          return { x, y, z, u: x / WORLD_WIDTH + 0.5, v: z / WORLD_DEPTH + 0.5 };
        }
        previous = t;
      }
      return undefined;
    }

    // --- Interaction ---------------------------------------------------------------------------
    const gestures = new Map<number, Gesture>();
    let hover: Hit | undefined;
    let brushAlpha = 0;

    /** The same bump the solver uses, so picking keeps up with the ground the visitor sculpts. */
    function sculptCpu(hit: Hit, rate: number, seconds: number) {
      const { width: gw, height: gh, cellSize } = sim.grid;
      const radius = Math.max(params.brushSize / cellSize, 1.5);
      const cx = hit.u * gw;
      const cy = hit.v * gh;
      const lo = sim.params.bedMin;
      const hi = sim.params.bedMax;
      for (
        let y = Math.max(0, Math.floor(cy - radius));
        y <= Math.min(gh - 1, Math.ceil(cy + radius));
        y++
      ) {
        for (
          let x = Math.max(0, Math.floor(cx - radius));
          x <= Math.min(gw - 1, Math.ceil(cx + radius));
          x++
        ) {
          const q = Math.hypot(x + 0.5 - cx, y + 0.5 - cy) / radius;
          if (q < 1) {
            const i = y * gw + x;
            bedCpu[i] = clamp(bedCpu[i]! + rate * seconds * (1 - q * q) ** 2, lo, hi);
          }
        }
      }
    }

    function handlePointers(delta: number) {
      const tool = params.tool as (typeof TOOLS)[number];
      const frame = pointers.poll();
      let orbitDx = 0;
      let orbitDy = 0;
      let orbiting = false;
      let pointerHover: Hit | undefined;

      for (const pointer of frame) {
        if (pointer.justPressed) {
          gestures.set(pointer.id, {
            until: clock + TAP_SECONDS,
            down: true,
            invert: inverted.get(pointer.id) ?? false,
            vx: 0,
            vz: 0,
          });
        }
        const gesture = gestures.get(pointer.id);
        if (gesture) {
          gesture.down = pointer.down;
          if (pointer.justReleased || !pointer.down) gesture.down = false;
        }
        if (tool === 'orbit camera') {
          if (pointer.down) {
            orbitDx += pointer.dx;
            orbitDy += pointer.dy;
            orbiting = true;
          }
          continue;
        }
        if (pointer.inside || pointer.down) {
          const hit = pick(pointer.x, pointer.y);
          if (hit) {
            pointerHover = hit;
            if (gesture) {
              if (gesture.lastHit && delta > 0) {
                const k = 1 - Math.exp(-delta * 18);
                gesture.vx += ((hit.x - gesture.lastHit.x) / delta - gesture.vx) * k;
                gesture.vz += ((hit.z - gesture.lastHit.z) / delta - gesture.vz) * k;
              }
              gesture.lastHit = hit;
            }
          }
        }
      }

      stepOrbit(
        orbit,
        { yaw: orbiting ? -orbitDx * 3 : 0, pitch: orbiting ? orbitDy * 2 : 0, dragging: orbiting },
        delta,
        orbitOptions,
      );

      hover = pointerHover ?? hover;
      const wantsRing =
        tool !== 'orbit camera' && (pointerHover !== undefined || gestures.size > 0);
      brushAlpha += ((wantsRing ? 1 : 0) - brushAlpha) * (1 - Math.exp(-delta * 10));

      // Apply the tool for every gesture still pressed (or inside its tap time).
      const seconds = delta * Math.max(0, params.speed);
      for (const [id, gesture] of gestures) {
        const active = gesture.down || clock < gesture.until;
        if (!active) {
          gestures.delete(id);
          inverted.delete(id);
          continue;
        }
        const hit = gesture.lastHit;
        if (!hit || tool === 'orbit camera') continue;
        const flip = gesture.invert ? -1 : 1;
        const radius = params.brushSize;
        switch (tool) {
          case 'add water':
          case 'remove water': {
            const sign = (tool === 'add water' ? 1 : -1) * flip;
            sim.addSource({ x: hit.u, y: hit.v, radius, volume: sign * params.brushStrength });
            break;
          }
          case 'push water': {
            const speed = Math.hypot(gesture.vx, gesture.vz);
            const gain = Math.min(1, speed / 1.5) * 14;
            const px = speed > 1e-3 ? (gesture.vx / speed) * gain : 0;
            const pz = speed > 1e-3 ? (gesture.vz / speed) * gain : 0;
            sim.addSource({ x: hit.u, y: hit.v, radius: radius * 1.3, pushX: px, pushY: pz });
            break;
          }
          case 'raise ground':
          case 'lower ground': {
            const rate = (tool === 'raise ground' ? 1 : -1) * flip * SCULPT_RATE;
            sim.addSource({ x: hit.u, y: hit.v, radius, bedRate: rate });
            sculptCpu(hit, rate, seconds);
            break;
          }
        }
      }
    }

    // --- Scenarios -----------------------------------------------------------------------------
    function startScenario(encoder: GPUCommandEncoder) {
      const next = params.scenario as ScenarioId;
      const changed = next !== scenarioId;
      scenarioId = next;
      scenario = SCENARIOS[scenarioId];
      sim.reset(encoder);
      sim.params.maxDepth = scenario.maxDepth;
      sim.params.edges = scenario.edges;
      const built = scenario.build(sim.grid.width, sim.grid.height);
      sim.setTerrain(built.terrain);
      sim.setWater(built.water);
      bedCpu = built.terrain.slice();
      releasedBed = built.released;
      released = false;
      releasedAt = 0;
      gestures.clear();
      if (changed || cameraReset) {
        orbit.yaw = scenario.camera.yaw;
        orbit.pitch = scenario.camera.pitch;
        orbit.distance = scenario.camera.distance;
        orbit.yawVelocity = 0;
        orbit.pitchVelocity = 0;
        cameraReset = false;
      }
      needsStart = false;
    }

    function syncParams() {
      const p = sim.params;
      p.gravity = params.gravity;
      p.friction = params.friction;
      p.rain = params.rain / 1000;
      p.timeScale = params.speed;
    }

    return {
      resize(w, h) {
        width = Math.max(1, Math.round(w));
        height = Math.max(1, Math.round(h));
        depthTexture?.destroy();
        depthTexture = device.createTexture({
          label: 'shallow water depth',
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

        pointers.setCapture(params.touch === TOUCH_MODES[1]);

        // Parameters that re-create the solver, then the ones that restart the scene.
        if (sim.kind !== params.solver) {
          sim.destroy();
          sim = makeSim();
          needsStart = true;
        }
        if (sim.setResolution(Number(params.gridResolution))) needsStart = true;
        if (params.scenario !== scenarioId) needsStart = true;
        if (params.restart !== lastRestart) {
          lastRestart = params.restart;
          needsStart = true;
        }
        if (params.autoRestart > 0 && released && sim.simTime - releasedAt > params.autoRestart) {
          needsStart = true;
        }
        if (needsStart) startScenario(encoder);
        syncParams();
        allocateMesh();

        // The camera comes first: picking needs this frame's matrices.
        const sway = params.autoRotate ? 0.34 * Math.sin(clock * 0.11) : 0;
        computeCamera(
          {
            position: orbitEye({ ...orbit, yaw: orbit.yaw + sway }),
            target: orbit.target,
            fovY: fitFovY(CAMERA_FOV_Y, aspect, MIN_HORIZONTAL_FOV),
            aspect,
            near: 0.1,
            far: 160,
          },
          camera,
        );
        setCameraUniform(cameraBuffer, camera, width, height);
        cameraBuffer.upload();

        handlePointers(delta);

        // The dam goes: the bed loses the wall and the water behind it is a column.
        if (
          scenario.releaseAfter !== undefined &&
          !released &&
          sim.simTime >= scenario.releaseAfter
        ) {
          if (releasedBed) {
            sim.setTerrain(releasedBed);
            bedCpu = releasedBed.slice();
          }
          released = true;
          releasedAt = sim.simTime;
        }
        scenario.forces(sim, sim.simTime, params.inflow);

        if (params.stepsPerFrame > 0) sim.advance(encoder, params.stepsPerFrame);
        else sim.update(encoder, delta);

        // A new solver starts counting generations again, so the solver itself is part of the key.
        if (!bindGroup || boundSim !== sim || boundGeneration !== sim.generation) {
          boundSim = sim;
          boundGeneration = sim.generation;
          bindGroup = device.createBindGroup({
            label: 'shallow water bind group',
            layout,
            entries: [
              { binding: 0, resource: { buffer: cameraBuffer.buffer } },
              { binding: 1, resource: { buffer: paramsBuffer.buffer } },
              { binding: 2, resource: sim.view },
              { binding: 3, resource: sim.groundView },
              { binding: 4, resource: sampler },
            ],
          });
        }

        const elevation = params.sunElevation * DEG;
        const azimuth = params.sunAzimuth * DEG;
        const { grid } = sim;
        paramsBuffer.set({
          sun: [
            Math.cos(elevation) * Math.sin(azimuth),
            Math.sin(elevation),
            Math.cos(elevation) * Math.cos(azimuth),
          ],
          sun_power: 2.1,
          brush: hover
            ? [hover.x, hover.z, params.brushSize, brushAlpha * 0.85]
            : [0, 0, params.brushSize, 0],
          palette: [scenario.palette[0], scenario.palette[1]],
          domain: [WORLD_WIDTH, WORLD_DEPTH],
          texel: [grid.texelSize[0], grid.texelSize[1]],
          mesh: [grid.width, grid.height],
          cell: grid.cellSize,
          time: clock,
          exposure: params.exposure,
          ripple: params.ripples,
          edge: 0.02,
          gravity: params.gravity,
          shine: params.shine,
          view: Math.max(0, VIEWS.indexOf(params.view as (typeof VIEWS)[number])),
          skirt: SKIRT_DEPTH,
          density: params.waterDensity,
          foam_amount: params.foamAmount,
          water_kind: Math.max(
            0,
            WATER_TYPES.indexOf(params.waterType as (typeof WATER_TYPES)[number]),
          ),
        });
        paramsBuffer.upload();

        const pass = encoder.beginRenderPass({
          label: 'shallow water scene',
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
        const repeats = Math.max(1, Math.round(params.sceneRepeats));
        for (let r = 0; r < repeats; r++) {
          pass.setPipeline(terrainPipeline);
          pass.setIndexBuffer(indexBuffer!, 'uint32');
          pass.drawIndexed(indexCount);
          pass.setPipeline(skirtPipeline);
          pass.draw(2 * (Math.max(grid.width, grid.height) + 1), 4);
          pass.setPipeline(waterPipeline);
          pass.setIndexBuffer(indexBuffer!, 'uint32');
          pass.drawIndexed(indexCount);
          pass.setPipeline(sectionPipeline);
          pass.draw(2 * (Math.max(grid.width, grid.height) + 1), 4);
        }
        pass.end();
      },

      destroy() {
        listeners.abort();
        pointers.destroy();
        sim.destroy();
        indexBuffer?.destroy();
        depthTexture?.destroy();
        cameraBuffer.destroy();
        paramsBuffer.destroy();
      },
    };
  },
});

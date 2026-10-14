import {
  createBillboardRenderer,
  createComputePass,
  createIntegrator,
  createNeighborCounter,
  createNeighborGrid,
  createParticleSystem,
  orthographicCamera,
  type Integrator,
  type NeighborGrid,
  type ParticleSystem,
  type Rgb,
} from '../../gpu/particles/index.ts';
import { trackPointer } from '../../gpu/pointer.ts';
import { createPostChain } from '../../gpu/post/index.ts';
import {
  CameraUniform,
  computeCamera,
  fitFovY,
  setCameraUniform,
  unprojectPoint,
  type CameraMatrices,
} from '../../gpu/scene/camera.ts';
import { createOrbitControl, createOrbitState, orbitEye } from '../../gpu/scene/orbit.ts';
import { createShader } from '../../gpu/shader.ts';
import { createUniformBuffer, defineUniformStruct } from '../../gpu/uniforms.ts';
import { defineEffect } from '../types.ts';
import fieldWgsl from './field.wgsl';
import { LANTERN_CAMERA, LANTERN_EMITTER_Y } from './lantern-layout.ts';
import { meta } from './meta.ts';
import pickWgsl from './pick.wgsl';
import sceneWgsl from './scene.wgsl';

// The particles demo: a spray of up to a million particles carried by a curl-noise flow, drawn as billboards,
// with the neighbour grid of src/gpu/particles/ built every frame when a view asks for it. The point of the
// demo is the framework (see the notes): the integrator does the physics, the grid does the neighbour search,
// and the neighbour views are the search made visible.

const SceneParams = defineUniformStruct('SceneParams', {
  top: 'vec3f',
  line_strength: 'f32',
  bottom: 'vec3f',
  box_min: 'vec3f',
  box_max: 'vec3f',
});

const PickParams = defineUniformStruct('PickParams', {
  pointer: 'vec2f',
  active_flag: 'u32',
});

const SCENES = ['fountain 3D', 'stream 2D'] as const;
const RENDERS = ['additive', 'alpha', 'opaque'] as const;
const VIEWS = ['speed', 'neighbour count', 'neighbours of picked'] as const;

/** Most particles the control allows. */
const MAX_COUNT = 1_000_000;
const NONE = 0xffffffff;

interface Box {
  min: [number, number, number];
  max: [number, number, number];
}

const BOX_3D: Box = { min: [-1.6, 0, -1.6], max: [1.6, 2.8, 1.6] };
const BOX_2D: Box = { min: [-2.6, 0, 0], max: [2.6, 3, 0] };

const SPEED_PALETTE: Rgb[] = [
  [0.01, 0.07, 0.25],
  [0.03, 0.35, 0.72],
  [0.35, 0.85, 1],
  [1, 1, 1],
];
const ORANGE_PALETTE: Rgb[] = [
  [0.18, 0.018, 0.001],
  [0.75, 0.11, 0.006],
  [1, 0.38, 0.025],
  [1, 0.7, 0.16],
];
const HEAT_PALETTE: Rgb[] = [
  [0.02, 0.03, 0.18],
  [0.05, 0.25, 0.7],
  [0.05, 0.7, 0.55],
  [0.95, 0.85, 0.15],
  [0.95, 0.2, 0.08],
  [1, 0.95, 0.95],
];
const PICK_PALETTE: Rgb[] = [
  [0.004, 0.012, 0.03],
  [1, 0.55, 0.08],
  [1, 1, 1],
];

const clamp = (x: number, lo: number, hi: number) => Math.min(hi, Math.max(lo, x));

interface World {
  scene: (typeof SCENES)[number];
  dimension: 2 | 3;
  box: Box;
  system: ParticleSystem;
  integrator: Integrator;
  grid: NeighborGrid;
  /** One f32 per particle: the neighbour count, or the highlight value. Vertex-readable for the renderer. */
  scalar: GPUBuffer;
  scalarCapacity: number;
  destroy(): void;
}

export const effect = defineEffect({
  meta,
  hdr: true,
  params: {
    glowEnergy: {
      type: 'number',
      default: 1,
      min: 0.5,
      max: 6,
      step: 0.1,
      label: 'Lantern radiance',
    },
    bloomIntensity: {
      type: 'number',
      default: 0.6,
      min: 0,
      max: 2,
      step: 0.05,
      label: 'Lantern bloom',
    },
    headroom: {
      type: 'number',
      default: 3,
      min: 1,
      max: 6,
      step: 0.1,
      label: 'HDR white multiplier',
    },
    pageColor: {
      type: 'color',
      default: [0.00605, 0.00651, 0.00478],
      label: 'Lantern page background',
    },
    presentation: {
      type: 'select',
      default: 'gallery',
      options: ['gallery', 'lantern'],
      label: 'Presentation (lantern locks camera and nozzle)',
    },
    palette: {
      type: 'select',
      default: 'ocean',
      options: ['ocean', 'orange'],
      label: 'Speed palette',
    },
    background: { type: 'boolean', default: true, label: 'Backdrop and simulation box' },
    scene: {
      type: 'select',
      default: SCENES[0],
      options: SCENES,
      label: 'Scene (re-creates the particles)',
    },
    count: {
      type: 'number',
      default: 250000,
      min: 10000,
      max: MAX_COUNT,
      step: 10000,
      label: 'Particles',
    },
    flow: {
      type: 'number',
      default: 1.5,
      min: 0,
      max: 8,
      step: 0.05,
      label: 'Curl-noise flow strength',
    },
    size: {
      type: 'number',
      default: 1,
      min: 0.25,
      max: 5,
      step: 0.05,
      label: 'Particle size',
    },
    brightness: {
      type: 'number',
      default: 0.15,
      min: 0.02,
      max: 2,
      step: 0.01,
      label: 'Brightness / opacity (additive, alpha)',
    },
    render: {
      type: 'select',
      default: RENDERS[0],
      options: RENDERS,
      label: 'Render mode',
    },
    view: {
      type: 'select',
      default: VIEWS[0],
      options: VIEWS,
      label: 'Colour by',
    },
    radius: {
      type: 'number',
      default: 0.07,
      min: 0.02,
      max: 0.4,
      step: 0.005,
      label: 'Neighbour search radius (world units)',
    },
    heatScale: {
      type: 'number',
      default: 64,
      min: 4,
      max: 400,
      step: 1,
      label: 'Neighbour count at full colour',
    },
    reorder: {
      type: 'boolean',
      default: true,
      label: 'Reorder particles into cell order (off: sorted index list)',
    },
    alwaysBuild: {
      type: 'boolean',
      default: false,
      label: 'Build the grid every frame even when no view shows it',
    },
    autoRotate: { type: 'boolean', default: true, label: 'Slowly orbit when idle (3D)' },
  },
  presets: {
    fountain: {},
    lantern: {
      presentation: 'lantern',
      palette: 'orange',
      background: false,
      autoRotate: false,
      count: 80000,
      brightness: 0.12,
    },
    mist: { flow: 5, size: 0.7, brightness: 0.08, count: 400000 },
    'a million': { count: 1000000, size: 0.6, brightness: 0.09 },
    'density map': { view: VIEWS[1], brightness: 0.35, size: 1.2 },
    'neighbours of the pointer': {
      view: VIEWS[2],
      count: 120000,
      radius: 0.14,
      size: 1.6,
      brightness: 0.5,
      autoRotate: false,
    },
    'solid spheres': { render: RENDERS[2], count: 60000, size: 2.2 },
    'stream 2D': { scene: SCENES[1], count: 200000, flow: 1, size: 0.8, brightness: 0.18 },
  },

  async create({ device, format, canvas, params, profiler, hdr = false }) {
    const cameraUniform = createUniformBuffer(device, CameraUniform, 'particles camera');
    const sceneUniform = createUniformBuffer(device, SceneParams, 'particles scene');
    const pickUniform = createUniformBuffer(device, PickParams, 'particles pick');
    const pickState = device.createBuffer({
      label: 'particles pick state',
      size: 8,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });

    // --- Backdrop and box --------------------------------------------------------------------------
    const sceneModule = createShader(device, {
      label: 'particles/scene.wgsl',
      code: sceneWgsl.code,
      chunks: { ...sceneWgsl.chunks, 'scene_params.wgsl': SceneParams.wgsl },
    });
    const backdropPipeline = device.createRenderPipeline({
      label: 'particles backdrop',
      layout: 'auto',
      vertex: { module: sceneModule, entryPoint: 'fullscreen_vs' },
      fragment: { module: sceneModule, entryPoint: 'fs_background', targets: [{ format }] },
    });
    const linesPipeline = device.createRenderPipeline({
      label: 'particles box',
      layout: 'auto',
      vertex: { module: sceneModule, entryPoint: 'vs_lines' },
      fragment: { module: sceneModule, entryPoint: 'fs_lines', targets: [{ format }] },
      primitive: { topology: 'line-list' },
    });
    // 'auto' layouts keep only the bindings a stage uses: the backdrop reads the scene colours, the box also the camera.
    const backdropGroup = device.createBindGroup({
      layout: backdropPipeline.getBindGroupLayout(0),
      entries: [{ binding: 1, resource: { buffer: sceneUniform.buffer } }],
    });
    const linesGroup = device.createBindGroup({
      layout: linesPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: cameraUniform.buffer } },
        { binding: 1, resource: { buffer: sceneUniform.buffer } },
      ],
    });

    // --- Picking -----------------------------------------------------------------------------------
    const pickChunks = { 'pick_params.wgsl': PickParams.wgsl };
    const pickDistance = createComputePass(device, {
      label: 'particles/pick.wgsl',
      code: pickWgsl.code,
      chunks: pickChunks,
      entryPoint: 'pick_distance',
      profiler,
      timestamp: 'pick',
    });
    const pickIndex = createComputePass(device, {
      label: 'particles/pick.wgsl',
      code: pickWgsl.code,
      chunks: pickChunks,
      entryPoint: 'pick_index',
      profiler,
      timestamp: 'pick',
    });

    const renderer = createBillboardRenderer({
      device,
      format,
      profiler,
      label: 'particles',
      style: { softness: 0.35, shading: 1 },
    });
    const counter = createNeighborCounter({ device, profiler, label: 'neighbours' });

    // Preserve linear radiance until bloom and HDR output. The gallery keeps its direct draw path.
    let lantern:
      | {
          chain: ReturnType<typeof createPostChain>;
          renderer: ReturnType<typeof createBillboardRenderer>;
        }
      | undefined;
    function ensureLantern() {
      lantern ??= {
        chain: createPostChain(device, {
          canvasFormat: format,
          width: 1,
          height: 1,
          label: 'lantern',
        }),
        renderer: createBillboardRenderer({
          device,
          format: 'rgba16float',
          encodeSrgb: false,
          profiler,
          label: 'lantern radiance',
          style: { softness: 0.35, shading: 1 },
        }),
      };
      lantern.chain.resize(width, height);
      lantern.renderer.resize(width, height);
      return lantern;
    }

    // --- Camera and input --------------------------------------------------------------------------
    const orbitState = createOrbitState({
      yaw: 0.6,
      pitch: 0.32,
      distance: 6.4,
      target: [0, 1.15, 0],
    });
    const orbit = createOrbitControl(canvas, orbitState, { minPitch: 0.02, maxPitch: 1.45 });
    const hover = trackPointer(canvas);
    let width = 1;
    let height = 1;
    let dpr = 1;
    let camera: CameraMatrices | undefined;
    // Where the nozzle (3D) or the vortex (2D) is, smoothed toward the pointer.
    const nozzle: [number, number] = [0, 0];
    let clock = 0;
    let lastHover = -10;

    // --- The particles -----------------------------------------------------------------------------
    let world: World | undefined;

    function createWorld(scene: (typeof SCENES)[number], count: number): World {
      const dimension: 2 | 3 = scene === SCENES[0] ? 3 : 2;
      const box = dimension === 3 ? BOX_3D : BOX_2D;
      const capacity = Math.min(MAX_COUNT, Math.max(count, 16384));
      const system = createParticleSystem({
        device,
        capacity,
        radius: dimension === 3 ? 0.012 : 0.014,
        dimension,
        profiler,
        label: 'particles',
      });
      const integrator = createIntegrator({
        system,
        profiler,
        label: 'integrate',
        field: { wgsl: fieldWgsl },
        scalar: 'speed',
        bounds: box,
        fixedStep: 1 / 120,
        maxSteps: 4,
        fieldEvery: 2,
      });
      const grid = createNeighborGrid({ device, dimension, profiler, label: 'grid', capacity });
      let scalarCapacity = capacity;
      let scalar = makeScalar(scalarCapacity);
      const made: World = {
        scene,
        dimension,
        box,
        system,
        integrator,
        grid,
        get scalar() {
          return scalar;
        },
        set scalar(value: GPUBuffer) {
          scalar = value;
        },
        get scalarCapacity() {
          return scalarCapacity;
        },
        set scalarCapacity(value: number) {
          scalarCapacity = value;
        },
        destroy() {
          integrator.destroy();
          grid.destroy();
          system.destroy();
          scalar.destroy();
        },
      };
      // Zero-filled particles have no life: the warm-up below sends them out of the nozzle one after another.
      system.setCount(count);
      warm(made);
      return made;
    }

    function makeScalar(capacity: number): GPUBuffer {
      return device.createBuffer({
        label: 'particles scalar',
        size: capacity * 4,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
      });
    }

    function configureIntegrator(w: World): void {
      const p = w.integrator.params;
      if (w.dimension === 3) {
        p.gravity = [0, -4.5, 0];
        p.damping = 0.3;
        p.restitution = 0.35;
        p.friction = 0;
        p.respawn = {
          shape: 'disc',
          center: [nozzle[0], LANTERN_EMITTER_Y, nozzle[1]],
          extent: [0.28, 0, 0],
          direction: [0, 1, 0],
          speed: 4,
          spread: 0.5,
          speedJitter: 0.35,
          lifetime: [0.5, 2.2],
          floor: 'kill',
        };
        p.user[0] = [params.flow, 0.9, 0.15, 0];
        p.user[1] = [0, 0, 0, 0];
      } else {
        p.gravity = [0, -2.6, 0];
        p.damping = 0.15;
        p.restitution = 0.15;
        p.friction = 0.2;
        const sweep = Math.sin(clock * 0.25) * 1.4;
        p.respawn = {
          shape: 'disc',
          center: [sweep, w.box.max[1] - 0.15, 0],
          extent: [0.3, 0, 0],
          direction: [0.1, -1, 0],
          speed: 2.4,
          spread: 0.3,
          speedJitter: 0.25,
          lifetime: [2.5, 5],
        };
        const active = clock - lastHover < 1.5;
        p.user[0] = [params.flow, 1.1, 0.2, 0];
        p.user[1] = [nozzle[0], nozzle[1], 0, active ? 2.5 : 0];
      }
    }

    /** Runs a couple of simulated seconds so the first frame already shows a developed spray. */
    function warm(w: World): void {
      configureIntegrator(w);
      const encoder = device.createCommandEncoder({ label: 'particles warm-up' });
      const every = w.integrator.params.fieldEvery;
      w.integrator.params.fieldEvery = 60;
      w.integrator.step(encoder, 600, 1 / 240);
      w.integrator.params.fieldEvery = every;
      device.queue.submit([encoder.finish()]);
    }

    function ensureWorld(): World {
      const scene = params.scene as (typeof SCENES)[number];
      const count = Math.round(clamp(params.count, 1, MAX_COUNT));
      if (!world || world.scene !== scene) {
        world?.destroy();
        world = createWorld(scene, count);
        return world;
      }
      if (count !== world.system.count) {
        if (count > world.system.capacity) {
          // Headroom, so dragging the slider does not reallocate on every tick.
          world.system.resize(Math.min(MAX_COUNT, Math.max(count, Math.ceil(count * 1.25))));
        }
        if (world.system.capacity > world.scalarCapacity) {
          world.scalar.destroy();
          world.scalarCapacity = world.system.capacity;
          world.scalar = makeScalar(world.scalarCapacity);
        }
        const grew = count > world.system.count;
        world.system.setCount(count);
        if (grew) warm(world);
      }
      return world;
    }

    // --- Cameras -----------------------------------------------------------------------------------
    function updateCamera(w: World, dt: number): CameraMatrices {
      const aspect = width / height;
      if (w.dimension === 3) {
        if (params.presentation === 'lantern') {
          camera = orthographicCamera({ ...LANTERN_CAMERA, aspect });
          return camera;
        }
        orbit.update(dt, params.autoRotate);
        camera = computeCamera({
          position: orbitEye(orbitState),
          target: orbitState.target,
          fovY: fitFovY(0.75, aspect, 1.1),
          aspect,
          near: 0.1,
          far: 60,
        });
      } else {
        // Keep the whole box in view whatever the canvas shape.
        const w2 = (w.box.max[0] - w.box.min[0]) / 2;
        const h2 = (w.box.max[1] - w.box.min[1]) / 2;
        const halfHeight = Math.max(h2, w2 / aspect) * 1.06;
        camera = orthographicCamera({
          center: [(w.box.min[0] + w.box.max[0]) / 2, (w.box.min[1] + w.box.max[1]) / 2],
          halfHeight,
          aspect,
        });
      }
      return camera;
    }

    /** Where the pointer is, in the world: on the floor in 3D, on the z = 0 plane in 2D. */
    function pointerWorld(cam: CameraMatrices, dimension: 2 | 3): [number, number] | undefined {
      const { x, y } = hover.state;
      if (dimension === 2) {
        const p = unprojectPoint(cam.invViewProj, x, y, 0.5);
        return [p[0]!, p[1]!];
      }
      const near = unprojectPoint(cam.invViewProj, x, y, 0);
      const far = unprojectPoint(cam.invViewProj, x, y, 1);
      const dy = far[1]! - near[1]!;
      if (Math.abs(dy) < 1e-6) return undefined;
      const t = -near[1]! / dy;
      if (t < 0) return undefined;
      return [near[0]! + (far[0]! - near[0]!) * t, near[2]! + (far[2]! - near[2]!) * t];
    }

    function steerPointer(w: World, cam: CameraMatrices, dt: number): void {
      if (params.presentation === 'lantern' && w.dimension === 3) {
        nozzle[0] = 0;
        nozzle[1] = 0;
        return;
      }
      const over = hover.state.inside && !hover.state.down;
      if (over) lastHover = clock;
      let target: [number, number] | undefined;
      if (over) target = pointerWorld(cam, w.dimension);
      if (!target) {
        // Nobody is pointing: the nozzle wanders in a slow loop (3D); the 2D vortex stays where it was.
        if (w.dimension === 3 && clock - lastHover > 1.5) {
          target = [Math.cos(clock * 0.4) * 0.55, Math.sin(clock * 0.31) * 0.55];
        } else {
          return;
        }
      }
      const limit = w.dimension === 3 ? 1.35 : 3;
      const k = 1 - Math.exp(-dt * 7);
      nozzle[0] += (clamp(target[0], -limit, limit) - nozzle[0]) * k;
      nozzle[1] +=
        (clamp(target[1], w.dimension === 3 ? -limit : 0, w.dimension === 3 ? limit : 3) -
          nozzle[1]) *
        k;
    }

    return {
      resize(w, h, ratio) {
        width = Math.max(1, w);
        height = Math.max(1, h);
        dpr = ratio;
        renderer.resize(width, height);
      },

      frame({ delta, frame }, encoder, target) {
        const dt = Math.min(delta, 0.1);
        clock += dt;
        const w = ensureWorld();
        const { system, integrator, grid } = w;
        const cam = updateCamera(w, dt);
        steerPointer(w, cam, dt);

        // 1. Move the particles.
        system.radius = (w.dimension === 3 ? 0.012 : 0.014) * params.size;
        configureIntegrator(w);
        integrator.advance(encoder, dt);

        // 2. The neighbour grid, when something wants it.
        const view = params.view;
        const showsNeighbours = view !== VIEWS[0];
        let scalar: GPUBuffer | undefined;
        if (showsNeighbours || params.alwaysBuild) {
          grid.mode = params.reorder ? 'reorder' : 'sorted';
          grid.configure({ min: w.box.min, max: w.box.max, radius: params.radius });
          grid.build(encoder, system);
        }
        if (view === VIEWS[1]) {
          counter.count(encoder, grid, system, w.scalar);
          scalar = w.scalar;
        } else if (view === VIEWS[2]) {
          setCameraUniform(cameraUniform, cam, width, height);
          cameraUniform.upload();
          pickUniform.set({
            pointer: [hover.state.x * width, hover.state.y * height],
            active_flag: 1,
          });
          pickUniform.upload();
          device.queue.writeBuffer(pickState, 0, new Uint32Array([NONE, NONE]));
          const resources = [
            cameraUniform.buffer,
            pickUniform.buffer,
            { buffer: system.positions, size: system.count * 16 },
            pickState,
          ];
          pickDistance.runCount(encoder, resources, system.count);
          pickIndex.runCount(encoder, resources, system.count);
          counter.mark(encoder, grid, system, pickState, w.scalar);
          scalar = w.scalar;
        }

        const glow = params.presentation === 'lantern' ? ensureLantern() : undefined;
        const activeRenderer = glow?.renderer ?? renderer;
        const particleTarget = glow?.chain.scene.view ?? target;

        // 3. The gallery backdrop; the lantern clears its linear scene to the page colour instead.
        if (!glow) {
          setCameraUniform(cameraUniform, cam, width, height);
          cameraUniform.upload();
          sceneUniform.set({
            top: [0.012, 0.045, 0.08],
            bottom: [0.002, 0.012, 0.03],
            line_strength: 0.7,
            box_min: w.box.min,
            box_max: w.box.max,
          });
          sceneUniform.upload();
          const backdrop = encoder.beginRenderPass({
            label: 'particles backdrop',
            colorAttachments: [
              {
                view: target,
                loadOp: 'clear',
                clearValue: { r: 0, g: 0, b: 0, a: params.background ? 1 : 0 },
                storeOp: 'store',
              },
            ],
            timestampWrites: profiler.pass('backdrop'),
          });
          if (params.background) {
            backdrop.setPipeline(backdropPipeline);
            backdrop.setBindGroup(0, backdropGroup);
            backdrop.draw(3);
            backdrop.setPipeline(linesPipeline);
            backdrop.setBindGroup(0, linesGroup);
            backdrop.draw(24);
          }
          backdrop.end();
        }

        const style = activeRenderer.style;
        style.mode = params.render as (typeof RENDERS)[number];
        style.intensity = params.brightness * (glow ? params.glowEnergy : 1);
        style.minPixels = 0.7 * dpr;
        if (view === VIEWS[0]) {
          style.palette = params.palette === 'orange' ? ORANGE_PALETTE : SPEED_PALETTE;
          style.scalarRange = [0, 6];
        } else if (view === VIEWS[1]) {
          style.palette = HEAT_PALETTE;
          style.scalarRange = [0, params.heatScale];
        } else {
          style.palette = PICK_PALETTE;
          style.scalarRange = [0, 2];
        }
        activeRenderer.draw(encoder, system.particles, cam, particleTarget, {
          width,
          height,
          clear: glow
            ? { r: params.pageColor[0], g: params.pageColor[1], b: params.pageColor[2], a: 1 }
            : null,
          scalar,
        });
        glow?.chain.encode(
          encoder,
          target,
          {
            bloom: {
              intensity: params.bloomIntensity,
              threshold: 0.8,
              knee: 0.4,
              radius: 0.7,
              levels: 5,
            },
            output: { hdr, headroom: params.headroom, tonemap: 'none', dither: 0, frame },
          },
          profiler,
        );
      },

      destroy() {
        world?.destroy();
        orbit.destroy();
        hover.destroy();
        renderer.destroy();
        lantern?.renderer.destroy();
        lantern?.chain.destroy();
        counter.destroy();
        pickDistance.destroy();
        pickIndex.destroy();
        cameraUniform.destroy();
        sceneUniform.destroy();
        pickUniform.destroy();
        pickState.destroy();
      },
    };
  },
});

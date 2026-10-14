import { trackDomRects, type DomRectsOptions } from '../../gpu/dom-rects.ts';
import { createFullscreenPipeline, renderFullscreen } from '../../gpu/fullscreen.ts';
import { trackPointers, type TrackedPointer } from '../../gpu/pointer.ts';
import {
  createFluid,
  type Fluid,
  type FluidPressureSolver,
  type FluidScheme,
} from '../../gpu/sim/fluid.ts';
import { createFluidStirrer, type FluidStirMode } from '../../gpu/sim/fluid-stir.ts';
import {
  OBSTACLE_CHUNKS,
  createObstacleShapes,
  pickObstacle,
  shapeFromRect,
  type ObstacleShape,
} from '../../gpu/sim/obstacle-shapes.ts';
import { createUniformBuffer, defineUniformStruct } from '../../gpu/uniforms.ts';
import { createDemoElements, type DemoElements } from '../ripple-dom/demo.ts';
import { defineEffect } from '../types.ts';
import displayShader from './display.wgsl';
import inletShader from './inlet.wgsl';
import { meta } from './meta.ts';
import { SCENES, sceneShapes, type Scene } from './scenes.ts';

const DisplayParams = defineUniformStruct('DisplayParams', {
  ground: 'vec3f',
  aspect: 'f32',
  body_edge: 'vec3f',
  skin: 'f32',
  body_core: 'vec3f',
  pixel: 'f32',
  gain: 'f32',
  debug_gain: 'f32',
  vignette: 'f32',
  view: 'u32',
  mode: 'u32',
  dom_start: 'u32',
});

const InletParams = defineUniformStruct('InletParams', {
  aspect: 'f32',
  width: 'f32',
  rows: 'f32',
  amount: 'f32',
  phase: 'f32',
  pulse: 'f32',
  kind: 'u32',
  mode: 'u32',
});

const RESOLUTIONS = ['96', '128', '144', '192', '256'] as const;
const DYE_PATTERNS = ['rows', 'bands', 'timelines', 'none'] as const;
const GROUNDS = ['dark', 'light'] as const;
const POINTER_MODES = [
  'drag shapes, stir elsewhere',
  'drag shapes only',
  'stir only',
  'off',
] as const;
const TOUCH_MODES = ['scrolls the page', 'grabs and stirs'] as const;
const SOLVERS = ['jacobi', 'sor'] as const;
const VIEWS = ['dye', 'velocity', 'vorticity', 'pressure', 'mask'] as const;

/** A dragged shape follows the pointer no faster than this, image heights per second. */
const MAX_DRAG_SPEED = 1.6;
/** Margin, image heights, within which a press grabs a shape. */
const GRAB_MARGIN = 0.025;
/** Dye is drawn into the band this wide at the inlet, image heights. */
const INLET_WIDTH = 0.07;
const DYE_DETAILS = ['1', '2', '3', '4'] as const;
/** Seconds between the timelines released at the inlet. */
const TIMELINE_PERIOD = 0.45;

export const effect = defineEffect({
  meta,
  params: {
    scene: {
      type: 'select',
      default: 'auto',
      options: SCENES,
      label: "Scene (auto: the page's own elements if it has any, else the cylinder)",
    },
    size: {
      type: 'number',
      default: 0.2,
      min: 0.06,
      max: 0.45,
      step: 0.005,
      label: 'Size (canvas heights: cylinder diameter, plate half-length)',
    },
    angle: {
      type: 'number',
      default: 10,
      min: -35,
      max: 35,
      step: 0.5,
      label: 'Wing angle of attack (degrees)',
    },
    speed: {
      type: 'number',
      default: 0.9,
      min: 0,
      max: 2.5,
      step: 0.01,
      label: 'Wind speed (canvas heights / s)',
    },
    viscosity: {
      type: 'number',
      default: 0.0002,
      min: 0,
      max: 0.01,
      step: 0.00005,
      label: 'Viscosity (heights² / s; lower = more turbulent)',
    },
    vorticity: {
      type: 'number',
      default: 3,
      min: 0,
      max: 30,
      step: 0.5,
      label: 'Vorticity confinement',
    },
    dissipation: {
      type: 'number',
      default: 0.05,
      min: 0,
      max: 2,
      step: 0.01,
      label: 'Flow dissipation (1 / s)',
    },
    dye: {
      type: 'select',
      default: 'rows',
      options: DYE_PATTERNS,
      label: 'Dye released at the inlet',
    },
    dyeLife: {
      type: 'number',
      default: 0.12,
      min: 0,
      max: 1,
      step: 0.01,
      label: 'Dye fade (1 / s)',
    },
    gain: { type: 'number', default: 1.3, min: 0.2, max: 4, step: 0.01, label: 'Dye strength' },
    ground: { type: 'select', default: 'dark', options: GROUNDS },
    dyeDetail: {
      type: 'select',
      default: '2',
      options: DYE_DETAILS,
      label: 'Dye texels per fluid cell (re-creates the fluid)',
    },
    resolution: {
      type: 'select',
      default: '144',
      options: RESOLUTIONS,
      label: 'Fluid cells (short side; re-creates the fluid)',
    },
    pressureSolver: {
      type: 'select',
      default: 'jacobi',
      options: SOLVERS,
      label: 'Pressure solver',
    },
    pressureIterations: {
      type: 'number',
      default: 50,
      min: 4,
      max: 160,
      step: 1,
      label: 'Pressure iterations',
    },
    pointer: { type: 'select', default: 'drag shapes, stir elsewhere', options: POINTER_MODES },
    touch: {
      type: 'select',
      default: 'scrolls the page',
      options: TOUCH_MODES,
      label: 'Touch drags',
    },
    view: {
      type: 'select',
      default: 'dye',
      options: VIEWS,
      label: "View (mask: the solver's cells in red, the exact outlines in green)",
    },
    debugGain: {
      type: 'number',
      default: 1,
      min: 0.1,
      max: 10,
      step: 0.1,
      label: 'Debug view gain',
    },
    stepsPerFrame: {
      type: 'number',
      default: 0,
      min: 0,
      max: 32,
      step: 1,
      label: 'Steps per frame (0 = by clock; for benchmarks)',
    },
  },
  presets: {
    'vortex street': {
      scene: 'cylinder',
      size: 0.2,
      speed: 0.9,
      viscosity: 0.0002,
      vorticity: 3,
      dissipation: 0.05,
      dye: 'rows',
      view: 'dye',
      resolution: '144',
    },
    'vortex street (vorticity view)': {
      scene: 'cylinder',
      size: 0.2,
      speed: 0.9,
      viscosity: 0.0002,
      vorticity: 3,
      dye: 'none',
      view: 'vorticity',
      resolution: '144',
      debugGain: 1.4,
    },
    'wing at an angle': {
      scene: 'wing',
      size: 0.18,
      angle: 14,
      speed: 1.1,
      viscosity: 0.0002,
      vorticity: 2,
      dye: 'rows',
      view: 'dye',
    },
    'peg slalom': {
      scene: 'slalom',
      size: 0.2,
      speed: 0.8,
      viscosity: 0.0003,
      vorticity: 5,
      dye: 'bands',
      view: 'dye',
    },
    'plates and funnel': {
      scene: 'plates',
      size: 0.2,
      speed: 0.9,
      viscosity: 0.0003,
      vorticity: 4,
      dye: 'timelines',
      view: 'dye',
    },
    'around the page': {
      scene: 'page elements',
      speed: 0.5,
      viscosity: 0.0005,
      vorticity: 4,
      dye: 'rows',
      gain: 1.1,
      resolution: '128',
      view: 'dye',
    },
  },

  async create({ device, format, canvas, params, profiler }) {
    const displayUniforms = createUniformBuffer(device, DisplayParams, 'fluid-obstacles display');
    const inletUniforms = createUniformBuffer(device, InletParams, 'fluid-obstacles inlet');
    const pointers = trackPointers(canvas);
    const stirrer = createFluidStirrer();
    const domOptions: DomRectsOptions = { outsetPx: 2, epsilon: 1e-3 };
    const dom = trackDomRects(canvas, domOptions);
    let demo: DemoElements | undefined;

    const sampler = device.createSampler({
      label: 'fluid-obstacles sampler',
      magFilter: 'linear',
      minFilter: 'linear',
      addressModeU: 'clamp-to-edge',
      addressModeV: 'clamp-to-edge',
    });
    const displayPipeline = createFullscreenPipeline(device, {
      fragment: {
        label: 'fluid-obstacles/display.wgsl',
        code: displayShader.code,
        chunks: {
          ...displayShader.chunks,
          ...OBSTACLE_CHUNKS,
          'display_params.wgsl': DisplayParams.wgsl,
        },
      },
      format,
    });
    const inletPipeline = createFullscreenPipeline(device, {
      fragment: {
        label: 'fluid-obstacles/inlet.wgsl',
        code: inletShader.code,
        chunks: { ...inletShader.chunks, 'inlet_params.wgsl': InletParams.wgsl },
      },
      format: 'rgba16float',
    });

    let width = Math.max(1, canvas.width || 1);
    let height = Math.max(1, canvas.height || 1);
    let simAspect = width / height;
    const cells = (shortSide: number, aspect: number) =>
      Math.round(shortSide * Math.max(aspect, 1 / aspect));

    const fluid: Fluid = createFluid({
      device,
      resolution: cells(Number(params.resolution), simAspect),
      dyeResolution: cells(Number(params.resolution) * Number(params.dyeDetail), simAspect),
      width,
      height,
      profiler,
      label: 'fluid-obstacles',
      // Open edges: the wind leaves at the right (and clear water enters elsewhere). The inflow is a
      // force band (see obstacle-drive.wgsl), because air cells outside the grid carry no velocity.
      params: { boundary: 'open', dyeScheme: 'maccormack', velocityScheme: 'maccormack' },
    });
    const shapes = createObstacleShapes({ device, fluid, profiler, label: 'fluid-obstacles' });
    shapes.attach();

    // --- The dye inlet (a `dye` hook) -------------------------------------------------------------------
    const inletGroups = new Map<GPUTextureView, GPUBindGroup>();
    fluid.hooks.dye = ({ encoder, read, write }) => {
      let group = inletGroups.get(read);
      if (!group) {
        if (inletGroups.size > 8) inletGroups.clear();
        group = device.createBindGroup({
          label: 'fluid-obstacles inlet',
          layout: inletPipeline.getBindGroupLayout(0),
          entries: [
            { binding: 0, resource: { buffer: inletUniforms.buffer } },
            { binding: 1, resource: read },
          ],
        });
        inletGroups.set(read, group);
      }
      renderFullscreen(encoder, write, inletPipeline, [group], {
        label: 'fluid-obstacles inlet',
        timestampWrites: profiler.pass('dye inlet'),
      });
      return true;
    };

    // --- Scene ----------------------------------------------------------------------------------------
    let analytic: ObstacleShape[] = [];
    let domShapes: ObstacleShape[] = [];
    let sceneKey = '';
    let effectiveScene: Exclude<Scene, 'auto'> = 'cylinder';
    let analyticDirty = true;
    let domKey = '';
    let kick = 0;
    /** Pointer id -> the shape it holds and where it holds it, in uv. */
    const grabs = new Map<number, { index: number; dx: number; dy: number }>();
    let hovering = false;
    let wind = 0;
    let pulseClock = 0;
    let hueClock = 0;
    const displayGroups = new Map<string, GPUBindGroup>();
    let groupGeneration = fluid.generation;

    function composeShapes() {
      shapes.set([...analytic, ...domShapes]);
    }

    function rebuildScene(aspect: number) {
      analytic = sceneShapes(effectiveScene, {
        size: params.size,
        angle: params.angle,
        aspect,
      });
      grabs.clear();
      analyticDirty = true;
      kick = analytic.length > 0 ? 1 : 0;
    }

    /** The page's own boxes as shapes; returns whether they changed. */
    function pageShapes(delta: number, aspect: number): boolean {
      domOptions.epsilon = 0.25 / Math.max(1, fluid.grid.width);
      const measured = dom.update(delta);
      const demoCounted = demo && dom.root?.contains(demo.container) ? demo.count : 0;
      const own = measured.found - demoCounted;
      const wantDemo = params.scene === 'page elements' && own <= 0;
      if (wantDemo && !demo) {
        demo = createDemoElements(canvas);
        if (demo) dom.invalidate();
      } else if (!wantDemo && demo) {
        demo.destroy();
        demo = undefined;
        dom.invalidate();
      }
      // `auto` is the page when it has elements of its own, and the cylinder otherwise.
      const resolved: Exclude<Scene, 'auto'> =
        params.scene === 'auto'
          ? own > 0
            ? 'page elements'
            : 'cylinder'
          : (params.scene as Exclude<Scene, 'auto'>);
      if (resolved !== effectiveScene) {
        effectiveScene = resolved;
        sceneKey = '';
      }
      if (effectiveScene !== 'page elements') {
        const had = domShapes.length > 0;
        domShapes = [];
        return had;
      }
      let moving = false;
      const next: ObstacleShape[] = measured.obstacles.map((o) => {
        if (o.motion) moving = true;
        return shapeFromRect(o.rect, aspect, o.motion);
      });
      const key = next
        .map(
          (s) =>
            `${s.x.toFixed(4)},${s.y.toFixed(4)},${s.size[0].toFixed(4)},${s.size[1].toFixed(4)}`,
        )
        .join('|');
      // A box that is moving changes the table every frame (its velocity does); one at rest only when it moves.
      if (key === domKey && !moving && !measured.changed) return false;
      domKey = key;
      domShapes = next;
      return true;
    }

    // --- Input ------------------------------------------------------------------------------------------
    function handlePointers(
      list: TrackedPointer[],
      aspect: number,
      delta: number,
    ): TrackedPointer[] {
      const mode = params.pointer;
      const canGrab = mode === 'drag shapes, stir elsewhere' || mode === 'drag shapes only';
      const dt = Math.min(Math.max(delta, 1 / 240), 0.1);
      const free: TrackedPointer[] = [];
      hovering = false;
      const moved = new Set<number>();
      for (const p of list) {
        if (canGrab && p.justPressed) {
          const index = pickObstacle(analytic, p.x, p.y, aspect, GRAB_MARGIN);
          if (index >= 0) {
            const s = analytic[index]!;
            grabs.set(p.id, { index, dx: s.x - p.x, dy: s.y - p.y });
          }
        }
        const grab = grabs.get(p.id);
        if (grab && (p.down || p.justReleased)) {
          if (p.down) {
            const s = analytic[grab.index]!;
            // Follow the pointer with a spring, no faster than MAX_DRAG_SPEED: a shape that teleports
            // would delete the fluid it lands on and start nothing.
            const tx = p.x + grab.dx - s.x;
            const ty = p.y + grab.dy - s.y;
            const distance = Math.hypot(tx * aspect, ty);
            const step = Math.min(distance, MAX_DRAG_SPEED * dt, distance * 14 * dt);
            if (distance > 1e-6) {
              const k = step / distance;
              s.x = Math.min(1.1, Math.max(-0.1, s.x + tx * k));
              s.y = Math.min(1.1, Math.max(-0.1, s.y + ty * k));
              s.velocity = [((tx * k) / dt) * aspect, (ty * k) / dt];
              moved.add(grab.index);
            }
            analyticDirty = true;
          }
          if (p.justReleased || !p.down) grabs.delete(p.id);
          continue;
        }
        if (grab) grabs.delete(p.id);
        if (
          p.inside &&
          !p.down &&
          canGrab &&
          pickObstacle(analytic, p.x, p.y, aspect, GRAB_MARGIN) >= 0
        ) {
          hovering = true;
        }
        free.push(p);
      }
      // A shape nobody is moving stands still: its velocity goes back to zero (one more upload).
      analytic.forEach((s, i) => {
        if (!moved.has(i) && s.velocity && (s.velocity[0] !== 0 || s.velocity[1] !== 0)) {
          s.velocity = [0, 0];
          analyticDirty = true;
        }
      });
      return free;
    }

    function sync(delta: number) {
      const p = fluid.params;
      p.boundary = 'open';
      p.pressureSolver = params.pressureSolver as FluidPressureSolver;
      p.pressureIterations = params.pressureIterations;
      p.velocityScheme = 'maccormack' as FluidScheme;
      p.vorticity = params.vorticity;
      p.viscosity = params.viscosity;
      p.velocityDissipation = params.dissipation;
      p.dyeDissipation = params.dyeLife;
      // The wind eases in over the first second, so a scene change does not slam the fluid.
      wind += (1 - wind) * Math.min(1, 1.5 * delta);
      shapes.drive.inlet = {
        velocity: [params.speed * wind, 0],
        width: INLET_WIDTH,
        strength: 0.45,
      };
      shapes.drive.band = 3;
      shapes.drive.coupling = 1;
    }

    return {
      resize(w, h) {
        width = Math.max(1, Math.round(w));
        height = Math.max(1, Math.round(h));
        const aspect = width / height;
        if (Math.abs(Math.log(aspect / simAspect)) > 0.06) {
          simAspect = aspect;
          fluid.resize(width, height);
          sceneKey = '';
        }
        dom.invalidate();
      },

      frame({ delta }, encoder, target) {
        const aspect = width / height;
        const long = Number(params.resolution);
        fluid.setResolution(
          cells(long, simAspect),
          cells(long * Number(params.dyeDetail), simAspect),
        );
        sync(delta);

        // 1. The page, read before anything is written.
        let domChanged = false;
        if (params.scene === 'auto' || params.scene === 'page elements') {
          domChanged = pageShapes(delta, aspect);
        } else {
          if (demo) {
            demo.destroy();
            demo = undefined;
            dom.invalidate();
          }
          if (effectiveScene !== params.scene) {
            effectiveScene = params.scene as Exclude<Scene, 'auto'>;
            sceneKey = '';
          }
          if (domShapes.length > 0) {
            domShapes = [];
            domChanged = true;
          }
        }

        // 2. The scene.
        const key = `${effectiveScene}:${params.size}:${params.angle}:${Math.round(aspect * 10)}`;
        if (key !== sceneKey) {
          sceneKey = key;
          rebuildScene(aspect);
        }

        // 3. Pointers: drag shapes, stir elsewhere.
        pointers.setCapture(params.touch === 'grabs and stirs');
        const free = handlePointers(pointers.poll(), aspect, delta);
        stirrer.stir(fluid, free, aspect, delta, {
          mode: (params.pointer === 'drag shapes, stir elsewhere' || params.pointer === 'stir only'
            ? 'press only'
            : 'off') as FluidStirMode,
          radius: 0.045,
          force: 0.9,
          dyeAmount: 0,
          colour: () => null,
          puff: 0,
        });
        if (canvas.style) canvas.style.cursor = hovering || grabs.size > 0 ? 'grab' : '';

        if (analyticDirty || domChanged) {
          composeShapes();
          analyticDirty = false;
        }

        // A little nudge off the axis when a scene starts, so the wake of a symmetric body does not
        // wait for numerical noise to start shedding.
        if (kick > 0 && wind > 0.5) {
          const s = analytic[0];
          if (s) fluid.addForce([s.x + 0.12, s.y], [s.x + 0.14, s.y + 0.015], 0.035, [0, 0.35]);
          kick = 0;
        }

        // 4. The inlet's dye pattern.
        pulseClock += delta;
        let pulse = 0;
        if (pulseClock >= TIMELINE_PERIOD) {
          pulseClock = 0;
          pulse = 1;
        }
        hueClock += delta * 0.04;
        inletUniforms.set({
          aspect,
          width: INLET_WIDTH,
          rows: 14,
          amount: 1,
          phase: hueClock,
          pulse,
          kind: Math.max(0, DYE_PATTERNS.indexOf(params.dye as (typeof DYE_PATTERNS)[number])),
          mode: params.ground === 'light' ? 1 : 0,
        });
        inletUniforms.upload();

        // 5. The solver.
        shapes.rasterize(encoder);
        if (params.stepsPerFrame > 0) fluid.advance(encoder, params.stepsPerFrame);
        else fluid.update(encoder, delta);

        // 6. Shading.
        const light = params.ground === 'light';
        displayUniforms.set({
          ground: light ? [0.93, 0.92, 0.88] : [0.012, 0.02, 0.034],
          aspect,
          body_edge: light ? [0.3, 0.32, 0.36] : [0.62, 0.68, 0.74],
          skin: 0.45 / fluid.grid.height,
          body_core: light ? [0.08, 0.09, 0.11] : [0.2, 0.24, 0.3],
          pixel: 1 / height,
          gain: params.gain,
          debug_gain: params.debugGain,
          vignette: light ? 0.1 : 0.3,
          view: Math.max(0, VIEWS.indexOf(params.view as (typeof VIEWS)[number])),
          mode: light ? 1 : 0,
          dom_start: analytic.length,
        });
        displayUniforms.upload();
        const groupKey = `${fluid.generation}:${fluid.dye.slot}`;
        let group = displayGroups.get(groupKey);
        if (!group) {
          if (groupGeneration !== fluid.generation) {
            displayGroups.clear();
            groupGeneration = fluid.generation;
          }
          group = device.createBindGroup({
            label: 'fluid-obstacles display',
            layout: displayPipeline.getBindGroupLayout(0),
            entries: [
              { binding: 0, resource: { buffer: displayUniforms.buffer } },
              { binding: 1, resource: fluid.dye.view },
              { binding: 2, resource: sampler },
              { binding: 3, resource: fluid.fieldsView },
              { binding: 4, resource: fluid.obstacles.view },
              { binding: 5, resource: { buffer: shapes.table } },
            ],
          });
          displayGroups.set(groupKey, group);
        }
        renderFullscreen(encoder, target, displayPipeline, [group], {
          timestampWrites: profiler.pass('shade'),
        });
      },

      destroy() {
        demo?.destroy();
        demo = undefined;
        dom.destroy();
        pointers.destroy();
        if (canvas.style) canvas.style.cursor = '';
        shapes.destroy();
        fluid.hooks.dye = undefined;
        fluid.destroy();
        displayUniforms.destroy();
        inletUniforms.destroy();
      },
    };
  },
});

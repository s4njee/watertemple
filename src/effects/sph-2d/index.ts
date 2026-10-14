import {
  createLiquid2d,
  type Liquid2d,
  type Liquid2dDomain,
  type Liquid2dParams,
} from '../../gpu/liquid-2d/index.ts';
import { createFullscreenPipeline, renderFullscreen } from '../../gpu/fullscreen.ts';
import { trackPointers, type TrackedPointer } from '../../gpu/pointer.ts';
import {
  createSph,
  type SphObstacle,
  type SphPointer,
  type SphScene,
  type SphSolver,
} from '../../gpu/sim/sph.ts';
import { createRenderTarget, type RenderTarget } from '../../gpu/targets.ts';
import { createUniformBuffer, defineUniformStruct, uniformArray } from '../../gpu/uniforms.ts';
import { defineEffect } from '../types.ts';
import backdropWgsl from './backdrop.wgsl';
import { meta } from './meta.ts';
import overlayWgsl from './overlay.wgsl';

// The 2D SPH demo: a tank of water simulated by src/gpu/sim/sph.ts and drawn by src/gpu/liquid-2d/ over a tiled wall.
// The solver steps on a fixed clock (several sub-steps per frame, see the notes); the renderer splats the particles
// into a field and shades its iso-surface as water. This file is the scenes, the pointer and the plumbing.

const SCENES = ['dam break', 'pour', 'two blobs'] as const;
const VIEWS = ['liquid', 'discs', 'density', 'pressure', 'velocity'] as const;
const WATERS = [
  'clear tropical',
  'coastal',
  'murky green',
  'tea',
  'glacial milk',
  'swimming pool',
  'open ocean',
] as const;
const POINTER_MODES = ['push', 'pull'] as const;

/** The tank, metres. Gravity 9.81 looks right at this size. */
const TANK = { min: [0, 0] as [number, number], max: [2, 1.2] as [number, number] };
const MAX_COUNT = 60_000;
/** Spare capacity for the blocks a tap drops. */
const HEADROOM = 1.3;

const BackdropParams = defineUniformStruct('BackdropParams', {
  center: 'vec2f',
  half_height: 'f32',
  aspect: 'f32',
  box_min: 'vec2f',
  box_max: 'vec2f',
  height_px: 'f32',
});

const OverlayParams = defineUniformStruct('OverlayParams', {
  center: 'vec2f',
  half_height: 'f32',
  aspect: 'f32',
  box_min: 'vec2f',
  box_max: 'vec2f',
  height_px: 'f32',
  obstacle_count: 'u32',
  pointer_count: 'u32',
  encode_srgb: 'u32',
  obstacles: uniformArray('vec4f', 8),
  pointers: uniformArray('vec4f', 4),
});

const clamp = (x: number, lo: number, hi: number) => Math.min(hi, Math.max(lo, x));

interface SceneDescription {
  scene: SphScene;
  /** Fluid area in m^2 the particle count is spread over (fixes the spacing). */
  area: number;
}

/** The three scenes. `count` is the particles to start with (the pour reaches it by the end). */
function describeScene(name: (typeof SCENES)[number], count: number): SceneDescription {
  if (name === 'pour') {
    const shelf: SphObstacle = {
      shape: 'box',
      center: [0.62, 0.6],
      halfSize: [0.3, 0.025],
      cornerRadius: 0.02,
    };
    const rock: SphObstacle = { shape: 'circle', center: [1.4, 0.28], radius: 0.2 };
    const area = 0.8;
    return {
      area,
      scene: {
        bounds: TANK,
        spacing: Math.sqrt(area / count),
        blocks: [],
        obstacles: [shelf, rock],
        emitter: { position: [0.5, 1.13], direction: [0, -1], width: 0.1, speed: 2.4 },
      },
    };
  }
  if (name === 'two blobs') {
    const area = 2 * Math.PI * 0.22 * 0.22;
    return {
      area,
      scene: {
        bounds: TANK,
        spacing: Math.sqrt(area / count),
        blocks: [
          { center: [0.62, 0.6], radius: 0.22, velocity: [0.4, 0] },
          { center: [1.38, 0.62], radius: 0.22, velocity: [-0.4, 0] },
        ],
        obstacles: [],
        emitter: null,
      },
    };
  }
  const area = 0.55 * 0.95;
  return {
    area,
    scene: {
      bounds: TANK,
      spacing: Math.sqrt(area / count),
      blocks: [{ min: [0, 0], max: [0.55, 0.95] }],
      obstacles: [],
      emitter: null,
    },
  };
}

interface TapState {
  x: number;
  y: number;
  time: number;
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
      label: 'Restart the scene (flip this switch)',
    },
    count: {
      type: 'number',
      default: 3000,
      min: 1000,
      max: MAX_COUNT,
      step: 500,
      label: 'Particles (restarts; sub-steps grow with their square root)',
    },
    view: { type: 'select', default: VIEWS[0], options: VIEWS, label: 'View' },
    gravity: {
      type: 'number',
      default: 9.81,
      min: 0,
      max: 30,
      step: 0.1,
      label: 'Gravity (m/s²)',
    },
    tilt: {
      type: 'number',
      default: 0,
      min: -90,
      max: 90,
      step: 1,
      label: 'Gravity tilt (degrees: the tank rolls)',
    },
    stiffness: {
      type: 'number',
      default: 4,
      min: 2,
      max: 10,
      step: 0.1,
      label: 'Stiffness (speed of sound / free-fall speed; more is stiffer and slower)',
    },
    viscosity: {
      type: 'number',
      default: 0.05,
      min: 0,
      max: 1,
      step: 0.005,
      label: 'Viscosity (m²/s)',
    },
    surfaceTension: {
      type: 'number',
      default: 0,
      min: 0,
      max: 200,
      step: 1,
      label: 'Surface tension (m/s²)',
    },
    subSteps: {
      type: 'number',
      default: 0,
      min: 0,
      max: 128,
      step: 1,
      label: 'Sub-steps per 1/60 s (0: as many as the time-step rule asks)',
    },
    pointerMode: {
      type: 'select',
      default: POINTER_MODES[0],
      options: POINTER_MODES,
      label: 'Drag does',
    },
    pointerSize: {
      type: 'number',
      default: 0.16,
      min: 0.05,
      max: 0.5,
      step: 0.01,
      label: 'Pointer radius (m)',
    },
    pointerStrength: {
      type: 'number',
      default: 30,
      min: 0,
      max: 100,
      step: 1,
      label: 'Pointer strength',
    },
    captureTouch: {
      type: 'boolean',
      default: false,
      label: 'Touch drags push the water (and stop the page scrolling over the canvas)',
    },
    blob: {
      type: 'number',
      default: 2.6,
      min: 1.4,
      max: 4,
      step: 0.05,
      label: 'Blob radius (in particle radii)',
    },
    threshold: {
      type: 'number',
      default: 0.5,
      min: 0.15,
      max: 0.9,
      step: 0.01,
      label: 'Surface threshold',
    },
    water: { type: 'select', default: WATERS[0], options: WATERS, label: 'Water type' },
    refraction: {
      type: 'number',
      default: 1,
      min: 0,
      max: 3,
      step: 0.05,
      label: 'Refraction',
    },
    depth: {
      type: 'number',
      default: 3,
      min: 0.5,
      max: 16,
      step: 0.1,
      label: 'Water depth (how fast it darkens)',
    },
    foam: { type: 'number', default: 0.6, min: 0, max: 1, step: 0.01, label: 'Foam (fast water)' },
  },
  presets: {
    'dam break': {},
    pour: { scene: SCENES[1], count: 10000, surfaceTension: 6 },
    'zero-g blobs': {
      scene: SCENES[2],
      gravity: 0,
      surfaceTension: 200,
      viscosity: 0.02,
      count: 6000,
      foam: 0,
    },
    honey: { viscosity: 0.6, stiffness: 5, foam: 0, water: WATERS[3] },
    'many particles': { count: 40000 },
    'plain discs': { view: VIEWS[1] },
    pressure: { view: VIEWS[3] },
  },

  async create({ device, format, canvas, params, profiler }) {
    // --- Plumbing ---------------------------------------------------------------------------------
    const backdropUniform = createUniformBuffer(device, BackdropParams, 'sph-2d backdrop');
    const overlayUniform = createUniformBuffer(device, OverlayParams, 'sph-2d overlay');
    const backdropPipeline = createFullscreenPipeline(device, {
      label: 'sph-2d backdrop',
      fragment: {
        label: 'sph-2d/backdrop.wgsl',
        code: backdropWgsl.code,
        chunks: { ...backdropWgsl.chunks, 'backdrop_params.wgsl': BackdropParams.wgsl },
      },
      format: 'rgba16float',
    });
    const overlayPipeline = createFullscreenPipeline(device, {
      label: 'sph-2d overlay',
      fragment: {
        label: 'sph-2d/overlay.wgsl',
        code: overlayWgsl.code,
        chunks: { ...overlayWgsl.chunks, 'overlay_params.wgsl': OverlayParams.wgsl },
      },
      format,
      blend: {
        color: { srcFactor: 'one', dstFactor: 'one-minus-src-alpha', operation: 'add' },
        alpha: { srcFactor: 'one', dstFactor: 'one-minus-src-alpha', operation: 'add' },
      },
    });
    const backdropGroup = device.createBindGroup({
      layout: backdropPipeline.getBindGroupLayout(0),
      entries: [{ binding: 0, resource: { buffer: backdropUniform.buffer } }],
    });
    const overlayGroup = device.createBindGroup({
      layout: overlayPipeline.getBindGroupLayout(0),
      entries: [{ binding: 0, resource: { buffer: overlayUniform.buffer } }],
    });

    let width = 1;
    let height = 1;
    let backdrop: RenderTarget | undefined;
    let backdropDrawn = false;
    let liquid: Liquid2d | undefined;
    let liquidScale = 1;

    const pointers = trackPointers(canvas, { capture: params.captureTouch });
    const taps = new Map<number, TapState>();
    let clock = 0;

    // --- The simulation --------------------------------------------------------------------------
    let solver: SphSolver | undefined;
    let sceneKey = '';
    let lastRestart = params.restart;

    function gravityVector(): [number, number] {
      const tilt = (params.tilt * Math.PI) / 180;
      return [params.gravity * Math.sin(tilt), -params.gravity * Math.cos(tilt)];
    }

    function ensureSolver(): SphSolver {
      const count = Math.round(clamp(params.count, 500, MAX_COUNT));
      const name = params.scene as (typeof SCENES)[number];
      const key = `${name}|${count}`;
      const restart = params.restart !== lastRestart;
      lastRestart = params.restart;
      if (key === sceneKey && !restart && solver) return solver;
      sceneKey = key;
      const described = describeScene(name, count);
      const capacity = Math.min(MAX_COUNT * HEADROOM, Math.ceil(count * HEADROOM));
      if (!solver) {
        solver = createSph({
          device,
          capacity,
          bounds: TANK,
          spacing: described.scene.spacing!,
          params: { gravity: gravityVector() },
          profiler,
          label: 'sph-2d',
        });
      } else {
        solver.setCapacity(capacity);
      }
      solver.reset(described.scene);
      return solver;
    }

    function syncParams(sph: SphSolver): void {
      const p = sph.params;
      p.gravity = gravityVector();
      p.stiffness = params.stiffness;
      p.viscosity = params.viscosity;
      p.surfaceTension = params.surfaceTension;
      p.subSteps = Math.round(params.subSteps);
      const view = params.view;
      p.scalar = view === 'density' ? 'density' : view === 'pressure' ? 'pressure' : 'speed';
      // A soft start: the column of a quiet scene should not ring when gravity switches on.
      p.gravityRamp = 0;
    }

    // --- Camera --------------------------------------------------------------------------------
    function domain(): Liquid2dDomain {
      const aspect = width / height;
      const halfWidth = (TANK.max[0] - TANK.min[0]) / 2;
      const halfHeight = (TANK.max[1] - TANK.min[1]) / 2;
      return {
        center: [(TANK.min[0] + TANK.max[0]) / 2, (TANK.min[1] + TANK.max[1]) / 2],
        halfHeight: Math.max(halfHeight, halfWidth / aspect) * 1.06,
      };
    }

    const toWorld = (u: number, v: number, d: Liquid2dDomain): [number, number] => {
      const aspect = width / height;
      return [
        d.center[0] + (2 * u - 1) * d.halfHeight * aspect,
        d.center[1] + (1 - 2 * v) * d.halfHeight,
      ];
    };

    /** Reads the pointers: pushes and pulls while pressed, a tap drops a block. Returns the rings to draw. */
    function handlePointers(sph: SphSolver, d: Liquid2dDomain): number[] {
      const aspect = width / height;
      const tracked: TrackedPointer[] = pointers.poll();
      const rings: number[] = [];
      const active: SphPointer[] = [];
      const makePointer = (p: TrackedPointer): SphPointer => {
        const [x, y] = toWorld(p.x, p.y, d);
        const vx = p.vx * 2 * d.halfHeight * aspect;
        const vy = -p.vy * 2 * d.halfHeight;
        const speed = Math.hypot(vx, vy);
        const pushing = params.pointerMode === 'push';
        // A push scales with how fast the pointer moves: a held, still pointer does nothing.
        const strength = pushing
          ? params.pointerStrength * clamp(speed / 0.5, 0, 1)
          : params.pointerStrength * 0.6;
        return {
          x,
          y,
          vx,
          vy,
          radius: params.pointerSize,
          strength,
          mode: pushing ? 'push' : 'pull',
        };
      };
      for (const p of tracked) {
        if (p.justPressed) taps.set(p.id, { x: p.x, y: p.y, time: clock });
        const tap = taps.get(p.id);
        if (tap && Math.hypot((p.x - tap.x) * aspect, p.y - tap.y) > 0.03) taps.delete(p.id);
        if (p.justReleased) {
          if (tap && !p.cancelled && clock - tap.time < 0.35) {
            const [x, y] = toWorld(tap.x, tap.y, d);
            sph.addBlock({ center: [x, y], radius: 0.09, velocity: [0, -1.5] });
          }
          taps.delete(p.id);
        }
        if (p.down && active.length < 4) active.push(makePointer(p));
        if ((p.down || p.inside) && rings.length < 16) {
          const [x, y] = toWorld(p.x, p.y, d);
          rings.push(x, y, params.pointerSize, p.down ? 0.9 : 0.35);
        }
      }
      sph.params.pointers = active;
      return rings;
    }

    // --- Rendering -----------------------------------------------------------------------------
    function ensureLiquid(): Liquid2d {
      // Half-resolution field on big canvases: the blobs are still several texels wide.
      const scale = width * height > 1_300_000 ? 0.5 : 1;
      if (!liquid || scale !== liquidScale) {
        liquid?.destroy();
        liquidScale = scale;
        liquid = createLiquid2d({
          device,
          format,
          width,
          height,
          fieldScale: scale,
          profiler,
          label: 'sph-2d liquid',
        });
      }
      return liquid;
    }

    const look: Partial<Liquid2dParams> = {};
    function lookFor(view: string, spacing: number, domainHeight: number): Partial<Liquid2dParams> {
      // Blur by about the particle spacing in field texels: enough to round off the lattice, no more.
      const spacingPx = (spacing * height) / (2 * domainHeight);
      look.smoothing = clamp(0.9 * spacingPx * liquidScale, 1, 4);
      look.rim = 0.6;
      look.view = view === 'liquid' ? 'final' : 'discs';
      look.blobRadius = params.blob;
      look.threshold = params.threshold;
      look.waterType = Math.max(0, WATERS.indexOf(params.water as (typeof WATERS)[number]));
      look.refraction = 0.12 * params.refraction;
      look.depthScale = params.depth;
      look.foam = params.foam;
      look.foamLow = 2.2;
      look.foamHigh = 4.6;
      if (view === 'density') {
        look.discRange = [0.75, 1.05];
        look.discPalette = HEAT;
      } else if (view === 'pressure') {
        look.discRange = [0, 0.35];
        look.discPalette = HEAT;
      } else if (view === 'velocity') {
        look.discRange = [0, 4];
        look.discPalette = SPEED;
      } else {
        look.discRange = [0, 3];
        look.discPalette = SPEED;
      }
      return look;
    }

    return {
      resize(w, h) {
        width = Math.max(1, Math.round(w));
        height = Math.max(1, Math.round(h));
        if (!backdrop) {
          backdrop = createRenderTarget(device, {
            label: 'sph-2d backdrop',
            format: 'rgba16float',
            width,
            height,
          });
        } else {
          backdrop.resize(width, height);
        }
        backdropDrawn = false;
        ensureLiquid().resize(width, height);
      },

      frame({ delta }, encoder, target) {
        const dt = Math.min(delta, 0.05);
        clock += dt;
        pointers.setCapture(params.captureTouch);
        const sph = ensureSolver();
        syncParams(sph);
        const d = domain();
        const rings = handlePointers(sph, d);

        sph.update(encoder, dt);

        // The wall behind the tank, once per size.
        const aspect = width / height;
        const frameFields = {
          center: [d.center[0], d.center[1]] as [number, number],
          half_height: d.halfHeight,
          aspect,
          box_min: [TANK.min[0], TANK.min[1]] as [number, number],
          box_max: [TANK.max[0], TANK.max[1]] as [number, number],
          height_px: height,
        };
        if (!backdropDrawn && backdrop) {
          backdropDrawn = true;
          backdropUniform.set(frameFields);
          backdropUniform.upload();
          renderFullscreen(encoder, backdrop.view, backdropPipeline, [backdropGroup], {
            label: 'sph-2d backdrop',
            timestampWrites: profiler.pass('backdrop (on resize)'),
          });
        }

        const renderer = ensureLiquid();
        renderer.render(encoder, {
          particles: sph.particles,
          domain: d,
          backdrop: backdrop!.view,
          target,
          params: lookFor(params.view, sph.derived.spacing, d.halfHeight),
          profiler,
        });

        // Obstacles, glass and pointer rings over the liquid.
        const obstacles: number[] = [];
        for (const o of sph.params.obstacles) {
          if (o.shape === 'circle') {
            obstacles.push(1, o.center[0], o.center[1], 0, o.radius, 0, 0, 0);
          } else {
            obstacles.push(
              2,
              o.center[0],
              o.center[1],
              0,
              o.halfSize[0],
              o.halfSize[1],
              o.cornerRadius ?? 0,
              0,
            );
          }
        }
        overlayUniform.set({
          ...frameFields,
          obstacle_count: sph.params.obstacles.length,
          pointer_count: Math.floor(rings.length / 4),
          encode_srgb: format.endsWith('-srgb') ? 0 : 1,
          obstacles,
          pointers: rings,
        });
        overlayUniform.upload();
        renderFullscreen(encoder, target, overlayPipeline, [overlayGroup], {
          label: 'sph-2d overlay',
          load: true,
          timestampWrites: profiler.pass('obstacles and glass'),
        });
      },

      destroy() {
        solver?.destroy();
        liquid?.destroy();
        backdrop?.destroy();
        pointers.destroy();
        backdropUniform.destroy();
        overlayUniform.destroy();
      },
    };
  },
});

const SPEED: [number, number, number][] = [
  [0.02, 0.1, 0.35],
  [0.05, 0.45, 0.8],
  [0.5, 0.9, 1],
  [1, 1, 1],
];
const HEAT: [number, number, number][] = [
  [0.03, 0.05, 0.3],
  [0.05, 0.4, 0.75],
  [0.1, 0.75, 0.5],
  [0.95, 0.85, 0.2],
  [0.95, 0.2, 0.1],
];

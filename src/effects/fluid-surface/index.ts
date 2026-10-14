import { mat4 } from 'wgpu-matrix';
import {
  createFluidSurface,
  FILTERS,
  type FluidDebugView,
  type FluidFilter,
  type FluidSurface,
  type FluidSurfaceMode,
  type FluidSurfaceParams,
} from '../../gpu/fluid-surface/index.ts';
import { createFullscreenPipeline, renderFullscreen } from '../../gpu/fullscreen.ts';
import { createOutputPass } from '../../gpu/post/index.ts';
import { computeCamera, createCameraMatrices, fitFovY } from '../../gpu/scene/camera.ts';
import { createDepthTarget } from '../../gpu/scene/depth.ts';
import {
  createForwardRenderer,
  SCENE_STRUCT_CHUNKS,
  type SceneLight,
  type SceneObject,
} from '../../gpu/scene/forward.ts';
import { boxGeometry, cylinderGeometry } from '../../gpu/scene/geometry.ts';
import { createMesh, type Mesh } from '../../gpu/scene/mesh.ts';
import { createOrbitControl, createOrbitState, orbitEye } from '../../gpu/scene/orbit.ts';
import { createShader } from '../../gpu/shader.ts';
import { createRenderTarget } from '../../gpu/targets.ts';
import { defineEffect } from '../types.ts';
import { meta } from './meta.ts';
import sceneShader from './scene.wgsl?raw';
import {
  createSources,
  naturalRadius,
  SOURCES,
  thicknessGain,
  type SourceName,
} from './sources.ts';
import splitShader from './split.wgsl?raw';

const COLOR_FORMAT: GPUTextureFormat = 'rgba16float';
const DEPTH_FORMAT = 'depth32float';

const VIEWS = [
  'final',
  'spheres',
  'split',
  'raw-depth',
  'smoothed-depth',
  'normals',
  'thickness',
] as const;
type View = (typeof VIEWS)[number];

/** Water types of absorption.wgsl, by the name the control shows. `custom` uses the colour. */
const WATER = ['clear', 'coastal', 'murky', 'tea', 'milk', 'pool', 'ocean', 'custom'] as const;

const SCALES: Record<string, number> = { '1': 1, '1/2': 0.5, '1/4': 0.25 };
const THICKNESS_SCALES: Record<string, number> = { '1/2': 0.5, '1/4': 0.25, '1/8': 0.125 };

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

const FOV_Y = (42 * Math.PI) / 180;
const MIN_HORIZONTAL_FOV = (62 * Math.PI) / 180;

/** The container: a thin frame round the 4 x 4 x 3 volume the sources fill. */
function buildScene(meshes: { box: Mesh; cylinder: Mesh }): SceneObject[] {
  const { box, cylinder } = meshes;
  const metal: [number, number, number] = [0.55, 0.58, 0.62];
  const bar = 0.07;
  const objects: SceneObject[] = [];
  const addBox = (
    position: [number, number, number],
    scale: [number, number, number],
    color: [number, number, number],
  ) => {
    const model = mat4.translation(position);
    mat4.scale(model, scale, model);
    objects.push({ mesh: box, model, color });
  };
  for (const y of [0.04, 3]) {
    for (const z of [-2, 2]) addBox([0, y, z], [4 + bar, bar, bar], metal);
    for (const x of [-2, 2]) addBox([x, y, 0], [bar, bar, 4 + bar], metal);
  }
  for (const x of [-2, 2]) {
    for (const z of [-2, 2]) addBox([x, 1.52, z], [bar, 3, bar], metal);
  }
  // Two posts: one in front of the liquid's reach, one behind it. The liquid must go behind the first
  // and in front of the second, and the floor pattern shows through it between them.
  const post = (position: [number, number, number], color: [number, number, number]) => {
    const model = mat4.translation(position);
    mat4.scale(model, [0.22, 2.4, 0.22], model);
    objects.push({ mesh: cylinder, model, color });
  };
  post([-0.9, 1.2, 1.35], [0.8, 0.08, 0.04]);
  post([1.1, 1.2, -1.2], [0.05, 0.45, 0.6]);
  return objects;
}

const PRESET_VIEW: Record<View, { surface: FluidSurfaceMode; debug: FluidDebugView }> = {
  final: { surface: 'smooth', debug: 'none' },
  spheres: { surface: 'raw', debug: 'none' },
  split: { surface: 'split', debug: 'none' },
  'raw-depth': { surface: 'smooth', debug: 'raw-depth' },
  'smoothed-depth': { surface: 'smooth', debug: 'smooth-depth' },
  normals: { surface: 'smooth', debug: 'normals' },
  thickness: { surface: 'smooth', debug: 'thickness' },
};

const indexOf = <T extends string>(list: readonly T[], value: string, fallback = 0): T =>
  list[Math.max(0, list.indexOf(value as T))] ?? list[fallback]!;

export const effect = defineEffect({
  meta,
  params: {
    source: {
      type: 'select',
      default: 'ball',
      options: SOURCES,
      label: 'Source (ball splits and merges, wave, pouring stream)',
    },
    view: {
      type: 'select',
      default: 'final',
      options: VIEWS,
      label: 'View (split: spheres left, surface right)',
    },
    count: {
      type: 'number',
      default: 60000,
      min: 1000,
      max: 1000000,
      step: 1000,
      label: 'Particles',
    },
    particleRadius: {
      type: 'number',
      default: 1,
      min: 0.4,
      max: 8,
      step: 0.05,
      label: 'Particle radius (x natural)',
    },
    distance: {
      type: 'number',
      default: 6.5,
      min: 3,
      max: 14,
      step: 0.1,
      label: 'Camera distance',
    },
    filter: {
      type: 'select',
      default: 'narrow-range',
      options: FILTERS,
      label: 'Depth filter',
    },
    compare: {
      type: 'boolean',
      default: false,
      label: 'Compare filters (bilateral left, narrow-range right)',
    },
    filterRadius: {
      type: 'number',
      default: 2.5,
      min: 0.25,
      max: 6,
      step: 0.05,
      label: 'Filter radius (particle radii)',
    },
    iterations: { type: 'number', default: 3, min: 1, max: 6, step: 1, label: 'Filter iterations' },
    smoothScale: {
      type: 'select',
      default: '1/2',
      options: ['1', '1/2', '1/4'],
      label: 'Smoothing resolution',
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
    water: { type: 'select', default: 'clear', options: WATER, label: 'Water type' },
    tint: {
      type: 'color',
      default: [0.25, 0.6, 0.85],
      label: 'Custom colour (after one unit of thickness)',
    },
    murk: { type: 'number', default: 0.1, min: 0, max: 3, step: 0.01, label: 'Custom turbidity' },
    density: { type: 'number', default: 4, min: 0.25, max: 24, step: 0.05, label: 'Density' },
    refraction: {
      type: 'number',
      default: 1,
      min: 0,
      max: 3,
      step: 0.01,
      label: 'Refraction strength',
    },
    ior: {
      type: 'number',
      default: 1.333,
      min: 1,
      max: 2,
      step: 0.005,
      label: 'Index of refraction',
    },
    roughness: { type: 'number', default: 0.04, min: 0.01, max: 0.4, step: 0.005, label: 'Gloss' },
    foam: { type: 'number', default: 0.6, min: 0, max: 1, step: 0.01, label: 'Foam (particle w)' },
    exposure: { type: 'number', default: 1, min: 0.3, max: 2.5, step: 0.01 },
    speed: { type: 'number', default: 1, min: 0, max: 2.5, step: 0.01 },
    autoRotate: { type: 'boolean', default: true, label: 'Idle rotation' },
  },
  presets: {
    'clear water': {
      water: 'clear',
      density: 6,
      refraction: 1,
      ior: 1.333,
      roughness: 0.04,
      foam: 0.6,
      filterRadius: 2.5,
      iterations: 3,
    },
    'deep blue': {
      water: 'ocean',
      density: 16,
      refraction: 0.8,
      ior: 1.333,
      roughness: 0.05,
      foam: 0.7,
      filterRadius: 2.5,
      iterations: 3,
    },
    'thick syrup': {
      water: 'custom',
      tint: [0.72, 0.38, 0.06],
      murk: 0.45,
      density: 1.1,
      refraction: 1.6,
      ior: 1.46,
      roughness: 0.02,
      foam: 0,
      filterRadius: 2.5,
      iterations: 3,
    },
    'milky tea': {
      water: 'tea',
      density: 3,
      refraction: 0.8,
      roughness: 0.08,
      foam: 0.4,
    },
    'bag of marbles': { view: 'spheres', water: 'clear', density: 4 },
    'spheres vs surface': { view: 'split' },
    'bilateral vs narrow-range': { compare: true, view: 'normals', source: 'ball' },
    'stream and splash': { source: 'stream', count: 120000, view: 'final' },
    'wave, 250k particles': { source: 'wave', count: 250000, view: 'final' },
    'cheap cut': {
      smoothScale: '1/4',
      thicknessScale: '1/8',
      thicknessStride: 2,
      iterations: 1,
      filterRadius: 1.25,
      count: 40000,
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
    const objects = buildScene(meshes);
    const view = renderer.createView('main');

    const sceneColor = createRenderTarget(device, {
      label: 'fluid-surface scene colour',
      format: COLOR_FORMAT,
      width: 1,
      height: 1,
    });
    const sceneDepth = createDepthTarget(device, {
      label: 'fluid-surface scene depth',
      width: 1,
      height: 1,
      format: DEPTH_FORMAT,
    });
    const liquid = createRenderTarget(device, {
      label: 'fluid-surface liquid',
      format: COLOR_FORMAT,
      width: 1,
      height: 1,
    });
    // The second and third images exist only for the filter comparison view.
    let comparison:
      | {
          second: FluidSurface;
          liquid: ReturnType<typeof createRenderTarget>;
          liquid2: ReturnType<typeof createRenderTarget>;
        }
      | undefined;

    // The sky and the floor are drawn inside the forward pass, before the container.
    const sceneModule = createShader(device, {
      label: 'fluid-surface/scene.wgsl',
      code: sceneShader,
      chunks: SCENE_STRUCT_CHUNKS,
    });
    const layout = device.createPipelineLayout({ bindGroupLayouts: [renderer.viewLayout] });
    const skyPipeline = device.createRenderPipeline({
      label: 'fluid-surface sky',
      layout,
      vertex: { module: sceneModule, entryPoint: 'fullscreen_vs' },
      fragment: { module: sceneModule, entryPoint: 'fs_sky', targets: [{ format: COLOR_FORMAT }] },
      depthStencil: { format: DEPTH_FORMAT, depthWriteEnabled: false, depthCompare: 'always' },
    });
    const floorPipeline = device.createRenderPipeline({
      label: 'fluid-surface floor',
      layout,
      vertex: { module: sceneModule, entryPoint: 'vs_floor' },
      fragment: {
        module: sceneModule,
        entryPoint: 'fs_floor',
        targets: [{ format: COLOR_FORMAT }],
      },
      depthStencil: { format: DEPTH_FORMAT, depthWriteEnabled: true, depthCompare: 'less' },
    });

    const splitPipeline = createFullscreenPipeline(device, {
      fragment: { label: 'fluid-surface/split.wgsl', code: splitShader },
      format: COLOR_FORMAT,
    });
    const splitTarget = createRenderTarget(device, {
      label: 'fluid-surface split',
      format: COLOR_FORMAT,
      width: 1,
      height: 1,
    });

    const output = createOutputPass(device, { format });
    const sources = createSources(device);

    const fluid = createFluidSurface({ device, format: COLOR_FORMAT, width: 1, height: 1 });

    const orbit = createOrbitControl(
      canvas,
      createOrbitState({ yaw: 0.55, pitch: 0.34, distance: 6.5, target: [0, 1.0, 0] }),
      { minPitch: 0.06, maxPitch: 1.3, autoSpeed: 0.1 },
    );
    const camera = createCameraMatrices();

    let width = 1;
    let height = 1;
    let phase = 0;
    let frameCount = 0;

    function materialParams(
      source: SourceName,
      count: number,
      radius: number,
      base: FluidSurfaceParams,
    ): FluidSurfaceParams {
      const water = indexOf(WATER, params.water);
      const custom = water === 'custom';
      const tint = params.tint.map((c) => Math.min(1, Math.max(1e-3, c)));
      return {
        ...base,
        waterType: custom ? -1 : WATER.indexOf(water),
        absorption: [-Math.log(tint[0]!), -Math.log(tint[1]!), -Math.log(tint[2]!)],
        scatter: [tint[0]! * 0.6, tint[1]! * 0.6, tint[2]! * 0.6],
        turbidity: params.murk,
        density: params.density,
        ior: params.ior,
        refraction: params.refraction,
        roughness: params.roughness,
        sunDirection: SUN_DIRECTION,
        thicknessGain: thicknessGain(source, count, radius),
        foamAmount: params.foam,
        maxRefractionThickness: 1.2,
      };
    }

    function ensureComparison(): NonNullable<typeof comparison> {
      comparison ??= {
        second: createFluidSurface({ device, format: COLOR_FORMAT, width, height }),
        liquid: createRenderTarget(device, {
          label: 'fluid-surface comparison A',
          format: COLOR_FORMAT,
          width,
          height,
        }),
        liquid2: createRenderTarget(device, {
          label: 'fluid-surface comparison B',
          format: COLOR_FORMAT,
          width,
          height,
        }),
      };
      return comparison;
    }

    return {
      resize(w, h) {
        width = Math.max(1, Math.round(w));
        height = Math.max(1, Math.round(h));
        sceneColor.resize(width, height);
        sceneDepth.resize(width, height);
        liquid.resize(width, height);
        splitTarget.resize(width, height);
        fluid.resize(width, height);
        if (comparison) {
          comparison.second.resize(width, height);
          comparison.liquid.resize(width, height);
          comparison.liquid2.resize(width, height);
        }
      },

      frame({ delta }, encoder, target) {
        phase += delta * params.speed;
        frameCount++;
        const state = orbit.update(delta, params.autoRotate);
        state.distance = params.distance;

        const aspect = width / height;
        computeCamera(
          {
            position: orbitEye(state),
            target: state.target,
            fovY: fitFovY(FOV_Y, aspect, MIN_HORIZONTAL_FOV),
            aspect,
            near: 0.1,
            far: 120,
          },
          camera,
        );

        const source = indexOf(SOURCES, params.source);
        const count = Math.round(params.count);
        const radius = naturalRadius(source, count) * params.particleRadius;
        const quality = {
          scale: SCALES[params.smoothScale] ?? 0.5,
          thicknessScale: THICKNESS_SCALES[params.thicknessScale] ?? 0.25,
          thicknessStride: Math.round(params.thicknessStride),
          filter: indexOf(FILTERS, params.filter) as FluidFilter,
          filterRadius: params.filterRadius,
          iterations: Math.round(params.iterations),
        };

        // 1. Particles.
        const particles = sources.update(encoder, source, count, radius, phase, profiler);

        // 2. The scene the liquid is drawn into.
        view.update(camera, width, height, LIGHT);
        const batch = renderer.upload(objects);
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
          label: 'fluid-surface scene pass',
          timestampWrites: profiler.pass('scene'),
        });

        // 3. The liquid.
        const mode = indexOf(VIEWS, params.view);
        const look = PRESET_VIEW[mode];
        const base: FluidSurfaceParams = { surface: look.surface, debug: look.debug };
        fluid.setQuality(quality);

        let shown = liquid;
        if (params.compare && mode !== 'split') {
          const pair = ensureComparison();
          fluid.setQuality({ filter: 'bilateral' });
          pair.second.setQuality({ ...quality, filter: 'narrow-range' });
          const material = materialParams(source, count, radius, base);
          fluid.render(encoder, {
            particles,
            camera,
            sceneColor: sceneColor.view,
            sceneDepth: sceneDepth.view,
            target: pair.liquid.view,
            params: material,
            profiler,
          });
          pair.second.render(encoder, {
            particles,
            camera,
            sceneColor: sceneColor.view,
            sceneDepth: sceneDepth.view,
            target: pair.liquid2.view,
            params: material,
          });
          const group = device.createBindGroup({
            layout: splitPipeline.getBindGroupLayout(0),
            entries: [
              { binding: 0, resource: pair.liquid.view },
              { binding: 1, resource: pair.liquid2.view },
            ],
          });
          renderFullscreen(encoder, splitTarget.view, splitPipeline, [group], {
            label: 'fluid-surface filter comparison',
          });
          shown = splitTarget;
        } else {
          fluid.render(encoder, {
            particles,
            camera,
            sceneColor: sceneColor.view,
            sceneDepth: sceneDepth.view,
            target: liquid.view,
            params: materialParams(source, count, radius, base),
            profiler,
          });
        }

        // 4. Tonemap into the canvas. The debug views are data, not light: no tonemap.
        const debug = look.debug !== 'none';
        output.encode(
          encoder,
          target,
          shown.view,
          undefined,
          {
            exposure: debug ? 1 : params.exposure,
            tonemap: debug ? 'none' : 'aces',
            frame: frameCount,
          },
          profiler.pass('output'),
        );
      },

      destroy() {
        orbit.destroy();
        sources.destroy();
        fluid.destroy();
        if (comparison) {
          comparison.second.destroy();
          comparison.liquid.destroy();
          comparison.liquid2.destroy();
        }
        output.destroy();
        sceneColor.destroy();
        sceneDepth.destroy();
        liquid.destroy();
        splitTarget.destroy();
        view.destroy();
        renderer.destroy();
        for (const mesh of Object.values(meshes)) mesh.destroy();
      },
    };
  },
});

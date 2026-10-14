import { mat4, type Mat4 } from 'wgpu-matrix';
import { createFullscreenPipeline, renderFullscreen } from '../../gpu/fullscreen.ts';
import {
  computeCamera,
  createCameraMatrices,
  fitFovY,
  horizontalPlane,
  mirrorCamera,
} from '../../gpu/scene/camera.ts';
import { createDepthTarget } from '../../gpu/scene/depth.ts';
import {
  createForwardRenderer,
  SCENE_STRUCT_CHUNKS,
  type SceneLight,
  type SceneObject,
} from '../../gpu/scene/forward.ts';
import { boxGeometry, cylinderGeometry, sphereGeometry } from '../../gpu/scene/geometry.ts';
import { createMesh, type Mesh } from '../../gpu/scene/mesh.ts';
import { createOrbitControl, createOrbitState, orbitEye } from '../../gpu/scene/orbit.ts';
import { createShader } from '../../gpu/shader.ts';
import { createRenderTarget } from '../../gpu/targets.ts';
import { createUniformBuffer, defineUniformStruct } from '../../gpu/uniforms.ts';
import { defineEffect } from '../types.ts';
import { meta } from './meta.ts';
import skyModel from './rf-sky.wgsl';
import skyPass from './sky-pass.wgsl';
import water from './water.wgsl';

const COLOR_FORMAT: GPUTextureFormat = 'rgba16float';
const DEPTH_FORMAT = 'depth32float';

/** Height of the water surface. The scene is built around it. */
const WATER_LEVEL = 0;
/** Half-width of the square basin, just inside the walls so the water edge never z-fights them. */
const BASIN = 5.98;

const STRATEGIES = ['planar', 'ssr', 'sky', 'split'] as const;
const DEBUG_VIEWS = ['off', 'buffer', 'mask'] as const;
const indexOf = (list: readonly string[], value: string) => Math.max(0, list.indexOf(value));
const SCALES: Record<string, number> = { '1': 1, '1/2': 0.5, '1/4': 0.25 };

const Params = defineUniformStruct('Params', {
  time: 'f32',
  strategy: 'u32',
  ssr_steps: 'u32',
  ssr_jitter: 'u32',
  ssr_distance: 'f32',
  ssr_thickness: 'f32',
  distortion: 'f32',
  ripple: 'f32',
  split: 'f32',
  debug: 'u32',
  water_level: 'f32',
  murk: 'f32',
  tint: 'vec3f',
  exposure: 'f32',
  basin: 'f32',
});

const LIGHT: SceneLight = {
  sunDirection: [-0.75, 0.45, -0.35],
  sunColor: [1, 0.93, 0.8],
  sunIntensity: 1.7,
  skyColor: [0.12, 0.32, 0.85],
  groundColor: [0.32, 0.27, 0.2],
  horizonColor: [0.6, 0.74, 0.92],
  ambient: 0.42,
  fogDensity: 0.009,
};

const FOV_Y = (45 * Math.PI) / 180;
/** Keep at least this much of the scene visible across a tall canvas (phone portrait). */
const MIN_HORIZONTAL_FOV = (62 * Math.PI) / 180;

interface Placed {
  mesh: Mesh;
  position: readonly [number, number, number];
  scale: readonly [number, number, number];
  color: readonly [number, number, number];
  yaw?: number;
}

function place({ mesh, position, scale, color, yaw = 0 }: Placed): SceneObject {
  const model = mat4.translation(position);
  mat4.rotateY(model, yaw, model);
  mat4.scale(model, scale, model);
  return { mesh, model, color };
}

/** The test scene: a square basin of calm water with objects chosen to expose each strategy. */
function buildScene(meshes: { box: Mesh; cylinder: Mesh; sphere: Mesh }) {
  const { box, cylinder, sphere } = meshes;
  const stone: [number, number, number] = [0.42, 0.36, 0.27];
  const top = 0.35;
  const bottom = -1.6;
  const bank = (x: number, z: number, w: number, d: number) =>
    place({
      mesh: box,
      position: [x, (top + bottom) / 2, z],
      scale: [w, top - bottom, d],
      color: stone,
    });

  const objects: SceneObject[] = [
    // The basin: four banks around a floor 1.2 below the waterline.
    bank(-103, 0, 194, 400),
    bank(103, 0, 194, 400),
    bank(0, -103, 12, 194),
    bank(0, 103, 12, 194),
    place({ mesh: box, position: [0, -1.4, 0], scale: [12, 0.4, 12], color: [0.16, 0.19, 0.17] }),

    // Crosses the waterline, tall and thin: the reference for "does the reflection meet its base".
    place({
      mesh: cylinder,
      position: [-2.4, 0.7, -1.2],
      scale: [0.7, 3.8, 0.7],
      color: [0.65, 0.015, 0.01],
    }),
    // A box sunk to just above the surface, turned so it shows two faces.
    place({
      mesh: box,
      position: [2.5, -0.2, 0.9],
      scale: [1.3, 1.2, 1.3],
      color: [0.85, 0.45, 0.005],
      yaw: 0.45,
    }),
    // Far and thin: small in screen space, a test for SSR's step size.
    place({
      mesh: cylinder,
      position: [3.7, 1.4, -4.1],
      scale: [0.45, 5.2, 0.45],
      color: [0.02, 0.5, 0.04],
    }),
    // Overhang: a lintel over the water on two posts. Its underside is invisible to the camera.
    place({
      mesh: box,
      position: [0, 2.05, -2.9],
      scale: [14.4, 0.3, 1.2],
      color: [0.75, 0.14, 0.01],
    }),
    // The posts end inside the lintel: coplanar tops would z-fight.
    place({ mesh: box, position: [-6.6, 1.175, -2.9], scale: [1.2, 1.65, 1.2], color: stone }),
    place({ mesh: box, position: [6.6, 1.175, -2.9], scale: [1.2, 1.65, 1.2], color: stone }),
    // On the bank, off to the side and far behind: partly off-screen from most angles.
    place({
      mesh: box,
      position: [-8.6, 1.1, 1.8],
      scale: [1.5, 1.5, 1.5],
      color: [0.35, 0.02, 0.6],
      yaw: 0.3,
    }),
    place({
      mesh: box,
      position: [-13, 2.2, -15],
      scale: [3, 3.7, 3],
      color: [0.7, 0.7, 0.75],
    }),
    place({
      mesh: cylinder,
      position: [7.8, 0.85, 4.6],
      scale: [0.9, 1, 0.9],
      color: [0.01, 0.45, 0.5],
    }),
  ];

  // The sphere bobs on the water, so the picture is alive and never repeats one frame.
  const floating = place({
    mesh: sphere,
    position: [0.1, 0.1, 2.3],
    scale: [1.4, 1.4, 1.4],
    color: [0.02, 0.12, 0.85],
  });
  objects.push(floating);
  const bobBase = [0.1, 0.1, 2.3] as const;
  return {
    objects,
    animate(phase: number) {
      const model: Mat4 = floating.model;
      mat4.translation([bobBase[0], bobBase[1] + Math.sin(phase * 0.9) * 0.1, bobBase[2]], model);
      mat4.scale(model, [1.4, 1.4, 1.4], model);
    },
  };
}

export const effect = defineEffect({
  meta,
  params: {
    strategy: {
      type: 'select',
      default: 'split',
      options: STRATEGIES,
      label: 'Strategy (split: planar left, SSR right)',
    },
    reflectionScale: {
      type: 'select',
      default: '1/2',
      options: ['1', '1/2', '1/4'],
      label: 'Planar resolution',
    },
    ssrSteps: { type: 'number', default: 48, min: 4, max: 128, step: 1, label: 'SSR steps' },
    ssrDistance: {
      type: 'number',
      default: 40,
      min: 4,
      max: 100,
      step: 1,
      label: 'SSR max distance',
    },
    ssrThickness: {
      type: 'number',
      default: 0.5,
      min: 0.05,
      max: 4,
      step: 0.05,
      label: 'SSR thickness',
    },
    ssrJitter: { type: 'boolean', default: true, label: 'SSR jitter' },
    distortion: { type: 'number', default: 0.5, min: 0, max: 1, step: 0.01 },
    ripple: { type: 'number', default: 0.35, min: 0, max: 2, step: 0.01, label: 'Ripple strength' },
    speed: { type: 'number', default: 1, min: 0, max: 3, step: 0.01 },
    split: { type: 'number', default: 0.5, min: 0.05, max: 0.95, step: 0.01, label: 'Divider' },
    debug: { type: 'select', default: 'off', options: DEBUG_VIEWS, label: 'Debug view' },
    murk: { type: 'number', default: 0.9, min: 0.05, max: 2, step: 0.01, label: 'Water murk' },
    tint: { type: 'color', default: [0.01, 0.12, 0.17], label: 'Water tint' },
    exposure: { type: 'number', default: 0.85, min: 0.4, max: 2, step: 0.01 },
    autoRotate: { type: 'boolean', default: true, label: 'Idle rotation' },
  },
  presets: {
    compare: { strategy: 'split', reflectionScale: '1/2', ssrSteps: 48, debug: 'off' },
    'planar-quarter': { strategy: 'planar', reflectionScale: '1/4', distortion: 0.5 },
    'ssr-cheap': { strategy: 'ssr', ssrSteps: 16, ssrJitter: true },
    'sky-only': { strategy: 'sky' },
    glassy: { strategy: 'split', ripple: 0.12, distortion: 0.2, murk: 0.3 },
  },

  async create({ device, format, canvas, params, profiler }) {
    const renderer = createForwardRenderer(device, {
      colorFormat: COLOR_FORMAT,
      depthFormat: DEPTH_FORMAT,
    });
    const meshes = {
      box: createMesh(device, boxGeometry(), 'box'),
      cylinder: createMesh(device, cylinderGeometry({ radialSegments: 32 }), 'cylinder'),
      sphere: createMesh(
        device,
        sphereGeometry({ radius: 0.5, widthSegments: 48, heightSegments: 24 }),
        'sphere',
      ),
    };
    const scene = buildScene(meshes);

    const mainView = renderer.createView('main');
    const mirrorView = renderer.createView('mirror');

    const sceneColor = createRenderTarget(device, {
      label: 'reflections scene colour',
      format: COLOR_FORMAT,
      width: 1,
      height: 1,
    });
    const sceneDepth = createDepthTarget(device, {
      label: 'reflections scene depth',
      width: 1,
      height: 1,
      format: DEPTH_FORMAT,
    });
    const reflColor = createRenderTarget(device, {
      label: 'reflections planar colour',
      format: COLOR_FORMAT,
      width: 1,
      height: 1,
    });
    const reflDepth = createDepthTarget(device, {
      label: 'reflections planar depth',
      width: 1,
      height: 1,
      format: DEPTH_FORMAT,
      sampled: false,
    });

    const uniforms = createUniformBuffer(device, Params, 'reflections params');
    uniforms.set({ water_level: WATER_LEVEL, basin: BASIN });

    // The sky is drawn inside the forward pass, before the objects, by the same pipeline for the real
    // and the mirrored camera.
    const skyModule = createShader(device, {
      label: 'reflections/sky-pass.wgsl',
      code: skyPass.code,
      chunks: {
        ...skyPass.chunks,
        ...SCENE_STRUCT_CHUNKS,
        'rf-sky.wgsl': skyModel.code,
        ...skyModel.chunks,
      },
    });
    const skyPipeline = device.createRenderPipeline({
      label: 'reflections sky',
      layout: device.createPipelineLayout({ bindGroupLayouts: [renderer.viewLayout] }),
      vertex: { module: skyModule, entryPoint: 'fullscreen_vs' },
      fragment: { module: skyModule, entryPoint: 'fs', targets: [{ format: COLOR_FORMAT }] },
      depthStencil: { format: DEPTH_FORMAT, depthWriteEnabled: false, depthCompare: 'always' },
    });

    const waterPipeline = createFullscreenPipeline(device, {
      fragment: {
        label: 'reflections/water.wgsl',
        code: water.code,
        chunks: {
          ...water.chunks,
          ...SCENE_STRUCT_CHUNKS,
          'params.wgsl': Params.wgsl,
          'rf-sky.wgsl': skyModel.code,
          ...skyModel.chunks,
        },
      },
      format,
    });
    const sampler = device.createSampler({
      label: 'reflections sampler',
      magFilter: 'linear',
      minFilter: 'linear',
      addressModeU: 'clamp-to-edge',
      addressModeV: 'clamp-to-edge',
    });

    // The water bind group references the targets' views, which change when a target is resized.
    let waterBindGroup: GPUBindGroup | undefined;
    let boundViews: GPUTextureView[] = [];
    function bindWater() {
      const views = [sceneColor.view, sceneDepth.view, reflColor.view];
      if (waterBindGroup && views.every((view, i) => view === boundViews[i])) return waterBindGroup;
      boundViews = views;
      waterBindGroup = device.createBindGroup({
        label: 'reflections water',
        layout: waterPipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: mainView.camera.buffer } },
          { binding: 1, resource: { buffer: mirrorView.camera.buffer } },
          { binding: 2, resource: { buffer: mainView.light.buffer } },
          { binding: 3, resource: { buffer: uniforms.buffer } },
          { binding: 4, resource: sceneColor.view },
          { binding: 5, resource: sceneDepth.view },
          { binding: 6, resource: reflColor.view },
          { binding: 7, resource: sampler },
        ],
      });
      return waterBindGroup;
    }

    const orbit = createOrbitControl(
      canvas,
      createOrbitState({ yaw: 0.5, pitch: 0.3, distance: 14, target: [0, 0.4, -0.6] }),
      { minPitch: 0.1, maxPitch: 1.2 },
    );
    const camera = createCameraMatrices();
    const mirror = createCameraMatrices();
    const plane = horizontalPlane(WATER_LEVEL);

    let width = 1;
    let height = 1;
    let phase = 0;

    const sky = (view: typeof mainView) => (pass: GPURenderPassEncoder) => {
      pass.setPipeline(skyPipeline);
      pass.setBindGroup(0, view.bindGroup);
      pass.draw(3);
    };

    return {
      resize(w, h) {
        width = Math.max(1, Math.round(w));
        height = Math.max(1, Math.round(h));
        sceneColor.resize(width, height);
        sceneDepth.resize(width, height);
      },

      frame({ delta }, encoder, target) {
        phase += delta * params.speed;
        scene.animate(phase);
        const state = orbit.update(delta, params.autoRotate);

        const aspect = width / height;
        computeCamera(
          {
            position: orbitEye(state),
            target: state.target,
            fovY: fitFovY(FOV_Y, aspect, MIN_HORIZONTAL_FOV),
            aspect,
            near: 0.2,
            far: 400,
          },
          camera,
        );
        mirrorCamera(camera, plane, mirror);

        const strategy = indexOf(STRATEGIES, params.strategy);
        const planarNeeded = strategy === 0 || strategy === 3;
        const scale = SCALES[params.reflectionScale] ?? 0.5;
        const reflWidth = Math.max(1, Math.round(width * scale));
        const reflHeight = Math.max(1, Math.round(height * scale));

        mainView.update(camera, width, height, LIGHT);
        const batch = renderer.upload(scene.objects);

        if (planarNeeded) {
          reflColor.resize(reflWidth, reflHeight);
          reflDepth.resize(reflWidth, reflHeight);
          mirrorView.update(mirror, reflWidth, reflHeight, LIGHT);
          renderer.draw(encoder, batch, {
            view: mirrorView,
            color: reflColor.view,
            depth: reflDepth.view,
            mirrored: true,
            background: sky(mirrorView),
            label: 'reflections planar pass',
            timestampWrites: profiler.pass('reflection'),
          });
        }

        renderer.draw(encoder, batch, {
          view: mainView,
          color: sceneColor.view,
          depth: sceneDepth.view,
          background: sky(mainView),
          label: 'reflections scene pass',
          timestampWrites: profiler.pass('scene'),
        });

        uniforms.set({
          time: phase,
          strategy,
          ssr_steps: Math.round(params.ssrSteps),
          ssr_jitter: params.ssrJitter ? 1 : 0,
          ssr_distance: params.ssrDistance,
          ssr_thickness: params.ssrThickness,
          distortion: params.distortion,
          ripple: params.ripple,
          split: params.split,
          debug: indexOf(DEBUG_VIEWS, params.debug),
          murk: params.murk,
          tint: params.tint,
          exposure: params.exposure,
        });
        uniforms.upload();

        renderFullscreen(encoder, target, waterPipeline, [bindWater()], {
          label: 'reflections water pass',
          timestampWrites: profiler.pass('water'),
        });
      },

      destroy() {
        orbit.destroy();
        uniforms.destroy();
        sceneColor.destroy();
        sceneDepth.destroy();
        reflColor.destroy();
        reflDepth.destroy();
        mainView.destroy();
        mirrorView.destroy();
        renderer.destroy();
        for (const mesh of Object.values(meshes)) mesh.destroy();
      },
    };
  },
});

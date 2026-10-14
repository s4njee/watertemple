import { createShader } from '../shader.ts';
import { createUniformBuffer, defineUniformStruct, uniformArray } from '../uniforms.ts';
import { createRenderTarget } from '../targets.ts';
import { createDepthTarget } from '../scene/depth.ts';
import { computeCamera, mirrorCamera, horizontalPlane } from '../scene/camera.ts';
import { createFullscreenPipeline, renderFullscreen } from '../fullscreen.ts';
import { createBillboardRenderer } from '../particles/billboards.ts';
import { defineEffect, type EffectMeta } from '../../effects/types.ts';
import { sampleWorldWaves } from './waves.ts';
import { oceanMesh } from './mesh.ts';
import { createSpectralOcean } from './fft.ts';
import { createBuoyancyReadback, BUOY_POINTS } from './buoyancy.ts';
import { createRiverFlowMap } from './flow.ts';
import { createWaterfallSpray } from './spray.ts';
import code from './scene.wgsl';
const Params = defineUniformStruct('OceanParams', {
  vp: 'mat4x4f',
  inv_vp: 'mat4x4f',
  mirror_vp: 'mat4x4f',
  mirror_inv: 'mat4x4f',
  eye: 'vec3f',
  time: 'f32',
  mode: 'u32',
  strategy: 'u32',
  view: 'u32',
  buoyancy: 'u32',
  strength: 'f32',
  chop: 'f32',
  foam: 'f32',
  sun: 'f32',
  fog: 'f32',
  exposure: 'f32',
  scattering: 'f32',
  buoys: uniformArray('vec4f', 6),
});
export type OceanMode = 0 | 1 | 2 | 3 | 4 | 5 | 6;
/** One renderer keeps camera, color and resource ownership consistent across the seven studies. */
export function oceanEffect(meta: EffectMeta, mode: OceanMode) {
  return defineEffect({
    meta,
    params: {
      mesh: {
        type: 'select',
        default: 'projected',
        options: ['projected', 'clipmap'],
        label: 'Surface mesh',
      },
      cells: { type: 'select', default: '96', options: ['48', '96', '160'], label: 'Mesh cells' },
      strength: {
        type: 'number',
        default: mode === 6 || mode === 1 ? 0.6 : 1,
        min: 0,
        max: 2,
        step: 0.01,
        label: mode === 3 ? 'Current detail' : 'Wave height',
      },
      ...(mode < 2 || mode === 5
        ? {
            chop: {
              type: 'number' as const,
              default: 0.85,
              min: 0,
              max: 1.8,
              step: 0.01,
              label: 'Wave steepness',
            },
          }
        : {}),
      ...(mode !== 6
        ? {
            foam: {
              type: 'number' as const,
              default: 1,
              min: 0,
              max: 3,
              step: 0.01,
              label: 'Foam strength',
            },
          }
        : {}),
      sun: {
        type: 'number',
        default: mode === 6 ? 0.14 : 0.48,
        min: -0.1,
        max: 2,
        step: 0.01,
        label: 'Sun height',
      },
      fog: { type: 'number', default: 0.008, min: 0, max: 0.04, step: 0.001, label: 'Horizon fog' },
      exposure: { type: 'number', default: 1, min: 0.3, max: 2, step: 0.01 },
      scattering: {
        type: 'number',
        default: 1,
        min: 0,
        max: 3,
        step: 0.01,
        label: 'Crest scattering',
      },
      cameraHeight: {
        type: 'number',
        default: mode === 5 ? 0.15 : mode === 4 ? 3.5 : mode === 3 ? 5 : mode === 1 ? 4.5 : 3,
        min: mode === 5 ? -1.2 : 1,
        max: 12,
        step: 0.01,
        label: 'Camera height (m)',
      },
      cameraPitch: {
        type: 'number',
        default: mode === 4 ? -4 : mode === 3 ? -20 : mode === 5 ? 0 : -7,
        min: -40,
        max: 12,
        step: 0.5,
        label: 'Camera pitch (deg)',
      },
      speed: { type: 'number', default: 1, min: 0, max: 2, step: 0.01 },
      view: {
        type: 'select',
        default: 'final',
        options: ['final', 'normals', 'foam', 'mesh'],
        label: 'Study view',
      },
      ...(mode === 0
        ? {
            buoyancy: {
              type: 'select' as const,
              default: 'cpu',
              options: ['off', 'cpu', 'gpu'],
              label: 'Buoyancy sampling',
            },
          }
        : {}),
      ...(mode === 1
        ? {
            fftSize: {
              type: 'select' as const,
              default: '64',
              options: ['64', '128', '256'],
              label: 'FFT samples per cascade',
            },
            foamThreshold: {
              type: 'number' as const,
              default: 0.75,
              min: 0.1,
              max: 1.2,
              step: 0.01,
              label: 'Folding threshold',
            },
            foamLifetime: {
              type: 'number' as const,
              default: 3,
              min: 0.1,
              max: 8,
              step: 0.1,
              label: 'Foam lifetime (s)',
            },
          }
        : {}),
    },
    presets: {
      ...(mode < 2
        ? {
            swell: { strength: 0.7, chop: 0.65, sun: 0.48, foam: 0.8 },
            storm: { strength: 1.65, chop: 1.4, sun: 0.24, foam: 2, fog: 0.018 },
            'golden hour': { sun: 0.06, exposure: 1.1, strength: 0.85 },
            'clipmap study': { mesh: 'clipmap', view: 'mesh' },
          }
        : {}),
      ...(mode === 2
        ? {
            'gentle surf': { strength: 0.6, foam: 1, cameraPitch: -14 },
            'breaking shore': { strength: 1.5, foam: 2, sun: 0.2, cameraPitch: -18 },
          }
        : {}),
      ...(mode === 3
        ? {
            'forest stream': { strength: 0.7, foam: 1, cameraPitch: -22 },
            'fast current': { strength: 1.6, foam: 2, speed: 1.7, cameraHeight: 4 },
          }
        : {}),
      ...(mode === 4
        ? {
            cascade: { cameraPitch: -4, exposure: 1, foam: 1 },
            'misty plunge': { cameraHeight: 2, cameraPitch: 2, foam: 2, fog: 0.025 },
          }
        : {}),
      ...(mode === 5
        ? {
            'half submerged': { cameraHeight: 0.15, cameraPitch: 0 },
            underwater: { cameraHeight: -0.8, cameraPitch: 6 },
            'above water': { cameraHeight: 1.5, cameraPitch: -6 },
          }
        : {}),
      ...(mode === 6
        ? {
            'morning mirror': { strength: 0.25, sun: 0.14, fog: 0.005 },
            'wind ripples': { strength: 1.4, sun: 0.5, cameraHeight: 2, cameraPitch: -8 },
          }
        : {}),
    },
    async create({ device, format, params, profiler }) {
      const uniforms = createUniformBuffer(device, Params, `${meta.id} scene`);
      const module = createShader(device, {
        label: 'ocean/scene.wgsl',
        code: code.code,
        chunks: { ...code.chunks, 'scene-params.wgsl': Params.wgsl },
      });
      const visibility = GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT;
      const layout = device.createBindGroupLayout({
        entries: [
          { binding: 0, visibility, buffer: { type: 'uniform' } },
          { binding: 1, visibility, texture: { viewDimension: '2d-array' } },
          { binding: 2, visibility, texture: { viewDimension: '2d-array' } },
          { binding: 3, visibility, sampler: { type: 'filtering' } },
          { binding: 4, visibility, texture: {} },
          { binding: 5, visibility, texture: {} },
        ],
      });
      const pipelineLayout = device.createPipelineLayout({ bindGroupLayouts: [layout] });
      const background = device.createRenderPipeline({
        label: 'open water environment',
        layout: pipelineLayout,
        vertex: { module, entryPoint: 'fullscreen_vs' },
        fragment: { module, entryPoint: 'background', targets: [{ format }] },
        depthStencil: { format: 'depth24plus', depthWriteEnabled: true, depthCompare: 'always' },
      });
      const water = device.createRenderPipeline({
        label: 'open water surface',
        layout: pipelineLayout,
        vertex: {
          module,
          entryPoint: 'water_vs',
          buffers: [
            {
              arrayStride: 16,
              attributes: [{ shaderLocation: 0, offset: 0, format: 'float32x4' }],
            },
          ],
        },
        fragment: { module, entryPoint: 'water_fs', targets: [{ format }] },
        primitive: { cullMode: 'none' },
        depthStencil: { format: 'depth24plus', depthWriteEnabled: true, depthCompare: 'less' },
      });
      const planarPipeline =
        mode === 6
          ? createFullscreenPipeline(device, {
              fragment: {
                label: 'lake planar reflection',
                code: code.code,
                chunks: { ...code.chunks, 'scene-params.wgsl': Params.wgsl },
              },
              entryPoint: 'reflection',
              format: 'rgba16float',
              layout: [layout],
            })
          : undefined;
      const falls =
        mode === 4
          ? device.createRenderPipeline({
              label: 'waterfall sheets',
              layout: pipelineLayout,
              vertex: { module, entryPoint: 'falls_vs' },
              fragment: {
                module,
                entryPoint: 'falls_fs',
                targets: [
                  {
                    format,
                    blend: {
                      color: { srcFactor: 'one', dstFactor: 'one-minus-src-alpha' },
                      alpha: { srcFactor: 'one', dstFactor: 'one-minus-src-alpha' },
                    },
                  },
                ],
              },
              depthStencil: {
                format: 'depth24plus',
                depthWriteEnabled: false,
                depthCompare: 'less',
              },
            })
          : undefined;
      const depth = createDepthTarget(device, {
        label: 'open water depth',
        width: 1,
        height: 1,
        format: 'depth24plus',
        sampled: false,
      });
      const reflectionTarget = createRenderTarget(device, {
        label: 'lake reflection',
        format: 'rgba16float',
        width: 1,
        height: 1,
      });
      const sceneTarget =
        mode === 5
          ? createRenderTarget(device, { label: 'waterline scene', format, width: 1, height: 1 })
          : undefined;
      const postLayout = device.createBindGroupLayout({
        entries: [{ binding: 0, visibility: GPUShaderStage.FRAGMENT, texture: {} }],
      });
      const post =
        mode === 5
          ? createFullscreenPipeline(device, {
              fragment: {
                label: 'waterline grading',
                code: code.code,
                chunks: { ...code.chunks, 'scene-params.wgsl': Params.wgsl },
              },
              entryPoint: 'waterline',
              format,
              layout: [layout, postLayout],
            })
          : undefined;
      const dummy = device.createTexture({
        size: [1, 1, 3],
        format: 'rgba16float',
        usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
      });
      const flow = createRiverFlowMap(device);
      const sampler = device.createSampler({
        magFilter: 'linear',
        minFilter: 'linear',
        addressModeU: 'repeat',
        addressModeV: 'repeat',
      });
      let spectral =
        mode === 1
          ? createSpectralOcean(device, Number(params.fftSize ?? 64), profiler)
          : undefined;
      const readback = mode === 0 ? createBuoyancyReadback(device, profiler) : undefined;
      const spray = mode === 4 ? createWaterfallSpray(device, profiler) : undefined;
      const sprites =
        mode === 4
          ? createBillboardRenderer({
              device,
              format,
              profiler,
              style: {
                mode: 'alpha',
                palette: [
                  [0.6, 0.8, 0.81],
                  [0.9, 0.95, 0.95],
                ],
                intensity: 0.11,
                softness: 1,
                sizeScale: 2.5,
                minPixels: 1,
              },
            })
          : undefined;
      let vertices: GPUBuffer | undefined,
        indices: GPUBuffer | undefined,
        indexCount = 0,
        meshKey = '',
        width = 1,
        height = 1,
        phase = 0;
      const sceneGroups = new Map<GPUTextureView, GPUBindGroup>();
      let reflectionGroup: GPUBindGroup | undefined, postGroup: GPUBindGroup | undefined;
      const dummyView = dummy.createView({ dimension: '2d-array' });
      let displacementView =
        spectral?.displacement.createView({ dimension: '2d-array' }) ?? dummyView;
      const makeGroup = (normal: GPUTextureView, reflection: GPUTextureView) =>
        device.createBindGroup({
          layout,
          entries: [
            { binding: 0, resource: { buffer: uniforms.buffer } },
            { binding: 1, resource: displacementView },
            { binding: 2, resource: normal },
            { binding: 3, resource: sampler },
            { binding: 4, resource: reflection },
            { binding: 5, resource: flow.createView() },
          ],
        });
      // Reflection pass cannot read its own attachment, even through an unused layout binding.
      const reflectionDummy = device.createTexture({
        size: [1, 1],
        format: 'rgba16float',
        usage: GPUTextureUsage.TEXTURE_BINDING,
      });
      const rebuildGroups = () => {
        sceneGroups.clear();
        reflectionGroup = makeGroup(dummyView, reflectionDummy.createView());
        if (sceneTarget)
          postGroup = device.createBindGroup({
            layout: postLayout,
            entries: [{ binding: 0, resource: sceneTarget.view }],
          });
      };
      rebuildGroups();
      return {
        resize(w, h) {
          width = w;
          height = h;
          depth.resize(w, h);
          if (mode === 6) reflectionTarget.resize(Math.max(1, w / 2), Math.max(1, h / 2));
          sceneTarget?.resize(w, h);
          sprites?.resize(w, h);
          rebuildGroups();
        },
        frame({ delta }, encoder, target) {
          phase += Math.min(delta, 0.1) * params.speed;
          if (spectral && spectral.size !== Number(params.fftSize)) {
            spectral.destroy();
            spectral = createSpectralOcean(device, Number(params.fftSize), profiler);
            displacementView = spectral.displacement.createView({ dimension: '2d-array' });
            rebuildGroups();
          }
          spectral?.run(encoder, phase, delta * params.speed, {
            strength: params.strength,
            chop: Number(params.chop ?? 0.85),
            foamThreshold: Number(params.foamThreshold ?? 0.75),
            foamLifetime: Number(params.foamLifetime ?? 3),
          });
          if (params.buoyancy === 'gpu')
            readback?.run(encoder, phase, params.strength, Number(params.chop ?? 0.85));
          const buoys = new Float32Array(BUOY_POINTS.length);
          if (
            mode === 0 &&
            params.buoyancy !== 'off' &&
            (params.buoyancy !== 'gpu' || !readback?.latest)
          )
            for (let i = 0; i < buoys.length; i += 4) {
              const w = sampleWorldWaves(
                BUOY_POINTS[i]!,
                BUOY_POINTS[i + 2]!,
                phase,
                params.strength,
                Number(params.chop ?? 0.85),
              );
              buoys.set([...w.position, phase], i);
            }
          if (params.buoyancy === 'gpu' && readback?.latest) buoys.set(readback.latest);
          const pitch = (params.cameraPitch * Math.PI) / 180;
          const eye: [number, number, number] = [
            mode === 2 ? 3 : 0,
            params.cameraHeight,
            mode === 4 ? 9 : 8,
          ];
          const camera = computeCamera({
            position: eye,
            target: [eye[0], eye[1] + Math.sin(pitch) * 20, eye[2] - Math.cos(pitch) * 20],
            fovY: Math.PI / 3,
            aspect: width / height,
            near: 0.1,
            far: 1200,
          });
          const mirrored = mirrorCamera(camera, horizontalPlane(0));
          uniforms.set({
            vp: camera.viewProj,
            inv_vp: camera.invViewProj,
            mirror_vp: mirrored.viewProj,
            mirror_inv: mirrored.invViewProj,
            eye,
            time: phase,
            mode,
            strategy: params.mesh === 'projected' ? 0 : 1,
            view: ['final', 'normals', 'foam', 'mesh'].indexOf(params.view),
            buoyancy: params.buoyancy && params.buoyancy !== 'off' ? 1 : 0,
            strength: params.strength,
            chop: Number(params.chop ?? 0.85),
            foam: Number(params.foam ?? 0),
            sun: params.sun,
            fog: params.fog,
            exposure: params.exposure,
            scattering: params.scattering,
            buoys,
          });
          uniforms.upload();
          const key = `${params.mesh}:${params.cells}`;
          if (key !== meshKey) {
            vertices?.destroy();
            indices?.destroy();
            const data = oceanMesh(
              params.mesh === 'clipmap' ? 'clipmap' : 'projected',
              Number(params.cells),
            );
            vertices = device.createBuffer({
              size: data.vertices.byteLength,
              usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
            });
            indices = device.createBuffer({
              size: data.indices.byteLength,
              usage: GPUBufferUsage.INDEX | GPUBufferUsage.COPY_DST,
            });
            device.queue.writeBuffer(vertices, 0, data.vertices);
            device.queue.writeBuffer(indices, 0, data.indices);
            indexCount = data.indices.length;
            meshKey = key;
          }
          const normal = spectral?.derivative ?? dummyView;
          let bind = sceneGroups.get(normal);
          if (!bind) {
            bind = makeGroup(normal, reflectionTarget.view);
            sceneGroups.set(normal, bind);
          }
          if (planarPipeline)
            renderFullscreen(encoder, reflectionTarget.view, planarPipeline, [reflectionGroup!], {
              timestampWrites: profiler.pass('lake planar reflection'),
            });
          const output = sceneTarget?.view ?? target;
          const pass = encoder.beginRenderPass({
            label: 'open water scene',
            colorAttachments: [{ view: output, loadOp: 'clear', storeOp: 'store' }],
            depthStencilAttachment: {
              view: depth.view,
              depthLoadOp: 'clear',
              depthStoreOp: 'store',
              depthClearValue: 1,
            },
            timestampWrites: profiler.pass('open water scene'),
          });
          pass.setBindGroup(0, bind);
          pass.setPipeline(background);
          pass.draw(3);
          pass.setPipeline(water);
          pass.setVertexBuffer(0, vertices!);
          pass.setIndexBuffer(indices!, 'uint32');
          pass.drawIndexed(indexCount);
          if (falls) {
            pass.setPipeline(falls);
            pass.draw(6);
          }
          pass.end();
          if (spray && sprites) {
            spray.run(encoder, phase, delta * params.speed);
            sprites.draw(encoder, spray.set, camera, output, {
              depth: depth.view,
              clearDepth: false,
              clear: null,
              timestamp: 'waterfall mist',
            });
          }
          if (post)
            renderFullscreen(encoder, target, post, [bind, postGroup!], {
              timestampWrites: profiler.pass('waterline meniscus'),
            });
        },
        destroy() {
          vertices?.destroy();
          indices?.destroy();
          uniforms.destroy();
          spectral?.destroy();
          readback?.destroy();
          spray?.destroy();
          sprites?.destroy();
          depth.destroy();
          reflectionTarget.destroy();
          sceneTarget?.destroy();
          dummy.destroy();
          reflectionDummy.destroy();
          flow.destroy();
          sceneGroups.clear();
        },
      };
    },
  });
}

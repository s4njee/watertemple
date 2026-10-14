import { createFullscreenPipeline, renderFullscreen } from '../../gpu/fullscreen.ts';
import { generateMipmaps, mipLevelCount } from '../../gpu/mipmap.ts';
import { createBlur, type Blur } from '../../gpu/passes/blur.ts';
import { downsample } from '../../gpu/passes/downsample.ts';
import { createRenderTarget, type RenderTarget } from '../../gpu/targets.ts';
import { createUniformBuffer, defineUniformStruct } from '../../gpu/uniforms.ts';
import { defineEffect } from '../types.ts';
import glassShader from './glass.wgsl?raw';
import { meta } from './meta.ts';
import sceneShader from './scene.wgsl?raw';

const SceneParams = defineUniformStruct('SceneParams', {
  aspect: 'f32',
  kind: 'i32',
  seed: 'i32',
});

const GlassParams = defineUniformStruct('GlassParams', {
  aspect: 'f32',
  aa: 'f32',
  time: 'f32',
  rain: 'f32',
  size: 'f32',
  fog: 'f32',
  refraction: 'f32',
  sharp_lod: 'f32',
  exposure: 'f32',
  layers: 'i32',
});

const SCENES = ['night street', 'dusk', 'grey day'] as const;

/** Offscreen HDR format: lights are far brighter than 1 and must survive the blur. */
const SCENE_FORMAT: GPUTextureFormat = 'rgba16float';

/** Width of one running-drop column, in image heights, at `dropSize` 1. */
const BASE_CELL = 0.085;

export const effect = defineEffect({
  meta,
  params: {
    rain: { type: 'number', default: 0.6, min: 0, max: 1, step: 0.01, label: 'Rain amount' },
    dropSize: { type: 'number', default: 1, min: 0.5, max: 2, step: 0.01, label: 'Drop size' },
    fog: { type: 'number', default: 0.8, min: 0, max: 1, step: 0.01, label: 'Fog / blur' },
    speed: { type: 'number', default: 1, min: 0, max: 3, step: 0.01 },
    refraction: {
      type: 'number',
      default: 3,
      min: 0.5,
      max: 6,
      step: 0.01,
      label: 'Refraction strength',
    },
    layers: {
      type: 'number',
      default: 3,
      min: 1,
      max: 3,
      step: 1,
      label: 'Drop layers (quality)',
    },
    scene: { type: 'select', default: 'night street', options: SCENES },
    brightness: { type: 'number', default: 1, min: 0.3, max: 2.5, step: 0.01 },
  },
  presets: {
    drizzle: {
      rain: 0.22,
      dropSize: 0.85,
      fog: 0.45,
      speed: 0.8,
      refraction: 2.6,
      layers: 3,
      scene: 'dusk',
      brightness: 1,
    },
    downpour: {
      rain: 1,
      dropSize: 1.15,
      fog: 0.7,
      speed: 1.9,
      refraction: 3.2,
      layers: 3,
      scene: 'night street',
      brightness: 1.1,
    },
    'fogged window': {
      rain: 0.35,
      dropSize: 1,
      fog: 1,
      speed: 0.7,
      refraction: 3,
      layers: 3,
      scene: 'night street',
      brightness: 1.2,
    },
    'grey day': {
      rain: 0.55,
      dropSize: 1.1,
      fog: 0.6,
      speed: 1,
      refraction: 2.6,
      layers: 2,
      scene: 'grey day',
      brightness: 1,
    },
  },

  async create({ device, format, params, profiler }) {
    const sceneUniforms = createUniformBuffer(device, SceneParams, 'rain-glass scene params');
    const glassUniforms = createUniformBuffer(device, GlassParams, 'rain-glass glass params');

    const scenePipeline = createFullscreenPipeline(device, {
      fragment: {
        label: 'rain-glass/scene.wgsl',
        code: sceneShader,
        chunks: { 'scene_params.wgsl': SceneParams.wgsl },
      },
      format: SCENE_FORMAT,
    });
    const glassPipeline = createFullscreenPipeline(device, {
      fragment: {
        label: 'rain-glass/glass.wgsl',
        code: glassShader,
        chunks: { 'glass_params.wgsl': GlassParams.wgsl },
      },
      format,
    });
    const sceneGroup = device.createBindGroup({
      label: 'rain-glass scene',
      layout: scenePipeline.getBindGroupLayout(0),
      entries: [{ binding: 0, resource: { buffer: sceneUniforms.buffer } }],
    });
    const sampler = device.createSampler({
      label: 'rain-glass sampler',
      magFilter: 'linear',
      minFilter: 'linear',
      mipmapFilter: 'linear',
    });

    let width = 1;
    let height = 1;
    let sharp: GPUTexture | undefined;
    let sharpMip0: GPUTextureView | undefined;
    let half: RenderTarget | undefined;
    let quarter: RenderTarget | undefined;
    let blurred: RenderTarget | undefined;
    let blur: Blur | undefined;
    let glassGroup: GPUBindGroup | undefined;
    let builtKey = '';
    let sceneWidth = 1;
    let sceneHeight = 1;
    let phase = 0;

    function releaseTargets() {
      sharp?.destroy();
      half?.destroy();
      quarter?.destroy();
      blurred?.destroy();
      blur?.destroy();
    }

    // The scene behind the glass is drawn at reduced size: a drop shows it shrunk, and the fog is
    // blurred anyway. Long edge at most 1024 texels, and half the canvas at most.
    function allocate() {
      releaseTargets();
      const scale = Math.min(0.5, 1024 / Math.max(width, height));
      sceneWidth = Math.max(8, Math.round(width * scale));
      sceneHeight = Math.max(8, Math.round(height * scale));
      sharp = device.createTexture({
        label: 'rain-glass scene',
        size: [sceneWidth, sceneHeight],
        format: SCENE_FORMAT,
        mipLevelCount: mipLevelCount(sceneWidth, sceneHeight),
        usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.RENDER_ATTACHMENT,
      });
      sharpMip0 = sharp.createView({ baseMipLevel: 0, mipLevelCount: 1 });
      const hw = Math.max(2, sceneWidth >> 1);
      const hh = Math.max(2, sceneHeight >> 1);
      const qw = Math.max(2, sceneWidth >> 2);
      const qh = Math.max(2, sceneHeight >> 2);
      half = createRenderTarget(device, {
        label: 'rain-glass half',
        format: SCENE_FORMAT,
        width: hw,
        height: hh,
      });
      quarter = createRenderTarget(device, {
        label: 'rain-glass quarter',
        format: SCENE_FORMAT,
        width: qw,
        height: qh,
      });
      blurred = createRenderTarget(device, {
        label: 'rain-glass fog',
        format: SCENE_FORMAT,
        width: qw,
        height: qh,
      });
      blur = createBlur(device, {
        label: 'rain-glass blur',
        format: SCENE_FORMAT,
        width: qw,
        height: qh,
      });
      glassGroup = device.createBindGroup({
        label: 'rain-glass glass',
        layout: glassPipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: glassUniforms.buffer } },
          { binding: 1, resource: sharp.createView() },
          { binding: 2, resource: blurred.view },
          { binding: 3, resource: sampler },
        ],
      });
      builtKey = '';
    }

    return {
      resize(w, h) {
        width = Math.max(1, Math.round(w));
        height = Math.max(1, Math.round(h));
        allocate();
      },

      frame({ delta }, encoder, target) {
        if (!sharp || !sharpMip0 || !half || !quarter || !blurred || !blur || !glassGroup) return;
        const aspect = width / height;
        const sceneIndex = Math.max(0, SCENES.indexOf(params.scene as (typeof SCENES)[number]));
        // Blur radius in texels of the quarter-size copy (an eighth of the canvas, at most).
        const radius = Math.round(1 + 9 * params.fog);

        // The scene and its blur chain are only redone when something they depend on changes.
        const key = `${sceneIndex}:${radius}:${width}x${height}`;
        if (key !== builtKey) {
          builtKey = key;
          sceneUniforms.set({ aspect, kind: sceneIndex, seed: 7 });
          sceneUniforms.upload();
          renderFullscreen(encoder, sharpMip0, scenePipeline, [sceneGroup], {
            timestampWrites: profiler.pass('scene (on change)'),
          });
          generateMipmaps(device, sharp, encoder);
          downsample(device, encoder, sharpMip0, half);
          downsample(device, encoder, half.view, quarter);
          blur.encode(encoder, quarter.view, blurred.view, { radius });
        }

        phase += delta * params.speed;
        const ratio = sceneWidth / width;
        glassUniforms.set({
          aspect,
          aa: 1 / height,
          time: phase,
          rain: params.rain,
          size: BASE_CELL * params.dropSize,
          fog: params.fog,
          refraction: params.refraction,
          sharp_lod: Math.log2(Math.max(1, (1 + params.refraction) * ratio)),
          exposure: params.brightness,
          layers: Math.round(params.layers),
        });
        glassUniforms.upload();
        renderFullscreen(encoder, target, glassPipeline, [glassGroup], {
          timestampWrites: profiler.pass('glass'),
        });
      },

      destroy() {
        releaseTargets();
        sceneUniforms.destroy();
        glassUniforms.destroy();
      },
    };
  },
});

import { createFullscreenPipeline, renderFullscreen } from '../../gpu/fullscreen.ts';
import { generateMipmaps, mipLevelCount } from '../../gpu/mipmap.ts';
import { trackPointer } from '../../gpu/pointer.ts';
import { createUniformBuffer, defineUniformStruct } from '../../gpu/uniforms.ts';
import { defineEffect } from '../types.ts';
import { meta } from './meta.ts';
import sceneShader from './scene.wgsl';
import waterShader from './water.wgsl';

const SceneParams = defineUniformStruct('SceneParams', {
  aspect: 'f32',
  aa: 'f32',
  density: 'f32',
  kind: 'i32',
});

const WaterParams = defineUniformStruct('WaterParams', {
  pointer: 'vec2f',
  aspect: 'f32',
  time: 'f32',
  tint: 'vec3f',
  ior: 'f32',
  thickness: 'f32',
  dispersion: 'f32',
  ripple_scale: 'f32',
  ripple_amount: 'f32',
  blur_px: 'f32',
  max_lod: 'f32',
  fade: 'f32',
  lens: 'f32',
  ring: 'f32',
  ring_phase: 'f32',
  shine: 'f32',
  taps: 'i32',
  model: 'u32',
  quality: 'u32',
  view: 'u32',
});

const BACKDROPS = ['pool tiles', 'test card', 'sand'] as const;
const MODELS = ['physical', 'cheap'] as const;
const QUALITIES = ['3 taps', 'spectral'] as const;
const VIEWS = ['refracted', 'split', 'normals', 'offset'] as const;

const BACKDROP_FORMAT: GPUTextureFormat = 'rgba8unorm-srgb';

export const effect = defineEffect({
  meta,
  params: {
    backdrop: { type: 'select', default: 'pool tiles', options: BACKDROPS },
    ior: { type: 'number', default: 1.333, min: 1, max: 2.4, step: 0.001, label: 'IOR' },
    depth: {
      type: 'number',
      default: 0.4,
      min: 0,
      max: 1.5,
      step: 0.01,
      label: 'Water depth (image heights)',
    },
    dispersion: {
      type: 'number',
      default: 0.02,
      min: 0,
      max: 0.12,
      step: 0.001,
      label: 'Dispersion (IOR spread)',
    },
    quality: { type: 'select', default: '3 taps', options: QUALITIES, label: 'Dispersion taps' },
    model: { type: 'select', default: 'physical', options: MODELS, label: 'Offset model' },
    rippleScale: { type: 'number', default: 3.2, min: 0.5, max: 12, step: 0.1 },
    rippleSpeed: { type: 'number', default: 0.5, min: 0, max: 3, step: 0.01 },
    rippleAmount: { type: 'number', default: 0.35, min: 0, max: 1.5, step: 0.01 },
    blur: { type: 'number', default: 0, min: 0, max: 1, step: 0.01, label: 'Roughness (mip blur)' },
    pointer: { type: 'number', default: 0.7, min: 0, max: 1.5, step: 0.01, label: 'Pointer lens' },
    shine: { type: 'number', default: 0.35, min: 0, max: 1, step: 0.01 },
    tint: { type: 'color', default: [0.8, 0.95, 1] },
    view: { type: 'select', default: 'refracted', options: VIEWS },
  },
  presets: {
    'still pool': {
      backdrop: 'pool tiles',
      ior: 1.333,
      depth: 0.4,
      dispersion: 0.012,
      quality: '3 taps',
      model: 'physical',
      rippleScale: 2.6,
      rippleSpeed: 0.35,
      rippleAmount: 0.3,
      blur: 0,
      pointer: 0.7,
      shine: 0.3,
    },
    'shallow stream': {
      backdrop: 'sand',
      ior: 1.333,
      depth: 0.22,
      dispersion: 0.006,
      quality: '3 taps',
      model: 'cheap',
      rippleScale: 7,
      rippleSpeed: 1.6,
      rippleAmount: 0.8,
      blur: 0,
      pointer: 0.5,
      shine: 0.6,
    },
    'heavy glass': {
      backdrop: 'test card',
      ior: 1.52,
      depth: 1.1,
      dispersion: 0.06,
      quality: 'spectral',
      model: 'physical',
      rippleScale: 1.6,
      rippleSpeed: 0.25,
      rippleAmount: 0.4,
      blur: 0,
      pointer: 1,
      shine: 0.15,
    },
    'rough deep water': {
      backdrop: 'pool tiles',
      ior: 1.333,
      depth: 1.2,
      dispersion: 0.02,
      quality: '3 taps',
      model: 'physical',
      rippleScale: 3.5,
      rippleSpeed: 0.7,
      rippleAmount: 0.7,
      blur: 0.7,
      pointer: 0.5,
      shine: 0.4,
    },
  },

  async create({ device, format, canvas, params, profiler }) {
    const sceneUniforms = createUniformBuffer(device, SceneParams, 'refraction scene params');
    const waterUniforms = createUniformBuffer(device, WaterParams, 'refraction water params');
    const pointer = trackPointer(canvas);

    const scenePipeline = createFullscreenPipeline(device, {
      fragment: {
        label: 'refraction/scene.wgsl',
        code: sceneShader.code,
        chunks: { ...sceneShader.chunks, 'scene_params.wgsl': SceneParams.wgsl },
      },
      format: BACKDROP_FORMAT,
    });
    const waterPipeline = createFullscreenPipeline(device, {
      fragment: {
        label: 'refraction/water.wgsl',
        code: waterShader.code,
        chunks: { ...waterShader.chunks, 'water_params.wgsl': WaterParams.wgsl },
      },
      format,
    });
    const sceneGroup = device.createBindGroup({
      label: 'refraction scene',
      layout: scenePipeline.getBindGroupLayout(0),
      entries: [{ binding: 0, resource: { buffer: sceneUniforms.buffer } }],
    });
    const sampler = device.createSampler({
      label: 'refraction backdrop sampler',
      magFilter: 'linear',
      minFilter: 'linear',
      mipmapFilter: 'linear',
    });

    let width = 1;
    let height = 1;
    let backdrop: GPUTexture | undefined;
    let backdropMip0: GPUTextureView | undefined;
    let waterGroup: GPUBindGroup | undefined;
    let drawnKey = '';

    // The backdrop is the "image": it is only redrawn when it changes. A video backdrop would be
    // re-uploaded (and its mips regenerated) every frame instead.
    function allocateBackdrop() {
      backdrop?.destroy();
      backdrop = device.createTexture({
        label: 'refraction backdrop',
        size: [width, height],
        format: BACKDROP_FORMAT,
        mipLevelCount: mipLevelCount(width, height),
        usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.RENDER_ATTACHMENT,
      });
      backdropMip0 = backdrop.createView({ baseMipLevel: 0, mipLevelCount: 1 });
      waterGroup = device.createBindGroup({
        label: 'refraction water',
        layout: waterPipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: waterUniforms.buffer } },
          { binding: 1, resource: backdrop.createView() },
          { binding: 2, resource: sampler },
        ],
      });
      drawnKey = '';
    }

    let phase = 0;
    let ringPhase = 0;
    let follow = 0;
    let energy = 0.1;

    return {
      resize(w, h) {
        width = Math.max(1, Math.round(w));
        height = Math.max(1, Math.round(h));
        allocateBackdrop();
      },

      frame({ delta, time }, encoder, target) {
        if (!backdrop || !backdropMip0 || !waterGroup) return;
        const aspect = width / height;

        const key = `${params.backdrop}:${width}x${height}`;
        if (key !== drawnKey) {
          drawnKey = key;
          sceneUniforms.set({
            aspect,
            aa: 1 / height,
            density: 1,
            kind: Math.max(0, BACKDROPS.indexOf(params.backdrop as (typeof BACKDROPS)[number])),
          });
          sceneUniforms.upload();
          renderFullscreen(encoder, backdropMip0, scenePipeline, [sceneGroup], {
            timestampWrites: profiler.pass('backdrop (on change)'),
          });
          generateMipmaps(device, backdrop, encoder);
        }

        // Ambient pointer: it follows the cursor while the cursor is over the canvas, and a slow
        // wander keeps the lens visible when nobody is pointing.
        const state = pointer.state;
        const moved = pointer.takeDelta();
        const speed = Math.hypot(moved.dx * aspect, moved.dy) / Math.max(delta, 1e-3);
        const blend = Math.min(1, delta * 4);
        follow += ((state.inside ? 1 : 0) - follow) * blend;
        energy += ((state.inside ? Math.min(1, speed * 1.2) + 0.12 : 0.1) - energy) * blend;
        const wanderX = 0.5 + 0.28 * Math.sin(time * 0.31);
        const wanderY = 0.5 + 0.2 * Math.sin(time * 0.47 + 1);
        const x = wanderX + (state.x - wanderX) * follow;
        const y = wanderY + (state.y - wanderY) * follow;

        phase += delta * params.rippleSpeed;
        ringPhase += delta * 7;

        const lensWeight = 0.45 + 0.55 * follow;
        waterUniforms.set({
          pointer: [x * aspect, y],
          aspect,
          time: phase,
          tint: params.tint,
          ior: params.ior,
          thickness: params.depth,
          dispersion: params.dispersion,
          ripple_scale: params.rippleScale,
          ripple_amount: params.rippleAmount,
          blur_px: params.blur * 0.035 * height * (0.3 + params.depth),
          max_lod: Math.max(0, backdrop.mipLevelCount - 1),
          fade: 0.05,
          lens: params.pointer * lensWeight * 0.9,
          ring: params.pointer * energy * 0.5,
          ring_phase: ringPhase,
          shine: params.shine,
          taps: 6,
          model: Math.max(0, MODELS.indexOf(params.model as (typeof MODELS)[number])),
          quality: Math.max(0, QUALITIES.indexOf(params.quality as (typeof QUALITIES)[number])),
          view: Math.max(0, VIEWS.indexOf(params.view as (typeof VIEWS)[number])),
        });
        waterUniforms.upload();
        renderFullscreen(encoder, target, waterPipeline, [waterGroup], {
          timestampWrites: profiler.pass('water'),
        });
      },

      destroy() {
        pointer.destroy();
        backdrop?.destroy();
        sceneUniforms.destroy();
        waterUniforms.destroy();
      },
    };
  },
});

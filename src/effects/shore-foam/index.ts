import { createFullscreenPipeline, renderFullscreen } from '../../gpu/fullscreen.ts';
import { createRenderTarget, type RenderTarget } from '../../gpu/targets.ts';
import { createUniformBuffer, defineUniformStruct } from '../../gpu/uniforms.ts';
import { defineEffect } from '../types.ts';
import groundShader from './ground.wgsl';
import { meta } from './meta.ts';
import surfShader from './surf.wgsl';

const GroundParams = defineUniformStruct('GroundParams', {
  aspect: 'f32',
  px: 'f32',
});

const SurfParams = defineUniformStruct('SurfParams', {
  aspect: 'f32',
  wave_cycles: 'f32',
  travel: 'f32',
  runup: 'f32',
  foam_time: 'f32',
  ripple_time: 'f32',
  foam_width: 'f32',
  breakup: 'f32',
  foam_scale: 'f32',
  foam_amount: 'f32',
  edge_soft: 'f32',
  wet_strength: 'f32',
  shimmer: 'f32',
  water_type: 'i32',
});

/** The first four `absorb_water` kinds suit a beach. Index = kind. */
const WATER_TYPES = ['clear tropical', 'coastal', 'murky green', 'tea lake'] as const;

const GROUND_FORMATS: [GPUTextureFormat, GPUTextureFormat] = ['rgba16float', 'rg16float'];

export const effect = defineEffect({
  meta,
  params: {
    waterType: { type: 'select', default: 'clear tropical', options: WATER_TYPES },
    wavePeriod: {
      type: 'number',
      default: 8,
      min: 2,
      max: 16,
      step: 0.1,
      label: 'Wave period (s)',
    },
    waveHeight: {
      type: 'number',
      default: 0.5,
      min: 0.05,
      max: 1.4,
      step: 0.01,
      label: 'Run-up (m)',
    },
    foamWidth: {
      type: 'number',
      default: 0.28,
      min: 0.04,
      max: 0.9,
      step: 0.01,
      label: 'Foam width (m)',
    },
    breakup: { type: 'number', default: 0.7, min: 0, max: 1, step: 0.01, label: 'Foam break-up' },
    foamScale: {
      type: 'number',
      default: 24,
      min: 8,
      max: 60,
      step: 0.5,
      label: 'Foam bubble scale',
    },
    foamAmount: { type: 'number', default: 1, min: 0, max: 1.6, step: 0.01, label: 'Foam amount' },
    foamSpeed: { type: 'number', default: 1, min: 0, max: 3, step: 0.01, label: 'Foam boil speed' },
    edgeSoftness: {
      type: 'number',
      default: 0.1,
      min: 0.005,
      max: 0.5,
      step: 0.005,
      label: 'Edge softness (m)',
    },
    wetSand: {
      type: 'number',
      default: 0.9,
      min: 0,
      max: 1.6,
      step: 0.01,
      label: 'Wet-sand darkening',
    },
    shimmer: { type: 'number', default: 0.3, min: 0, max: 1, step: 0.01, label: 'Light shimmer' },
  },
  presets: {
    'calm lagoon': {
      waterType: 'clear tropical',
      wavePeriod: 11,
      waveHeight: 0.28,
      foamWidth: 0.16,
      breakup: 0.55,
      foamScale: 30,
      foamAmount: 0.85,
      foamSpeed: 0.6,
      edgeSoftness: 0.14,
      wetSand: 0.7,
      shimmer: 0.35,
    },
    'lively surf': {
      waterType: 'coastal',
      wavePeriod: 5,
      waveHeight: 0.95,
      foamWidth: 0.5,
      breakup: 0.85,
      foamScale: 20,
      foamAmount: 1.25,
      foamSpeed: 1.6,
      edgeSoftness: 0.07,
      wetSand: 1.1,
      shimmer: 0.2,
    },
    'murky tide': {
      waterType: 'murky green',
      wavePeriod: 9,
      waveHeight: 0.6,
      foamWidth: 0.32,
      breakup: 0.75,
      foamScale: 26,
      foamAmount: 1,
      foamSpeed: 0.9,
      edgeSoftness: 0.2,
      wetSand: 0.9,
      shimmer: 0.1,
    },
  },

  async create({ device, format, params, profiler }) {
    const groundUniforms = createUniformBuffer(device, GroundParams, 'shore-foam ground params');
    const surfUniforms = createUniformBuffer(device, SurfParams, 'shore-foam surf params');

    const groundPipeline = createFullscreenPipeline(device, {
      fragment: {
        label: 'shore-foam/ground.wgsl',
        code: groundShader.code,
        chunks: { ...groundShader.chunks, 'ground_params.wgsl': GroundParams.wgsl },
      },
      format: GROUND_FORMATS,
    });
    const surfPipeline = createFullscreenPipeline(device, {
      fragment: {
        label: 'shore-foam/surf.wgsl',
        code: surfShader.code,
        chunks: { ...surfShader.chunks, 'surf_params.wgsl': SurfParams.wgsl },
      },
      format,
    });
    const groundGroup = device.createBindGroup({
      label: 'shore-foam ground',
      layout: groundPipeline.getBindGroupLayout(0),
      entries: [{ binding: 0, resource: { buffer: groundUniforms.buffer } }],
    });
    const sampler = device.createSampler({
      label: 'shore-foam sampler',
      magFilter: 'linear',
      minFilter: 'linear',
    });

    let width = 1;
    let height = 1;
    let ground: RenderTarget | undefined;
    let aux: RenderTarget | undefined;
    let surfGroup: GPUBindGroup | undefined;
    let baked = false;
    // Accumulated, so changing a period or speed slider never makes the waves jump.
    let waveCycles = 0;
    let foamTime = 0;
    let rippleTime = 0;

    return {
      resize(w, h) {
        width = Math.max(1, Math.round(w));
        height = Math.max(1, Math.round(h));
        if (!ground || !aux) {
          ground = createRenderTarget(device, {
            label: 'shore-foam ground',
            format: GROUND_FORMATS[0],
            width,
            height,
          });
          aux = createRenderTarget(device, {
            label: 'shore-foam aux',
            format: GROUND_FORMATS[1],
            width,
            height,
          });
        } else {
          ground.resize(width, height);
          aux.resize(width, height);
        }
        surfGroup = device.createBindGroup({
          label: 'shore-foam surf',
          layout: surfPipeline.getBindGroupLayout(0),
          entries: [
            { binding: 0, resource: { buffer: surfUniforms.buffer } },
            { binding: 1, resource: ground.view },
            { binding: 2, resource: aux.view },
            { binding: 3, resource: sampler },
          ],
        });
        baked = false;
      },

      frame({ delta }, encoder, target) {
        if (!ground || !aux || !surfGroup) return;
        const aspect = width / height;

        if (!baked) {
          baked = true;
          groundUniforms.set({ aspect, px: 1 / height });
          groundUniforms.upload();
          renderFullscreen(encoder, [ground.view, aux.view], groundPipeline, [groundGroup], {
            timestampWrites: profiler.pass('bake beach (on resize)'),
          });
        }

        waveCycles += delta / params.wavePeriod;
        foamTime += delta * params.foamSpeed;
        rippleTime += delta * 0.6;
        surfUniforms.set({
          aspect,
          wave_cycles: waveCycles,
          travel: 0.7,
          runup: params.waveHeight,
          foam_time: foamTime,
          ripple_time: rippleTime,
          foam_width: params.foamWidth,
          breakup: params.breakup,
          foam_scale: params.foamScale,
          foam_amount: params.foamAmount,
          edge_soft: params.edgeSoftness,
          wet_strength: params.wetSand,
          shimmer: params.shimmer,
          water_type: Math.max(
            0,
            WATER_TYPES.indexOf(params.waterType as (typeof WATER_TYPES)[number]),
          ),
        });
        surfUniforms.upload();
        renderFullscreen(encoder, target, surfPipeline, [surfGroup], {
          timestampWrites: profiler.pass('surf'),
        });
      },

      destroy() {
        ground?.destroy();
        aux?.destroy();
        groundUniforms.destroy();
        surfUniforms.destroy();
      },
    };
  },
});

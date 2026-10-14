import { createFullscreenPipeline, renderFullscreen } from '../../gpu/fullscreen.ts';
import { createUniformBuffer, defineUniformStruct } from '../../gpu/uniforms.ts';
import { defineEffect } from '../types.ts';
import shader from './crest-sss.wgsl';
import { meta } from './meta.ts';

const Params = defineUniformStruct('Params', {
  time: 'f32',
  aspect: 'f32',
  pixel_angle: 'f32',
  pitch: 'f32',
  fov: 'f32',
  cam_height: 'f32',
  sun_elevation: 'f32',
  sun_azimuth: 'f32',
  wave_strength: 'f32',
  wave_scale: 'f32',
  exposure: 'f32',
  haze: 'f32',
  sss_strength: 'f32',
  sss_distortion: 'f32',
  sss_power: 'f32',
  sss_on: 'u32',
  view_mode: 'u32',
  body: 'vec3f',
  scatter: 'vec3f',
});

const DEG = Math.PI / 180;
const FOV = 60 * DEG;
const VIEWS = ['final', 'sss only', 'crest mask'] as const;

export const effect = defineEffect({
  meta,
  params: {
    sss: { type: 'boolean', default: true, label: 'Subsurface scattering' },
    sssStrength: { type: 'number', default: 2, min: 0, max: 8, step: 0.05, label: 'SSS strength' },
    scatterColor: { type: 'color', default: [0.03, 0.5, 0.32], label: 'Scatter colour' },
    sssDistortion: {
      type: 'number',
      default: 0.6,
      min: 0,
      max: 1,
      step: 0.01,
      label: 'SSS distortion',
    },
    sssPower: { type: 'number', default: 3, min: 1, max: 16, step: 0.1, label: 'SSS power' },
    view: { type: 'select', default: 'final', options: VIEWS, label: 'View' },
    sunElevation: {
      type: 'number',
      default: 12,
      min: 1,
      max: 80,
      step: 0.5,
      label: 'Sun elevation (deg)',
    },
    sunAzimuth: {
      type: 'number',
      default: 8,
      min: -90,
      max: 90,
      step: 1,
      label: 'Sun azimuth (deg)',
    },
    waveStrength: {
      type: 'number',
      default: 2,
      min: 0,
      max: 3,
      step: 0.01,
      label: 'Wave strength',
    },
    waveScale: { type: 'number', default: 2.2, min: 0.5, max: 4, step: 0.05, label: 'Wave size' },
    cameraHeight: {
      type: 'number',
      default: 5.5,
      min: 1,
      max: 30,
      step: 0.1,
      label: 'Camera height (m)',
    },
    cameraPitch: {
      type: 'number',
      default: -5,
      min: -30,
      max: 10,
      step: 0.5,
      label: 'Camera pitch (deg)',
    },
    haze: { type: 'number', default: 0.35, min: 0, max: 1, step: 0.01 },
    deepColor: { type: 'color', default: [0.006, 0.04, 0.07], label: 'Deep water colour' },
    exposure: { type: 'number', default: 1, min: 0.2, max: 3, step: 0.01 },
    speed: { type: 'number', default: 1, min: 0, max: 2, step: 0.01 },
  },
  presets: {
    'backlit swell': {},
    tropical: {
      sunElevation: 18,
      sunAzimuth: 15,
      waveStrength: 1.5,
      sssStrength: 1.4,
      scatterColor: [0.02, 0.62, 0.42],
      sssDistortion: 0.6,
      sssPower: 5,
      deepColor: [0.004, 0.05, 0.08],
      haze: 0.25,
    },
    'storm light': {
      sunElevation: 6,
      sunAzimuth: -10,
      waveStrength: 3,
      waveScale: 2.8,
      sssStrength: 2.4,
      scatterColor: [0.1, 0.35, 0.22],
      sssPower: 4,
      deepColor: [0.01, 0.03, 0.04],
      haze: 0.6,
      cameraHeight: 8,
    },
    'sss off': { sss: false },
  },

  async create({ device, format, params, profiler }) {
    const uniforms = createUniformBuffer(device, Params, 'crest-sss params');
    const pipeline = createFullscreenPipeline(device, {
      fragment: {
        label: 'crest-sss/crest-sss.wgsl',
        code: shader.code,
        chunks: { ...shader.chunks, 'params.wgsl': Params.wgsl },
      },
      format,
    });
    const bindGroup = device.createBindGroup({
      label: 'crest-sss',
      layout: pipeline.getBindGroupLayout(0),
      entries: [{ binding: 0, resource: { buffer: uniforms.buffer } }],
    });

    let phase = 0;

    return {
      resize(width, height) {
        uniforms.set({
          aspect: width / Math.max(1, height),
          pixel_angle: (2 * Math.tan(FOV / 2)) / Math.max(1, height),
        });
      },
      frame({ delta }, encoder, target) {
        phase += delta * params.speed;
        uniforms.set({
          time: phase,
          pitch: params.cameraPitch * DEG,
          fov: FOV,
          cam_height: params.cameraHeight,
          sun_elevation: params.sunElevation * DEG,
          sun_azimuth: params.sunAzimuth * DEG,
          wave_strength: params.waveStrength,
          wave_scale: params.waveScale,
          exposure: params.exposure,
          haze: params.haze,
          sss_strength: params.sssStrength,
          sss_distortion: params.sssDistortion,
          sss_power: params.sssPower,
          sss_on: params.sss ? 1 : 0,
          view_mode: Math.max(0, VIEWS.indexOf(params.view as (typeof VIEWS)[number])),
          body: params.deepColor,
          scatter: params.scatterColor,
        });
        uniforms.upload();
        renderFullscreen(encoder, target, pipeline, [bindGroup], {
          timestampWrites: profiler.pass('water'),
        });
      },
      destroy() {
        uniforms.destroy();
      },
    };
  },
});

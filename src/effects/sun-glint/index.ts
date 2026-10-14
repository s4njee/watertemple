import { createFullscreenPipeline, renderFullscreen } from '../../gpu/fullscreen.ts';
import { createUniformBuffer, defineUniformStruct } from '../../gpu/uniforms.ts';
import { defineEffect } from '../types.ts';
import { meta } from './meta.ts';
import shader from './sun-glint.wgsl?raw';

const Params = defineUniformStruct('Params', {
  time: 'f32',
  aspect: 'f32',
  pixel_angle: 'f32',
  pitch: 'f32',
  fov: 'f32',
  cam_height: 'f32',
  drift: 'f32',
  sun_elevation: 'f32',
  sun_azimuth: 'f32',
  sun_gain: 'f32',
  wave_strength: 'f32',
  wave_scale: 'f32',
  exposure: 'f32',
  haze: 'f32',
  roughness: 'f32',
  sparkle_density: 'f32',
  sparkle_intensity: 'f32',
  sparkle_sharpness: 'f32',
  sparkle_cell: 'f32',
  view_mode: 'u32',
  body: 'vec3f',
});

const DEG = Math.PI / 180;
const FOV = 60 * DEG;
const VIEWS = ['final', 'lobe only', 'sparkle only'] as const;

export const effect = defineEffect({
  meta,
  params: {
    sunElevation: {
      type: 'number',
      default: 14,
      min: 1,
      max: 80,
      step: 0.5,
      label: 'Sun elevation (deg)',
    },
    sunAzimuth: {
      type: 'number',
      default: 0,
      min: -90,
      max: 90,
      step: 1,
      label: 'Sun azimuth (deg)',
    },
    sunIntensity: {
      type: 'number',
      default: 1,
      min: 0,
      max: 3,
      step: 0.01,
      label: 'Sun intensity',
    },
    roughness: {
      type: 'number',
      default: 0.12,
      min: 0,
      max: 0.6,
      step: 0.005,
      label: 'Lobe roughness',
    },
    waveStrength: {
      type: 'number',
      default: 0.7,
      min: 0,
      max: 3,
      step: 0.01,
      label: 'Wave strength',
    },
    sparkleDensity: {
      type: 'number',
      default: 0.5,
      min: 0,
      max: 1,
      step: 0.01,
      label: 'Sparkle density',
    },
    sparkleIntensity: {
      type: 'number',
      default: 6,
      min: 0,
      max: 30,
      step: 0.1,
      label: 'Sparkle intensity',
    },
    sparkleSharpness: {
      type: 'number',
      default: 1,
      min: 0.25,
      max: 8,
      step: 0.05,
      label: 'Sparkle sharpness',
    },
    sparkleSize: {
      type: 'number',
      default: 0.06,
      min: 0.02,
      max: 0.3,
      step: 0.005,
      label: 'Sparkle cell size (m)',
    },
    view: { type: 'select', default: 'final', options: VIEWS, label: 'View' },
    cameraHeight: {
      type: 'number',
      default: 2.5,
      min: 0.5,
      max: 20,
      step: 0.1,
      label: 'Camera height (m)',
    },
    cameraPitch: {
      type: 'number',
      default: -3,
      min: -30,
      max: 10,
      step: 0.5,
      label: 'Camera pitch (deg)',
    },
    drift: { type: 'number', default: 0, min: -4, max: 4, step: 0.05, label: 'Camera drift (m/s)' },
    haze: { type: 'number', default: 0.35, min: 0, max: 1, step: 0.01 },
    deepColor: { type: 'color', default: [0.01, 0.06, 0.09], label: 'Deep water colour' },
    exposure: { type: 'number', default: 1, min: 0.2, max: 3, step: 0.01 },
    speed: { type: 'number', default: 1, min: 0, max: 2, step: 0.01 },
  },
  presets: {
    'calm noon': {
      sunElevation: 40,
      sunAzimuth: 0,
      cameraPitch: -20,
      roughness: 0.05,
      waveStrength: 0.35,
      sparkleDensity: 0.35,
      sparkleIntensity: 5,
      haze: 0.25,
    },
    'golden hour': {
      sunElevation: 7,
      sunAzimuth: 0,
      roughness: 0.14,
      waveStrength: 0.8,
      sparkleDensity: 0.6,
      sparkleIntensity: 7,
      haze: 0.45,
      exposure: 1.05,
    },
    'glitter path': {
      sunElevation: 16,
      sunAzimuth: 0,
      roughness: 0.2,
      waveStrength: 1.2,
      sparkleDensity: 0.85,
      sparkleIntensity: 9,
      sparkleSharpness: 1.6,
      sparkleSize: 0.04,
    },
    'no sparkle': { sparkleDensity: 0, sparkleIntensity: 0 },
  },

  async create({ device, format, params, profiler }) {
    const uniforms = createUniformBuffer(device, Params, 'sun-glint params');
    const pipeline = createFullscreenPipeline(device, {
      fragment: {
        label: 'sun-glint/sun-glint.wgsl',
        code: shader,
        chunks: { 'params.wgsl': Params.wgsl },
      },
      format,
    });
    const bindGroup = device.createBindGroup({
      label: 'sun-glint',
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
          drift: params.drift,
          sun_elevation: params.sunElevation * DEG,
          sun_azimuth: params.sunAzimuth * DEG,
          sun_gain: params.sunIntensity,
          wave_strength: params.waveStrength,
          wave_scale: 1,
          exposure: params.exposure,
          haze: params.haze,
          roughness: params.roughness,
          sparkle_density: params.sparkleDensity,
          sparkle_intensity: params.sparkleIntensity,
          sparkle_sharpness: params.sparkleSharpness,
          sparkle_cell: params.sparkleSize,
          view_mode: Math.max(0, VIEWS.indexOf(params.view as (typeof VIEWS)[number])),
          body: params.deepColor,
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

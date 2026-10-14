import { createCubemap } from '../../gpu/cubemap.ts';
import { createFullscreenPipeline, renderFullscreen } from '../../gpu/fullscreen.ts';
import { createUniformBuffer, defineUniformStruct } from '../../gpu/uniforms.ts';
import { defineEffect } from '../types.ts';
import shader from './fresnel-sky.wgsl';
import { meta } from './meta.ts';
import skyCube from './sky-cube.wgsl';

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
  f0: 'f32',
  haze: 'f32',
  use_cube: 'u32',
  view_mode: 'u32',
  body: 'vec3f',
});

const DEG = Math.PI / 180;
const FOV = 60 * DEG;
const VIEWS = ['final', 'fresnel', 'reflection', 'body'] as const;
const CUBE_SIZE = 256;

export const effect = defineEffect({
  meta,
  params: {
    sunElevation: {
      type: 'number',
      default: 28,
      min: -10,
      max: 85,
      step: 0.5,
      label: 'Sun elevation (deg)',
    },
    sunAzimuth: {
      type: 'number',
      default: 20,
      min: -180,
      max: 180,
      step: 1,
      label: 'Sun azimuth (deg)',
    },
    waveStrength: {
      type: 'number',
      default: 1,
      min: 0,
      max: 3,
      step: 0.01,
      label: 'Wave strength',
    },
    source: {
      type: 'select',
      default: 'procedural',
      options: ['procedural', 'cubemap'],
      label: 'Sky source',
    },
    view: { type: 'select', default: 'final', options: VIEWS, label: 'View' },
    ior: {
      type: 'number',
      default: 1.333,
      min: 1.05,
      max: 2.4,
      step: 0.001,
      label: 'Index of refraction',
    },
    haze: { type: 'number', default: 0.3, min: 0, max: 1, step: 0.01 },
    cameraHeight: {
      type: 'number',
      default: 3,
      min: 0.5,
      max: 30,
      step: 0.1,
      label: 'Camera height (m)',
    },
    cameraPitch: {
      type: 'number',
      default: -4,
      min: -30,
      max: 15,
      step: 0.5,
      label: 'Camera pitch (deg)',
    },
    deepColor: { type: 'color', default: [0.01, 0.07, 0.1], label: 'Deep water colour' },
    exposure: { type: 'number', default: 1, min: 0.2, max: 3, step: 0.01 },
    speed: { type: 'number', default: 1, min: 0, max: 2, step: 0.01 },
  },
  presets: {
    noon: { sunElevation: 62, sunAzimuth: 30, haze: 0.25, waveStrength: 0.8 },
    sunset: { sunElevation: 3, sunAzimuth: 12, haze: 0.4, waveStrength: 0.9, exposure: 1.1 },
    glassy: { sunElevation: 35, waveStrength: 0.12, cameraHeight: 2 },
    choppy: { sunElevation: 40, waveStrength: 2.4, cameraHeight: 6, cameraPitch: -8 },
    'cubemap sky': { source: 'cubemap', sunElevation: 20, waveStrength: 0.6 },
  },

  async create({ device, format, params, profiler }) {
    const uniforms = createUniformBuffer(device, Params, 'fresnel-sky params');
    const cube = createCubemap(device, {
      label: 'fresnel-sky sky cubemap',
      size: CUBE_SIZE,
      source: {
        label: 'fresnel-sky/sky-cube.wgsl',
        code: skyCube.code,
        chunks: { ...skyCube.chunks },
      },
    });
    const sampler = device.createSampler({
      label: 'fresnel-sky cube sampler',
      magFilter: 'linear',
      minFilter: 'linear',
      mipmapFilter: 'linear',
    });
    const pipeline = createFullscreenPipeline(device, {
      fragment: {
        label: 'fresnel-sky/fresnel-sky.wgsl',
        code: shader.code,
        chunks: { ...shader.chunks, 'params.wgsl': Params.wgsl },
      },
      format,
    });
    const bindGroup = device.createBindGroup({
      label: 'fresnel-sky',
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: uniforms.buffer } },
        { binding: 1, resource: cube.view },
        { binding: 2, resource: sampler },
      ],
    });

    let phase = 0;
    let height = 1;
    let cubeKey = '';

    return {
      resize(width, canvasHeight) {
        height = Math.max(1, canvasHeight);
        uniforms.set({
          aspect: width / height,
          pixel_angle: (2 * Math.tan(FOV / 2)) / height,
        });
      },
      frame({ delta }, encoder, target) {
        phase += delta * params.speed;
        const sunElevation = params.sunElevation * DEG;
        const sunAzimuth = params.sunAzimuth * DEG;
        const useCube = params.source === 'cubemap';
        const f0 = ((1 - params.ior) / (1 + params.ior)) ** 2;
        uniforms.set({
          time: phase,
          pitch: params.cameraPitch * DEG,
          fov: FOV,
          cam_height: params.cameraHeight,
          sun_elevation: sunElevation,
          sun_azimuth: sunAzimuth,
          wave_strength: params.waveStrength,
          wave_scale: 1,
          exposure: params.exposure,
          f0,
          haze: params.haze,
          use_cube: useCube ? 1 : 0,
          view_mode: Math.max(0, VIEWS.indexOf(params.view as (typeof VIEWS)[number])),
          body: params.deepColor,
        });
        uniforms.upload();

        // The cubemap only needs redrawing when the sky itself changes, not every frame.
        if (useCube) {
          const key = `${sunElevation}|${sunAzimuth}|${params.haze}`;
          if (key !== cubeKey) {
            cubeKey = key;
            cube.render(encoder, [sunElevation, sunAzimuth, params.haze, 0], {
              profiler,
              label: 'sky cubemap',
            });
          }
        } else {
          cubeKey = '';
        }

        renderFullscreen(encoder, target, pipeline, [bindGroup], {
          timestampWrites: profiler.pass('water'),
        });
      },
      destroy() {
        uniforms.destroy();
        cube.destroy();
      },
    };
  },
});

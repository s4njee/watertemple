import { createFullscreenPipeline, renderFullscreen } from '../../gpu/fullscreen.ts';
import { generateMipmaps, mipLevelCount } from '../../gpu/mipmap.ts';
import { createUniformBuffer, defineUniformStruct } from '../../gpu/uniforms.ts';
import { defineEffect } from '../types.ts';
import groundShader from './ground.wgsl';
import { meta } from './meta.ts';
import puddleShader from './puddle.wgsl';

const GroundParams = defineUniformStruct('GroundParams', {
  surface: 'i32',
  size: 'f32',
  tile_m: 'f32',
  height_m: 'f32',
});

const PuddleParams = defineUniformStruct('PuddleParams', {
  cam_pos: 'vec3f',
  tan_half: 'f32',
  cam_right: 'vec3f',
  aspect: 'f32',
  cam_up: 'vec3f',
  pixel: 'f32',
  cam_fwd: 'vec3f',
  level: 'f32',
  sun: 'vec3f',
  rain: 'f32',
  tile_m: 'f32',
  height_m: 'f32',
  damp_reach: 'f32',
  wetness: 'f32',
  roughness: 'f32',
  time: 'f32',
  ripple_cell: 'f32',
  ripple_life: 'f32',
  ripple_crest: 'f32',
  lamps: 'f32',
  exposure: 'f32',
});

const SURFACES = ['asphalt', 'cobbles', 'flagstones'] as const;
/** Dry roughness per surface: asphalt is the dullest. */
const ROUGHNESS = [0.82, 0.78, 0.7] as const;

/** Edge of the baked tile in metres, and the height range it spans. */
const TILE_M = 6;
const HEIGHT_M = 0.05;
const GROUND_SIZE = 1024;
const FOV = (45 * Math.PI) / 180;
/** The camera looks at this point of the tile, a little right of the lane line. */
const TARGET_X = 3.7;
const DISTANCE = 3.6;

const DEG = Math.PI / 180;

export const effect = defineEffect({
  meta,
  params: {
    waterLevel: {
      type: 'number',
      default: 0.32,
      min: 0,
      max: 0.8,
      step: 0.005,
      label: 'Water level',
    },
    rain: { type: 'number', default: 0.55, min: 0, max: 1, step: 0.01, label: 'Rain intensity' },
    rippleSize: {
      type: 'number',
      default: 0.4,
      min: 0.15,
      max: 1,
      step: 0.01,
      label: 'Ripple cell (m)',
    },
    rippleStrength: { type: 'number', default: 1, min: 0, max: 2.5, step: 0.01 },
    wetness: {
      type: 'number',
      default: 0.75,
      min: 0,
      max: 1,
      step: 0.01,
      label: 'Ground wetness',
    },
    surface: { type: 'select', default: 'asphalt', options: SURFACES },
    sunAzimuth: {
      type: 'number',
      default: 0,
      min: -90,
      max: 90,
      step: 1,
      label: 'Sun azimuth (deg)',
    },
    sunElevation: {
      type: 'number',
      default: 12,
      min: -8,
      max: 75,
      step: 1,
      label: 'Sun elevation (deg)',
    },
    lamps: { type: 'number', default: 0.6, min: 0, max: 1, step: 0.01, label: 'Street lights' },
    tilt: {
      type: 'number',
      default: 20,
      min: 12,
      max: 90,
      step: 1,
      label: 'Camera tilt (deg below horizon)',
    },
    brightness: { type: 'number', default: 1, min: 0.3, max: 2.5, step: 0.01 },
  },
  presets: {
    'after the storm': {
      waterLevel: 0.42,
      rain: 0.05,
      rippleSize: 0.45,
      rippleStrength: 0.9,
      wetness: 0.9,
      surface: 'asphalt',
      sunAzimuth: -8,
      sunElevation: 6,
      lamps: 0.35,
      tilt: 16,
      brightness: 1,
    },
    'steady rain': {
      waterLevel: 0.36,
      rain: 0.75,
      rippleSize: 0.38,
      rippleStrength: 1.2,
      wetness: 0.95,
      surface: 'asphalt',
      sunAzimuth: 0,
      sunElevation: 30,
      lamps: 0.9,
      tilt: 17,
      brightness: 1.25,
    },
    'drying out': {
      waterLevel: 0.28,
      rain: 0,
      rippleSize: 0.5,
      rippleStrength: 1,
      wetness: 0.45,
      surface: 'flagstones',
      sunAzimuth: 25,
      sunElevation: 38,
      lamps: 0,
      tilt: 30,
      brightness: 0.85,
    },
    'cobbled night': {
      waterLevel: 0.4,
      rain: 0.45,
      rippleSize: 0.34,
      rippleStrength: 1.2,
      wetness: 1,
      surface: 'cobbles',
      sunAzimuth: 0,
      sunElevation: -8,
      lamps: 1,
      tilt: 16,
      brightness: 1.2,
    },
  },

  async create({ device, format, params, profiler }) {
    const groundUniforms = createUniformBuffer(device, GroundParams, 'puddles ground params');
    const puddleUniforms = createUniformBuffer(device, PuddleParams, 'puddles params');

    const groundPipeline = createFullscreenPipeline(device, {
      fragment: {
        label: 'puddles/ground.wgsl',
        code: groundShader.code,
        chunks: { ...groundShader.chunks, 'ground_params.wgsl': GroundParams.wgsl },
      },
      format: ['rgba16float', 'rg16float'],
    });
    const puddlePipeline = createFullscreenPipeline(device, {
      fragment: {
        label: 'puddles/puddle.wgsl',
        code: puddleShader.code,
        chunks: { ...puddleShader.chunks, 'puddle_params.wgsl': PuddleParams.wgsl },
      },
      format,
    });
    const groundGroup = device.createBindGroup({
      label: 'puddles ground',
      layout: groundPipeline.getBindGroupLayout(0),
      entries: [{ binding: 0, resource: { buffer: groundUniforms.buffer } }],
    });
    // Repeating, trilinear and anisotropic: the ground foreshortens a lot at a shallow camera angle.
    const sampler = device.createSampler({
      label: 'puddles ground sampler',
      addressModeU: 'repeat',
      addressModeV: 'repeat',
      magFilter: 'linear',
      minFilter: 'linear',
      mipmapFilter: 'linear',
      maxAnisotropy: 4,
    });

    const levels = mipLevelCount(GROUND_SIZE, GROUND_SIZE);
    const usage = GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.RENDER_ATTACHMENT;
    const albedo = device.createTexture({
      label: 'puddles albedo+height',
      size: [GROUND_SIZE, GROUND_SIZE],
      format: 'rgba16float',
      mipLevelCount: levels,
      usage,
    });
    const normals = device.createTexture({
      label: 'puddles normals',
      size: [GROUND_SIZE, GROUND_SIZE],
      format: 'rg16float',
      mipLevelCount: levels,
      usage,
    });
    const albedoMip0 = albedo.createView({ baseMipLevel: 0, mipLevelCount: 1 });
    const normalsMip0 = normals.createView({ baseMipLevel: 0, mipLevelCount: 1 });
    const puddleGroup = device.createBindGroup({
      label: 'puddles',
      layout: puddlePipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: puddleUniforms.buffer } },
        { binding: 1, resource: albedo.createView() },
        { binding: 2, resource: normals.createView() },
        { binding: 3, resource: sampler },
      ],
    });

    let width = 1;
    let height = 1;
    let bakedSurface = -1;
    let rippleTime = 0;

    return {
      resize(w, h) {
        width = Math.max(1, Math.round(w));
        height = Math.max(1, Math.round(h));
      },

      frame({ delta }, encoder, target) {
        const aspect = width / height;
        const surface = Math.max(0, SURFACES.indexOf(params.surface as (typeof SURFACES)[number]));

        // The ground is baked once, and again only when the surface changes.
        if (surface !== bakedSurface) {
          bakedSurface = surface;
          groundUniforms.set({
            surface,
            size: GROUND_SIZE,
            tile_m: TILE_M,
            height_m: HEIGHT_M,
          });
          groundUniforms.upload();
          renderFullscreen(encoder, [albedoMip0, normalsMip0], groundPipeline, [groundGroup], {
            timestampWrites: profiler.pass('bake ground (on change)'),
          });
          generateMipmaps(device, albedo, encoder);
          generateMipmaps(device, normals, encoder);
        }

        rippleTime += delta;
        // A camera on a tripod looking at a fixed spot, `tilt` degrees below the horizon.
        const tilt = params.tilt * DEG;
        const forward = [0, -Math.sin(tilt), Math.cos(tilt)] as const;
        const up = [0, Math.cos(tilt), Math.sin(tilt)] as const;
        const position = [TARGET_X, DISTANCE * Math.sin(tilt), -DISTANCE * Math.cos(tilt)] as const;
        const az = params.sunAzimuth * DEG;
        const el = params.sunElevation * DEG;
        const tanHalf = Math.tan(FOV / 2);

        puddleUniforms.set({
          cam_pos: position,
          tan_half: tanHalf,
          cam_right: [1, 0, 0],
          aspect,
          cam_up: up,
          pixel: (2 * tanHalf) / height,
          cam_fwd: forward,
          level: params.waterLevel,
          sun: [Math.sin(az) * Math.cos(el), Math.sin(el), Math.cos(az) * Math.cos(el)],
          rain: params.rain,
          tile_m: TILE_M,
          height_m: HEIGHT_M,
          damp_reach: 0.004 + 0.02 * params.wetness,
          wetness: params.wetness * 0.8,
          roughness: ROUGHNESS[surface] ?? 0.8,
          time: rippleTime,
          ripple_cell: params.rippleSize,
          ripple_life: 1.1,
          ripple_crest: 0.012 * params.rippleStrength,
          lamps: params.lamps,
          exposure: params.brightness,
        });
        puddleUniforms.upload();
        renderFullscreen(encoder, target, puddlePipeline, [puddleGroup], {
          timestampWrites: profiler.pass('puddles'),
        });
      },

      destroy() {
        albedo.destroy();
        normals.destroy();
        groundUniforms.destroy();
        puddleUniforms.destroy();
      },
    };
  },
});

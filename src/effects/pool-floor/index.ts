import { createWaterNormalMap, createWaterSampler } from '../../gpu/bake.ts';
import {
  CAUSTICS_MAP_CELLS,
  createCausticsMap,
  createCausticsSampler,
} from '../../gpu/bake-caustics.ts';
import { createFullscreenPipeline, renderFullscreen } from '../../gpu/fullscreen.ts';
import { generateMipmaps, mipLevelCount } from '../../gpu/mipmap.ts';
import { trackPointer } from '../../gpu/pointer.ts';
import { createUniformBuffer, defineUniformStruct } from '../../gpu/uniforms.ts';
import { defineEffect } from '../types.ts';
import floorShader from './floor.wgsl?raw';
import { meta } from './meta.ts';
import poolShader from './pool.wgsl?raw';

const FloorParams = defineUniformStruct('FloorParams', {
  aspect: 'f32',
  aa: 'f32',
  density: 'f32',
});

const PoolParams = defineUniformStruct('PoolParams', {
  pointer: 'vec2f',
  aspect: 'f32',
  time: 'f32',
  caustic_time: 'f32',
  thickness: 'f32',
  depth: 'f32',
  ripple: 'f32',
  ripple_scale: 'f32',
  caustic: 'f32',
  caustic_scale: 'f32',
  caustic_lod: 'f32',
  shine: 'f32',
  lens: 'f32',
  ring: 'f32',
  ring_phase: 'f32',
  blur_px: 'f32',
  max_lod: 'f32',
  normal_lod: 'f32',
  water_kind: 'i32',
});

/** Index = the `absorb_water` kind. */
const WATER_TYPES = [
  'clear tropical',
  'coastal',
  'murky green',
  'tea lake',
  'glacial milk',
  'swimming pool',
  'open ocean',
] as const;

const FLOOR_FORMAT: GPUTextureFormat = 'rgba8unorm-srgb';
/** One image height of pool floor is this many metres across. */
const METRES_PER_HEIGHT = 5;

export const effect = defineEffect({
  meta,
  params: {
    waterType: { type: 'select', default: 'swimming pool', options: WATER_TYPES },
    depth: { type: 'number', default: 1.4, min: 0.2, max: 6, step: 0.01, label: 'Depth (m)' },
    rippleStrength: { type: 'number', default: 0.45, min: 0, max: 1.6, step: 0.01 },
    rippleScale: { type: 'number', default: 0.8, min: 0.5, max: 5, step: 0.01 },
    rippleSpeed: { type: 'number', default: 0.8, min: 0, max: 3, step: 0.01 },
    caustics: {
      type: 'number',
      default: 1,
      min: 0,
      max: 2,
      step: 0.01,
      label: 'Caustic intensity',
    },
    shine: { type: 'number', default: 0.5, min: 0, max: 1, step: 0.01, label: 'Surface highlight' },
    pointer: {
      type: 'number',
      default: 0.7,
      min: 0,
      max: 1.5,
      step: 0.01,
      label: 'Pointer ripple',
    },
  },
  presets: {
    'morning pool': {
      waterType: 'swimming pool',
      depth: 1.4,
      rippleStrength: 0.45,
      rippleScale: 0.8,
      rippleSpeed: 0.8,
      caustics: 1,
      shine: 0.5,
      pointer: 0.7,
    },
    'deep end': {
      waterType: 'swimming pool',
      depth: 4.8,
      rippleStrength: 0.3,
      rippleScale: 0.6,
      rippleSpeed: 0.55,
      caustics: 0.8,
      shine: 0.35,
      pointer: 0.6,
    },
    'kiddie pool': {
      waterType: 'clear tropical',
      depth: 0.35,
      rippleStrength: 0.8,
      rippleScale: 1.4,
      rippleSpeed: 1.2,
      caustics: 1.3,
      shine: 0.7,
      pointer: 1,
    },
    'green lido': {
      waterType: 'murky green',
      depth: 2.2,
      rippleStrength: 0.5,
      rippleScale: 1.0,
      rippleSpeed: 0.7,
      caustics: 1.1,
      shine: 0.4,
      pointer: 0.7,
    },
  },

  async create({ device, format, canvas, params, profiler }) {
    const floorUniforms = createUniformBuffer(device, FloorParams, 'pool-floor floor params');
    const poolUniforms = createUniformBuffer(device, PoolParams, 'pool-floor water params');
    const pointer = trackPointer(canvas);

    const floorPipeline = createFullscreenPipeline(device, {
      fragment: {
        label: 'pool-floor/floor.wgsl',
        code: floorShader,
        chunks: { 'floor_params.wgsl': FloorParams.wgsl },
      },
      format: FLOOR_FORMAT,
    });
    const poolPipeline = createFullscreenPipeline(device, {
      fragment: {
        label: 'pool-floor/pool.wgsl',
        code: poolShader,
        chunks: { 'pool_params.wgsl': PoolParams.wgsl },
      },
      format,
    });
    const floorGroup = device.createBindGroup({
      label: 'pool-floor floor',
      layout: floorPipeline.getBindGroupLayout(0),
      entries: [{ binding: 0, resource: { buffer: floorUniforms.buffer } }],
    });

    // Baked once: ripple normals and the caustic map. The per-frame pass only reads them.
    const normals = createWaterNormalMap(device, { size: 512, seed: 5, label: 'pool normals' });
    const caustics = createCausticsMap(device, {
      cells: CAUSTICS_MAP_CELLS,
      method: 'sine',
      sharpness: 0.78,
      label: 'pool caustics',
    });
    const normalSampler = createWaterSampler(device, { anisotropy: 4 });
    const causticSampler = createCausticsSampler(device);
    const floorSampler = device.createSampler({
      label: 'pool-floor floor sampler',
      magFilter: 'linear',
      minFilter: 'linear',
      mipmapFilter: 'linear',
    });

    let width = 1;
    let height = 1;
    let floorTexture: GPUTexture | undefined;
    let floorMip0: GPUTextureView | undefined;
    let poolGroup: GPUBindGroup | undefined;
    let floorDrawn = false;

    let phase = 0;
    let causticPhase = 0;
    let ringPhase = 0;
    let follow = 0;
    let energy = 0.1;

    return {
      resize(w, h) {
        width = Math.max(1, Math.round(w));
        height = Math.max(1, Math.round(h));
        floorTexture?.destroy();
        floorTexture = device.createTexture({
          label: 'pool-floor tiles',
          size: [width, height],
          format: FLOOR_FORMAT,
          mipLevelCount: mipLevelCount(width, height),
          usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.RENDER_ATTACHMENT,
        });
        floorMip0 = floorTexture.createView({ baseMipLevel: 0, mipLevelCount: 1 });
        poolGroup = device.createBindGroup({
          label: 'pool-floor water',
          layout: poolPipeline.getBindGroupLayout(0),
          entries: [
            { binding: 0, resource: { buffer: poolUniforms.buffer } },
            { binding: 1, resource: floorTexture.createView() },
            { binding: 2, resource: floorSampler },
            { binding: 3, resource: normals.view },
            { binding: 4, resource: normalSampler },
            { binding: 5, resource: caustics.view },
            { binding: 6, resource: causticSampler },
          ],
        });
        floorDrawn = false;
      },

      frame({ delta, time }, encoder, target) {
        if (!floorTexture || !floorMip0 || !poolGroup) return;
        const aspect = width / height;

        if (!floorDrawn) {
          floorDrawn = true;
          floorUniforms.set({ aspect, aa: 1 / height, density: 0.8 });
          floorUniforms.upload();
          renderFullscreen(encoder, floorMip0, floorPipeline, [floorGroup], {
            timestampWrites: profiler.pass('floor (on resize)'),
          });
          generateMipmaps(device, floorTexture, encoder);
        }

        // Ambient pointer: follows the cursor while it is over the canvas, wanders slowly otherwise.
        const state = pointer.state;
        const moved = pointer.takeDelta();
        const speed = Math.hypot(moved.dx * aspect, moved.dy) / Math.max(delta, 1e-3);
        const blend = Math.min(1, delta * 4);
        follow += ((state.inside ? 1 : 0) - follow) * blend;
        energy += ((state.inside ? Math.min(1, speed * 1.2) + 0.12 : 0.1) - energy) * blend;
        const wanderX = 0.5 + 0.3 * Math.sin(time * 0.23);
        const wanderY = 0.5 + 0.2 * Math.sin(time * 0.37 + 1);
        const x = wanderX + (state.x - wanderX) * follow;
        const y = wanderY + (state.y - wanderY) * follow;

        phase += delta * params.rippleSpeed * 0.2;
        causticPhase += delta * params.rippleSpeed * 0.9;
        ringPhase += delta * 7;

        const depth = params.depth;
        poolUniforms.set({
          pointer: [x * aspect, y],
          aspect,
          time: phase,
          caustic_time: causticPhase,
          thickness: depth / METRES_PER_HEIGHT,
          depth,
          ripple: params.rippleStrength,
          ripple_scale: params.rippleScale,
          caustic: params.caustics,
          // Deeper water spreads the light: bigger caustic cells that are blurrier and fainter.
          caustic_scale: 7 / 12 / (1 + 0.18 * depth),
          caustic_lod: 0.4 + 0.45 * depth,
          shine: params.shine,
          lens: params.pointer * (0.35 + 0.65 * follow) * 0.5,
          ring: params.pointer * energy * 0.3,
          ring_phase: ringPhase,
          blur_px: 0.0025 * height * Math.min(depth, 3),
          max_lod: Math.max(0, mipLevelCount(width, height) - 1),
          // The ripple map's finest octaves are too busy for a pool: read a coarser mip.
          normal_lod: 1.6,
          water_kind: Math.max(
            0,
            WATER_TYPES.indexOf(params.waterType as (typeof WATER_TYPES)[number]),
          ),
        });
        poolUniforms.upload();
        renderFullscreen(encoder, target, poolPipeline, [poolGroup], {
          timestampWrites: profiler.pass('pool'),
        });
      },

      destroy() {
        pointer.destroy();
        floorTexture?.destroy();
        normals.destroy();
        caustics.destroy();
        floorUniforms.destroy();
        poolUniforms.destroy();
      },
    };
  },
});

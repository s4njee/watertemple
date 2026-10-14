import { createFullscreenPipeline, renderFullscreen } from '../../gpu/fullscreen.ts';
import { createRenderTarget, type RenderTarget } from '../../gpu/targets.ts';
import { createUniformBuffer, defineUniformStruct } from '../../gpu/uniforms.ts';
import { defineEffect } from '../types.ts';
import bedShader from './bed.wgsl';
import { meta } from './meta.ts';
import waterShader from './water.wgsl';

const BedParams = defineUniformStruct('BedParams', {
  aspect: 'f32',
  aa: 'f32',
  density: 'f32',
  kind: 'i32',
});

const ViewParams = defineUniformStruct('ViewParams', {
  aspect: 'f32',
  time: 'f32',
  depth_scale: 'f32',
  turbidity_scale: 'f32',
  cos_view: 'f32',
  cos_sun: 'f32',
  parallax: 'f32',
  ripple: 'f32',
  sun: 'f32',
  water_type: 'i32',
  view: 'u32',
  ramp: 'u32',
});

/** Index = the `kind` argument of `absorb_water`. */
const WATER_TYPES = [
  'clear tropical',
  'coastal',
  'murky green',
  'tea lake',
  'glacial milk',
  'swimming pool',
  'open ocean',
] as const;
const BEDS = ['sand', 'pebbles', 'pool tiles', 'test card'] as const;
const BED_KINDS: Record<(typeof BEDS)[number], number> = {
  sand: 3,
  pebbles: 2,
  'pool tiles': 0,
  'test card': 1,
};
const VIEWS = ['final', 'depth', 'transmittance'] as const;

const WATER_IOR = 1.333;
const BED_FORMAT: GPUTextureFormat = 'rgba16float';

export const effect = defineEffect({
  meta,
  params: {
    waterType: { type: 'select', default: 'clear tropical', options: WATER_TYPES },
    depthScale: {
      type: 'number',
      default: 10,
      min: 0.5,
      max: 40,
      step: 0.1,
      label: 'Deepest point (m)',
    },
    turbidity: {
      type: 'number',
      default: 1,
      min: 0,
      max: 4,
      step: 0.01,
      label: 'Turbidity (x type)',
    },
    viewAngle: { type: 'number', default: 30, min: 0, max: 75, step: 1, label: 'View angle (deg)' },
    sunAngle: { type: 'number', default: 25, min: 0, max: 80, step: 1, label: 'Sun angle (deg)' },
    bed: { type: 'select', default: 'sand', options: BEDS },
    ripple: {
      type: 'number',
      default: 0.35,
      min: 0,
      max: 1,
      step: 0.01,
      label: 'Surface caustics',
    },
    view: { type: 'select', default: 'final', options: VIEWS, label: 'Debug view' },
    ramp: { type: 'boolean', default: true, label: 'Depth ramp strip' },
  },
  presets: {
    'clear tropical': { waterType: 'clear tropical', depthScale: 14, turbidity: 1, bed: 'sand' },
    coastal: { waterType: 'coastal', depthScale: 8, turbidity: 1, bed: 'sand' },
    'murky green': { waterType: 'murky green', depthScale: 3, turbidity: 1, bed: 'sand' },
    'tea lake': { waterType: 'tea lake', depthScale: 4, turbidity: 1, bed: 'sand' },
    'glacial milk': { waterType: 'glacial milk', depthScale: 10, turbidity: 1, bed: 'sand' },
    'swimming pool': {
      waterType: 'swimming pool',
      depthScale: 3.5,
      turbidity: 1,
      bed: 'pool tiles',
      viewAngle: 20,
    },
  },

  async create({ device, format, params, profiler }) {
    const bedUniforms = createUniformBuffer(device, BedParams, 'depth-absorption bed params');
    const viewUniforms = createUniformBuffer(device, ViewParams, 'depth-absorption view params');

    const bedPipeline = createFullscreenPipeline(device, {
      fragment: {
        label: 'depth-absorption/bed.wgsl',
        code: bedShader.code,
        chunks: { ...bedShader.chunks, 'bed_params.wgsl': BedParams.wgsl },
      },
      format: BED_FORMAT,
    });
    const waterPipeline = createFullscreenPipeline(device, {
      fragment: {
        label: 'depth-absorption/water.wgsl',
        code: waterShader.code,
        chunks: { ...waterShader.chunks, 'view_params.wgsl': ViewParams.wgsl },
      },
      format,
    });
    const bedGroup = device.createBindGroup({
      label: 'depth-absorption bed',
      layout: bedPipeline.getBindGroupLayout(0),
      entries: [{ binding: 0, resource: { buffer: bedUniforms.buffer } }],
    });
    const sampler = device.createSampler({
      label: 'depth-absorption bed sampler',
      magFilter: 'linear',
      minFilter: 'linear',
      addressModeU: 'mirror-repeat',
      addressModeV: 'mirror-repeat',
    });

    let width = 1;
    let height = 1;
    let bed: RenderTarget | undefined;
    let waterGroup: GPUBindGroup | undefined;
    let bakedKey = '';
    let phase = 0;

    return {
      resize(w, h) {
        width = Math.max(1, Math.round(w));
        height = Math.max(1, Math.round(h));
        if (!bed) {
          bed = createRenderTarget(device, {
            label: 'depth-absorption bed',
            format: BED_FORMAT,
            width,
            height,
          });
        } else {
          bed.resize(width, height);
        }
        waterGroup = device.createBindGroup({
          label: 'depth-absorption water',
          layout: waterPipeline.getBindGroupLayout(0),
          entries: [
            { binding: 0, resource: { buffer: viewUniforms.buffer } },
            { binding: 1, resource: bed.view },
            { binding: 2, resource: sampler },
          ],
        });
        bakedKey = '';
      },

      frame({ delta }, encoder, target) {
        if (!bed || !waterGroup) return;
        const aspect = width / height;

        // The bed and its depth never change while the view stays put: bake them once per resize
        // or bed choice, and let the per-frame pass do only the water maths.
        const key = `${params.bed}:${width}x${height}`;
        if (key !== bakedKey) {
          bakedKey = key;
          bedUniforms.set({
            aspect,
            aa: 1 / height,
            density: params.bed === 'pool tiles' ? 1 : 1.6,
            kind: BED_KINDS[params.bed as (typeof BEDS)[number]] ?? 3,
          });
          bedUniforms.upload();
          renderFullscreen(encoder, bed.view, bedPipeline, [bedGroup], {
            timestampWrites: profiler.pass('bake bed (on change)'),
          });
        }

        phase += delta;
        const view = (params.viewAngle * Math.PI) / 180;
        const sinInside = Math.sin(view) / WATER_IOR;
        const tanInside = sinInside / Math.sqrt(1 - sinInside * sinInside);
        viewUniforms.set({
          aspect,
          time: phase,
          depth_scale: params.depthScale,
          turbidity_scale: params.turbidity,
          cos_view: Math.cos(view),
          cos_sun: Math.cos((params.sunAngle * Math.PI) / 180),
          // How far up the screen the bed appears to shift at full depth, in uv.
          parallax: tanInside * 0.12,
          ripple: params.ripple,
          sun: 1,
          water_type: Math.max(
            0,
            WATER_TYPES.indexOf(params.waterType as (typeof WATER_TYPES)[number]),
          ),
          view: Math.max(0, VIEWS.indexOf(params.view as (typeof VIEWS)[number])),
          ramp: params.ramp ? 1 : 0,
        });
        viewUniforms.upload();
        renderFullscreen(encoder, target, waterPipeline, [waterGroup], {
          timestampWrites: profiler.pass('water'),
        });
      },

      destroy() {
        bed?.destroy();
        bedUniforms.destroy();
        viewUniforms.destroy();
      },
    };
  },
});

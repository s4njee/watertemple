import { createFullscreenPipeline, renderFullscreen } from '../../gpu/fullscreen.ts';
import { createRenderTarget, type RenderTarget } from '../../gpu/targets.ts';
import { createUniformBuffer, defineUniformStruct } from '../../gpu/uniforms.ts';
import { defineEffect } from '../types.ts';
import composeShader from './compose.wgsl?raw';
import fieldShader from './field.wgsl?raw';
import { meta } from './meta.ts';
import paperShader from './paper.wgsl?raw';

const FieldParams = defineUniformStruct('FieldParams', {
  origin: 'vec2f',
  extent: 'vec2f',
  scale: 'f32',
  time: 'f32',
  warp: 'f32',
});

const ComposeParams = defineUniformStruct('ComposeParams', {
  paper_scale: 'vec2f',
  threshold: 'f32',
  threshold_b: 'f32',
  edge_soft: 'f32',
  pool_width: 'f32',
  edge: 'f32',
  contrast: 'f32',
  granulation: 'f32',
  fibre: 'f32',
  ragged: 'f32',
  transparent: 'u32',
  paper: 'vec3f',
  ink_a: 'vec3f',
  ink_b: 'vec3f',
});

type Rgb = [number, number, number];

const srgbToLinear = (c: number) => (c <= 0.04045 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4);
const linearToSrgb = (c: number) => {
  const x = Math.min(1, Math.max(0, c));
  return x <= 0.0031308 ? x * 12.92 : 1.055 * x ** (1 / 2.4) - 0.055;
};
/** sRGB 0..1 from a hex colour. The shader mixes pigments in sRGB, like paint on a screen. */
const srgb = (css: string): Rgb => {
  const n = parseInt(css.replace('#', ''), 16);
  return [(n >> 16) & 255, (n >> 8) & 255, n & 255].map((v) => v / 255) as Rgb;
};
/** Linear RGB, the form colour parameters take. */
const linear = (css: string): Rgb => srgb(css).map(srgbToLinear) as Rgb;
const encode = (c: readonly number[]): Rgb => [
  linearToSrgb(c[0] ?? 0),
  linearToSrgb(c[1] ?? 0),
  linearToSrgb(c[2] ?? 0),
];

type Palette = readonly [paper: string, inkA: string, inkB: string];
/** Muted by design: the ink colours sit close to the paper's brightness so text stays readable. */
const PALETTES: Record<string, { light: Palette; dark: Palette }> = {
  indigo: {
    light: ['#f1f1ee', '#8397c1', '#a498c4'],
    dark: ['#0b1520', '#2c4876', '#4a3f78'],
  },
  tea: {
    light: ['#f3ecdf', '#c9a577', '#a3ad85'],
    dark: ['#15110c', '#6a4b2c', '#4a5a35'],
  },
  storm: {
    light: ['#eceeef', '#98a8b5', '#7f93a4'],
    dark: ['#0e1316', '#2f404e', '#455767'],
  },
  rose: {
    light: ['#f7efed', '#dca0a4', '#bba6c8'],
    dark: ['#171012', '#6a3a44', '#4b3a5a'],
  },
  moss: {
    light: ['#eef0e6', '#97ad7d', '#6f9a95'],
    dark: ['#0e130d', '#3b5a33', '#2d5a58'],
  },
};
const PALETTE_NAMES = [...Object.keys(PALETTES), 'custom'] as const;
const RESOLUTIONS = ['1/4', '1/2', 'full'] as const;
const RESOLUTION_DIVISOR: Record<string, number> = { '1/4': 4, '1/2': 2, full: 1 };

const PAPER_SIZE = 256;
/** One paper tile covers this many CSS pixels, so the grain keeps its size at every DPR. */
const PAPER_CSS_PX = 200;

export const effect = defineEffect({
  meta,
  params: {
    palette: { type: 'select', default: 'indigo', options: PALETTE_NAMES },
    tone: { type: 'select', default: 'light', options: ['light', 'dark'], label: 'Paper tone' },
    transparentPaper: {
      type: 'boolean',
      default: false,
      label: 'Transparent paper (alpha)',
    },
    contrast: { type: 'number', default: 0.55, min: 0.1, max: 1.5, step: 0.01, label: 'Contrast' },
    coverage: { type: 'number', default: 0.5, min: 0.1, max: 0.9, step: 0.01, label: 'Coverage' },
    bloomScale: {
      type: 'number',
      default: 1.5,
      min: 0.5,
      max: 5,
      step: 0.05,
      label: 'Bloom scale',
    },
    warp: { type: 'number', default: 0.7, min: 0, max: 1.5, step: 0.01, label: 'Warp' },
    edgeDarkening: {
      type: 'number',
      default: 0.4,
      min: 0,
      max: 1.2,
      step: 0.01,
      label: 'Edge darkening',
    },
    edgeSoftness: {
      type: 'number',
      default: 2.5,
      min: 0.5,
      max: 10,
      step: 0.1,
      label: 'Edge softness (px)',
    },
    granulation: {
      type: 'number',
      default: 0.5,
      min: 0,
      max: 1,
      step: 0.01,
      label: 'Granulation',
    },
    paperGrain: { type: 'number', default: 0.5, min: 0, max: 1, step: 0.01, label: 'Paper fibres' },
    speed: { type: 'number', default: 0.25, min: 0, max: 2, step: 0.01, label: 'Speed' },
    fieldResolution: {
      type: 'select',
      default: '1/4',
      options: RESOLUTIONS,
      label: 'Field resolution',
    },
    paperColor: { type: 'color', default: linear('#f1f1ee'), label: 'Custom: paper' },
    inkColorA: { type: 'color', default: linear('#8397c1'), label: 'Custom: ink A' },
    inkColorB: { type: 'color', default: linear('#a498c4'), label: 'Custom: ink B' },
  },
  presets: {
    indigo: { palette: 'indigo', tone: 'light', contrast: 0.6, coverage: 0.5, bloomScale: 1.5 },
    tea: {
      palette: 'tea',
      tone: 'light',
      contrast: 0.7,
      coverage: 0.45,
      bloomScale: 2,
      warp: 0.5,
      granulation: 0.7,
      speed: 0.2,
    },
    storm: {
      palette: 'storm',
      tone: 'dark',
      contrast: 0.8,
      coverage: 0.55,
      bloomScale: 1.2,
      warp: 1.1,
      edgeDarkening: 0.5,
      speed: 0.4,
    },
    'indigo night': {
      palette: 'indigo',
      tone: 'dark',
      contrast: 0.8,
      coverage: 0.5,
      bloomScale: 1.6,
      warp: 0.8,
    },
    'ink over page': {
      palette: 'storm',
      tone: 'light',
      transparentPaper: true,
      contrast: 0.7,
      coverage: 0.5,
      bloomScale: 1.4,
    },
  },

  async create({ device, format, params, profiler }) {
    const fieldUniforms = createUniformBuffer(device, FieldParams, 'ink-wash field params');
    const composeUniforms = createUniformBuffer(device, ComposeParams, 'ink-wash compose params');

    const fieldPipeline = createFullscreenPipeline(device, {
      fragment: {
        label: 'ink-wash/field.wgsl',
        code: fieldShader,
        chunks: { 'field_params.wgsl': FieldParams.wgsl },
      },
      format: 'rgba16float',
    });
    const paperPipeline = createFullscreenPipeline(device, {
      fragment: { label: 'ink-wash/paper.wgsl', code: paperShader },
      format: 'rgba8unorm',
    });
    const composePipeline = createFullscreenPipeline(device, {
      fragment: {
        label: 'ink-wash/compose.wgsl',
        code: composeShader,
        chunks: { 'compose_params.wgsl': ComposeParams.wgsl },
      },
      format,
    });

    const fieldGroup = device.createBindGroup({
      label: 'ink-wash field',
      layout: fieldPipeline.getBindGroupLayout(0),
      entries: [{ binding: 0, resource: { buffer: fieldUniforms.buffer } }],
    });
    const clampSampler = device.createSampler({
      label: 'ink-wash field sampler',
      magFilter: 'linear',
      minFilter: 'linear',
    });
    const repeatSampler = device.createSampler({
      label: 'ink-wash paper sampler',
      magFilter: 'linear',
      minFilter: 'linear',
      addressModeU: 'repeat',
      addressModeV: 'repeat',
    });
    const paper = createRenderTarget(device, {
      label: 'ink-wash paper',
      format: 'rgba8unorm',
      width: PAPER_SIZE,
      height: PAPER_SIZE,
    });

    let width = 1;
    let height = 1;
    let dpr = 1;
    let paperBaked = false;
    let field: RenderTarget | undefined;
    let composeGroup: GPUBindGroup | undefined;
    let divisor = 4;
    // Accumulated, so a speed change never makes the pigment jump.
    let phase = 0;

    const sizeField = () => {
      const w = Math.max(2, Math.ceil(width / divisor));
      const h = Math.max(2, Math.ceil(height / divisor));
      if (!field) {
        field = createRenderTarget(device, {
          label: 'ink-wash field',
          format: 'rgba16float',
          width: w,
          height: h,
        });
      } else if (!field.resize(w, h) && composeGroup) {
        return;
      }
      composeGroup = device.createBindGroup({
        label: 'ink-wash compose',
        layout: composePipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: composeUniforms.buffer } },
          { binding: 1, resource: field.view },
          { binding: 2, resource: paper.view },
          { binding: 3, resource: clampSampler },
          { binding: 4, resource: repeatSampler },
        ],
      });
    };

    return {
      resize(w, h, ratio) {
        width = Math.max(1, Math.round(w));
        height = Math.max(1, Math.round(h));
        dpr = Math.max(ratio, 0.1);
        sizeField();
      },

      frame({ delta }, encoder, target) {
        const wanted = RESOLUTION_DIVISOR[params.fieldResolution] ?? 4;
        if (wanted !== divisor) {
          divisor = wanted;
          sizeField();
        }
        if (!field || !composeGroup) return;

        if (!paperBaked) {
          paperBaked = true;
          renderFullscreen(encoder, paper.view, paperPipeline, [], {
            timestampWrites: profiler.pass('bake paper (once)'),
          });
        }

        phase += delta * params.speed;
        fieldUniforms.set({
          origin: [3.7, 1.3],
          extent: width >= height ? [width / height, 1] : [1, height / width],
          scale: params.bloomScale,
          time: phase,
          warp: params.warp,
        });
        fieldUniforms.upload();
        renderFullscreen(encoder, field.view, fieldPipeline, [fieldGroup], {
          timestampWrites: profiler.pass('pigment field'),
        });

        const named = PALETTES[params.palette];
        const colors: Palette | undefined = named
          ? named[params.tone === 'dark' ? 'dark' : 'light']
          : undefined;
        // `coverage` moves the bloom level: higher coverage = lower threshold = more of the page inked.
        const threshold = 0.55 - params.coverage * 0.9;
        composeUniforms.set({
          paper_scale: [1 / (PAPER_CSS_PX * dpr), 1 / (PAPER_CSS_PX * dpr)],
          threshold,
          threshold_b: threshold + 0.1,
          edge_soft: params.edgeSoftness,
          pool_width: params.edgeSoftness * 2.2 + 3,
          edge: params.edgeDarkening,
          contrast: params.contrast,
          granulation: params.granulation,
          fibre: params.paperGrain,
          ragged: 1,
          transparent: params.transparentPaper ? 1 : 0,
          paper: colors ? srgb(colors[0]) : encode(params.paperColor),
          ink_a: colors ? srgb(colors[1]) : encode(params.inkColorA),
          ink_b: colors ? srgb(colors[2]) : encode(params.inkColorB),
        });
        composeUniforms.upload();
        renderFullscreen(encoder, target, composePipeline, [composeGroup], {
          timestampWrites: profiler.pass('compose'),
        });
      },

      destroy() {
        field?.destroy();
        paper.destroy();
        fieldUniforms.destroy();
        composeUniforms.destroy();
      },
    };
  },
});

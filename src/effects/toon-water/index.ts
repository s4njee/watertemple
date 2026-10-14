import { createFullscreenPipeline, renderFullscreen } from '../../gpu/fullscreen.ts';
import { createUniformBuffer, defineUniformStruct, uniformArray } from '../../gpu/uniforms.ts';
import { defineEffect } from '../types.ts';
import { meta } from './meta.ts';
import bakeShader from './sdf-bake.wgsl';
import shader from './toon-water.wgsl';

const Params = defineUniformStruct('ToonParams', {
  resolution: 'vec2f',
  aspect: 'f32',
  zoom: 'f32',
  time: 'f32',
  foam_time: 'f32',
  bands: 'f32',
  depth: 'f32',
  foam_scale: 'f32',
  foam_width: 'f32',
  shore: 'f32',
  highlight: 'f32',
  outline: 'f32',
  deep_foam: 'f32',
  palette: uniformArray('vec4f', 12),
});

type Rgb = readonly [number, number, number];

/** sRGB hex, as a designer would pick it, to the linear RGB the shader and the colour params use. */
function hex(value: string): Rgb {
  const n = parseInt(value.slice(1), 16);
  const channel = (shift: number) => {
    const c = ((n >> shift) & 255) / 255;
    return c <= 0.04045 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4;
  };
  return [channel(16), channel(8), channel(0)];
}

/** Twelve slots, in the order the shader reads them. */
interface Palette {
  deep: Rgb;
  midDeep: Rgb;
  midShallow: Rgb;
  shallow: Rgb;
  foam: Rgb;
  ink: Rgb;
  sand: Rgb;
  grass: Rgb;
  bush: Rgb;
  rock: Rgb;
  rockDark: Rgb;
  sparkle: Rgb;
}

const make = (colours: Record<keyof Palette, string>): Palette =>
  Object.fromEntries(
    Object.entries(colours).map(([key, value]) => [key, hex(value)]),
  ) as unknown as Palette;

const PALETTES = {
  tropical: make({
    deep: '#1d5fb0',
    midDeep: '#2a8fcf',
    midShallow: '#3fc1d8',
    shallow: '#7fe8da',
    foam: '#f4fffb',
    ink: '#0d3b66',
    sand: '#f7e3a1',
    grass: '#5fc45a',
    bush: '#2f8f4a',
    rock: '#8d8f9c',
    rockDark: '#5c5f72',
    sparkle: '#ffffff',
  }),
  pastel: make({
    deep: '#8fb3e8',
    midDeep: '#a9c6ee',
    midShallow: '#c2daf2',
    shallow: '#dcecf5',
    foam: '#ffffff',
    ink: '#7d8fc4',
    sand: '#fbe8d5',
    grass: '#bfe3b4',
    bush: '#98cfa0',
    rock: '#cfc6dc',
    rockDark: '#b3a8c8',
    sparkle: '#ffffff',
  }),
  night: make({
    deep: '#0b1030',
    midDeep: '#131c4d',
    midShallow: '#1d2f73',
    shallow: '#2c4a9a',
    foam: '#bcd4ff',
    ink: '#040616',
    sand: '#6a6a9c',
    grass: '#1f4d5a',
    bush: '#123642',
    rock: '#3b3f66',
    rockDark: '#272a4a',
    sparkle: '#fff6c8',
  }),
  sunset: make({
    deep: '#5b2a86',
    midDeep: '#a23b8c',
    midShallow: '#ef6b73',
    shallow: '#ffb27a',
    foam: '#fff1d6',
    ink: '#3a1450',
    sand: '#ffd9a0',
    grass: '#7a9b4c',
    bush: '#4f7440',
    rock: '#7a4a63',
    rockDark: '#55324a',
    sparkle: '#fff5b0',
  }),
} satisfies Record<string, Palette>;

const PALETTE_NAMES = [...Object.keys(PALETTES), 'custom'] as const;

function paletteFor(name: string, custom: { deep: Rgb; shallow: Rgb; foam: Rgb }): Palette {
  if (name in PALETTES) return PALETTES[name as keyof typeof PALETTES];
  const mix = (a: Rgb, b: Rgb, k: number): Rgb => [
    a[0] + (b[0] - a[0]) * k,
    a[1] + (b[1] - a[1]) * k,
    a[2] + (b[2] - a[2]) * k,
  ];
  return {
    ...PALETTES.tropical,
    deep: custom.deep,
    midDeep: mix(custom.deep, custom.shallow, 1 / 3),
    midShallow: mix(custom.deep, custom.shallow, 2 / 3),
    shallow: custom.shallow,
    foam: custom.foam,
    ink: mix(custom.deep, [0, 0, 0], 0.55),
  };
}

const SLOTS: readonly (keyof Palette)[] = [
  'deep',
  'midDeep',
  'midShallow',
  'shallow',
  'foam',
  'ink',
  'sand',
  'grass',
  'bush',
  'rock',
  'rockDark',
  'sparkle',
];

export const effect = defineEffect({
  meta,
  params: {
    palette: { type: 'select', default: 'tropical', options: PALETTE_NAMES },
    bands: { type: 'number', default: 4, min: 2, max: 8, step: 1, label: 'Depth bands' },
    depth: { type: 'number', default: 0.42, min: 0.12, max: 1, step: 0.01, label: 'Water depth' },
    foamScale: { type: 'number', default: 9, min: 3, max: 22, step: 0.1, label: 'Foam cell scale' },
    foamThickness: {
      type: 'number',
      default: 2.4,
      min: 0,
      max: 7,
      step: 0.1,
      label: 'Foam line width (px at 600 px tall)',
    },
    foamSpeed: { type: 'number', default: 0.6, min: 0, max: 2.5, step: 0.01 },
    shoreFoam: { type: 'number', default: 0.7, min: 0, max: 1, step: 0.01, label: 'Shore foam' },
    highlight: { type: 'number', default: 0.6, min: 0, max: 1, step: 0.01, label: 'Glints' },
    outline: { type: 'boolean', default: true, label: 'Ink outline' },
    deepFoam: {
      type: 'boolean',
      default: true,
      label: 'Second foam layer (about a third of the cost)',
    },
    customDeep: { type: 'color', default: hex('#1b3f9b'), label: 'Custom: deep water' },
    customShallow: { type: 'color', default: hex('#7ad7f0'), label: 'Custom: shallow water' },
    customFoam: { type: 'color', default: hex('#ffffff'), label: 'Custom: foam' },
  },
  presets: {
    'tropical cartoon': {
      palette: 'tropical',
      bands: 4,
      depth: 0.42,
      foamScale: 9,
      foamThickness: 2.4,
      foamSpeed: 0.6,
      shoreFoam: 0.7,
      highlight: 0.6,
      outline: true,
    },
    pastel: {
      palette: 'pastel',
      bands: 5,
      depth: 0.5,
      foamScale: 7,
      foamThickness: 3.2,
      foamSpeed: 0.4,
      shoreFoam: 0.55,
      highlight: 0.4,
      outline: false,
    },
    night: {
      palette: 'night',
      bands: 4,
      depth: 0.4,
      foamScale: 11,
      foamThickness: 1.8,
      foamSpeed: 0.5,
      shoreFoam: 0.6,
      highlight: 0.95,
      outline: true,
    },
    'poster sunset': {
      palette: 'sunset',
      bands: 3,
      depth: 0.45,
      foamScale: 6,
      foamThickness: 3.6,
      foamSpeed: 0.7,
      shoreFoam: 0.9,
      highlight: 0.5,
      outline: true,
    },
  },

  async create({ device, format, params, profiler }) {
    const uniforms = createUniformBuffer(device, Params, 'toon-water params');
    const pipeline = createFullscreenPipeline(device, {
      fragment: {
        label: 'toon-water/toon-water.wgsl',
        code: shader.code,
        chunks: { ...shader.chunks, 'toon_params.wgsl': Params.wgsl },
      },
      format,
    });
    // The island and rocks never move, so their distance fields are drawn once per canvas size into a
    // half-resolution rg16float texture (island in r, rocks in g) and read with one bilinear fetch.
    const bakeFormat: GPUTextureFormat = 'rg16float';
    const bakePipeline = createFullscreenPipeline(device, {
      fragment: {
        label: 'toon-water/sdf-bake.wgsl',
        code: bakeShader.code,
        chunks: { ...bakeShader.chunks, 'toon_params.wgsl': Params.wgsl },
      },
      format: bakeFormat,
    });
    const bakeGroup = device.createBindGroup({
      label: 'toon-water bake',
      layout: bakePipeline.getBindGroupLayout(0),
      entries: [{ binding: 0, resource: { buffer: uniforms.buffer } }],
    });
    const sampler = device.createSampler({
      label: 'toon-water sdf sampler',
      magFilter: 'linear',
      minFilter: 'linear',
    });
    let field: GPUTexture | undefined;
    let group: GPUBindGroup | undefined;
    let needsBake = true;

    let width = 1;
    let height = 1;
    // Accumulated so the speed slider never makes the foam jump.
    let time = 0;
    let foamTime = 0;
    const palette = new Float32Array(12 * 4);

    return {
      resize(w, h) {
        width = Math.max(1, Math.round(w));
        height = Math.max(1, Math.round(h));
        field?.destroy();
        field = device.createTexture({
          label: 'toon-water distance field',
          size: [Math.max(16, Math.ceil(width / 2)), Math.max(16, Math.ceil(height / 2))],
          format: bakeFormat,
          usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
        });
        group = device.createBindGroup({
          label: 'toon-water',
          layout: pipeline.getBindGroupLayout(0),
          entries: [
            { binding: 0, resource: { buffer: uniforms.buffer } },
            { binding: 1, resource: field.createView() },
            { binding: 2, resource: sampler },
          ],
        });
        needsBake = true;
      },

      frame({ delta }, encoder, target) {
        if (!field || !group) return;
        time += delta;
        foamTime += delta * params.foamSpeed * 1.4;
        const aspect = width / height;
        const colours = paletteFor(params.palette, {
          deep: params.customDeep,
          shallow: params.customShallow,
          foam: params.customFoam,
        });
        SLOTS.forEach((slot, i) => palette.set([...colours[slot], 1], i * 4));

        uniforms.set({
          resolution: [width, height],
          aspect,
          // Narrow canvases (phones in portrait) zoom out so the island still fits.
          zoom: Math.max(1, 1.25 / aspect),
          time,
          foam_time: foamTime,
          bands: params.bands,
          depth: params.depth,
          foam_scale: params.foamScale,
          foam_width: params.foamThickness,
          shore: params.shoreFoam,
          highlight: params.highlight,
          outline: params.outline ? 1 : 0,
          deep_foam: params.deepFoam ? 1 : 0,
          palette,
        });
        uniforms.upload();
        if (needsBake) {
          needsBake = false;
          renderFullscreen(encoder, field.createView(), bakePipeline, [bakeGroup], {
            timestampWrites: profiler.pass('distance field (on resize)'),
          });
        }
        renderFullscreen(encoder, target, pipeline, [group], {
          timestampWrites: profiler.pass('toon water'),
        });
      },

      destroy() {
        field?.destroy();
        uniforms.destroy();
      },
    };
  },
});

import { createFullscreenPipeline, renderFullscreen } from '../../gpu/fullscreen.ts';
import { trackScroll } from '../../gpu/scroll.ts';
import { createRenderTarget, type RenderTarget } from '../../gpu/targets.ts';
import { createUniformBuffer, defineUniformStruct } from '../../gpu/uniforms.ts';
import { defineEffect } from '../types.ts';
import composeShader from './compose.wgsl?raw';
import layersShader from './layers.wgsl?raw';
import { meta } from './meta.ts';

const LayerParams = defineUniformStruct('LayerParams', {
  drift: 'vec2f',
  extent: 'vec2f',
  phase: 'f32',
  time: 'f32',
  warp: 'f32',
  coverage: 'f32',
  scale: 'f32',
  stretch: 'f32',
  scroll: 'f32',
  height: 'f32',
  falloff: 'f32',
  source_edge: 'u32',
});

const ComposeParams = defineUniformStruct('ComposeParams', {
  density: 'f32',
  max_opacity: 'f32',
  tint: 'vec3f',
});

type Rgb = [number, number, number];

const srgbToLinear = (c: number) => (c <= 0.04045 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4);
const linearToSrgb = (c: number) => {
  const x = Math.min(1, Math.max(0, c));
  return x <= 0.0031308 ? x * 12.92 : 1.055 * x ** (1 / 2.4) - 0.055;
};
/** Linear RGB from a CSS hex colour, the form every colour parameter takes. */
const hex = (css: string): Rgb => {
  const n = parseInt(css.replace('#', ''), 16);
  return [(n >> 16) & 255, (n >> 8) & 255, n & 255].map((v) => srgbToLinear(v / 255)) as Rgb;
};

/** Where the mist comes from; the order is the shader's switch. */
const EDGES = ['bottom', 'top', 'left', 'right', 'none'] as const;
const RESOLUTIONS = ['1/4', '1/2', 'full'] as const;
const RESOLUTION_DIVISOR: Record<string, number> = { '1/4': 4, '1/2': 2, full: 1 };

/** Highest opacity the control allows; mist.wgsl clamps to the same value. */
export const MIST_MAX_OPACITY = 0.8;

export const effect = defineEffect({
  meta,
  params: {
    from: { type: 'select', default: 'bottom', options: EDGES, label: 'Rises from' },
    density: { type: 'number', default: 1.15, min: 0.1, max: 2, step: 0.01, label: 'Density' },
    maxOpacity: {
      type: 'number',
      default: 0.38,
      min: 0,
      max: MIST_MAX_OPACITY,
      step: 0.01,
      label: 'Max opacity (cap)',
    },
    height: { type: 'number', default: 0.6, min: 0.1, max: 1.2, step: 0.01, label: 'Height' },
    falloff: { type: 'number', default: 0.35, min: 0.02, max: 0.8, step: 0.01, label: 'Falloff' },
    coverage: { type: 'number', default: 0.62, min: 0, max: 1, step: 0.01, label: 'Coverage' },
    wispScale: { type: 'number', default: 1.6, min: 0.5, max: 4, step: 0.05, label: 'Wisp scale' },
    stretch: {
      type: 'number',
      default: 2,
      min: 0.3,
      max: 4,
      step: 0.01,
      label: 'Stretch (wide > 1, tall < 1)',
    },
    warp: { type: 'number', default: 0.8, min: 0, max: 1.5, step: 0.01, label: 'Billow' },
    driftAngle: {
      type: 'number',
      default: 15,
      min: 0,
      max: 360,
      step: 1,
      label: 'Drift direction (deg, 90 = up)',
    },
    speed: { type: 'number', default: 0.3, min: 0, max: 2, step: 0.01, label: 'Drift speed' },
    parallax: {
      type: 'number',
      default: 0.6,
      min: 0,
      max: 1.5,
      step: 0.01,
      label: 'Scroll parallax',
    },
    tint: { type: 'color', default: hex('#c3d2db'), label: 'Tint' },
    fieldResolution: {
      type: 'select',
      default: '1/4',
      options: RESOLUTIONS,
      label: 'Field resolution',
    },
  },
  presets: {
    'morning mist': {
      from: 'bottom',
      density: 1.15,
      maxOpacity: 0.38,
      height: 0.6,
      falloff: 0.35,
      coverage: 0.62,
      wispScale: 1.6,
      stretch: 2,
      warp: 0.8,
      driftAngle: 15,
      speed: 0.3,
      parallax: 0.6,
      tint: hex('#c3d2db'),
    },
    'hot spring steam': {
      from: 'bottom',
      density: 1.2,
      maxOpacity: 0.42,
      height: 0.85,
      falloff: 0.6,
      coverage: 0.5,
      wispScale: 1.4,
      stretch: 0.65,
      warp: 1.2,
      driftAngle: 88,
      speed: 0.7,
      parallax: 0.9,
      tint: hex('#d9d6d2'),
    },
    'low fog': {
      from: 'bottom',
      density: 1.4,
      maxOpacity: 0.5,
      height: 0.3,
      falloff: 0.25,
      coverage: 0.7,
      wispScale: 1.2,
      stretch: 3.2,
      warp: 0.5,
      driftAngle: 0,
      speed: 0.15,
      parallax: 0.4,
      tint: hex('#b4c3cc'),
    },
    'top haze': {
      from: 'top',
      density: 0.9,
      maxOpacity: 0.34,
      height: 0.5,
      falloff: 0.5,
      coverage: 0.5,
      wispScale: 1.8,
      stretch: 1.6,
      warp: 0.9,
      driftAngle: 200,
      speed: 0.25,
      parallax: 0.5,
      tint: hex('#cfcde0'),
    },
  },

  async create({ device, format, canvas, params, profiler }) {
    const layerUniforms = createUniformBuffer(device, LayerParams, 'mist layer params');
    const composeUniforms = createUniformBuffer(device, ComposeParams, 'mist compose params');

    const layerPipeline = createFullscreenPipeline(device, {
      fragment: {
        label: 'mist/layers.wgsl',
        code: layersShader,
        chunks: { 'layers_params.wgsl': LayerParams.wgsl },
      },
      format: 'rgba16float',
    });
    const composePipeline = createFullscreenPipeline(device, {
      fragment: {
        label: 'mist/compose.wgsl',
        code: composeShader,
        chunks: { 'compose_params.wgsl': ComposeParams.wgsl },
      },
      format,
    });
    const layerGroup = device.createBindGroup({
      label: 'mist layers',
      layout: layerPipeline.getBindGroupLayout(0),
      entries: [{ binding: 0, resource: { buffer: layerUniforms.buffer } }],
    });
    const sampler = device.createSampler({
      label: 'mist sampler',
      magFilter: 'linear',
      minFilter: 'linear',
    });
    const scroll = trackScroll(canvas);

    let width = 1;
    let height = 1;
    let cssHeight = 1;
    let divisor = 4;
    let layers: RenderTarget | undefined;
    let composeGroup: GPUBindGroup | undefined;
    // Accumulated, so changing speed or direction never makes the mist jump.
    let phase = 0;
    let time = 0;

    const sizeTarget = () => {
      const w = Math.max(2, Math.ceil(width / divisor));
      const h = Math.max(2, Math.ceil(height / divisor));
      if (!layers) {
        layers = createRenderTarget(device, {
          label: 'mist layers',
          format: 'rgba16float',
          width: w,
          height: h,
        });
      } else if (!layers.resize(w, h) && composeGroup) {
        return;
      }
      composeGroup = device.createBindGroup({
        label: 'mist compose',
        layout: composePipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: composeUniforms.buffer } },
          { binding: 1, resource: layers.view },
          { binding: 2, resource: sampler },
        ],
      });
    };

    return {
      resize(w, h, ratio) {
        width = Math.max(1, Math.round(w));
        height = Math.max(1, Math.round(h));
        cssHeight = height / Math.max(ratio, 0.1);
        sizeTarget();
      },

      frame({ delta }, encoder, target) {
        const wanted = RESOLUTION_DIVISOR[params.fieldResolution] ?? 4;
        if (wanted !== divisor) {
          divisor = wanted;
          sizeTarget();
        }
        if (!layers || !composeGroup) return;

        const state = scroll.update(delta);
        // Scrolling fast stirs the mist a little; at rest it adds nothing.
        const stir = Math.min(Math.abs(state.velocity) / 1500, 1) * Math.min(params.parallax, 1);
        phase += delta * params.speed * 0.12 * (1 + 0.8 * stir);
        time += delta * (0.2 + params.speed * 0.5);

        const angle = (params.driftAngle * Math.PI) / 180;
        layerUniforms.set({
          // Screen y points down, so "up" (90 degrees) is -y.
          drift: [Math.cos(angle), -Math.sin(angle)],
          extent: width >= height ? [width / height, 1] : [1, height / width],
          phase,
          time,
          warp: params.warp,
          coverage: params.coverage,
          scale: params.wispScale,
          stretch: params.stretch,
          scroll: (state.offset / cssHeight) * params.parallax * 0.5,
          height: params.height,
          falloff: params.falloff,
          source_edge: Math.max(0, EDGES.indexOf(params.from as (typeof EDGES)[number])),
        });
        layerUniforms.upload();
        renderFullscreen(encoder, layers.view, layerPipeline, [layerGroup], {
          timestampWrites: profiler.pass('layers'),
        });

        composeUniforms.set({
          density: params.density,
          max_opacity: Math.min(params.maxOpacity, MIST_MAX_OPACITY),
          tint: [
            linearToSrgb(params.tint[0]),
            linearToSrgb(params.tint[1]),
            linearToSrgb(params.tint[2]),
          ],
        });
        composeUniforms.upload();
        renderFullscreen(encoder, target, composePipeline, [composeGroup], {
          timestampWrites: profiler.pass('compose'),
        });
      },

      destroy() {
        scroll.destroy();
        layers?.destroy();
        layerUniforms.destroy();
        composeUniforms.destroy();
      },
    };
  },
});

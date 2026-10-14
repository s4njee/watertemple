import { createFullscreenPipeline, renderFullscreen } from '../../gpu/fullscreen.ts';
import { trackScroll } from '../../gpu/scroll.ts';
import { createUniformBuffer, defineUniformStruct } from '../../gpu/uniforms.ts';
import { defineEffect } from '../types.ts';
import shader from './divider.wgsl?raw';
import { meta } from './meta.ts';

const DividerParams = defineUniformStruct('DividerParams', {
  size: 'vec2f',
  px: 'f32',
  phase: 'f32',
  time: 'f32',
  layers: 'i32',
  flip: 'u32',
  amplitude: 'f32',
  wavelength: 'f32',
  spread: 'f32',
  roughness: 'f32',
  foam: 'f32',
  foam_width: 'f32',
  back_alpha: 'f32',
  shift: 'f32',
  lift: 'f32',
  boost: 'f32',
  color_back: 'vec3f',
  color_front: 'vec3f',
  color_foam: 'vec3f',
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
const encode = (c: readonly number[]): Rgb => [
  linearToSrgb(c[0] ?? 0),
  linearToSrgb(c[1] ?? 0),
  linearToSrgb(c[2] ?? 0),
];

/** The CSS variable the demo reads for "match the page"; the band context fills the section below with it. */
const PAGE_VARIABLE = '--wt-surface';

/**
 * Demo convenience, not part of the technique: a hidden element whose background is the page's
 * section colour, so the front layer can match it in both site themes. A real site would paste its
 * own colour into `colorFront` instead. Returns sRGB 0..1, or undefined without a DOM or a colour.
 */
function createPageColorProbe(canvas: HTMLCanvasElement) {
  const parent = typeof document === 'undefined' ? null : canvas.parentElement;
  if (!parent) return { read: (): Rgb | undefined => undefined, destroy() {} };
  const probe = document.createElement('div');
  probe.setAttribute('aria-hidden', 'true');
  probe.style.cssText = `position:absolute;width:0;height:0;visibility:hidden;pointer-events:none;background-color:var(${PAGE_VARIABLE})`;
  parent.append(probe);
  return {
    read(): Rgb | undefined {
      const text = getComputedStyle(probe).backgroundColor;
      const m = /rgba?\(\s*([\d.]+)[ ,]+([\d.]+)[ ,]+([\d.]+)(?:[ ,/]+([\d.%]+))?/.exec(text);
      if (!m || (m[4] !== undefined && Number.parseFloat(m[4]) < 1)) return undefined;
      return [Number(m[1]) / 255, Number(m[2]) / 255, Number(m[3]) / 255];
    },
    destroy() {
      probe.remove();
    },
  };
}

export const effect = defineEffect({
  meta,
  params: {
    layers: { type: 'number', default: 4, min: 2, max: 5, step: 1, label: 'Layers' },
    amplitude: {
      type: 'number',
      default: 0.16,
      min: 0.02,
      max: 0.3,
      step: 0.005,
      label: 'Wave height (of band)',
    },
    wavelength: {
      type: 'number',
      default: 340,
      min: 80,
      max: 1200,
      step: 5,
      label: 'Wavelength (CSS px)',
    },
    spread: {
      type: 'number',
      default: 0.6,
      min: 0,
      max: 1,
      step: 0.01,
      label: 'Layer spread',
    },
    speed: { type: 'number', default: 0.5, min: 0, max: 3, step: 0.01, label: 'Speed' },
    roughness: { type: 'number', default: 0.3, min: 0, max: 1, step: 0.01, label: 'Roughness' },
    foam: { type: 'number', default: 0.55, min: 0, max: 1, step: 0.01, label: 'Foam line' },
    foamWidth: {
      type: 'number',
      default: 6,
      min: 2,
      max: 28,
      step: 0.5,
      label: 'Foam width (CSS px)',
    },
    parallax: {
      type: 'number',
      default: 0.5,
      min: 0,
      max: 1,
      step: 0.01,
      label: 'Scroll parallax',
    },
    scrollResponse: {
      type: 'number',
      default: 0.4,
      min: 0,
      max: 1,
      step: 0.01,
      label: 'Scroll-speed response',
    },
    backAlpha: {
      type: 'number',
      default: 0.85,
      min: 0.2,
      max: 1,
      step: 0.01,
      label: 'Back layer opacity',
    },
    flip: { type: 'boolean', default: false, label: 'Flip (cap the bottom of a section)' },
    matchPage: {
      type: 'boolean',
      default: true,
      label: 'Front = demo page colour',
    },
    colorBack: { type: 'color', default: hex('#4f9db0'), label: 'Back layer' },
    colorFront: { type: 'color', default: hex('#ffffff'), label: 'Front layer (exact)' },
    colorFoam: { type: 'color', default: hex('#eaf7f9'), label: 'Foam' },
  },
  presets: {
    'calm footer': {
      layers: 4,
      amplitude: 0.1,
      wavelength: 520,
      spread: 0.5,
      speed: 0.35,
      roughness: 0.25,
      foam: 0.55,
      foamWidth: 8,
      parallax: 0.5,
      scrollResponse: 0.3,
      backAlpha: 0.9,
      flip: false,
      matchPage: true,
      colorBack: hex('#4f9db0'),
      colorFoam: hex('#eaf7f9'),
    },
    'stormy break': {
      layers: 5,
      amplitude: 0.2,
      wavelength: 260,
      spread: 0.7,
      speed: 1.5,
      roughness: 0.85,
      foam: 1,
      foamWidth: 16,
      parallax: 0.8,
      scrollResponse: 1,
      backAlpha: 0.95,
      flip: false,
      matchPage: true,
      colorBack: hex('#0e2b38'),
      colorFoam: hex('#e1f4f7'),
    },
    pastel: {
      layers: 3,
      amplitude: 0.13,
      wavelength: 380,
      spread: 0.45,
      speed: 0.5,
      roughness: 0.12,
      foam: 0.35,
      foamWidth: 7,
      parallax: 0.4,
      scrollResponse: 0.2,
      backAlpha: 0.8,
      flip: false,
      matchPage: false,
      colorBack: hex('#bfe3ea'),
      colorFront: hex('#f8dccf'),
      colorFoam: hex('#ffffff'),
    },
    'section cap': {
      layers: 4,
      amplitude: 0.12,
      wavelength: 440,
      spread: 0.55,
      speed: 0.45,
      roughness: 0.3,
      foam: 0.7,
      foamWidth: 9,
      flip: true,
      matchPage: true,
      colorBack: hex('#2f7f93'),
      colorFoam: hex('#eaf7f9'),
    },
  },

  async create({ device, format, canvas, params, profiler }) {
    const uniforms = createUniformBuffer(device, DividerParams, 'wave-divider params');
    const pipeline = createFullscreenPipeline(device, {
      fragment: {
        label: 'wave-divider/divider.wgsl',
        code: shader,
        chunks: { 'divider_params.wgsl': DividerParams.wgsl },
      },
      format,
    });
    const bindGroup = device.createBindGroup({
      label: 'wave-divider',
      layout: pipeline.getBindGroupLayout(0),
      entries: [{ binding: 0, resource: { buffer: uniforms.buffer } }],
    });
    const scroll = trackScroll(canvas);
    const probe = createPageColorProbe(canvas);

    let cssWidth = 1;
    let cssHeight = 1;
    let dpr = 1;
    let frames = 0;
    let pageColor: Rgb | undefined;
    // Accumulated, so changing speed never makes the waves jump.
    let phase = 0;
    let time = 0;

    return {
      resize(width, height, ratio) {
        dpr = Math.max(ratio, 0.1);
        cssWidth = width / dpr;
        cssHeight = height / dpr;
      },
      frame({ delta }, encoder, target) {
        const state = scroll.update(delta);
        // Scroll speed swells the waves and quickens them a little; at rest it adds nothing.
        const rush = Math.min(Math.abs(state.velocity) / 1200, 1) * params.scrollResponse;
        phase += delta * (params.speed + rush * 1.5);
        time += delta * (0.5 + params.speed);

        if (params.matchPage && frames++ % 15 === 0) pageColor = probe.read();
        const front = params.matchPage && pageColor ? pageColor : encode(params.colorFront);

        uniforms.set({
          size: [cssWidth, cssHeight],
          px: 1 / dpr,
          phase,
          time,
          layers: Math.round(params.layers),
          flip: params.flip ? 1 : 0,
          amplitude: params.amplitude * cssHeight,
          wavelength: params.wavelength,
          spread: params.spread,
          roughness: params.roughness,
          foam: params.foam,
          foam_width: params.foamWidth,
          back_alpha: params.backAlpha,
          shift: state.offset * params.parallax * 0.5,
          lift: (state.progress - 0.5) * params.parallax * 0.2,
          boost: 1 + rush * 0.45,
          color_back: encode(params.colorBack),
          color_front: front,
          color_foam: encode(params.colorFoam),
        });
        uniforms.upload();
        renderFullscreen(encoder, target, pipeline, [bindGroup], {
          timestampWrites: profiler.pass('waves'),
        });
      },
      destroy() {
        scroll.destroy();
        probe.destroy();
        uniforms.destroy();
      },
    };
  },
});

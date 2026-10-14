import { createFullscreenPipeline, renderFullscreen } from '../../gpu/fullscreen.ts';
import {
  classifyFile,
  createMediaFromFile,
  createMediaTexture,
  mediaFromTexture,
  type MediaFit,
  type MediaTexture,
} from '../../gpu/media.ts';
import { generateMipmaps, mipLevelCount } from '../../gpu/mipmap.ts';
import { trackPointers } from '../../gpu/pointer.ts';
import {
  createFluid,
  type Fluid,
  type FluidBoundary,
  type FluidScheme,
} from '../../gpu/sim/fluid.ts';
import { createFluidStirrer, type FluidStirMode } from '../../gpu/sim/fluid-stir.ts';
import {
  createFluidWarp,
  createFluidWarpPresenter,
  type WarpView,
} from '../../gpu/sim/fluid-warp.ts';
import { createUniformBuffer, defineUniformStruct } from '../../gpu/uniforms.ts';
import { defineEffect } from '../types.ts';
// Read-only reuse of the Epic 4 effect's helpers: the harbour still, the animated canvas behind the
// live <video> (both DOM-only) and the shared test card that stands in for them without a DOM.
import sceneShader from '../_shared/media-scene.wgsl?raw';
import { createLiveVideo, createStillCanvas, type LiveVideo } from '../media-distortion/sources.ts';
import { HEADLINES, renderHeadline } from './headline.ts';
import { meta } from './meta.ts';

const SOURCES = ['image', 'video', 'headline'] as const;
const FITS = ['cover', 'contain'] as const;
const RESOLUTIONS = ['64', '96', '128', '192'] as const;
const BOUNDARIES = ['wall', 'open'] as const;
const POINTER_MODES = ['hover and press', 'press only', 'hover only', 'off'] as const;
const TOUCH_MODES = ['scrolls the page', 'stirs the content'] as const;
const VIEWS = ['content', 'displacement', 'grid', 'velocity'] as const;
const PROCEDURAL_SIZE = { width: 640, height: 360 };
/** The solver's dye is unused here; this is the smallest grid it will make. */
const DYE_CELLS = 16;
/** A resized canvas rebuilds the headline once the size has been steady for this long, seconds. */
const TEXT_SETTLE = 0.2;

const SceneParams = defineUniformStruct('SceneParams', { aspect: 'f32', aa: 'f32' });

/** mulberry32: a small seeded generator, so the idle path of a preset repeats. */
function createRandom(seed: number): () => number {
  let state = seed >>> 0;
  return () => {
    state = (state + 0x6d2b79f5) >>> 0;
    let t = state;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

export const effect = defineEffect({
  meta,
  params: {
    source: {
      type: 'select',
      default: 'image',
      options: SOURCES,
      label: 'Content (or drop a file on the canvas)',
    },
    headline: { type: 'select', default: HEADLINES[0], options: HEADLINES, label: 'Headline' },
    fit: { type: 'select', default: 'cover', options: FITS, label: 'Image fit' },
    relaxation: {
      type: 'number',
      default: 1,
      min: 0,
      max: 8,
      step: 0.01,
      label: 'Spring back (1 / s; 0 = a smear is permanent, like paint)',
    },
    maxDisplacement: {
      type: 'number',
      default: 0.4,
      min: 0.05,
      max: 1,
      step: 0.01,
      label: 'Largest displacement (canvas heights)',
    },
    radius: {
      type: 'number',
      default: 0.07,
      min: 0.015,
      max: 0.25,
      step: 0.001,
      label: 'Stir radius (canvas heights)',
    },
    force: {
      type: 'number',
      default: 0.8,
      min: 0,
      max: 3,
      step: 0.01,
      label: 'Push (fraction of pointer velocity)',
    },
    viscosity: {
      type: 'number',
      default: 0.0005,
      min: 0,
      max: 0.01,
      step: 0.0001,
      label: 'Viscosity (heights² / s; smooths the smear)',
    },
    velocityDissipation: {
      type: 'number',
      default: 1.2,
      min: 0,
      max: 6,
      step: 0.01,
      label: 'Flow dissipation (1 / s; how long a push keeps going)',
    },
    vorticity: {
      type: 'number',
      default: 5,
      min: 0,
      max: 30,
      step: 0.5,
      label: 'Vorticity confinement (swirl)',
    },
    boundary: { type: 'select', default: 'wall', options: BOUNDARIES, label: 'Edges' },
    chroma: {
      type: 'number',
      default: 0.25,
      min: 0,
      max: 1,
      step: 0.01,
      label: 'Chromatic split (follows the stretch)',
    },
    sheen: { type: 'number', default: 0.35, min: 0, max: 1, step: 0.01, label: 'Liquid sheen' },
    ambient: {
      type: 'number',
      default: 0.5,
      min: 0,
      max: 1,
      step: 0.01,
      label: 'Idle stir (needs spring back > 0)',
    },
    calm: {
      type: 'number',
      default: 0.8,
      min: 0,
      max: 1,
      step: 0.01,
      label: 'Quiet down while touched',
    },
    pointer: { type: 'select', default: 'hover and press', options: POINTER_MODES },
    touch: {
      type: 'select',
      default: 'scrolls the page',
      options: TOUCH_MODES,
      label: 'Touch drags',
    },
    resolution: {
      type: 'select',
      default: '96',
      options: RESOLUTIONS,
      label: 'Fluid cells (short side; re-creates the fluid)',
    },
    view: { type: 'select', default: 'content', options: VIEWS, label: 'View' },
    debugGain: {
      type: 'number',
      default: 1,
      min: 0.1,
      max: 10,
      step: 0.1,
      label: 'Debug view gain',
    },
    stepsPerFrame: {
      type: 'number',
      default: 0,
      min: 0,
      max: 32,
      step: 1,
      label: 'Steps per frame (0 = by clock; for benchmarks)',
    },
  },
  presets: {
    'wet paint': {
      source: 'image',
      relaxation: 0,
      maxDisplacement: 0.6,
      radius: 0.06,
      force: 0.9,
      viscosity: 0.0008,
      velocityDissipation: 1.6,
      vorticity: 4,
      chroma: 0.1,
      sheen: 0.55,
      ambient: 0,
    },
    jelly: {
      source: 'image',
      relaxation: 4.5,
      maxDisplacement: 0.3,
      radius: 0.1,
      force: 1.3,
      viscosity: 0.0003,
      velocityDissipation: 0.4,
      vorticity: 0,
      chroma: 0.15,
      sheen: 0.25,
      ambient: 0.35,
    },
    'liquid headline': {
      source: 'headline',
      relaxation: 1.1,
      maxDisplacement: 0.4,
      radius: 0.07,
      force: 0.85,
      viscosity: 0.0005,
      velocityDissipation: 1.2,
      vorticity: 5,
      chroma: 0.3,
      sheen: 0.4,
      ambient: 0.5,
    },
    'smeared video': {
      source: 'video',
      relaxation: 0.5,
      maxDisplacement: 0.5,
      radius: 0.08,
      force: 1,
      viscosity: 0.0006,
      velocityDissipation: 1,
      vorticity: 6,
      chroma: 0.35,
      sheen: 0.3,
      ambient: 0.3,
    },
  },

  async create({ device, format, canvas, params, profiler }) {
    const pointers = trackPointers(canvas);
    const stirrer = createFluidStirrer();
    const random = createRandom(0x3a7c);
    const reducedMotion =
      typeof matchMedia === 'function' && matchMedia('(prefers-reduced-motion: reduce)').matches;
    const presenter = createFluidWarpPresenter({ device, format, label: 'fluid-warp view' });

    let width = Math.max(1, canvas.width || 1);
    let height = Math.max(1, canvas.height || 1);
    /** The aspect the fluid was last sized for; it only follows the canvas when that changes a lot. */
    let simAspect = width / height;
    const cells = (shortSide: number, aspect: number) =>
      Math.round(shortSide * Math.max(aspect, 1 / aspect));

    const fluid: Fluid = createFluid({
      device,
      resolution: cells(Number(params.resolution), simAspect),
      dyeResolution: DYE_CELLS,
      width,
      height,
      profiler,
      label: 'fluid-warp',
      // The dye is not used; the cheapest scheme keeps its pass negligible.
      params: { dyeScheme: 'semi-lagrangian', dyeDissipation: 0, pressureIterations: 24 },
    });
    const warp = createFluidWarp({
      device,
      fluid,
      scale: 2,
      params: { relaxation: params.relaxation },
      profiler,
      label: 'fluid-warp',
    });
    warp.attach();

    // --- Content ----------------------------------------------------------------------------------
    let destroyed = false;
    let media: MediaTexture | undefined;
    let mediaKey = '';
    let live: LiveVideo | undefined;
    let loading = false;
    let dropped = false;
    let seenSource = `${params.source}:${params.headline}`;
    let sizeChangedAt = 0;
    let now = 0;

    /** The test card drawn once into a texture: the stand-in for every source when there is no DOM. */
    function createProceduralMedia(): MediaTexture {
      const { width: w, height: h } = PROCEDURAL_SIZE;
      const texture = device.createTexture({
        label: 'fluid-warp procedural picture',
        size: [w, h],
        format: 'rgba8unorm-srgb',
        mipLevelCount: mipLevelCount(w, h),
        usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.RENDER_ATTACHMENT,
      });
      const sceneUniforms = createUniformBuffer(device, SceneParams, 'fluid-warp scene params');
      sceneUniforms.set({ aspect: w / h, aa: 1 / h });
      sceneUniforms.upload();
      const scenePipeline = createFullscreenPipeline(device, {
        fragment: { label: 'media-distortion/scene.wgsl', code: sceneShader },
        format: 'rgba8unorm-srgb',
      });
      const group = device.createBindGroup({
        layout: scenePipeline.getBindGroupLayout(0),
        entries: [{ binding: 0, resource: { buffer: sceneUniforms.buffer } }],
      });
      const encoder = device.createCommandEncoder({ label: 'fluid-warp procedural picture' });
      renderFullscreen(
        encoder,
        texture.createView({ baseMipLevel: 0, mipLevelCount: 1 }),
        scenePipeline,
        [group],
      );
      generateMipmaps(device, texture, encoder);
      device.queue.submit([encoder.finish()]);
      return mediaFromTexture(texture, () => sceneUniforms.destroy());
    }

    const textKey = () => `headline:${params.headline}:${width}x${height}`;
    const wantedKey = () => (params.source === 'headline' ? textKey() : params.source);

    async function makeMedia(which: string): Promise<MediaTexture> {
      if (which === 'video') {
        const next = await createLiveVideo().catch(() => undefined);
        if (next) {
          try {
            const result = await createMediaTexture(device, next.video, {
              label: 'fluid-warp live video',
              onDestroy: () => next.destroy(),
            });
            live = next;
            return result;
          } catch (error) {
            next.destroy();
            console.warn('fluid-warp: the live video could not be used', error);
          }
        }
      }
      if (which.startsWith('headline')) {
        const card = renderHeadline(params.headline, width, height);
        if (card) {
          // 1:1 texels on the canvas, so no mip chain: type stays as crisp as the 2D canvas drew it.
          return createMediaTexture(device, card, {
            label: 'fluid-warp headline',
            mipmaps: false,
          });
        }
      } else {
        const painted = createStillCanvas();
        if (painted) return createMediaTexture(device, painted, { label: 'fluid-warp harbour' });
      }
      return createProceduralMedia();
    }

    function swapIn(next: MediaTexture, key: string) {
      if (destroyed) {
        next.destroy();
        return;
      }
      const previous = media;
      media = next;
      mediaKey = key;
      if (key !== 'video') live = undefined;
      previous?.destroy();
    }

    async function load(which: string) {
      if (loading) return;
      loading = true;
      try {
        const next = await makeMedia(which);
        // The key says what was asked for, so a video that could not start is not asked for again.
        swapIn(next, which);
      } catch (error) {
        console.warn('fluid-warp: could not load the content', error);
        mediaKey = which;
      } finally {
        loading = false;
      }
    }

    async function loadFile(file: File) {
      if (!classifyFile(file)) {
        console.warn(`fluid-warp: ${file.name || 'that file'} is not an image or a video`);
        return;
      }
      try {
        const next = await createMediaFromFile(device, file);
        dropped = true;
        swapIn(next, `file:${file.name}`);
      } catch (error) {
        console.warn('fluid-warp: could not use the dropped file', error);
      }
    }

    // Drag and drop on the canvas itself: the file is read in this page and goes nowhere else.
    const hasFiles = (event: DragEvent) => event.dataTransfer?.types.includes('Files') ?? false;
    const controller = new AbortController();
    const { signal } = controller;
    canvas.addEventListener(
      'dragover',
      (event) => {
        if (!hasFiles(event as DragEvent)) return;
        event.preventDefault();
      },
      { signal },
    );
    canvas.addEventListener(
      'drop',
      (event) => {
        event.preventDefault();
        const file = (event as DragEvent).dataTransfer?.files[0];
        if (file) void loadFile(file);
      },
      { signal },
    );

    // The first content is made now so the first frame has something; the rest load as they are asked for.
    {
      const first = wantedKey();
      swapIn(await makeMedia(first), first);
    }

    // --- Idle stir ----------------------------------------------------------------------------------
    // One emitter wanders on a slow Lissajous path pushing as it goes; its strength follows the spring
    // back (a smear that never relaxes must not be wrecked by an idle hand) and it eases off while
    // someone is touching the content.
    const emitter = {
      phase: random() * 100,
      last: undefined as [number, number] | undefined,
    };
    let interaction = 0;

    function ambient(delta: number, aspect: number) {
      const spring = Math.min(1, params.relaxation / 0.6);
      const level =
        params.ambient * spring * (1 - params.calm * interaction) * (reducedMotion ? 0.3 : 1);
      const dt = Math.min(Math.max(delta, 1 / 240), 0.1);
      emitter.phase += dt;
      const at: [number, number] = [
        0.5 + 0.38 * Math.sin(emitter.phase * 0.43) * Math.cos(emitter.phase * 0.11),
        0.5 + 0.34 * Math.sin(emitter.phase * 0.61 + 1.3),
      ];
      const from = emitter.last;
      emitter.last = at;
      if (!from || level < 0.01) return;
      const dx = (at[0] - from[0]) * aspect;
      const dy = at[1] - from[1];
      const length = Math.hypot(dx, dy);
      if (length < 1e-6) return;
      // A gentle push along the path (about 0.08 heights per second at full level): the picture breathes
      // by a few percent of its height and is plainly legible, which a visitor's own stroke (1 or more) is not.
      const speed = 0.1 * level;
      const ramp = params.radius * 0.5;
      fluid.addForce(
        from,
        at,
        params.radius * 1.3,
        [(dx / length) * speed, (dy / length) * speed],
        ramp,
      );
    }

    function sync() {
      const p = fluid.params;
      p.velocityScheme = 'maccormack' as FluidScheme;
      p.vorticity = params.vorticity;
      p.viscosity = params.viscosity;
      p.velocityDissipation = params.velocityDissipation;
      p.boundary = params.boundary as FluidBoundary;
      warp.params.relaxation = params.relaxation;
      warp.params.maxDisplacement = params.maxDisplacement;
    }

    return {
      resize(w, h) {
        width = Math.max(1, Math.round(w));
        height = Math.max(1, Math.round(h));
        sizeChangedAt = now;
        const aspect = width / height;
        // Following every pixel of a window drag would re-create the grids (and clear the smear)
        // each time; follow the canvas only when its shape has really changed.
        if (Math.abs(Math.log(aspect / simAspect)) > 0.06) {
          simAspect = aspect;
          fluid.resize(width, height);
        }
      },

      frame({ delta, time }, encoder, target) {
        now = time;
        sync();
        const aspect = width / height;
        fluid.setResolution(cells(Number(params.resolution), simAspect), DYE_CELLS);

        // Content. A different source choice discards a dropped file; the headline is redrawn at the
        // new size once the canvas has stopped resizing.
        const sourceKey = `${params.source}:${params.headline}`;
        if (sourceKey !== seenSource) {
          seenSource = sourceKey;
          dropped = false;
        }
        const wanted = wantedKey();
        const settled = time - sizeChangedAt > TEXT_SETTLE || !media;
        if (
          !dropped &&
          wanted !== mediaKey &&
          !loading &&
          (params.source !== 'headline' || settled)
        ) {
          void load(wanted);
        }
        live?.tick(time);
        media?.update(encoder);

        // Input: a press-drag stirs; touch drags scroll the page unless the visitor asked the content to take them.
        pointers.setCapture(params.touch === 'stirs the content');
        const stir = stirrer.stir(fluid, pointers.poll(), aspect, delta, {
          mode: params.pointer as FluidStirMode,
          radius: params.radius,
          force: params.force,
          dyeAmount: 0,
          colour: () => null,
        });
        interaction +=
          ((stir.active ? 1 : 0) - interaction) * (1 - Math.exp(-delta * (stir.active ? 5 : 0.7)));
        ambient(delta, aspect);

        if (params.stepsPerFrame > 0) fluid.advance(encoder, params.stepsPerFrame);
        else fluid.update(encoder, delta);

        if (!media) return;
        presenter.draw(encoder, target, {
          warp,
          content: media,
          fit: mediaKey.startsWith('headline') ? 'stretch' : (params.fit as MediaFit),
          canvasSize: [width, height],
          bars: [0.003, 0.012, 0.025],
          chroma: params.chroma * 0.12,
          sheen: params.sheen,
          view: params.view as WarpView,
          debugGain: params.debugGain,
          fields: fluid.fieldsView,
          timestampWrites: profiler.pass('shade'),
        });
      },

      destroy() {
        destroyed = true;
        controller.abort();
        pointers.destroy();
        warp.destroy();
        fluid.destroy();
        presenter.destroy();
        media?.destroy();
        media = undefined;
      },
    };
  },
});

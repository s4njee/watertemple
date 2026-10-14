import { createAudioAnalyser, type AudioFrame } from '../../gpu/audio.ts';
import { createFullscreenPipeline, renderFullscreen } from '../../gpu/fullscreen.ts';
import { trackPointers } from '../../gpu/pointer.ts';
import { trackScroll } from '../../gpu/scroll.ts';
import {
  createHeightfield,
  type BoundaryMode,
  type Heightfield,
} from '../../gpu/sim/heightfield.ts';
import { createStirrer, type StirMode } from '../../gpu/sim/stir.ts';
import { createUniformBuffer, defineUniformStruct } from '../../gpu/uniforms.ts';
import { defineEffect } from '../types.ts';
import { meta } from './meta.ts';
import sceneShader from './scene.wgsl?raw';
import waterShader from './water.wgsl?raw';

const SceneParams = defineUniformStruct('SceneParams', {
  aspect: 'f32',
  aa: 'f32',
  density: 'f32',
  kind: 'i32',
});

const WaterParams = defineUniformStruct('WaterParams', {
  tint: 'vec3f',
  aspect: 'f32',
  strength: 'f32',
  ior: 'f32',
  thickness: 'f32',
  dispersion: 'f32',
  fade: 'f32',
  shine: 'f32',
  relief: 'f32',
  height_gain: 'f32',
  clock: 'f32',
  level: 'f32',
  flash: 'f32',
  pixel: 'f32',
  view: 'u32',
  dot_mode: 'u32',
});

const SOURCES = ['scroll', 'audio (built-in)', 'both'] as const;
const RESOLUTIONS = ['192', '256', '384', '512'] as const;
const BOUNDARIES = ['absorb', 'reflect'] as const;
const BACKDROPS = ['pool tiles', 'test card', 'sand'] as const;
const POINTER_MODES = ['hover and press', 'press only', 'hover only', 'off'] as const;
const VIEWS = ['water', 'height'] as const;

const BACKDROP_FORMAT: GPUTextureFormat = 'rgba8unorm-srgb';

/** Scroll speed (CSS px / s) that counts as "fast" (1.0). Reading is a fifth of this or less. */
const FAST_SCROLL = 1500;
/** Most impulses one frame may queue from each driver, leaving room for the pointer in the 64. */
const MAX_SCROLL_DROPS = 8;
const MAX_AUDIO_DROPS = 16;

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

/** A handle on the analyser for the console and for tests of the real page: `canvas.__wtAudio`. */
interface DebugCanvas {
  __wtAudio?: ReturnType<typeof createAudioAnalyser>;
}

const clamp = (x: number, lo: number, hi: number) => Math.min(hi, Math.max(lo, x));

function prefersReducedMotion(): boolean {
  try {
    return (
      typeof globalThis.matchMedia === 'function' &&
      globalThis.matchMedia('(prefers-reduced-motion: reduce)').matches
    );
  } catch {
    return false;
  }
}

export const effect = defineEffect({
  meta,
  params: {
    source: {
      type: 'select',
      default: 'both',
      options: SOURCES,
      label: 'Driven by (click the water once to start audio; drop a file to replace the beat)',
    },
    scrollSensitivity: {
      type: 'number',
      default: 1,
      min: 0,
      max: 3,
      step: 0.01,
      label: 'Scroll sensitivity',
    },
    audioSensitivity: {
      type: 'number',
      default: 1,
      min: 0,
      max: 3,
      step: 0.01,
      label: 'Audio sensitivity',
    },
    audible: {
      type: 'boolean',
      default: false,
      label: 'Audible (the analysis never needs sound out)',
    },
    indicator: { type: 'boolean', default: true, label: 'Audio status dot' },
    idle: { type: 'number', default: 0.4, min: 0, max: 1, step: 0.01, label: 'Idle swell' },
    rainRate: {
      type: 'number',
      default: 1.2,
      min: 0,
      max: 20,
      step: 0.1,
      label: 'Ambient rain (drops / s)',
    },
    damping: { type: 'number', default: 0.7, min: 0, max: 4, step: 0.01, label: 'Damping (1 / s)' },
    waveSpeed: {
      type: 'number',
      default: 0.45,
      min: 0.1,
      max: 1.5,
      step: 0.01,
      label: 'Wave speed (canvas heights / s)',
    },
    boundary: { type: 'select', default: 'absorb', options: BOUNDARIES, label: 'Edges' },
    resolution: {
      type: 'select',
      default: '256',
      options: RESOLUTIONS,
      label: 'Grid cells on the long side',
    },
    pointer: { type: 'select', default: 'hover and press', options: POINTER_MODES },
    strength: {
      type: 'number',
      default: 2,
      min: 0.2,
      max: 8,
      step: 0.01,
      label: 'Refraction strength (slope gain)',
    },
    depth: {
      type: 'number',
      default: 0.35,
      min: 0.05,
      max: 1.2,
      step: 0.01,
      label: 'Water depth (canvas heights)',
    },
    shine: { type: 'number', default: 0.45, min: 0, max: 1, step: 0.01 },
    relief: { type: 'number', default: 0.3, min: 0, max: 1, step: 0.01, label: 'Relief shading' },
    tint: { type: 'color', default: [0.8, 0.95, 1] },
    backdrop: { type: 'select', default: 'pool tiles', options: BACKDROPS },
    view: { type: 'select', default: 'water', options: VIEWS, label: 'View' },
  },
  presets: {
    'reading page': {
      source: 'scroll',
      scrollSensitivity: 1,
      idle: 0.4,
      rainRate: 1.2,
      damping: 0.8,
      waveSpeed: 0.4,
      strength: 1.8,
      depth: 0.35,
      shine: 0.4,
      relief: 0.3,
    },
    'lo-fi beat': {
      source: 'audio (built-in)',
      audioSensitivity: 1,
      idle: 0.3,
      rainRate: 0.4,
      damping: 0.8,
      waveSpeed: 0.5,
      strength: 2.2,
      depth: 0.35,
      shine: 0.5,
      relief: 0.3,
    },
    storm: {
      source: 'both',
      scrollSensitivity: 2,
      audioSensitivity: 2,
      idle: 0.5,
      rainRate: 3,
      damping: 0.5,
      waveSpeed: 0.65,
      strength: 2,
      depth: 0.45,
      shine: 0.55,
      relief: 0.35,
    },
    'glass lake': {
      source: 'both',
      scrollSensitivity: 0.6,
      audioSensitivity: 0.7,
      idle: 0.2,
      rainRate: 0.3,
      damping: 0.35,
      waveSpeed: 0.35,
      boundary: 'reflect',
      strength: 1.6,
      depth: 0.5,
      shine: 0.35,
      relief: 0.25,
    },
  },

  async create({ device, format, canvas, params, profiler }) {
    const sceneUniforms = createUniformBuffer(device, SceneParams, 'ripple-reactive scene params');
    const waterUniforms = createUniformBuffer(device, WaterParams, 'ripple-reactive water params');
    const pointers = trackPointers(canvas);
    const stirrer = createStirrer();
    const scroll = trackScroll(canvas);
    const audio = createAudioAnalyser();
    (canvas as unknown as DebugCanvas).__wtAudio = audio;
    const random = createRandom(0xa0d10);
    const reduced = prefersReducedMotion();

    const scenePipeline = createFullscreenPipeline(device, {
      fragment: {
        label: 'ripple-reactive/scene.wgsl',
        code: sceneShader,
        chunks: { 'scene_params.wgsl': SceneParams.wgsl },
      },
      format: BACKDROP_FORMAT,
    });
    const waterPipeline = createFullscreenPipeline(device, {
      fragment: {
        label: 'ripple-reactive/water.wgsl',
        code: waterShader,
        chunks: { 'water_params.wgsl': WaterParams.wgsl },
      },
      format,
    });
    const sceneGroup = device.createBindGroup({
      label: 'ripple-reactive scene',
      layout: scenePipeline.getBindGroupLayout(0),
      entries: [{ binding: 0, resource: { buffer: sceneUniforms.buffer } }],
    });
    const sampler = device.createSampler({
      label: 'ripple-reactive sampler',
      magFilter: 'linear',
      minFilter: 'linear',
    });

    let width = Math.max(1, canvas.width || 1);
    let height = Math.max(1, canvas.height || 1);
    let backdrop: GPUTexture | undefined;
    let backdropView: GPUTextureView | undefined;
    let waterGroup: GPUBindGroup | undefined;
    let waterGroupGeneration = -1;
    let drawnKey = '';

    const solver: Heightfield = createHeightfield({
      device,
      kind: 'render',
      resolution: Number(params.resolution),
      width,
      height,
      profiler,
      label: 'ripple-reactive',
    });

    function allocateBackdrop() {
      backdrop?.destroy();
      backdrop = device.createTexture({
        label: 'ripple-reactive backdrop',
        size: [width, height],
        format: BACKDROP_FORMAT,
        usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.RENDER_ATTACHMENT,
      });
      backdropView = backdrop.createView();
      waterGroup = undefined;
      drawnKey = '';
    }

    // --- Input from the page: drag and drop of a file, and the first gesture for audio ---------------
    const controller = new AbortController();
    const { signal } = controller;
    const usesAudio = () => params.source !== 'scroll';
    // `pointerdown` and `pointerup` both: the browser counts one or the other as the activation,
    // depending on the pointer type. Resuming is cheap and idempotent.
    const gesture = () => {
      if (usesAudio()) void audio.resume();
    };
    canvas.addEventListener('pointerdown', gesture, { signal });
    canvas.addEventListener('pointerup', gesture, { signal });
    const hasFiles = (event: Event) =>
      (event as DragEvent).dataTransfer?.types?.includes('Files') ?? false;
    canvas.addEventListener(
      'dragover',
      (event) => {
        if (hasFiles(event)) event.preventDefault();
      },
      { signal },
    );
    canvas.addEventListener(
      'drop',
      (event) => {
        event.preventDefault();
        const file = (event as DragEvent).dataTransfer?.files?.[0];
        if (!file) return;
        // The drop is a gesture, so the context may start here; the file stays in this page.
        audio.useFile(file).catch((error: unknown) => {
          console.warn('ripple-reactive: could not use the dropped file', error);
        });
      },
      { signal },
    );

    // --- Drivers -----------------------------------------------------------------------------------
    let clock = 0;
    let interaction = 0;
    let rainDebt = 0;
    let swellPhase = random() * 100;
    let swellPoint: [number, number] | undefined;
    let seeded = false;

    // Scroll.
    let scrollDebt = 0;
    let scrollRainDebt = 0;
    let recentPeak = 0;
    let sloshCooldown = 0;
    let lastDirection = 1;

    // Audio.
    let midDebt = 0;
    let highDebt = 0;
    let bassDebt = 0;
    let flash = 0;
    let lastAudible = false;
    let lastActive = true;
    let lastFrame: AudioFrame | undefined;

    const drop = (x: number, y: number, radius: number, strength: number) =>
      solver.addDrop(
        clamp(x, 0.01, 0.99),
        clamp(y, 0.01, 0.99),
        radius,
        clamp(strength, -2.5, 2.5),
      );

    /** Waves from the page moving: fed in at the leading edge, never louder than a gentle splash. */
    function driveScroll(delta: number, sensitivity: number) {
      const state = scroll.update(delta);
      if (sloshCooldown > 0) sloshCooldown -= delta;
      if (!state.visible || sensitivity <= 0) return;
      const speed = clamp(state.velocity / FAST_SCROLL, -1.5, 1.5);
      const magnitude = Math.abs(speed);
      if (magnitude > 0.02) lastDirection = Math.sign(speed);

      // Waves at the edge the content is arriving from: scrolling down brings the page up from the
      // bottom, so the bottom edge is leading. The rate follows the speed; each splash stays small.
      const level = Math.pow(magnitude, 0.8);
      scrollDebt += delta * level * 12 * sensitivity;
      let queued = 0;
      while (scrollDebt >= 1 && queued < MAX_SCROLL_DROPS) {
        scrollDebt -= 1;
        queued++;
        const y = lastDirection > 0 ? 0.97 : 0.03;
        drop(
          0.06 + random() * 0.88,
          y,
          0.035 + 0.02 * random(),
          -(0.6 + 0.9 * Math.min(level, 1)) * sensitivity,
        );
      }
      scrollDebt = Math.min(scrollDebt, 2);

      // A light rain over the whole surface while moving, so the disturbance is not only at the edge.
      scrollRainDebt += delta * level * 5 * sensitivity;
      while (scrollRainDebt >= 1 && queued < MAX_SCROLL_DROPS) {
        scrollRainDebt -= 1;
        queued++;
        drop(
          0.03 + random() * 0.94,
          0.05 + random() * 0.9,
          0.022 + 0.014 * random(),
          -(0.4 + 0.4 * level) * sensitivity,
        );
      }
      scrollRainDebt = Math.min(scrollRainDebt, 2);

      // A slosh on an abrupt stop: the water keeps going, heaps up against the edge the page was
      // moving toward and leaves a trough behind it, then the two rock back and forth and settle.
      recentPeak = Math.max(magnitude, recentPeak * Math.exp(-delta / 0.5));
      if (magnitude < 0.06 && recentPeak > 0.4 && sloshCooldown <= 0) {
        const heap = lastDirection > 0 ? 0.88 : 0.12;
        const size = Math.min(recentPeak, 1);
        drop(0.5, heap, 0.17, 1.1 * size * sensitivity);
        drop(0.5, 1 - heap, 0.17, -0.7 * size * sensitivity);
        sloshCooldown = 0.7;
        recentPeak = 0;
      }
    }

    /** Sound as water: bass swells, mids and highs rain, onsets drop. */
    function driveAudio(a: AudioFrame, delta: number, sensitivity: number) {
      flash = Math.max(a.onsetStrength, flash * Math.exp(-delta / 0.18));
      if (!a.running || sensitivity <= 0) return;
      let queued = 0;
      const place = (radius: number, strength: number, band: number) => {
        if (queued >= MAX_AUDIO_DROPS) return;
        queued++;
        // Lower sounds sit lower in the pool, brighter ones higher up, spread across the width.
        const y = clamp(0.88 - band * 0.78 + (random() - 0.5) * 0.25, 0.05, 0.95);
        drop(0.05 + random() * 0.9, y, radius, strength);
      };

      // A kick: a broad dent near the middle that rebounds into a ring, and a push from the bottom edge.
      if (a.lowOnset > 0) {
        drop(
          0.5 + (random() - 0.5) * 0.3,
          0.5 + (random() - 0.5) * 0.3,
          0.13 + 0.04 * a.lowOnset,
          -(0.7 + 0.9 * a.lowOnset) * sensitivity,
        );
        for (let i = 0; i < 3; i++) {
          drop(0.2 + 0.3 * i + 0.1 * random(), 0.96, 0.07, 0.5 * a.lowOnset * sensitivity);
        }
        queued += 4;
      }
      // Hats and claps: single drops.
      if (a.highOnset > 0) {
        place(0.022, -(0.6 + 0.8 * a.highOnset) * sensitivity, 0.85);
      }
      // Sustained bass: broad slow waves from the edges, as often as the bass is strong.
      bassDebt += delta * a.bass * a.bass * 5 * sensitivity;
      while (bassDebt >= 1 && queued < MAX_AUDIO_DROPS) {
        bassDebt -= 1;
        queued++;
        const side = random() < 0.5 ? 0.03 : 0.97;
        drop(side, 0.2 + random() * 0.6, 0.08, 0.5 * (0.5 + a.bass) * sensitivity);
      }
      bassDebt = Math.min(bassDebt, 2);
      // Mids: moderate rain. Highs: fine, quick rain.
      midDebt += delta * Math.pow(a.mids, 1.5) * 14 * sensitivity;
      while (midDebt >= 1 && queued < MAX_AUDIO_DROPS) {
        midDebt -= 1;
        place(0.028, -(0.4 + 0.6 * a.mids) * sensitivity, 0.45);
      }
      midDebt = Math.min(midDebt, 2);
      highDebt += delta * Math.pow(a.highs, 1.5) * 24 * sensitivity;
      while (highDebt >= 1 && queued < MAX_AUDIO_DROPS) {
        highDebt -= 1;
        place(0.016, -(0.3 + 0.4 * a.highs) * sensitivity, 0.8);
      }
      highDebt = Math.min(highDebt, 2);
    }

    /** What keeps the surface alive with nothing else going on: a little rain and a wandering swell. */
    function ambient(delta: number) {
      const quiet = 1 - 0.6 * interaction;
      const calm = reduced ? 0.4 : 1;
      rainDebt += delta * params.rainRate * quiet * calm;
      let drops = 0;
      while (rainDebt >= 1 && drops < 6) {
        rainDebt -= 1;
        drops++;
        drop(
          0.02 + random() * 0.96,
          0.02 + random() * 0.96,
          0.03 * (0.7 + 0.6 * random()),
          -2 * (0.7 + 0.7 * random()),
        );
      }
      rainDebt = Math.min(rainDebt, 1);

      swellPhase += delta;
      const next: [number, number] = [
        0.5 + 0.44 * Math.sin(swellPhase * 0.19 + 1.3) * Math.cos(swellPhase * 0.07),
        0.5 + 0.4 * Math.sin(swellPhase * 0.27),
      ];
      if (swellPoint && params.idle > 0)
        solver.displace(swellPoint, next, 0.16, params.idle * quiet * calm);
      swellPoint = next;

      // So a first frame, or a screenshot, already shows moving water.
      if (!seeded) {
        seeded = true;
        for (let i = 0; i < 5; i++) {
          drop(0.1 + random() * 0.8, 0.1 + random() * 0.8, 0.035, -1.6 * (0.6 + 0.6 * random()));
        }
      }
    }

    return {
      resize(w, h) {
        width = Math.max(1, Math.round(w));
        height = Math.max(1, Math.round(h));
        allocateBackdrop();
        solver.resize(width, height);
      },

      frame({ delta }, encoder, target) {
        if (!backdrop || !backdropView) return;
        const aspect = width / height;
        clock += delta;

        solver.setResolution(Number(params.resolution));
        const wave = solver.params;
        wave.waveSpeed = params.waveSpeed;
        wave.damping = params.damping;
        wave.boundary = params.boundary as BoundaryMode;

        const key = `${params.backdrop}:${width}x${height}`;
        if (key !== drawnKey) {
          drawnKey = key;
          sceneUniforms.set({
            aspect,
            aa: 1 / height,
            density: 1,
            kind: Math.max(0, BACKDROPS.indexOf(params.backdrop as (typeof BACKDROPS)[number])),
          });
          sceneUniforms.upload();
          renderFullscreen(encoder, backdropView, scenePipeline, [sceneGroup], {
            timestampWrites: profiler.pass('backdrop (on change)'),
          });
        }

        // Pointer: a gentle hover and press, as in `ripples`, so the surface still answers a visitor.
        const stir = stirrer.stir(solver, pointers.poll(), aspect, {
          mode: params.pointer as StirMode,
          radius: 0.03,
          strength: 0.8,
        });
        interaction += ((stir.active ? 1 : 0) - interaction) * (1 - Math.exp(-delta * 4));

        // Drivers. Scroll is always tracked so it has no jump when it is switched on.
        const useScroll = params.source !== 'audio (built-in)';
        const useAudio = usesAudio();
        const calm = reduced ? 0.4 : 1;
        driveScroll(delta, useScroll ? params.scrollSensitivity * calm : 0);

        if (useAudio !== lastActive) {
          lastActive = useAudio;
          audio.setActive(useAudio);
        }
        if (params.audible !== lastAudible) {
          lastAudible = params.audible;
          audio.setAudible(params.audible);
        }
        if (useAudio) {
          lastFrame = audio.update(delta);
          driveAudio(lastFrame, delta, params.audioSensitivity * calm);
        } else {
          lastFrame = undefined;
          flash = 0;
        }

        ambient(delta);
        solver.update(encoder, delta);

        if (!waterGroup || waterGroupGeneration !== solver.generation) {
          waterGroupGeneration = solver.generation;
          waterGroup = device.createBindGroup({
            label: 'ripple-reactive water',
            layout: waterPipeline.getBindGroupLayout(0),
            entries: [
              { binding: 0, resource: { buffer: waterUniforms.buffer } },
              { binding: 1, resource: backdropView },
              { binding: 2, resource: sampler },
              { binding: 3, resource: solver.view },
              { binding: 4, resource: sampler },
            ],
          });
        }
        const dotMode =
          !params.indicator || !useAudio || audio.state === 'unsupported'
            ? 0
            : audio.state === 'running'
              ? 2
              : 1;
        waterUniforms.set({
          tint: params.tint,
          aspect,
          strength: params.strength,
          ior: 1.333,
          thickness: params.depth,
          dispersion: 0.012,
          fade: 0.05,
          shine: params.shine,
          relief: params.relief,
          height_gain: 1.5,
          clock,
          level: lastFrame?.level ?? 0,
          flash,
          pixel: 1 / height,
          view: Math.max(0, VIEWS.indexOf(params.view as (typeof VIEWS)[number])),
          dot_mode: dotMode,
        });
        waterUniforms.upload();
        renderFullscreen(encoder, target, waterPipeline, [waterGroup], {
          timestampWrites: profiler.pass('shade'),
        });
      },

      destroy() {
        controller.abort();
        audio.destroy();
        scroll.destroy();
        pointers.destroy();
        solver.destroy();
        backdrop?.destroy();
        sceneUniforms.destroy();
        waterUniforms.destroy();
        delete (canvas as unknown as DebugCanvas).__wtAudio;
      },
    };
  },
});

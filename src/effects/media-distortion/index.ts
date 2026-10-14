import { createWaterNormalMap } from '../../gpu/bake.ts';
import { createFullscreenPipeline, renderFullscreen } from '../../gpu/fullscreen.ts';
import {
  classifyFile,
  createMediaFromFile,
  createMediaTexture,
  fitScale,
  mediaFromTexture,
  type MediaFit,
  type MediaTexture,
} from '../../gpu/media.ts';
import { generateMipmaps, mipLevelCount } from '../../gpu/mipmap.ts';
import { trackPointer } from '../../gpu/pointer.ts';
import { createUniformBuffer, defineUniformStruct, uniformArray } from '../../gpu/uniforms.ts';
import { defineEffect } from '../types.ts';
import { meta } from './meta.ts';
import shader from './media-distortion.wgsl?raw';
import sceneShader from '../_shared/media-scene.wgsl?raw';
import { createLiveVideo, createStillCanvas, type LiveVideo } from './sources.ts';

const Params = defineUniformStruct('DistortionParams', {
  resolution: 'vec2f',
  media_px: 'vec2f',
  fit: 'vec2f',
  cover_fit: 'vec2f',
  pointer: 'vec2f',
  aspect: 'f32',
  phase: 'f32',
  clock: 'f32',
  strength: 'f32',
  scale: 'f32',
  dispersion: 'f32',
  highlight: 'f32',
  lens: 'f32',
  ring: 'f32',
  ring_phase: 'f32',
  amp: 'f32',
  focus: 'f32',
  drag: 'f32',
  bar_lod: 'f32',
  contain: 'u32',
  compare: 'u32',
  tint: 'vec3f',
  drops: uniformArray('vec4f', 4),
});

const SOURCES = ['still', 'video'] as const;
const FITS = ['cover', 'contain'] as const;
const MODES = ['ambient', 'ripple', 'hover'] as const;
const FALLBACK_SIZE = { width: 640, height: 360 };

export const effect = defineEffect({
  meta,
  params: {
    source: {
      type: 'select',
      default: 'still',
      options: SOURCES,
      label: 'Source (or drop a file on the canvas)',
    },
    fit: { type: 'select', default: 'cover', options: FITS },
    interaction: {
      type: 'select',
      default: 'ripple',
      options: MODES,
      label: 'Interaction',
    },
    strength: { type: 'number', default: 0.45, min: 0, max: 2, step: 0.01 },
    scale: { type: 'number', default: 1, min: 0.25, max: 4, step: 0.01, label: 'Wave scale' },
    speed: { type: 'number', default: 0.6, min: 0, max: 3, step: 0.01 },
    dispersion: {
      type: 'number',
      default: 0.012,
      min: 0,
      max: 0.1,
      step: 0.001,
      label: 'Dispersion (colour fringe)',
    },
    highlight: { type: 'number', default: 0.3, min: 0, max: 1, step: 0.01 },
    tint: { type: 'color', default: [1, 1, 1], label: 'Water tint (white = none)' },
    compare: { type: 'boolean', default: false, label: 'Compare (left half: original)' },
  },
  presets: {
    'gentle hero': {
      source: 'still',
      fit: 'cover',
      interaction: 'ripple',
      strength: 0.35,
      scale: 0.8,
      speed: 0.45,
      dispersion: 0.008,
      highlight: 0.2,
      tint: [1, 1, 1],
    },
    'hover ripple': {
      source: 'still',
      fit: 'cover',
      interaction: 'hover',
      strength: 0.7,
      scale: 1.4,
      speed: 0.9,
      dispersion: 0.02,
      highlight: 0.35,
      tint: [1, 1, 1],
    },
    'deep pool': {
      source: 'still',
      fit: 'cover',
      interaction: 'ripple',
      strength: 1.5,
      scale: 0.7,
      speed: 0.3,
      dispersion: 0.04,
      highlight: 0.5,
      tint: [0.62, 0.92, 1],
    },
    'product shot': {
      source: 'still',
      fit: 'contain',
      interaction: 'hover',
      strength: 0.6,
      scale: 1.3,
      speed: 0.7,
      dispersion: 0.015,
      highlight: 0.25,
      tint: [1, 1, 1],
    },
    'live video': {
      source: 'video',
      fit: 'cover',
      interaction: 'ripple',
      strength: 0.6,
      scale: 1.2,
      speed: 0.6,
      dispersion: 0.015,
      highlight: 0.3,
      tint: [1, 1, 1],
    },
  },

  async create({ device, format, canvas, params, profiler }) {
    const uniforms = createUniformBuffer(device, Params, 'media-distortion params');
    const pointer = trackPointer(canvas);
    // Baked once: 512 x 512 rg16float with mips. Three smooth octaves, not the default five: a picture
    // under water wants broad, lazy wobbles, and fine crests read as cracked glass.
    const normals = createWaterNormalMap(device, {
      size: 512,
      seed: 5,
      period: 3,
      octaves: 3,
      choppiness: 0.1,
    });
    const pipeline = createFullscreenPipeline(device, {
      fragment: {
        label: 'media-distortion/media-distortion.wgsl',
        code: shader,
        chunks: { 'distortion_params.wgsl': Params.wgsl },
      },
      format,
    });
    const mediaSampler = device.createSampler({
      label: 'media sampler',
      magFilter: 'linear',
      minFilter: 'linear',
      mipmapFilter: 'linear',
      addressModeU: 'clamp-to-edge',
      addressModeV: 'clamp-to-edge',
    });
    const normalSampler = device.createSampler({
      label: 'media normal sampler',
      magFilter: 'linear',
      minFilter: 'linear',
      mipmapFilter: 'linear',
      addressModeU: 'repeat',
      addressModeV: 'repeat',
    });

    // --- Media -------------------------------------------------------------------------------------
    let destroyed = false;
    let media: MediaTexture | undefined;
    let mediaKey = '';
    let bindGroup: GPUBindGroup | undefined;
    let boundView: GPUTextureView | undefined;
    let wanted = 'still';
    let seenSource: string = params.source;
    let dropped = false;
    let loading = false;
    let live: LiveVideo | undefined;
    let dragging = 0;
    let dragTarget = 0;

    /** The same picture the browser path paints, drawn by a shader for when there is no DOM. */
    function createProceduralMedia(): MediaTexture {
      const { width, height } = FALLBACK_SIZE;
      const texture = device.createTexture({
        label: 'media-distortion procedural picture',
        size: [width, height],
        format: 'rgba8unorm-srgb',
        mipLevelCount: mipLevelCount(width, height),
        usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.RENDER_ATTACHMENT,
      });
      const sceneUniforms = createUniformBuffer(
        device,
        defineUniformStruct('SceneParams', { aspect: 'f32', aa: 'f32' }),
        'media-distortion scene params',
      );
      sceneUniforms.set({ aspect: width / height, aa: 1 / height });
      sceneUniforms.upload();
      const scenePipeline = createFullscreenPipeline(device, {
        fragment: { label: 'media-distortion/scene.wgsl', code: sceneShader },
        format: 'rgba8unorm-srgb',
      });
      const group = device.createBindGroup({
        layout: scenePipeline.getBindGroupLayout(0),
        entries: [{ binding: 0, resource: { buffer: sceneUniforms.buffer } }],
      });
      const encoder = device.createCommandEncoder({ label: 'media-distortion procedural picture' });
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

    async function makeMedia(which: string): Promise<MediaTexture> {
      if (which === 'video') {
        const next = await createLiveVideo().catch(() => undefined);
        if (next) {
          try {
            const result = await createMediaTexture(device, next.video, {
              label: 'media-distortion live video',
              onDestroy: () => next.destroy(),
            });
            live = next;
            return result;
          } catch (error) {
            next.destroy();
            console.warn('media-distortion: the live video could not be used', error);
          }
        }
      }
      const painted = createStillCanvas();
      if (painted) {
        return createMediaTexture(device, painted, { label: 'media-distortion harbour' });
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
      boundView = undefined; // the bind group points at the old view
      previous?.destroy();
    }

    async function load(which: string) {
      if (loading) return;
      loading = true;
      try {
        const next = await makeMedia(which);
        swapIn(next, which === 'video' && next.kind === 'video' ? 'video' : 'still');
      } finally {
        loading = false;
      }
    }

    async function loadFile(file: File) {
      if (!classifyFile(file)) {
        console.warn(`media-distortion: ${file.name || 'that file'} is not an image or a video`);
        return;
      }
      try {
        const next = await createMediaFromFile(device, file);
        dropped = true;
        swapIn(next, `file:${file.name}`);
      } catch (error) {
        console.warn('media-distortion: could not use the dropped file', error);
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
        dragTarget = 1;
      },
      { signal },
    );
    canvas.addEventListener('dragleave', () => (dragTarget = 0), { signal });
    canvas.addEventListener(
      'drop',
      (event) => {
        event.preventDefault();
        dragTarget = 0;
        const file = (event as DragEvent).dataTransfer?.files[0];
        if (file) void loadFile(file);
      },
      { signal },
    );

    swapIn(await makeMedia('still'), 'still');

    // --- Per-frame state ---------------------------------------------------------------------------
    let width = 1;
    let height = 1;
    let phase = 0;
    let ringPhase = 0;
    let follow = 0;
    let energy = 0;
    // Hover starts still; every other mode starts moving.
    let amp = params.interaction === 'hover' ? 0 : 1;
    let smoothX = 0.5;
    let smoothY = 0.5;
    let wasDown = false;
    const drops = new Float32Array(16);
    let nextDrop = 0;

    return {
      resize(w, h) {
        width = Math.max(1, Math.round(w));
        height = Math.max(1, Math.round(h));
      },

      frame({ delta, time }, encoder, target) {
        if (!media) return;
        const aspect = width / height;

        // A different `source` choice discards a dropped file; otherwise load what is asked for.
        if (params.source !== seenSource) {
          seenSource = params.source;
          dropped = false;
        }
        wanted = params.source === 'video' ? 'video' : 'still';
        if (!dropped && wanted !== mediaKey && !loading) void load(wanted);
        live?.tick(time);

        media.update(encoder);
        if (boundView !== media.view) {
          boundView = media.view;
          bindGroup = device.createBindGroup({
            label: 'media-distortion',
            layout: pipeline.getBindGroupLayout(0),
            entries: [
              { binding: 0, resource: { buffer: uniforms.buffer } },
              { binding: 1, resource: media.view },
              { binding: 2, resource: mediaSampler },
              { binding: 3, resource: normals.view },
              { binding: 4, resource: normalSampler },
            ],
          });
        }

        // Pointer. Ambient: it is read, never captured, so touch still scrolls the page.
        const mode = params.interaction;
        const state = pointer.state;
        const moved = pointer.takeDelta();
        const speed = Math.hypot(moved.dx * aspect, moved.dy) / Math.max(delta, 1e-3);
        const blend = Math.min(1, delta * 5);
        follow += ((state.inside ? 1 : 0) - follow) * blend;
        smoothX += (state.x - smoothX) * Math.min(1, delta * 12);
        smoothY += (state.y - smoothY) * Math.min(1, delta * 12);
        const interactive = mode !== 'ambient';
        energy += ((state.inside && interactive ? Math.min(1, speed * 1.2) : 0) - energy) * blend;
        ringPhase += delta * 8;

        // Hover: still until the pointer moves over it, then agitated near the pointer, settling
        // slowly (over a second or two) once it stops or leaves.
        if (mode === 'hover') {
          const wanted_amp = state.inside ? Math.min(1, speed * 1.6) : 0;
          const rate = wanted_amp > amp ? 7 : 1.1;
          amp += (wanted_amp - amp) * Math.min(1, delta * rate);
        } else {
          amp += (1 - amp) * Math.min(1, delta * 3);
        }

        // A press drops a ring (ripple and hover modes).
        if (state.down && !wasDown && interactive && state.inside) {
          drops.set([state.x * aspect, state.y, time, 1], nextDrop * 4);
          nextDrop = (nextDrop + 1) % 4;
        }
        wasDown = state.down;

        dragging += (dragTarget - dragging) * Math.min(1, delta * 12);
        phase += delta * params.speed;

        const fit = fitScale(aspect, media.aspect, params.fit as MediaFit);
        const coverFit = fitScale(aspect, media.aspect, 'cover');
        uniforms.set({
          resolution: [width, height],
          media_px: [media.width, media.height],
          fit,
          cover_fit: coverFit,
          pointer: [smoothX * aspect, smoothY],
          aspect,
          phase,
          clock: time,
          strength: params.strength * 0.5,
          scale: params.scale,
          dispersion: params.dispersion,
          highlight: params.highlight,
          lens: interactive ? (mode === 'ripple' ? 0.9 : 0.5) * follow : 0,
          ring: energy * 0.5,
          ring_phase: ringPhase,
          amp,
          focus: mode === 'hover' ? 1 : 0,
          drag: dragging * 0.6,
          bar_lod: Math.min(5, Math.max(0, media.mipLevelCount - 1)),
          contain: params.fit === 'contain' ? 1 : 0,
          compare: params.compare ? 1 : 0,
          tint: params.tint,
          drops,
        });
        uniforms.upload();

        renderFullscreen(encoder, target, pipeline, [bindGroup!], {
          timestampWrites: profiler.pass('water over media'),
        });
      },

      destroy() {
        destroyed = true;
        controller.abort();
        pointer.destroy();
        media?.destroy();
        media = undefined;
        normals.destroy();
        uniforms.destroy();
      },
    };
  },
});

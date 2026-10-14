import { defineEffect, type EffectMeta, type ParamSchema } from '../../effects/types.ts';
import { createFullscreenPipeline, renderFullscreen } from '../fullscreen.ts';
import { trackPointer, trackPointers } from '../pointer.ts';
import { trackScroll } from '../scroll.ts';
import { createPingPongTarget } from '../targets.ts';
import { createUniformBuffer, defineUniformStruct, uniformArray } from '../uniforms.ts';
import { uploadBackdrop } from './capture.ts';
import { createTextureFromData } from '../texture.ts';
import { createTextAtlas } from './atlas.ts';
import {
  advanceDrops,
  advanceSpring,
  MAX_DROPS,
  seedDrops,
  transitionProgress,
  type Spring,
} from './dynamics.ts';
import shader from './display.wgsl';
import trailShader from './trail.wgsl';
import backdropShader from './backdrop.wgsl';
import textShader from './text.wgsl';

const Params = defineUniformStruct('UIParams', {
  image_fit: 'vec2f',
  has_image: 'f32',
  resolution: 'vec2f',
  pointer: 'vec2f',
  previous: 'vec2f',
  press: 'vec2f',
  tint: 'vec3f',
  clock: 'f32',
  strength: 'f32',
  dispersion: 'f32',
  blur: 'f32',
  radius: 'f32',
  extra: 'f32',
  level: 'f32',
  spring: 'f32',
  ripple: 'f32',
  interaction: 'f32',
  pressed: 'f32',
  delta: 'f32',
  dry: 'f32',
  style: 'u32',
  segment_count: 'u32',
  segments: uniformArray('vec4f', 16),
  drops: uniformArray('vec4f', MAX_DROPS),
});
export type UIMode = 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10;
const number = (value: number, min: number, max: number, label: string) => ({
  type: 'number' as const,
  default: value,
  min,
  max,
  step: 0.01,
  label,
});
const choice = (options: string[], label: string) => ({
  type: 'select' as const,
  default: options[0]!,
  options,
  label,
});
export function uiSchema(mode: UIMode): ParamSchema {
  const params: ParamSchema = {
    strength: number(1, 0, 2, 'Water strength'),
    speed: number(0.7, 0, 2, 'Animation speed'),
  };
  if (mode === 0 || mode === 8) delete params.speed;
  if ([0, 4, 8, 10].includes(mode)) params.dispersion = number(0.009, 0, 0.06, 'Colour dispersion');
  if ([0, 1, 2, 6].includes(mode))
    params.tint = { type: 'color', default: [0.55, 0.9, 0.82], label: 'Liquid tint' };
  if (mode === 0) {
    params.blur = number(6, 0, 24, 'Backdrop blur (px)');
    params.radius = number(0.07, 0.01, 0.2, 'Corner radius');
  }
  if (mode === 1) {
    params.merge = number(0.12, 0.01, 0.3, 'Merge distance');
    params.radius = number(0.09, 0.03, 0.18, 'Follower radius');
  }
  if (mode === 2) {
    params.frequency = number(3.5, 1, 8, 'Spring frequency (Hz)');
    params.damping = number(0.25, 0.05, 1.5, 'Spring damping');
  }
  if (mode === 3) {
    params.style = choice(['drip', 'wave', 'dissolve'], 'Transition');
    params.automatic = { type: 'boolean', default: true, label: 'Cycle between sections' };
    params.progress = number(0.4, 0, 1, 'Transition progress');
  }
  if (mode === 5) {
    params.style = choice(['ripple', 'melt', 'fill'], 'Typography');
    params.level = number(0.5, 0, 1, 'Water fill');
  }
  if (mode === 6) {
    params.style = choice(['tank', 'bar', 'loader'], 'Component');
    params.level = number(0.64, 0, 1, 'Fill level');
  }
  if (mode === 7) params.merge = number(0.35, 0.08, 0.6, 'Blob merge distance');
  if (mode === 8) {
    params.radius = number(0.027, 0.008, 0.08, 'Stroke radius');
    params.spread = number(1, 0, 4, 'Wet spread');
    params.dry = number(0.55, 0.05, 3, 'Drying rate');
  }
  if (mode === 9) {
    params.scrollRange = number(640, 100, 2000, 'Scroll to submerge (px)');
    params.preview = { type: 'boolean', default: false, label: 'Preview water level' };
    params.level = number(0.45, 0, 1, 'Preview level');
  }
  if (mode === 10) {
    params.gravity = number(0.45, 0, 2, 'Gravity');
    params.condensation = number(3, 0, 12, 'New drops / second');
  }
  return params;
}
function uniformLayout(device: GPUDevice) {
  return device.createBindGroupLayout({
    entries: [
      { binding: 0, visibility: GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } },
      { binding: 1, visibility: GPUShaderStage.FRAGMENT, sampler: { type: 'filtering' } },
      ...[2, 3, 4, 5, 6].map((binding) => ({
        binding,
        visibility: GPUShaderStage.FRAGMENT,
        texture: { sampleType: 'float' as const },
      })),
    ],
  });
}
/** Eleven component studies share backdrop, input and resource ownership; MODE is compile-time. */
export function uiEffect(meta: EffectMeta, mode: UIMode) {
  const schema = uiSchema(mode);
  const presets: Record<string, Record<string, number | string | boolean>> = {
    quiet: { strength: 0.5, speed: 0.4 },
    expressive: { strength: 1.55, speed: 1.1 },
  };
  if (mode === 0 || mode === 8) {
    delete presets.quiet!.speed;
    delete presets.expressive!.speed;
  }
  if (mode === 0) {
    presets.frosted = { blur: 18, dispersion: 0.004, strength: 0.5 };
    presets.clear = { blur: 0, dispersion: 0.02, strength: 1.3 };
  }
  if (mode === 2) {
    presets.bouncy = { frequency: 3, damping: 0.12, strength: 1.5 };
    presets.settled = { frequency: 5, damping: 1, strength: 0.5 };
  }
  if (mode === 3) {
    presets['wave reveal'] = { style: 'wave' };
    presets['dissolve'] = { style: 'dissolve' };
  }
  if (mode === 5) {
    presets['melting type'] = { style: 'melt', strength: 1.4 };
    presets['water fill'] = { style: 'fill', level: 0.6 };
  }
  if (mode === 6) {
    presets['progress bar'] = { style: 'bar', level: 0.7 };
    presets['round loader'] = { style: 'loader', level: 0.5 };
  }
  if (mode === 9) {
    presets['half submerged'] = { preview: true, level: 0.5 };
    presets['scroll page'] = { preview: false, scrollRange: 640 };
  }
  return defineEffect({
    meta,
    params: schema,
    presets,
    async create({ device, format, canvas, params, profiler }) {
      const n = (key: string, fallback: number) =>
        typeof params[key] === 'number' ? (params[key] as number) : fallback;
      const atlas = await createTextAtlas(device);
      const sampler = device.createSampler({
        magFilter: 'linear',
        minFilter: 'linear',
        addressModeU: 'clamp-to-edge',
        addressModeV: 'clamp-to-edge',
      });
      const uniforms = createUniformBuffer(device, Params, 'Liquid UI params');
      const layout = uniformLayout(device);
      const pipeline = createFullscreenPipeline(device, {
        format,
        layout: [layout],
        fragment: {
          label: 'ui/display.wgsl',
          code: `diagnostic(off, derivative_uniformity);\nconst MODE: u32 = ${mode}u;\n${shader.code}`,
          chunks: {
            'params.wgsl': Params.wgsl,
            'text.wgsl': textShader.code,
            ...textShader.chunks,
          },
        },
      });
      let image = createTextureFromData(device, {
        width: 1,
        height: 1,
        data: new Uint8Array([0, 0, 0, 255]),
      });
      let hasImage = false,
        imageAspect = 1,
        uploadVersion = 0;
      const pages: GPUTexture[] = [];
      const Backdrop = defineUniformStruct('BackdropParams', { page: 'f32' });
      const bake = createFullscreenPipeline(device, {
        format: 'rgba16float',
        fragment: {
          label: 'ui/backdrop.wgsl',
          code: backdropShader.code,
          chunks: {
            ...backdropShader.chunks,
            'params.wgsl': Backdrop.wgsl,
            'text.wgsl': textShader.code,
            ...textShader.chunks,
          },
        },
      });
      const bakeEncoder = device.createCommandEncoder({ label: 'Bake two UI sections' });
      const bakeUniforms = [];
      for (let i = 0; i < (mode === 3 ? 2 : 1); i++) {
        const texture = device.createTexture({
          label: `UI section ${i}`,
          size: [1024, 640],
          format: 'rgba16float',
          usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
        });
        pages.push(texture);
        const uniform = createUniformBuffer(device, Backdrop, `UI section ${i} params`);
        uniform.set({ page: i });
        uniform.upload();
        bakeUniforms.push(uniform);
        const group = device.createBindGroup({
          layout: bake.getBindGroupLayout(0),
          entries: [
            { binding: 0, resource: { buffer: uniform.buffer } },
            { binding: 1, resource: sampler },
            { binding: 2, resource: atlas.createView() },
          ],
        });
        renderFullscreen(bakeEncoder, texture.createView(), bake, [group]);
      }
      device.queue.submit([bakeEncoder.finish()]);
      for (const uniform of bakeUniforms) uniform.destroy();
      // A tiny inert texture for one-pass effects; a half-resolution wet field only for trails.
      const trail = createPingPongTarget(device, {
        label: 'Persistent wetness',
        width: 1,
        height: 1,
        format: 'r16float',
      });
      const trailPipeline =
        mode === 8
          ? createFullscreenPipeline(device, {
              format: 'r16float',
              fragment: {
                label: 'ui/trail.wgsl',
                code: trailShader.code,
                chunks: { ...trailShader.chunks, 'params.wgsl': Params.wgsl },
              },
            })
          : undefined;
      let boundImage: GPUTexture | undefined;
      const displayGroups = new Map<GPUTextureView, GPUBindGroup>();
      const pageViews = pages.map((page) => page.createView());
      const controller = new AbortController();
      const pointer = trackPointer(canvas);
      let capturedPointer: number | undefined;
      if (mode === 2) {
        canvas.addEventListener(
          'pointerdown',
          (event) => {
            const e = event as PointerEvent;
            const x = ((pointer.state.x - 0.5) * width) / height,
              y = pointer.state.y - 0.5;
            if (Math.abs(x) < 0.23 && Math.abs(y) < 0.105) {
              canvas.setPointerCapture(e.pointerId);
              capturedPointer = e.pointerId;
            }
          },
          { signal: controller.signal },
        );
        for (const type of ['pointerup', 'pointercancel'])
          canvas.addEventListener(
            type,
            () => {
              capturedPointer = undefined;
            },
            { signal: controller.signal },
          );
      }
      const paths = mode === 8 ? trackPointers(canvas, { maxPointers: 1 }) : undefined;
      const scroll = mode === 9 ? trackScroll(canvas) : undefined;
      if (mode === 4) {
        canvas.addEventListener('dragover', (event) => event.preventDefault(), {
          signal: controller.signal,
        });
        canvas.addEventListener(
          'drop',
          (event) => {
            event.preventDefault();
            const file = (event as DragEvent).dataTransfer?.files[0];
            if (!file?.type.startsWith('image/')) return;
            const version = ++uploadVersion;
            void (async () => {
              let bitmap: ImageBitmap | undefined;
              try {
                bitmap = await createImageBitmap(file);
                if (destroyed || version !== uploadVersion) return;
                const texture = uploadBackdrop(device, bitmap);
                image.destroy();
                image = texture;
                imageAspect = bitmap.width / bitmap.height;
                hasImage = true;
                canvas.setAttribute?.('aria-label', 'Liquid image hover. Your image is loaded.');
              } catch {
                if (!destroyed)
                  canvas.setAttribute?.(
                    'aria-label',
                    'Image could not be decoded or uploaded. The coastal illustration is still available.',
                  );
              } finally {
                bitmap?.close();
              }
            })();
          },
          { signal: controller.signal },
        );
      }
      const spring: Spring = { value: 0, velocity: 0 };
      let clock = 0,
        width = 1,
        height = 1,
        previous = { x: 0.5, y: 0.5 },
        ripple = 20,
        press: [number, number] = [0.5, 0.5],
        pressed = false,
        wasDown = false;
      let keyboardHover = false,
        keyboardPress = false;
      let clickedFrom = 0.4,
        clickedAge = 1;
      let transitionTarget = n('progress', 0.4),
        clickedProgress: number | undefined,
        lastProgress = transitionTarget;
      const drops = seedDrops();
      let spawn = 0,
        spawnIndex = 0,
        destroyed = false;
      // Canvas is the accessible control surface; restore attributes when another effect mounts.
      const original = new Map<string, string | null>();
      if (typeof canvas.setAttribute === 'function') {
        for (const key of [
          'role',
          'aria-label',
          'tabindex',
          'aria-valuemin',
          'aria-valuemax',
          'aria-valuenow',
        ])
          original.set(key, canvas.getAttribute(key));
        canvas.setAttribute('role', mode === 2 ? 'button' : mode === 6 ? 'progressbar' : 'img');
        canvas.setAttribute(
          'aria-label',
          mode === 2
            ? 'Flow — liquid button. Press Enter or Space.'
            : `${meta.title}. ${mode === 8 ? 'Move the pointer to leave a drying water trail.' : (meta.summary ?? '')}`,
        );
        if (mode === 2) canvas.setAttribute('tabindex', '0');
        if (mode === 6) {
          canvas.setAttribute('aria-valuemin', '0');
          canvas.setAttribute('aria-valuemax', '100');
        }
      }
      if (mode === 2) {
        const key = (event: KeyboardEvent, down: boolean) => {
          if (event.key === 'Enter' || event.key === ' ') {
            event.preventDefault();
            keyboardPress = down;
          }
        };
        canvas.addEventListener('keydown', (event) => key(event as KeyboardEvent, true), {
          signal: controller.signal,
        });
        canvas.addEventListener('keyup', (event) => key(event as KeyboardEvent, false), {
          signal: controller.signal,
        });
        canvas.addEventListener(
          'focus',
          () => {
            keyboardHover = true;
          },
          { signal: controller.signal },
        );
        canvas.addEventListener(
          'blur',
          () => {
            keyboardHover = false;
            keyboardPress = false;
          },
          { signal: controller.signal },
        );
      }
      return {
        resize(w, h) {
          width = Math.max(1, w);
          height = Math.max(1, h);
          if (mode === 8 && trail.resize(Math.ceil(width / 2), Math.ceil(height / 2)))
            displayGroups.clear();
          previous = { x: pointer.state.x, y: pointer.state.y };
        },
        frame(time, encoder, target) {
          if (destroyed) return;
          const dt = Math.max(0, Math.min(time.delta, 0.1));
          const speed = n('speed', 0.7);
          clock += dt * speed;
          ripple += dt;
          const mouse = pointer.state,
            aspect = width / height;
          const hover =
            keyboardHover ||
            (mouse.inside &&
              Math.abs((mouse.x - 0.5) * aspect) < 0.23 &&
              Math.abs(mouse.y - 0.5) < 0.105);
          const down = keyboardPress || (mouse.down && hover);
          if (down && !wasDown) {
            ripple = 0;
            press = keyboardPress ? [0.5, 0.5] : [mouse.x, mouse.y];
            pressed = true;
            spring.velocity += 3;
          }
          advanceSpring(spring, hover ? 1 : 0, dt, n('frequency', 3.5), n('damping', 0.25));
          if (mode === 3) {
            if (n('progress', 0.4) !== lastProgress) {
              lastProgress = n('progress', 0.4);
              clickedProgress = undefined;
            }
            if (mouse.down && !wasDown && params.automatic === false) {
              clickedFrom = transitionTarget;
              clickedAge = 0;
              clickedProgress = (clickedProgress ?? lastProgress) < 0.5 ? 1 : 0;
            }
            clickedAge = Math.min(1, clickedAge + dt * speed);
            const ease = clickedAge * clickedAge * (3 - 2 * clickedAge);
            transitionTarget =
              clickedProgress === undefined
                ? lastProgress
                : clickedFrom + (clickedProgress - clickedFrom) * ease;
          }
          wasDown = mode === 3 ? mouse.down : down;
          let level = n('level', 0.5);
          if (mode === 3)
            level = params.automatic === true ? transitionProgress(clock) : transitionTarget;
          if (mode === 9) {
            const state = scroll!.update(dt);
            level =
              params.preview === true
                ? level
                : Math.min(1, Math.max(0, state.offset / n('scrollRange', 640)));
          }
          if (mode === 6) canvas.setAttribute?.('aria-valuenow', String(Math.round(level * 100)));
          if (mode === 10) {
            advanceDrops(drops, dt * speed, n('gravity', 0.45), aspect);
            spawn += dt * speed * n('condensation', 3);
            while (spawn >= 1) {
              spawn--;
              if (drops.length < MAX_DROPS) {
                drops.push({
                  x: 0.16 + ((spawnIndex++ * 0.618034) % 1) * 0.68,
                  y: 0.16,
                  radius: 0.006,
                  velocity: 0,
                });
              }
            }
          }
          const dropData = Array.from({ length: MAX_DROPS }, (_, i) => {
            const d = drops[i];
            return d ? [d.x, d.y, d.radius, d.velocity] : [0, 0, 0, 0];
          });
          const styles = ['drip', 'wave', 'dissolve'];
          if (mode === 5) styles.splice(0, 3, 'ripple', 'melt', 'fill');
          if (mode === 6) styles.splice(0, 3, 'bar', 'loader', 'tank');
          let active = mouse.inside ? 1 : 0;
          const segments = new Float32Array(64);
          let segmentCount = 0;
          if (mode === 8) {
            const point = paths!.poll(time.time * 1000)[0];
            if (point?.path.length) {
              const first = point.path[0]!;
              previous = { x: first.x, y: first.y };
            }
            if (point?.justEntered) previous = { x: mouse.x, y: mouse.y };
            active = point?.inside ? 1 : 0;
            if (point?.path.length) {
              const path = point.path;
              segmentCount = Math.min(16, Math.max(1, path.length - 1));
              for (let i = 0; i < segmentCount; i++) {
                const a = path[Math.floor((i * (path.length - 1)) / segmentCount)]!;
                const b = path[Math.floor(((i + 1) * (path.length - 1)) / segmentCount)]!;
                segments.set([a.x, a.y, b.x, b.y], i * 4);
              }
            }
          }
          const regionAspect = (aspect * 0.46) / 0.76;
          uniforms.set({
            image_fit:
              imageAspect > regionAspect
                ? [regionAspect / imageAspect, 1]
                : [1, imageAspect / regionAspect],
            has_image: hasImage ? 1 : 0,
            resolution: [width, height],
            pointer: [mouse.x, mouse.y],
            previous: [previous.x, previous.y],
            press,
            tint: Array.isArray(params.tint) ? params.tint : [0.55, 0.9, 0.82],
            clock,
            strength: n('strength', 1),
            dispersion: n('dispersion', 0),
            blur: n('blur', 0),
            radius: n('radius', 0.027),
            extra: n('merge', n('spread', 1)),
            level,
            spring: spring.value,
            ripple,
            interaction: active,
            pressed: pressed ? 1 : 0,
            delta: dt,
            dry: n('dry', 0.55),
            style: Math.max(0, styles.indexOf(params.style as string)),
            segment_count: segmentCount,
            segments,
            drops: dropData.flat(),
          });
          uniforms.upload();
          if (trailPipeline) {
            const group = device.createBindGroup({
              layout: trailPipeline.getBindGroupLayout(0),
              entries: [
                { binding: 0, resource: { buffer: uniforms.buffer } },
                { binding: 1, resource: sampler },
                { binding: 2, resource: trail.read.view },
              ],
            });
            renderFullscreen(encoder, trail.write.view, trailPipeline, [group], {
              timestampWrites: profiler.pass('Wet spread + dry'),
            });
            trail.swap();
          }
          if (boundImage !== image) {
            displayGroups.clear();
            boundImage = image;
          }
          let group = displayGroups.get(trail.read.view);
          if (!group) {
            group = device.createBindGroup({
              layout,
              entries: [
                { binding: 0, resource: { buffer: uniforms.buffer } },
                { binding: 1, resource: sampler },
                { binding: 2, resource: pageViews[0]! },
                { binding: 3, resource: pageViews[1] ?? pageViews[0]! },
                { binding: 4, resource: atlas.createView() },
                { binding: 5, resource: trail.read.view },
                { binding: 6, resource: image.createView() },
              ],
            });
            displayGroups.set(trail.read.view, group);
          }
          renderFullscreen(encoder, target, pipeline, [group], {
            timestampWrites: profiler.pass(meta.title),
          });
          previous = { x: mouse.x, y: mouse.y };
        },
        destroy() {
          if (destroyed) return;
          destroyed = true;
          controller.abort();
          if (capturedPointer !== undefined) {
            try {
              canvas.releasePointerCapture(capturedPointer);
            } catch {
              /* Pointer may already have been cancelled. */
            }
          }
          displayGroups.clear();
          pointer.destroy();
          paths?.destroy();
          scroll?.destroy();
          trail.destroy();
          uniforms.destroy();
          atlas.destroy();
          image.destroy();
          uploadVersion++;
          pages.forEach((p) => p.destroy());
          for (const [key, value] of original) {
            if (value === null) canvas.removeAttribute(key);
            else canvas.setAttribute(key, value);
          }
        },
      };
    },
  });
}

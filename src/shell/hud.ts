import type { PassTiming } from '../gpu/profiler.ts';
import { getTextureMemory } from '../gpu/texture-memory.ts';
import type { Tier } from '../effects/types.ts';
import type { EffectHost, FrameStats } from './host.ts';
import type { ShellContext, Unmount } from './types.ts';
import './hud.css';

/* ---- Statistics: pure, so they can be tested without a DOM or a GPU ---- */

/** Samples from the last `spanMs`, newest last. */
export interface RollingWindow {
  push(time: number, value: number): void;
  /** Samples still inside the window at `now`. */
  values(now: number): number[];
  clear(): void;
}

export function createRollingWindow(spanMs: number, maxSamples = 600): RollingWindow {
  let times: number[] = [];
  let samples: number[] = [];
  let head = 0;

  function prune(now: number) {
    const cutoff = now - spanMs;
    while (head < times.length && (times[head] ?? 0) < cutoff) head++;
    if (head > 256 && head * 2 > times.length) {
      times = times.slice(head);
      samples = samples.slice(head);
      head = 0;
    }
  }

  return {
    push(time, value) {
      times.push(time);
      samples.push(value);
      prune(time);
      if (times.length - head > maxSamples) head = times.length - maxSamples;
    },
    values(now) {
      prune(now);
      return samples.slice(head);
    },
    clear() {
      times = [];
      samples = [];
      head = 0;
    },
  };
}

export interface Summary {
  count: number;
  mean: number;
  min: number;
  max: number;
  /** Nearest-rank 95th percentile: the frame time 1 in 20 frames exceeds. */
  p95: number;
}

export function summarize(values: readonly number[]): Summary | undefined {
  if (values.length === 0) return undefined;
  const sorted = [...values].sort((a, b) => a - b);
  const sum = sorted.reduce((total, value) => total + value, 0);
  return {
    count: sorted.length,
    mean: sum / sorted.length,
    min: sorted[0] ?? 0,
    max: sorted[sorted.length - 1] ?? 0,
    p95: sorted[Math.max(0, Math.ceil(sorted.length * 0.95) - 1)] ?? 0,
  };
}

export interface PassSummary {
  label: string;
  mean: number;
  max: number;
}

export interface PassWindow {
  /**
   * Offer the profiler's `latest` each frame; the same array again is ignored. `frameMs` is the
   * frame's real GPU time (first pass start to last pass end). Without it the passes are summed,
   * which overstates the total wherever pass intervals overlap.
   */
  push(time: number, passes: readonly PassTiming[], frameMs?: number): void;
  summary(now: number): { total: Summary | undefined; passes: PassSummary[]; durations: number[] };
  clear(): void;
}

/**
 * Collects GPU timings. The profiler hands back the same `latest` array until a newer frame's
 * results arrive, so counting by identity keeps one sample per GPU frame instead of one per
 * rendered frame.
 */
export function createPassWindow(spanMs: number): PassWindow {
  const total = createRollingWindow(spanMs);
  const perPass = new Map<string, RollingWindow>();
  let last: readonly PassTiming[] | undefined;

  return {
    push(time, passes, frameMs) {
      if (passes === last || passes.length === 0) return;
      last = passes;
      total.push(time, frameMs ?? passes.reduce((sum, pass) => sum + pass.ms, 0));
      for (const { label, ms } of passes) {
        let window = perPass.get(label);
        if (!window) perPass.set(label, (window = createRollingWindow(spanMs)));
        window.push(time, ms);
      }
    },
    summary(now) {
      const passes: PassSummary[] = [];
      const durations: number[] = [];
      for (const [label, window] of perPass) {
        const values = window.values(now);
        const summary = summarize(values);
        if (!summary) continue;
        passes.push({ label, mean: summary.mean, max: summary.max });
        durations.push(...values);
      }
      return { total: summarize(total.values(now)), passes, durations };
    },
    clear() {
      total.clear();
      perPass.clear();
      last = undefined;
    },
  };
}

/** GPU budgets from plan.md, in milliseconds at the 1080p reference. */
export const TIER_BUDGET_MS: Readonly<Record<Tier, number>> = { A: 2, B: 6, C: 12 };
export const REFERENCE_PIXELS = 1920 * 1080;

export type BudgetLevel = 'within' | 'near' | 'over';

export interface BudgetStatus {
  budgetMs: number;
  /** Measured GPU time as a fraction of the budget. */
  ratio: number;
  level: BudgetLevel;
}

export function budgetStatus(tier: Tier, gpuMs: number): BudgetStatus {
  const budgetMs = TIER_BUDGET_MS[tier];
  const ratio = gpuMs / budgetMs;
  return { budgetMs, ratio, level: ratio > 1 ? 'over' : ratio >= 0.8 ? 'near' : 'within' };
}

/** Chrome rounds timestamps to this unless "WebGPU Developer Features" is enabled. */
export const CHROME_TIMER_STEP_MS = 0.1;

/**
 * True when enough non-zero durations are all whole multiples of `stepMs`: a pass duration is the
 * difference of two quantised timestamps, so it lands on the same grid. Random real durations
 * would almost never do that.
 */
export function looksQuantised(
  durationsMs: readonly number[],
  stepMs = CHROME_TIMER_STEP_MS,
  minSamples = 8,
): boolean {
  const nonZero = durationsMs.filter((ms) => ms > 0);
  if (nonZero.length < minSamples) return false;
  return nonZero.every((ms) => {
    const steps = ms / stepMs;
    return Math.abs(steps - Math.round(steps)) < 1e-3;
  });
}

export function formatMs(ms: number): string {
  return ms >= 100 ? ms.toFixed(0) : ms >= 10 ? ms.toFixed(1) : ms.toFixed(2);
}

export function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${Math.round(bytes)} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  if (bytes < 1024 ** 3) return `${(bytes / 1024 ** 2).toFixed(1)} MB`;
  return `${(bytes / 1024 ** 3).toFixed(2)} GB`;
}

export function formatMegapixels(width: number, height: number): string {
  const mpx = (width * height) / 1e6;
  return `${mpx < 1 ? mpx.toFixed(2) : mpx.toFixed(1)} Mpx`;
}

/** Polyline points for a frame-time trace, newest on the right; values above `yMax` are clipped. */
export function sparklinePoints(
  values: readonly number[],
  capacity: number,
  width: number,
  height: number,
  yMax: number,
): string {
  const step = capacity > 1 ? width / (capacity - 1) : 0;
  const offset = capacity - values.length;
  return values
    .map((value, index) => {
      const x = (offset + index) * step;
      const y = height - Math.min(1, Math.max(0, value / yMax)) * height;
      return `${x.toFixed(1)},${y.toFixed(1)}`;
    })
    .join(' ');
}

/* ---- Panel ---- */

const WINDOW_MS = 3000;
const REFRESH_MS = 250;
const SPARK_SAMPLES = 120;
const SPARK_WIDTH = 120;
const SPARK_HEIGHT = 32;
const FRAME_60_MS = 1000 / 60;
const SCALES = [0.25, 0.5, 0.75, 1] as const;
const KEY_LIMITS: readonly { key: string; label: string; bytes?: boolean }[] = [
  { key: 'maxTextureDimension2D', label: 'Max texture' },
  { key: 'maxBufferSize', label: 'Max buffer', bytes: true },
  { key: 'maxStorageBufferBindingSize', label: 'Max storage binding', bytes: true },
  { key: 'maxComputeInvocationsPerWorkgroup', label: 'Workgroup invocations' },
  { key: 'maxComputeWorkgroupStorageSize', label: 'Workgroup memory', bytes: true },
  { key: 'maxStorageTexturesPerShaderStage', label: 'Storage textures' },
  { key: 'maxColorAttachmentBytesPerSample', label: 'Attachment bytes' },
  { key: 'maxBindGroups', label: 'Bind groups' },
];
const SVG_NS = 'http://www.w3.org/2000/svg';

function el<K extends keyof HTMLElementTagNameMap>(
  tag: K,
  className: string,
  text?: string,
): HTMLElementTagNameMap[K] {
  const node = document.createElement(tag);
  node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
}

function setText(node: Node, text: string) {
  if (node.textContent !== text) node.textContent = text;
}

/** A label/value pair inside a `<dl>`; the value node is returned for updating. */
function row(list: HTMLElement, label: string, extraClass = ''): HTMLElement {
  const wrapper = el('div', `wt-hud__row ${extraClass}`.trim());
  const value = el('dd', 'wt-hud__value wt-data');
  wrapper.append(el('dt', 'wt-hud__term', label), value);
  list.append(wrapper);
  return value;
}

export function mountHud(container: HTMLElement, host: EffectHost, shell: ShellContext): Unmount {
  const frames = createRollingWindow(WINDOW_MS);
  const cpu = createRollingWindow(WINDOW_MS);
  const gpu = createPassWindow(WINDOW_MS);
  let size = { width: 0, height: 0, dpr: 1 };
  let tier: Tier | undefined;
  let lastStatsAt = -Infinity;
  let startedAt = performance.now();
  let passKey = '';
  const passRows = new Map<string, HTMLElement>();
  let detailsDevice: GPUDevice | undefined;

  // Timing costs a little GPU time, so it only runs while this panel is open.
  const releaseTiming = host.requestTiming();

  /* DOM */
  const root = el('section', 'wt-hud');
  root.setAttribute('aria-label', 'Performance');
  const head = el('header', 'wt-hud__head');
  head.append(el('h2', 'wt-label', 'Performance'));
  const summary = el('span', 'wt-hud__summary wt-data');
  const toggle = el('button', 'wt-button wt-button--quiet wt-hud__toggle', 'Hide');
  toggle.type = 'button';
  toggle.setAttribute('aria-expanded', 'true');
  head.append(summary, toggle);

  const body = el('div', 'wt-hud__body');
  body.id = `wt-hud-body-${Math.random().toString(36).slice(2, 8)}`;
  toggle.setAttribute('aria-controls', body.id);

  const readout = el('div', 'wt-hud__big');
  const fpsValue = el('span', 'wt-hud__fps');
  readout.append(fpsValue, el('span', 'wt-muted', ' fps'));

  const list = el('dl', 'wt-hud__rows');
  const frameValue = row(list, 'Frame');
  const worstValue = row(list, 'p95 / max ms');
  const cpuValue = row(list, 'CPU');
  const gpuValue = row(list, 'GPU');
  const passList = el('div', 'wt-hud__passes');
  list.append(passList);
  const budgetValue = row(list, 'Budget');
  const status = el('span', 'wt-chip wt-hud__status');
  const budgetText = el('span', '');
  budgetValue.append(budgetText, ' ', status);
  const resolutionValue = row(list, 'Buffer');
  const dprValue = row(list, 'DPR');
  const memoryValue = row(list, 'Textures');

  const note = el('p', 'wt-hud__note wt-muted');
  const budgetNote = el('p', 'wt-hud__note wt-muted');
  const quantNote = el('p', 'wt-hud__note wt-muted');
  quantNote.hidden = true;

  const spark = document.createElementNS(SVG_NS, 'svg');
  spark.setAttribute('class', 'wt-hud__spark');
  spark.setAttribute('viewBox', `0 0 ${SPARK_WIDTH} ${SPARK_HEIGHT}`);
  spark.setAttribute('preserveAspectRatio', 'none');
  spark.setAttribute('role', 'img');
  const sparkRef = document.createElementNS(SVG_NS, 'line');
  sparkRef.setAttribute('class', 'wt-hud__spark-ref');
  sparkRef.setAttribute('x1', '0');
  sparkRef.setAttribute('x2', String(SPARK_WIDTH));
  const sparkLine = document.createElementNS(SVG_NS, 'polyline');
  sparkLine.setAttribute('class', 'wt-hud__spark-line');
  spark.append(sparkRef, sparkLine);
  const sparkCaption = el('p', 'wt-hud__note wt-muted');

  const scaleGroup = el('div', 'wt-hud__scales');
  scaleGroup.setAttribute('role', 'group');
  scaleGroup.setAttribute('aria-label', 'Resolution scale');
  const scaleButtons = SCALES.map((scale) => {
    const button = el('button', 'wt-button wt-hud__scale', `${scale * 100}%`);
    button.type = 'button';
    button.addEventListener('click', () => {
      host.setResolutionScale(scale);
      refresh();
    });
    scaleGroup.append(button);
    return { scale, button };
  });
  const scaleLabel = el('p', 'wt-hud__note wt-muted');

  const details = el('details', 'wt-hud__details');
  details.append(el('summary', 'wt-hud__details-summary', 'Device details'));
  const detailsBody = el('div', 'wt-hud__details-body');
  details.append(detailsBody);

  body.append(
    readout,
    list,
    note,
    budgetNote,
    quantNote,
    spark,
    sparkCaption,
    el('div', 'wt-label', 'Resolution scale'),
    scaleGroup,
    scaleLabel,
    details,
  );
  root.append(head, body);
  container.append(root);

  toggle.addEventListener('click', () => {
    const open = body.hidden;
    body.hidden = !open;
    toggle.textContent = open ? 'Hide' : 'Show';
    toggle.setAttribute('aria-expanded', String(open));
    refresh();
  });

  function buildDetails() {
    const state = shell.session.state;
    detailsBody.replaceChildren();
    if (!state.ok) {
      detailsBody.append(el('p', 'wt-muted', 'No GPU device.'));
      detailsDevice = undefined;
      return;
    }
    const { info } = state.gpu;
    detailsDevice = state.gpu.device;
    const facts = el('dl', 'wt-hud__rows');
    const adapter = info.adapter;
    const named = [adapter.vendor, adapter.architecture, adapter.device].filter(Boolean).join(' ');
    row(facts, 'Adapter').textContent = named || 'Hidden by the browser';
    if (adapter.description) row(facts, 'Description').textContent = adapter.description;
    for (const { key, label, bytes } of KEY_LIMITS) {
      const value = info.limits[key];
      if (value === undefined) continue;
      const node = row(facts, label);
      node.textContent = bytes ? formatBytes(value) : String(value);
      node.title = key;
    }
    const features = el('p', 'wt-hud__features wt-data', info.features.join(', ') || 'none');
    detailsBody.append(
      facts,
      el('div', 'wt-label', `Features (${info.features.length})`),
      features,
    );
  }
  details.addEventListener('toggle', () => {
    if (details.open) buildDetails();
  });

  /* Data in */
  const stopStats = host.onStats((stats: FrameStats) => {
    const now = performance.now();
    lastStatsAt = now;
    size = { width: stats.width, height: stats.height, dpr: stats.dpr };
    // The first frame after a start or a pause reports a meaningless delta.
    if (stats.frameMs > 0) frames.push(now, stats.frameMs);
    cpu.push(now, stats.cpuMs);
    gpu.push(now, stats.passes, stats.gpuMs);
  });
  const stopLoad = host.onLoad((loaded) => {
    tier = loaded.effect.meta.tier;
    // A rebuilt effect may label its passes differently, and its old timings no longer apply.
    gpu.clear();
    startedAt = performance.now();
  });

  /* Rendering */
  function syncPasses(passes: readonly PassSummary[]) {
    const key = passes.map((pass) => pass.label).join('\n');
    if (key !== passKey) {
      passKey = key;
      passList.replaceChildren();
      passRows.clear();
      for (const { label } of passes) {
        const value = row(passList, label, 'wt-hud__row--pass');
        passRows.set(label, value);
      }
    }
    for (const pass of passes) {
      const value = passRows.get(pass.label);
      if (value) setText(value, `${formatMs(pass.mean)} ms`);
    }
  }

  function refresh() {
    const now = performance.now();
    const profiler = host.profiler;
    const frameSummary = summarize(frames.values(now));
    const cpuSummary = summarize(cpu.values(now));
    const live = now - lastStatsAt < 1000 && frameSummary !== undefined;
    const gpuData = gpu.summary(now);

    const fps = frameSummary ? 1000 / frameSummary.mean : 0;
    const gpuMean = gpuData.total?.mean;
    // Collapsed, the header carries the two numbers worth seeing at a glance.
    setText(
      summary,
      live
        ? `${fps.toFixed(0)} fps${gpuMean !== undefined ? ` · ${formatMs(gpuMean)} ms GPU` : ''}`
        : host.paused
          ? 'paused'
          : 'idle',
    );
    summary.hidden = !body.hidden;
    if (body.hidden) return;

    setText(fpsValue, live ? fps.toFixed(0) : '—');
    setText(frameValue, live && frameSummary ? `${formatMs(frameSummary.mean)} ms avg` : '—');
    setText(
      worstValue,
      live && frameSummary ? `${formatMs(frameSummary.p95)} / ${formatMs(frameSummary.max)}` : '—',
    );
    setText(cpuValue, cpuSummary && live ? `${formatMs(cpuSummary.mean)} ms` : '—');

    // GPU
    const supported = profiler?.supported ?? false;
    syncPasses(supported ? gpuData.passes : []);
    let gpuNote = '';
    if (!profiler) {
      setText(gpuValue, '—');
      gpuNote = 'No GPU device.';
    } else if (!supported) {
      setText(gpuValue, 'n/a');
      gpuNote =
        'GPU timing is unavailable: this device did not offer the timestamp-query feature ' +
        '(Chrome and Edge 121+, Safari 26+ do). Frame and CPU time still work.';
    } else if (gpuMean === undefined) {
      setText(gpuValue, '…');
      gpuNote =
        now - startedAt > 3000 && live
          ? 'No timed passes reported: this effect does not label its passes.'
          : 'Waiting for GPU timings.';
    } else {
      setText(gpuValue, `${formatMs(gpuMean)} ms`);
      if (gpuData.passes.length > 1) {
        gpuNote =
          'GPU is the time from the first pass starting to the last one ending. The per-pass ' +
          'figures overlap on tile-based GPUs, so they do not add up to it.';
      }
    }
    setText(note, gpuNote);
    note.hidden = gpuNote === '';

    const quantised = supported && looksQuantised(gpuData.durations);
    quantNote.hidden = !quantised;
    if (quantised) {
      setText(
        quantNote,
        `GPU times land on ${CHROME_TIMER_STEP_MS} ms steps: Chrome rounds timestamps unless the ` +
          '"WebGPU Developer Features" flag is on. Averages remain usable; single passes do not.',
      );
    }
    gpuValue.title = quantised ? `Resolution ${CHROME_TIMER_STEP_MS} ms (quantised)` : '';

    // Budget
    const pixels = size.width * size.height;
    if (tier && gpuMean !== undefined && supported) {
      const result = budgetStatus(tier, gpuMean);
      setText(budgetText, `${Math.round(result.ratio * 100)}% of ${result.budgetMs} ms`);
      setText(status, result.level);
      status.dataset.level = result.level;
      status.hidden = false;
      const atReference = pixels > 0 ? (gpuMean * REFERENCE_PIXELS) / pixels : undefined;
      budgetValue.title =
        atReference === undefined
          ? ''
          : `If cost scaled with pixel count, about ${formatMs(atReference)} ms at 1080p. It often does not.`;
      setText(
        budgetNote,
        `Tier ${tier} budget, for 1080p (${formatMegapixels(1920, 1080)}) on a reference GPU. ` +
          `A rough indicator: this is rendering ${formatMegapixels(size.width, size.height)}.`,
      );
      budgetNote.hidden = false;
    } else {
      setText(budgetText, tier ? `Tier ${tier}: ${TIER_BUDGET_MS[tier]} ms` : '—');
      status.hidden = true;
      budgetValue.title = '';
      budgetNote.hidden = true;
    }

    // Resolution
    setText(
      resolutionValue,
      size.width > 0
        ? `${size.width}×${size.height} · ${formatMegapixels(size.width, size.height)}`
        : '—',
    );
    setText(dprValue, size.width > 0 ? size.dpr.toFixed(2) : '—');
    dprValue.title =
      'Drawing-buffer pixels per CSS pixel actually in effect, after the DPR cap, the resolution scale and size limits.';
    const current = host.resolutionScale;
    for (const { scale, button } of scaleButtons) {
      button.setAttribute('aria-pressed', String(Math.abs(scale - current) < 0.005));
    }
    setText(
      scaleLabel,
      `Scale applies per axis, so ${Math.round(current * 100)}% draws about ` +
        `${Math.round(current * current * 100)}% of the pixels.`,
    );

    // Texture memory: device-wide, because the one device serves every canvas on the page.
    const state = shell.session.state;
    const memory = state.ok ? getTextureMemory(state.gpu.device) : undefined;
    if (memory) {
      const unknown = memory.unknownFormats.length > 0;
      setText(
        memoryValue,
        `≈ ${formatBytes(memory.bytes)}${unknown ? ' +?' : ''} · ${memory.textures} tex`,
      );
      memoryValue.title =
        'Estimate from createTexture calls on the shared GPU device, so it covers every canvas ' +
        'on this page. Excludes buffers, swap-chain textures and driver padding.' +
        (unknown ? ` Counted as 4 bytes per pixel: ${memory.unknownFormats.join(', ')}.` : '');
    } else {
      setText(memoryValue, '—');
      memoryValue.title = '';
    }
    if (details.open && state.ok && state.gpu.device !== detailsDevice) buildDetails();

    // Frame-time trace
    const recent = frames.values(now).slice(-SPARK_SAMPLES);
    const peak = recent.reduce((max, value) => Math.max(max, value), 0);
    const yMax = Math.min(100, Math.max(FRAME_60_MS * 2, peak * 1.1));
    sparkLine.setAttribute(
      'points',
      sparklinePoints(recent, SPARK_SAMPLES, SPARK_WIDTH, SPARK_HEIGHT, yMax),
    );
    const refY = SPARK_HEIGHT - (FRAME_60_MS / yMax) * SPARK_HEIGHT;
    sparkRef.setAttribute('y1', refY.toFixed(1));
    sparkRef.setAttribute('y2', refY.toFixed(1));
    const caption = `Frame time, 0 to ${yMax.toFixed(0)} ms. Dashed line: 16.7 ms (60 fps).`;
    setText(sparkCaption, caption);
    spark.setAttribute('aria-label', caption);
  }

  refresh();
  const timer = setInterval(refresh, REFRESH_MS);

  return () => {
    clearInterval(timer);
    stopStats();
    stopLoad();
    releaseTiming();
    root.remove();
  };
}

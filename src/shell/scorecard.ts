import type { EffectMeta, Scorecard, Tier } from '../effects/types.ts';
import type { PassTiming } from '../gpu/profiler.ts';
import type { EffectHost, FrameStats } from './host.ts';
import type { Unmount } from './types.ts';
import './scorecard.css';

export const NOT_MEASURED = 'not measured yet';

/** The tiers and budgets from plan.md ("Practicality tiers"). */
export const TIERS: Readonly<
  Record<Tier, { name: string; shape: string; budgetMs: number; reference: string }>
> = {
  A: {
    name: 'Drop-in',
    shape: 'One or two fragment passes, no compute. Safe as ambient decoration, including mobile.',
    budgetMs: 2,
    reference: 'an integrated GPU',
  },
  B: {
    name: 'Feature',
    shape: 'Multi-pass or a small compute sim. For hero sections and interactive set pieces.',
    budgetMs: 6,
    reference: 'an integrated GPU',
  },
  C: {
    name: 'Showpiece',
    shape: 'Heavy compute and/or a full 3D scene. A dedicated page, desktop-first.',
    budgetMs: 12,
    reference: 'a discrete GPU or Apple Silicon',
  },
};

export function tierLabel(tier: Tier): string {
  return `Tier ${tier} · ${TIERS[tier].name}`;
}

export function budgetText(tier: Tier): string {
  const { budgetMs, reference } = TIERS[tier];
  return `≤ ${budgetMs} ms per frame at 1080p on ${reference}`;
}

export function formatMs(ms: number): string {
  return `${ms < 10 ? ms.toFixed(2) : ms.toFixed(1)} ms`;
}

export function formatKb(kb: number | undefined): string {
  return kb === undefined ? NOT_MEASURED : `${kb < 10 ? kb.toFixed(1) : Math.round(kb)} kB gzipped`;
}

export function computeText(compute: boolean): string {
  return compute ? 'Yes. Compute shaders, so there is no WebGL2 fallback path.' : 'No';
}

export function mobileText(mobile: Scorecard['mobile']): string {
  if (mobile === 'untested') return 'Untested';
  return mobile === 'yes' ? 'Yes, holds its tier budget on phones' : 'No';
}

export type FeatureStatus = 'available' | 'missing' | 'unknown';

/** `deviceFeatures` is undefined while it is still being looked up. */
export function featureStatus(
  feature: string,
  deviceFeatures: ReadonlySet<string> | undefined,
): FeatureStatus {
  if (!deviceFeatures) return 'unknown';
  return deviceFeatures.has(feature) ? 'available' : 'missing';
}

export function featureStatusText(status: FeatureStatus): string {
  return { available: 'available here', missing: 'not available here', unknown: 'checking' }[
    status
  ];
}

export interface CostRow {
  device: string;
  value: string;
  /** Against the tier budget; the reference devices are the ones the budget is defined on. */
  verdict: string;
}

/** One row per measured reference device; empty when nothing has been measured. */
export function costRows(tier: Tier, gpuMs: Scorecard['gpuMs']): CostRow[] {
  const { budgetMs } = TIERS[tier];
  return (gpuMs ?? []).map(({ device, ms }) => ({
    device,
    value: formatMs(ms),
    verdict: ms <= budgetMs ? `within the ${budgetMs} ms budget` : `over the ${budgetMs} ms budget`,
  }));
}

export function sumPasses(passes: readonly PassTiming[]): number {
  return passes.reduce((total, pass) => total + pass.ms, 0);
}

/** Exponential moving average; the first reading is taken as is. */
export function smooth(previous: number | undefined, next: number, alpha = 0.15): number {
  return previous === undefined ? next : previous + (next - previous) * alpha;
}

export function resolutionText(width: number, height: number): string {
  const megapixels = (width * height) / 1e6;
  return `${width} × ${height} (${megapixels.toFixed(2)} MP)`;
}

function el<K extends keyof HTMLElementTagNameMap>(
  tag: K,
  className: string,
  text?: string,
): HTMLElementTagNameMap[K] {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
}

function row(list: HTMLElement, label: string, ...value: (Node | string)[]): HTMLElement {
  const wrapper = el('div', 'wt-scorecard__row');
  const dd = el('dd', 'wt-scorecard__value');
  dd.append(...value);
  wrapper.append(el('dt', 'wt-label', label), dd);
  list.append(wrapper);
  return dd;
}

function unknown(text: string): HTMLElement {
  return el('span', 'wt-muted', text);
}

function panel(title: string): { section: HTMLElement; list: HTMLElement } {
  const section = el('section', 'wt-panel wt-scorecard__panel');
  const list = el('dl', 'wt-scorecard__list');
  section.append(el('h2', 'wt-label', title), list);
  return { section, list };
}

let adapterFeatures: Promise<ReadonlySet<string>> | undefined;

/** Features this browser's adapter offers; empty if there is no WebGPU at all. */
function loadDeviceFeatures(): Promise<ReadonlySet<string>> {
  adapterFeatures ??= (async () => {
    try {
      const adapter = await navigator.gpu?.requestAdapter();
      return new Set<string>(adapter?.features ?? []);
    } catch {
      return new Set<string>();
    }
  })();
  return adapterFeatures;
}

function featureList(
  features: readonly GPUFeatureName[] | undefined,
  none: string,
): { node: HTMLElement; mark(set: ReadonlySet<string>): void } {
  if (!features || features.length === 0) return { node: unknown(none), mark() {} };
  const list = el('ul', 'wt-scorecard__features');
  const marks = features.map((feature) => {
    const item = el('li', '');
    const status = el('span', 'wt-muted');
    item.append(el('code', 'wt-data', feature), ' ', status);
    list.append(item);
    return { feature, item, status };
  });
  const update = (set: ReadonlySet<string> | undefined) => {
    for (const { feature, item, status } of marks) {
      const state = featureStatus(feature, set);
      item.dataset.state = state;
      status.textContent = featureStatusText(state);
    }
  };
  update(undefined);
  return { node: list, mark: update };
}

export function mountScorecard(
  container: HTMLElement,
  meta: EffectMeta,
  host?: EffectHost,
): Unmount {
  const root = el('section', 'wt-scorecard');
  root.setAttribute('aria-label', 'Practicality scorecard');
  const grid = el('div', 'wt-scorecard__grid');
  root.append(grid);
  container.append(root);

  const card = meta.scorecard;
  if (!card) {
    root.append(
      el('p', 'wt-muted', 'No scorecard yet. Add one to this effect’s meta.ts once it is known.'),
    );
    return () => root.remove();
  }

  // Verdict
  const verdict = panel('Verdict');
  const tier = el('span', 'wt-chip', tierLabel(meta.tier));
  tier.dataset.tier = meta.tier;
  const tierRow = row(verdict.list, 'Tier', tier);
  tierRow.append(
    el('span', 'wt-muted', ` ${TIERS[meta.tier].shape}`),
    el('span', 'wt-data wt-scorecard__budget', `Budget: ${budgetText(meta.tier)}`),
  );
  row(verdict.list, 'Needs compute', computeText(card.compute));
  row(verdict.list, 'Mobile-safe', mobileText(card.mobile));
  row(verdict.list, 'Fallback', card.fallback);
  const uses = el('ul', 'wt-scorecard__uses');
  for (const useCase of card.useCases) uses.append(el('li', '', useCase));
  row(verdict.list, 'Best for', card.useCases.length > 0 ? uses : unknown('not decided yet'));

  // Requirements and cost
  const cost = panel('Requirements and cost');
  const required = featureList(card.features?.required, 'none');
  const optional = featureList(card.features?.optional, 'none');
  row(cost.list, 'Required features', required.node);
  row(cost.list, 'Optional features', optional.node);
  row(
    cost.list,
    'Bundle',
    card.bundleKb === undefined ? unknown(NOT_MEASURED) : formatKb(card.bundleKb),
  );

  const rows = costRows(meta.tier, card.gpuMs);
  if (rows.length === 0) {
    row(cost.list, 'GPU time at 1080p', unknown(NOT_MEASURED));
  } else {
    const table = el('table', 'wt-scorecard__table');
    const head = el('tr', '');
    for (const title of ['Device', 'Time', 'Against budget']) head.append(el('th', '', title));
    const thead = el('thead', '');
    thead.append(head);
    table.append(el('caption', 'wt-muted', 'Per frame, per reference device'), thead);
    const body = el('tbody', '');
    for (const entry of rows) {
      const tr = el('tr', '');
      tr.append(
        el('th', '', entry.device),
        el('td', 'wt-data', entry.value),
        el('td', 'wt-muted', entry.verdict),
      );
      body.append(tr);
    }
    table.append(body);
    row(cost.list, 'GPU time at 1080p', table);
  }

  grid.append(verdict.section, cost.section);

  let disposed = false;
  void loadDeviceFeatures().then((set) => {
    if (disposed) return;
    required.mark(set);
    optional.mark(set);
  });

  // Live reading: only when the host reports pass timings.
  const live = panel('This device, right now');
  live.section.hidden = true;
  live.section.classList.add('wt-scorecard__live');
  const liveTime = row(live.list, 'GPU time', '');
  const liveSize = row(live.list, 'Resolution', '');
  const liveNote = el(
    'p',
    'wt-muted wt-scorecard__note',
    'Measured on this device at this resolution. Not comparable with the 1080p reference numbers.',
  );
  live.section.append(liveNote);
  grid.append(live.section);

  let averaged: number | undefined;
  let lastPaint = 0;
  const releaseTiming = host?.requestTiming();
  const stopStats = host?.onStats((stats: FrameStats) => {
    if (stats.passes.length === 0) return;
    // Not the sum of the passes: their intervals overlap on tile-based GPUs.
    averaged = smooth(averaged, stats.gpuMs ?? sumPasses(stats.passes));
    const now = performance.now();
    if (now - lastPaint < 500) return;
    lastPaint = now;
    liveTime.textContent = `${formatMs(averaged)} across ${stats.passes.length} ${stats.passes.length === 1 ? 'pass' : 'passes'}`;
    liveTime.classList.add('wt-data');
    liveSize.textContent = resolutionText(stats.width, stats.height);
    liveSize.classList.add('wt-data');
    live.section.hidden = false;
  });

  return () => {
    disposed = true;
    stopStats?.();
    releaseTiming?.();
    root.remove();
  };
}

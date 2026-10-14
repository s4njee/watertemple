import type { EffectMeta } from '../types.ts';
export const meta: EffectMeta = {
  id: 'ui-droplets',
  title: 'Droplets on UI',
  epic: '9',
  tier: 'B',
  tags: ['ui', 'web-native', 'interactive'],
  summary:
    'Condensation droplets grow, coalesce with conserved volume, and slide down a card under gravity.',
  scorecard: {
    compute: false,
    mobile: 'untested',
    bundleKb: 54.0,
    gpuMs: [{ device: 'Apple M1 Max (Metal)', ms: 1.049, resolution: '1920 × 1080' }],
    features: { optional: ['timestamp-query'] },
    fallback: 'A static poster with semantic HTML for the component and its content.',
    useCases: ['Website components', 'Liquid interface research'],
  },
};

import type { EffectMeta } from '../types.ts';

export const meta: EffectMeta = {
  id: 'rain-glass',
  title: 'Rain on glass',
  epic: '4',
  tier: 'A',
  tags: ['background', 'weather', 'rain', 'glass', 'droplets', 'refraction', 'blur'],
  summary:
    'A rainy window at night: every drop is a tiny lens showing a sharp, flipped piece of the street while the misted glass around it is out of focus, and running drops wipe clean trails.',
  scorecard: {
    compute: false,
    mobile: 'untested',
    features: { required: [] },
    fallback:
      'A short looping video of the rainy window, or a blurred still of the scene with a CSS blur and a few SVG-masked drops.',
    useCases: [
      'Mood background for a hero, a login page or a weather or music app',
      'A window or windscreen overlay on top of a photo or video you already have (swap the procedural scene for a texture)',
      'The droplet field for rain on a camera lens or visor (story 10.5 builds on it)',
    ],
  },
};

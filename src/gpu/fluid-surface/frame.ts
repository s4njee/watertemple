import { defineUniformStruct } from '../uniforms.ts';

/**
 * The one uniform block every pass of the fluid surface reads (`fluid-frame.wgsl` is generated from
 * it). Sizes are in pixels. The camera is a separate `Camera` block, as everywhere else.
 */
export const FluidFrame = defineUniformStruct('FluidFrame', {
  /** Depth / smoothing / normals chain, the thickness targets, and the output target. */
  chain_size: 'vec2f',
  thick_size: 'vec2f',
  full_size: 'vec2f',
  /** World radius of a particle, of the sphere the thickness pass integrates, and its gain. */
  radius: 'f32',
  thick_radius: 'f32',
  thick_gain: 'f32',
  /** 0 when the set is empty: the composite then returns the scene untouched. */
  live: 'u32',

  // Smoothing.
  filter_kind: 'u32',
  filter_scale: 'f32',
  max_radius: 'f32',
  range: 'f32',
  blur_radius: 'i32',

  // Composite.
  surface: 'u32',
  split: 'f32',
  debug: 'u32',
  output_srgb: 'u32',
  sigma: 'vec3f',
  density: 'f32',
  scatter: 'vec3f',
  turbidity: 'f32',
  sun_dir: 'vec3f',
  ior: 'f32',
  foam_color: 'vec3f',
  foam_amount: 'f32',
  refraction: 'f32',
  reflection: 'f32',
  roughness: 'f32',
  specular: 'f32',
  haze: 'f32',
  water_type: 'i32',
  max_refract: 'f32',
});

/** Per-draw data of the separable passes: the step between taps, as (1, 0) or (0, 1). */
export const FluidPass = defineUniformStruct('FluidPass', {
  direction: 'vec2i',
});

/** Effect-local include chunks that carry the struct declarations. */
export const FLUID_STRUCT_CHUNKS: Readonly<Record<string, string>> = {
  'fluid-frame.wgsl': FluidFrame.wgsl + FluidPass.wgsl,
};

export const FILTERS = ['bilateral', 'narrow-range'] as const;
export type FluidFilter = (typeof FILTERS)[number];

export const SURFACES = ['smooth', 'raw', 'split'] as const;
/** `raw` shades the unsmoothed sphere depths (the bag of marbles); `split` is raw left, smooth right. */
export type FluidSurfaceMode = (typeof SURFACES)[number];

export const DEBUG_VIEWS = [
  'none',
  'raw-depth',
  'smooth-depth',
  'normals',
  'thickness',
  'foam',
] as const;
export type FluidDebugView = (typeof DEBUG_VIEWS)[number];

export const STAGES = ['depth', 'smooth', 'normals', 'thickness', 'composite'] as const;
/** The renderer's stages, for the `stages` measurement hook of `render`. */
export type FluidStage = (typeof STAGES)[number];

type Vec3 = readonly [number, number, number];

/** What costs resolution or time; changeable at any moment with `setQuality`. */
export interface FluidSurfaceQuality {
  /**
   * Size of the depth / smoothing / normals chain relative to the target, 0.25 to 1. Half
   * resolution (the default) is usually free visually: the surface is smooth by construction, and
   * the composite interpolates it back up with silhouette-aware weights.
   */
  scale: number;
  /** Size of the thickness targets relative to the target, 0.125 to 1. Default 0.25. */
  thicknessScale: number;
  /**
   * Draw only every n-th particle into the thickness image, with n times the weight: 1 to 16,
   * default 1. Thickness is a smooth integral that is blurred anyway, so a regular subsample is
   * nearly indistinguishable and the pass costs 1/n. Needs the particles in an order that does not
   * cluster by position within runs of n, which no solver's output does.
   */
  thicknessStride: number;
  /** Radius of the Gaussian that blurs the thickness, in thickness-target pixels (0 to 16). Default 3. */
  thicknessBlur: number;
  /** Default `narrow-range`. */
  filter: FluidFilter;
  /**
   * Kernel radius in projected particle radii: the filter follows the particle size on screen, so
   * the look is the same near and far. Default 1.5.
   */
  filterRadius: number;
  /** Cap on the kernel radius in chain pixels (the cost bound when a particle fills the screen). Default 16 (beyond 16 the kernel skips texels). */
  maxFilterRadius: number;
  /** Horizontal-plus-vertical rounds, 1 to 8. Default 2. */
  iterations: number;
  /**
   * Depth tolerance in particle radii. Bilateral: sigma of the range weight. Narrow-range: half
   * width of the window samples are clamped to. Default 0.75.
   */
  rangeScale: number;
}

export const DEFAULT_QUALITY: FluidSurfaceQuality = {
  scale: 0.5,
  thicknessScale: 0.25,
  thicknessStride: 1,
  thicknessBlur: 3,
  filter: 'narrow-range',
  filterRadius: 1.5,
  maxFilterRadius: 16,
  iterations: 2,
  rangeScale: 0.75,
};

/** How the liquid looks. Every field is optional; the defaults are clear water. */
export interface FluidSurfaceParams {
  /** Default `smooth`. */
  surface?: FluidSurfaceMode;
  /** Where `split` divides, as a fraction of the width; raw is on the left. Default 0.5. */
  split?: number;
  /** Show an intermediate instead of the liquid. Default `none`. */
  debug?: FluidDebugView;

  /**
   * Named water of `absorption.wgsl`: 0 clear tropical, 1 coastal, 2 murky green, 3 tea lake,
   * 4 glacial milk, 5 swimming pool, 6 open ocean. -1 (default) uses `absorption` and `scatter`.
   */
  waterType?: number;
  /** Absorption per world unit, linear rgb (custom water). Default: slightly blue-green. */
  absorption?: Vec3;
  /** Colour of the in-scattered light (custom water). */
  scatter?: Vec3;
  /** Scattering per world unit (custom water). Default 0.05. */
  turbidity?: number;
  /**
   * Multiplier on the thickness the optics see. The named water types are in metres; a small tank
   * of a few centimetres needs a density to look as deep as the sea. Default 1.
   */
  density?: number;
  /** Index of refraction: sets the Fresnel reflectance and the refraction offset. Default 1.333. */
  ior?: number;
  /** Scale of the refraction offset; 0 shows the background unbent. Default 1. */
  refraction?: number;
  /** Largest thickness, in world units, that bends the background (keeps deep water from scrambling it). Default 1. */
  maxRefractionThickness?: number;
  /** Strength of the sky reflection; 0 turns it off. Default 1. */
  reflection?: number;
  /** Perceptual roughness of the sun highlight. Default 0.05. */
  roughness?: number;
  /** Strength of the sun highlight. Default 1. */
  specular?: number;
  /** Sky haze for the reflection, 0 to 1. Default 0.3. */
  haze?: number;
  /** Unit vector towards the sun (normalised on upload). Must match the sky the scene shows. */
  sunDirection?: Vec3;
  /**
   * Thickness gain: the volume a particle stands for divided by the volume of its sphere,
   * (mass / rest density) / (4/3 pi r^3). 1 sums the exact chords of the spheres; use more than 1
   * when the spheres of a dense fluid overlap less than the fluid does (a packed set with
   * spacing 2r needs 1.9), less when they overlap more. Default 1.
   */
  thicknessGain?: number;
  /** Radius of the sphere the thickness pass sums, in particle radii. Default 1. */
  thicknessRadius?: number;
  /**
   * The per-particle scalar `w` as foam: the thickness-weighted mean of `w` along the view ray,
   * times `foamAmount`, mixes the surface towards `foamColor`. 0 (default) leaves it out.
   */
  foamAmount?: number;
  foamColor?: Vec3;
  /** Encode the result as sRGB (for a canvas target). Default false: linear, for an HDR target. */
  srgb?: boolean;
}

export type ResolvedParams = Required<FluidSurfaceParams>;

export const DEFAULT_PARAMS: ResolvedParams = {
  surface: 'smooth',
  split: 0.5,
  debug: 'none',
  waterType: -1,
  absorption: [0.35, 0.09, 0.05],
  scatter: [0.05, 0.25, 0.35],
  turbidity: 0.05,
  density: 1,
  ior: 1.333,
  refraction: 1,
  maxRefractionThickness: 1,
  reflection: 1,
  roughness: 0.05,
  specular: 1,
  haze: 0.3,
  sunDirection: [0.5, 0.7, 0.35],
  thicknessGain: 1,
  thicknessRadius: 1,
  foamAmount: 0,
  foamColor: [0.95, 0.97, 1],
  srgb: false,
};

export function resolveParams(params: FluidSurfaceParams = {}): ResolvedParams {
  const defined = Object.fromEntries(
    Object.entries(params).filter(([, value]) => value !== undefined),
  );
  return { ...DEFAULT_PARAMS, ...defined } as ResolvedParams;
}

const clamp = (x: number, lo: number, hi: number) => Math.min(hi, Math.max(lo, x));

/** Clamps a quality to what the renderer supports. */
export function clampQuality(quality: FluidSurfaceQuality): FluidSurfaceQuality {
  return {
    scale: clamp(quality.scale, 0.25, 1),
    thicknessScale: clamp(quality.thicknessScale, 0.125, 1),
    thicknessStride: Math.round(clamp(quality.thicknessStride, 1, 16)),
    thicknessBlur: Math.round(clamp(quality.thicknessBlur, 0, 16)),
    filter: quality.filter === 'bilateral' ? 'bilateral' : 'narrow-range',
    filterRadius: clamp(quality.filterRadius, 0.1, 16),
    maxFilterRadius: clamp(quality.maxFilterRadius, 1, 64),
    iterations: Math.round(clamp(quality.iterations, 1, 8)),
    rangeScale: clamp(quality.rangeScale, 0.05, 16),
  };
}

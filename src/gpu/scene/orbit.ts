import { trackPointer } from '../pointer.ts';

/*
 * Orbit camera control, split in two so the maths can be tested without a canvas:
 * `stepOrbit` is a pure state update (drag deltas in, new angles out); `createOrbitControl` feeds
 * it from `trackPointer`. Yaw 0 puts the eye on the +Z side of the target; pitch is the elevation
 * above the horizontal plane through the target.
 */

export interface OrbitState {
  /** Radians around +Y; increasing moves the eye from +Z towards +X. */
  yaw: number;
  /** Elevation in radians; 0 is level with the target, positive is above. */
  pitch: number;
  distance: number;
  target: [number, number, number];
  /** Radians per second. Carries a flick on after release, then decays. */
  yawVelocity: number;
  pitchVelocity: number;
  /** Seconds since the visitor last moved the camera; drives the idle auto-rotation. */
  idleTime: number;
}

export interface OrbitOptions {
  /** Keep the eye above a surface: pitch is clamped to this range. Defaults 0.05 and 1.3. */
  minPitch: number;
  maxPitch: number;
  /** Exponential decay of the release velocity, per second. Higher stops sooner. */
  damping: number;
  /** Seconds of stillness before the camera starts to drift. */
  idleDelay: number;
  /** Seconds over which the drift eases in, so it never starts with a jolt. */
  idleRamp: number;
  /** Auto-rotation speed in radians per second; 0 disables it. */
  autoSpeed: number;
}

export const DEFAULT_ORBIT_OPTIONS: OrbitOptions = {
  minPitch: 0.05,
  maxPitch: 1.3,
  damping: 4,
  idleDelay: 2.5,
  idleRamp: 2,
  autoSpeed: 0.12,
};

export interface OrbitInput {
  /** Camera rotation requested this frame, in radians (already scaled from pointer movement). */
  yaw: number;
  pitch: number;
  /** A drag is in progress. */
  dragging: boolean;
}

export function createOrbitState(initial: Partial<OrbitState> = {}): OrbitState {
  return {
    yaw: 0,
    pitch: 0.3,
    distance: 10,
    yawVelocity: 0,
    pitchVelocity: 0,
    idleTime: 0,
    ...initial,
    target: initial.target ? [...initial.target] : [0, 0, 0],
  };
}

const clamp = (x: number, lo: number, hi: number) => Math.min(hi, Math.max(lo, x));
const smooth = (x: number) => {
  const t = clamp(x, 0, 1);
  return t * t * (3 - 2 * t);
};

/** Below this speed (rad/s) the camera counts as still, so inertia ends and the idle clock runs. */
const STILL_SPEED = 0.02;

/**
 * Advances the state by `dt` seconds. While dragging the angles follow the input directly and the
 * velocity tracks the drag; after release the velocity decays exponentially (damping). Once
 * nothing has moved for `idleDelay` seconds the yaw starts drifting, easing in over `idleRamp`.
 */
export function stepOrbit(
  state: OrbitState,
  input: OrbitInput,
  dt: number,
  options: OrbitOptions = DEFAULT_ORBIT_OPTIONS,
): OrbitState {
  if (input.dragging) {
    state.yaw += input.yaw;
    state.pitch += input.pitch;
    if (dt > 0) {
      // Smoothed so one jittery frame does not decide the flick on release.
      state.yawVelocity += (input.yaw / dt - state.yawVelocity) * 0.5;
      state.pitchVelocity += (input.pitch / dt - state.pitchVelocity) * 0.5;
    }
    state.idleTime = 0;
  } else if (dt > 0) {
    state.yaw += state.yawVelocity * dt;
    state.pitch += state.pitchVelocity * dt;
    const decay = Math.exp(-options.damping * dt);
    state.yawVelocity *= decay;
    state.pitchVelocity *= decay;
    const speed = Math.hypot(state.yawVelocity, state.pitchVelocity);
    if (speed < STILL_SPEED) {
      state.yawVelocity = 0;
      state.pitchVelocity = 0;
      state.idleTime += dt;
    } else {
      state.idleTime = 0;
    }
    if (options.autoSpeed !== 0) {
      const weight = smooth(
        (state.idleTime - options.idleDelay) / Math.max(options.idleRamp, 1e-6),
      );
      state.yaw += options.autoSpeed * weight * dt;
    }
  }

  const pitch = clamp(state.pitch, options.minPitch, options.maxPitch);
  if (pitch !== state.pitch) state.pitchVelocity = 0;
  state.pitch = pitch;
  // Keep the angle bounded so a long session does not lose float precision.
  state.yaw = ((((state.yaw + Math.PI) % (Math.PI * 2)) + Math.PI * 2) % (Math.PI * 2)) - Math.PI;
  return state;
}

/** World-space eye position for the state. */
export function orbitEye(state: OrbitState): [number, number, number] {
  const horizontal = Math.cos(state.pitch) * state.distance;
  return [
    state.target[0] + Math.sin(state.yaw) * horizontal,
    state.target[1] + Math.sin(state.pitch) * state.distance,
    state.target[2] + Math.cos(state.yaw) * horizontal,
  ];
}

export interface OrbitControlOptions extends Partial<OrbitOptions> {
  /** Radians of yaw for a drag across the full canvas width. Default 3. */
  yawPerWidth?: number;
  /** Radians of pitch for a drag across the full canvas height. Default 2. */
  pitchPerHeight?: number;
}

export interface OrbitControl {
  readonly state: OrbitState;
  /** Call once per frame with the frame's delta in seconds. Auto-rotation can be switched off live. */
  update(dt: number, autoRotate?: boolean): OrbitState;
  destroy(): void;
}

/** Drag to orbit, built on `trackPointer` with capture so a drag survives leaving the canvas. */
export function createOrbitControl(
  canvas: HTMLCanvasElement,
  state: OrbitState,
  options: OrbitControlOptions = {},
): OrbitControl {
  const { yawPerWidth = 3, pitchPerHeight = 2, ...rest } = options;
  const merged: OrbitOptions = { ...DEFAULT_ORBIT_OPTIONS, ...rest };
  const pointer = trackPointer(canvas, { capture: true });
  return {
    state,
    update(dt, autoRotate = true) {
      // Always drained, so movement while hovering is not applied when a drag starts.
      const { dx, dy } = pointer.takeDelta();
      const dragging = pointer.state.down;
      return stepOrbit(
        state,
        {
          yaw: dragging ? -dx * yawPerWidth : 0,
          pitch: dragging ? dy * pitchPerHeight : 0,
          dragging,
        },
        dt,
        autoRotate ? merged : { ...merged, autoSpeed: 0 },
      );
    },
    destroy() {
      pointer.destroy();
    },
  };
}

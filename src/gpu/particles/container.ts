// Effective gravity in a container's local frame. Pointer displacement tilts the vessel and
// pointer acceleration / shaking contributes the opposite inertial acceleration to the liquid.
const clamp = (x: number, a: number, b: number) => Math.min(b, Math.max(a, x));
const finite = (x: number) => (Number.isFinite(x) ? x : 0);
type V3 = [number, number, number];

/** Device tilt mapped to tank +x right, +y up, +z towards the viewer; beta=0 is neutral. */
export function tiltGravity(
  beta: number,
  gamma: number,
  screenAngle: number,
  magnitude: number,
): V3 {
  const b = (clamp(finite(beta), -60, 60) * Math.PI) / 180;
  const c = (clamp(finite(gamma), -60, 60) * Math.PI) / 180;
  const a = (finite(screenAngle) * Math.PI) / 180;
  const x = Math.sin(c),
    z = Math.sin(b) * Math.cos(c);
  const y = -Math.cos(b) * Math.cos(c);
  return [
    (x * Math.cos(a) - z * Math.sin(a)) * magnitude,
    y * magnitude,
    (x * Math.sin(a) + z * Math.cos(a)) * magnitude,
  ];
}

export function createContainerController(canvas: HTMLCanvasElement) {
  const owner = canvas.ownerDocument?.defaultView;
  let tilt: [number, number] | undefined;
  let enabled = false;
  let attached = false;
  let disposed = false;
  let requesting = false;
  let clock = 0;
  let roll = 0,
    pitch = 0;
  let previous: [number, number] = [0, 0];
  const orientation = (event: DeviceOrientationEvent) => {
    if (event.beta !== null && event.gamma !== null) tilt = [event.beta, event.gamma];
  };
  const attach = () => {
    if (!owner || attached || disposed || !enabled) return;
    owner.addEventListener('deviceorientation', orientation);
    attached = true;
  };
  // iOS requires permission from a user gesture; the next canvas tap requests it only when opted in.
  const request = () => {
    if (!enabled || !owner || attached || requesting) return;
    const ctor = (
      owner as unknown as { DeviceOrientationEvent?: { requestPermission?: () => Promise<string> } }
    ).DeviceOrientationEvent;
    if (ctor?.requestPermission) {
      requesting = true;
      void ctor
        .requestPermission()
        .then((result) => {
          if (result === 'granted') attach();
        })
        .catch(() => {})
        .finally(() => {
          requesting = false;
        });
    } else attach();
  };
  canvas.addEventListener('pointerup', request);
  return {
    setTiltEnabled(value: boolean) {
      enabled = value;
      if (!value) {
        if (attached) owner?.removeEventListener('deviceorientation', orientation);
        attached = false;
        tilt = undefined;
      } else if (
        !(
          owner as unknown as
            { DeviceOrientationEvent?: { requestPermission?: unknown } } | undefined
        )?.DeviceOrientationEvent?.requestPermission
      )
        attach();
    },
    gravity(
      delta: number,
      magnitude: number,
      input: { dx?: number; dy?: number; dragging?: boolean; shake?: number } = {},
    ): V3 {
      const dt = clamp(finite(delta), 0, 0.05);
      if (dt === 0) return tiltGravity(pitch, roll, 0, magnitude);
      clock += dt;
      const dx = input.dragging ? clamp(finite(input.dx ?? 0), -0.1, 0.1) : 0;
      const dy = input.dragging ? clamp(finite(input.dy ?? 0), -0.1, 0.1) : 0;
      roll = clamp(roll + dx * 160, -45, 45) * Math.exp(-dt * 1.5);
      pitch = clamp(pitch + dy * 160, -45, 45) * Math.exp(-dt * 1.5);
      const g =
        enabled && tilt
          ? tiltGravity(tilt[0], tilt[1], owner?.screen.orientation?.angle ?? 0, magnitude)
          : tiltGravity(pitch, roll, 0, magnitude);
      const vx = dx / dt,
        vz = dy / dt;
      const shake = clamp(finite(input.shake ?? 0), 0, 1);
      g[0] += clamp((-(vx - previous[0]) / dt) * 0.12, -15, 15) + shake * 12 * Math.sin(clock * 16);
      g[2] += clamp((-(vz - previous[1]) / dt) * 0.12, -15, 15) + shake * 8 * Math.cos(clock * 19);
      previous = [vx, vz];
      return g;
    },
    reset() {
      roll = pitch = clock = 0;
      previous = [0, 0];
    },
    destroy() {
      disposed = true;
      canvas.removeEventListener('pointerup', request);
      if (attached) owner?.removeEventListener('deviceorientation', orientation);
    },
  };
}

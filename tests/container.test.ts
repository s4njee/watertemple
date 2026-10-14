import { describe, expect, it } from 'vitest';
import { createContainerController, tiltGravity } from '../src/gpu/particles/container.ts';

const canvas = () => new EventTarget() as unknown as HTMLCanvasElement;
describe('container forces', () => {
  it('rotates gravity without changing its magnitude, including landscape orientation', () => {
    expect(tiltGravity(0, 0, 0, 9.81)).toEqual([0, -9.81, 0]);
    for (const angle of [0, 90, 180, 270]) {
      expect(Math.hypot(...tiltGravity(30, 20, angle, 9.81))).toBeCloseTo(9.81);
    }
    expect(tiltGravity(0, 30, 90, 10)[2]).toBeCloseTo(5);
    expect(tiltGravity(NaN, Infinity, NaN, 10)).toEqual([0, -10, 0]);
  });
  it('responds to dragging and shaking, relaxes, resets, and supports headless teardown', () => {
    const controller = createContainerController(canvas());
    expect(controller.gravity(1 / 60, 10)).toEqual([0, -10, 0]);
    expect(controller.gravity(1 / 60, 10, { dragging: true, dx: 0.05 })[0]).toBeLessThan(0);
    let g = [0, 0, 0];
    for (let i = 0; i < 300; i++) g = controller.gravity(1 / 60, 10);
    expect(g[0]).toBeCloseTo(0, 1);
    expect(controller.gravity(1 / 60, 10, { shake: 1 })[0]).not.toBeCloseTo(0);
    controller.reset();
    expect(controller.gravity(1 / 60, 10)).toEqual([0, -10, 0]);
    controller.setTiltEnabled(true);
    controller.setTiltEnabled(false);
    controller.destroy();
  });
  it('removes device orientation and gesture listeners on teardown', () => {
    const owner = Object.assign(new EventTarget(), { screen: { orientation: { angle: 0 } } });
    const element = Object.assign(new EventTarget(), { ownerDocument: { defaultView: owner } });
    const controller = createContainerController(element as unknown as HTMLCanvasElement);
    controller.setTiltEnabled(true);
    const event = Object.assign(new Event('deviceorientation'), { beta: 0, gamma: 30 });
    owner.dispatchEvent(event);
    expect(controller.gravity(1 / 60, 10)[0]).toBeCloseTo(5);
    controller.setTiltEnabled(false);
    owner.dispatchEvent(event);
    expect(controller.gravity(1 / 60, 10)[0]).toBe(0);
    controller.destroy();
  });
  it('requests sensor permission only after opt-in and a gesture, and ignores a late grant after disabling', async () => {
    let requests = 0;
    let grant: (value: string) => void = () => {};
    const owner = Object.assign(new EventTarget(), {
      screen: { orientation: { angle: 0 } },
      DeviceOrientationEvent: {
        requestPermission() {
          requests++;
          return new Promise<string>((resolve) => {
            grant = resolve;
          });
        },
      },
    });
    const element = Object.assign(new EventTarget(), { ownerDocument: { defaultView: owner } });
    const controller = createContainerController(element as unknown as HTMLCanvasElement);
    element.dispatchEvent(new Event('pointerup'));
    expect(requests).toBe(0);
    controller.setTiltEnabled(true);
    expect(requests).toBe(0);
    element.dispatchEvent(new Event('pointerup'));
    expect(requests).toBe(1);
    controller.setTiltEnabled(false);
    grant('granted');
    await Promise.resolve();
    controller.setTiltEnabled(true);
    owner.dispatchEvent(Object.assign(new Event('deviceorientation'), { beta: 0, gamma: 30 }));
    expect(controller.gravity(1 / 60, 10)[0]).toBe(0);
    controller.destroy();
  });
});

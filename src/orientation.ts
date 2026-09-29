export type Quaternion = readonly [x: number, y: number, z: number, w: number];

export type OrientationSample = {
  capturedAt: number;
  alpha: number;
  beta: number;
  gamma: number;
  absolute: boolean;
  quaternion: Quaternion;
};

type DeviceOrientationEventConstructorWithPermission = typeof DeviceOrientationEvent & {
  requestPermission?: (absolute?: boolean) => Promise<"granted" | "denied">;
};

const DEG2RAD = Math.PI / 180;
const RAD2DEG = 180 / Math.PI;

function multiplyQuaternion(a: Quaternion, b: Quaternion): Quaternion {
  const [ax, ay, az, aw] = a;
  const [bx, by, bz, bw] = b;
  return [
    aw * bx + ax * bw + ay * bz - az * by,
    aw * by - ax * bz + ay * bw + az * bx,
    aw * bz + ax * by - ay * bx + az * bw,
    aw * bw - ax * bx - ay * by - az * bz,
  ];
}

function normalizeQuaternion(q: Quaternion): Quaternion {
  const length = Math.hypot(q[0], q[1], q[2], q[3]);
  if (length < 1e-9) return [0, 0, 0, 1];
  return [q[0] / length, q[1] / length, q[2] / length, q[3] / length];
}

function axisQuaternion(axis: "x" | "y" | "z", degrees: number): Quaternion {
  const half = degrees * DEG2RAD * 0.5;
  const s = Math.sin(half);
  const c = Math.cos(half);
  if (axis === "x") return [s, 0, 0, c];
  if (axis === "y") return [0, s, 0, c];
  return [0, 0, s, c];
}

/**
 * DeviceOrientation uses intrinsic Z-X'-Y'' rotations: alpha, beta, gamma.
 * We only use the resulting quaternion for short-term relative angular distance
 * during registration. It is never persisted as verification input.
 */
function orientationQuaternion(alpha: number, beta: number, gamma: number): Quaternion {
  const qz = axisQuaternion("z", alpha);
  const qx = axisQuaternion("x", beta);
  const qy = axisQuaternion("y", gamma);
  return normalizeQuaternion(multiplyQuaternion(multiplyQuaternion(qz, qx), qy));
}

export function orientationAngleDeg(a?: OrientationSample, b?: OrientationSample): number | undefined {
  if (!a || !b) return undefined;
  const qa = a.quaternion;
  const qb = b.quaternion;
  // q and -q describe the same rotation, hence abs(dot).
  const dot = Math.min(1, Math.max(-1, Math.abs(
    qa[0] * qb[0] + qa[1] * qb[1] + qa[2] * qb[2] + qa[3] * qb[3],
  )));
  return 2 * Math.acos(dot) * RAD2DEG;
}

export class RegistrationOrientationSensor {
  private latest: OrientationSample | undefined;
  private listening = false;

  private readonly onOrientation = (event: DeviceOrientationEvent): void => {
    if (event.alpha == null || event.beta == null || event.gamma == null) return;
    if (![event.alpha, event.beta, event.gamma].every(Number.isFinite)) return;
    this.latest = {
      capturedAt: performance.now(),
      alpha: event.alpha,
      beta: event.beta,
      gamma: event.gamma,
      absolute: event.absolute,
      quaternion: orientationQuaternion(event.alpha, event.beta, event.gamma),
    };
  };

  static get supported(): boolean {
    return typeof window !== "undefined" && "DeviceOrientationEvent" in window;
  }

  async start(): Promise<boolean> {
    if (!RegistrationOrientationSensor.supported || !window.isSecureContext) return false;

    const ctor = DeviceOrientationEvent as DeviceOrientationEventConstructorWithPermission;
    if (typeof ctor.requestPermission === "function") {
      let permission: "granted" | "denied";
      try {
        // Relative orientation is enough; avoid requesting magnetometer/absolute orientation.
        permission = await ctor.requestPermission(false);
      } catch (error) {
        // Older WebKit implementations expose the no-argument form only.
        if (!(error instanceof TypeError)) return false;
        try {
          permission = await ctor.requestPermission();
        } catch {
          return false;
        }
      }
      if (permission !== "granted") return false;
    }

    window.addEventListener("deviceorientation", this.onOrientation, { passive: true });
    this.listening = true;

    // Distinguish "API exists" from "this device actually delivers samples".
    const started = performance.now();
    while (!this.latest && performance.now() - started < 420) {
      await new Promise(resolve => setTimeout(resolve, 35));
    }
    return Boolean(this.latest);
  }

  sample(maxAgeMs = 500): OrientationSample | undefined {
    if (!this.latest) return undefined;
    if (performance.now() - this.latest.capturedAt > maxAgeMs) return undefined;
    return { ...this.latest, quaternion: [...this.latest.quaternion] as Quaternion };
  }

  stop(): void {
    if (this.listening) window.removeEventListener("deviceorientation", this.onOrientation);
    this.listening = false;
    this.latest = undefined;
  }
}

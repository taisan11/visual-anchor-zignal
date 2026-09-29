import { orientationAngleDeg, type OrientationSample } from "./orientation";
import type { FeatureSet, MatchGeometry, VisualAnchorJS } from "./vision";

const KEYPOINT_STRIDE = 6;

export type SerializedFeatureSetV2 = {
  width: number;
  height: number;
  count: number;
  /** Pixel x/y only, rounded to uint16. Descriptor computation already happened at registration. */
  keypointsXYU16: string;
  descriptorsU8: string;
};

export type SerializedFeatureSet = SerializedFeatureSetV2;

export type VisualAnchor = {
  format: "visual-anchor-zignal";
  version: 2;
  createdAt: string;
  processingWidth: number;
  /** Legacy anchors may include the original Zig/Zignal implementation versions. */
  zignal?: "0.10.0";
  zig?: "0.16.0";
  engine?: "javascript-native";
  views: SerializedFeatureSet[];
  registration: {
    pairScores: number[];
    medianPairScore: number;
    parallaxSignatures: number[];
    medianParallaxSignature: number;
    /** Registration-only diagnostics. Verification intentionally ignores these fields. */
    orientationAssisted?: boolean;
    candidateViews?: number;
    selectedCandidateIndices?: number[];
    orientationSpanDeg?: number;
    meanOrientationStepDeg?: number;
    inputFeatureCount?: number;
    outputFeatureCount?: number;
  };
};

export type RegistrationCandidate = {
  features: FeatureSet;
  orientation?: OrientationSample;
};

export type RegistrationSelection = {
  frames: FeatureSet[];
  indices: number[];
  orientationAssisted: boolean;
  orientationSpanDeg: number;
  meanOrientationStepDeg: number;
};

export type VerificationResult = {
  ok: boolean;
  score: number;
  bestView: number;
  geometry: MatchGeometry;
  burstScores: number[];
  planarReplayRisk: boolean;
  temporalGeometry?: MatchGeometry;
};

function bytesToBase64(bytes: Uint8Array): string {
  let binary = "";
  const chunk = 0x8000;
  for (let i = 0; i < bytes.length; i += chunk) {
    binary += String.fromCharCode(...bytes.subarray(i, Math.min(i + chunk, bytes.length)));
  }
  return btoa(binary);
}

function base64ToBytes(text: string): Uint8Array {
  const binary = atob(text);
  const out = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i++) out[i] = binary.charCodeAt(i);
  return out;
}

/**
 * V2 keeps only x/y and quantizes them to whole pixels. ORB descriptors are already
 * rotation/scale-normalized, while matching + homography only read x/y.
 * This changes keypoint storage from 24 bytes/feature to 4 bytes/feature.
 */
export function serializeFeatures(features: FeatureSet): SerializedFeatureSetV2 {
  const xy = new Uint16Array(features.count * 2);
  for (let i = 0; i < features.count; i++) {
    const base = i * KEYPOINT_STRIDE;
    xy[i * 2] = Math.max(0, Math.min(65535, Math.round(features.keypoints[base])));
    xy[i * 2 + 1] = Math.max(0, Math.min(65535, Math.round(features.keypoints[base + 1])));
  }
  return {
    width: features.width,
    height: features.height,
    count: features.count,
    keypointsXYU16: bytesToBase64(new Uint8Array(xy.buffer)),
    descriptorsU8: bytesToBase64(features.descriptors),
  };
}

export function deserializeFeatures(features: SerializedFeatureSet): FeatureSet {
  const xyBytes = base64ToBytes(features.keypointsXYU16);
  const aligned = new Uint8Array(xyBytes.length);
  aligned.set(xyBytes);
  const xy = new Uint16Array(aligned.buffer);
  const keypoints = new Float32Array(features.count * KEYPOINT_STRIDE);
  for (let i = 0; i < features.count; i++) {
    keypoints[i * KEYPOINT_STRIDE] = xy[i * 2];
    keypoints[i * KEYPOINT_STRIDE + 1] = xy[i * 2 + 1];
  }
  return {
    width: features.width,
    height: features.height,
    count: features.count,
    keypoints,
    descriptors: base64ToBytes(features.descriptorsU8),
  };
}

export function geometryScore(g: MatchGeometry): number {
  if (g.inliers < 4 || !Number.isFinite(g.meanError)) return 0;
  const inlierCount = Math.min(1, g.inliers / 42);
  const ratio = Math.min(1, Math.max(0, g.inlierRatio));
  const reprojection = Math.min(1, Math.max(0, 1 - g.meanError / 7));
  const hamming = Math.min(1, Math.max(0, 1 - Math.max(0, g.avgDistance - 24) / 72));
  return inlierCount * 0.4 + ratio * 0.3 + reprojection * 0.2 + hamming * 0.1;
}

function median(values: number[]): number {
  if (!values.length) return 0;
  const v = [...values].sort((a, b) => a - b);
  const m = Math.floor(v.length / 2);
  return v.length % 2 ? v[m] : (v[m - 1] + v[m]) / 2;
}

export function homographyTranslation(g: MatchGeometry): number {
  return g.homography.length >= 6 ? Math.hypot(g.homography[2], g.homography[5]) : 0;
}

function parallaxSignature(g: MatchGeometry): number {
  const motion = Math.max(1, homographyTranslation(g));
  return g.p90Error > 0 ? g.p90Error / motion : 0;
}

/**
 * Spatially distributes the strongest ORB features instead of merely taking the
 * globally strongest points. This reduces output size while retaining coverage.
 */
export function pruneFeatureSetSpatial(features: FeatureSet, targetCount = 520): FeatureSet {
  if (features.count <= targetCount) return features;
  const stride = features.keypoints.length / features.count;
  const descriptorBytes = features.descriptors.length / features.count;
  if (!Number.isInteger(stride) || stride < 5 || !Number.isInteger(descriptorBytes)) return features;

  const gridX = 8;
  const gridY = 6;
  const buckets: number[][] = Array.from({ length: gridX * gridY }, () => []);
  for (let i = 0; i < features.count; i++) {
    const base = i * stride;
    const x = features.keypoints[base];
    const y = features.keypoints[base + 1];
    const gx = Math.min(gridX - 1, Math.max(0, Math.floor((x / Math.max(1, features.width)) * gridX)));
    const gy = Math.min(gridY - 1, Math.max(0, Math.floor((y / Math.max(1, features.height)) * gridY)));
    buckets[gy * gridX + gx].push(i);
  }
  for (const bucket of buckets) {
    bucket.sort((a, b) => features.keypoints[b * stride + 4] - features.keypoints[a * stride + 4]);
  }

  const selected: number[] = [];
  let depth = 0;
  while (selected.length < targetCount) {
    let added = false;
    for (const bucket of buckets) {
      if (depth < bucket.length) {
        selected.push(bucket[depth]);
        added = true;
        if (selected.length >= targetCount) break;
      }
    }
    if (!added) break;
    depth++;
  }

  const keypoints = new Float32Array(selected.length * stride);
  const descriptors = new Uint8Array(selected.length * descriptorBytes);
  selected.forEach((sourceIndex, outputIndex) => {
    keypoints.set(
      features.keypoints.subarray(sourceIndex * stride, (sourceIndex + 1) * stride),
      outputIndex * stride,
    );
    descriptors.set(
      features.descriptors.subarray(sourceIndex * descriptorBytes, (sourceIndex + 1) * descriptorBytes),
      outputIndex * descriptorBytes,
    );
  });
  return { ...features, count: selected.length, keypoints, descriptors };
}

function viewUtility(
  vision: VisualAnchorJS,
  from: RegistrationCandidate,
  to: RegistrationCandidate,
): number {
  const geometry = vision.match(from.features, to.features, {
    crossCheck: true,
    ransacIterations: 150,
  });
  const base = geometryScore(geometry);
  const motion = Math.min(1, homographyTranslation(geometry) / 36);
  const nonPlanar = Math.min(1, Math.max(0, geometry.p90Error) / 3.5);
  const angle = orientationAngleDeg(from.orientation, to.orientation);
  // Large rotation is a poor substitute for lateral translation when building a parallax-rich anchor.
  const rotationPenalty = angle == null ? 0 : Math.min(1, Math.max(0, angle - 7) / 18);
  const weakGeometryPenalty = geometry.inliers < 14 ? 0.35 : 0;
  return base * 0.52 + motion * 0.27 + nonPlanar * 0.21 - rotationPenalty * 0.28 - weakGeometryPenalty;
}

/**
 * Select temporally spread registration views. Orientation only influences this
 * registration-time choice; no orientation values are consumed by verification.
 */
export function selectRegistrationViews(
  vision: VisualAnchorJS,
  candidates: RegistrationCandidate[],
  targetViews = 5,
): RegistrationSelection {
  if (candidates.length <= targetViews) {
    const indices = candidates.map((_, i) => i);
    return selectionStats(candidates, indices);
  }

  const indices = [0];
  for (let slot = 1; slot < targetViews; slot++) {
    const center = Math.round((slot * (candidates.length - 1)) / (targetViews - 1));
    const previous = indices[indices.length - 1];
    const minIndex = previous + 1;
    const maxIndex = candidates.length - (targetViews - slot);
    const lo = Math.max(minIndex, center - 1);
    const hi = Math.min(maxIndex, center + 1);

    let bestIndex = Math.max(minIndex, Math.min(maxIndex, center));
    let bestUtility = Number.NEGATIVE_INFINITY;
    for (let i = lo; i <= hi; i++) {
      const utility = viewUtility(vision, candidates[previous], candidates[i]);
      if (utility > bestUtility) {
        bestUtility = utility;
        bestIndex = i;
      }
    }
    indices.push(bestIndex);
  }
  return selectionStats(candidates, indices);
}

function selectionStats(candidates: RegistrationCandidate[], indices: number[]): RegistrationSelection {
  const selected = indices.map(i => candidates[i]);
  const validOrientations = selected.filter(c => c.orientation).length;
  const orientationAssisted = validOrientations >= Math.max(2, Math.ceil(selected.length * 0.6));

  const reference = selected[0]?.orientation;
  const fromReference = selected
    .map(c => orientationAngleDeg(reference, c.orientation))
    .filter((v): v is number => v != null && Number.isFinite(v));
  const steps: number[] = [];
  for (let i = 1; i < selected.length; i++) {
    const step = orientationAngleDeg(selected[i - 1].orientation, selected[i].orientation);
    if (step != null && Number.isFinite(step)) steps.push(step);
  }

  return {
    frames: selected.map(c => c.features),
    indices,
    orientationAssisted,
    orientationSpanDeg: fromReference.length ? Math.max(...fromReference) : 0,
    meanOrientationStepDeg: steps.length ? steps.reduce((a, b) => a + b, 0) / steps.length : 0,
  };
}

export function createAnchor(
  vision: VisualAnchorJS,
  frames: FeatureSet[],
  processingWidth: number,
  metadata: Partial<VisualAnchor["registration"]> = {},
): VisualAnchor {
  const pairScores: number[] = [];
  const parallaxSignatures: number[] = [];
  for (let i = 1; i < frames.length; i++) {
    const geometry = vision.match(frames[i - 1], frames[i], { crossCheck: true });
    pairScores.push(geometryScore(geometry));
    parallaxSignatures.push(parallaxSignature(geometry));
  }
  return {
    format: "visual-anchor-zignal",
    version: 2,
    createdAt: new Date().toISOString(),
    processingWidth,
    engine: "javascript-native",
    views: frames.map(serializeFeatures),
    registration: {
      pairScores,
      medianPairScore: median(pairScores),
      parallaxSignatures,
      medianParallaxSignature: median(parallaxSignatures),
      ...metadata,
    },
  };
}

export function verifyBurst(
  vision: VisualAnchorJS,
  anchor: VisualAnchor,
  queryFrames: FeatureSet[],
): VerificationResult {
  // Deliberately image-only: verification does not read DeviceOrientation or registration sensor values.
  const views = anchor.views.map(deserializeFeatures);
  const burstScores: number[] = [];
  let bestScore = -1;
  let bestView = -1;
  let bestGeometry: MatchGeometry | undefined;

  for (const query of queryFrames) {
    let localBest = 0;
    for (let i = 0; i < views.length; i++) {
      const geometry = vision.match(query, views[i], { crossCheck: true });
      const score = geometryScore(geometry);
      if (score > localBest) localBest = score;
      if (score > bestScore) {
        bestScore = score;
        bestView = i;
        bestGeometry = geometry;
      }
    }
    burstScores.push(localBest);
  }

  let temporalGeometry: MatchGeometry | undefined;
  let planarReplayRisk = false;
  if (queryFrames.length >= 2) {
    temporalGeometry = vision.match(queryFrames[0], queryFrames[queryFrames.length - 1], {
      crossCheck: true,
      ransacThreshold: 3,
    });
    const translation = homographyTranslation(temporalGeometry);
    const currentParallax = parallaxSignature(temporalGeometry);
    const registeredParallax = anchor.registration.medianParallaxSignature ?? 0;
    // Heuristic only: compare live burst non-planarity against the scene's registration baseline.
    planarReplayRisk =
      registeredParallax >= 0.015 &&
      temporalGeometry.inliers >= 28 &&
      temporalGeometry.inlierRatio >= 0.75 &&
      translation >= 5 &&
      temporalGeometry.p90Error > 0 &&
      temporalGeometry.p90Error < 1.0 &&
      currentParallax < registeredParallax * 0.35;
  }

  const score = median(burstScores);
  const geometry = bestGeometry ?? {
    matches: 0,
    inliers: 0,
    inlierRatio: 0,
    meanError: 0,
    p90Error: 0,
    avgDistance: 0,
    homography: [],
    pairs: [],
  };

  const ok = score >= 0.58 && geometry.inliers >= 22 && geometry.inlierRatio >= 0.32;
  return { ok, score, bestView, geometry, burstScores, planarReplayRisk, temporalGeometry };
}

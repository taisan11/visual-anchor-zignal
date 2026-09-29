import { ORB_PATTERN } from "./orb-pattern";

export type FeatureSet = {
  width: number;
  height: number;
  count: number;
  /** x, y, size, angle, response, octave per keypoint. */
  keypoints: Float32Array;
  descriptors: Uint8Array;
};

export type MatchPair = { queryIndex: number; trainIndex: number; distance: number };
export type MatchGeometry = {
  matches: number;
  /** Best geometric inlier count across homography and fundamental models. */
  inliers: number;
  inlierRatio: number;
  /** Error corresponding to the best geometric model (px for H, sqrt Sampson px for F). */
  meanError: number;
  p90Error: number;
  avgDistance: number;
  homography: number[];
  pairs: MatchPair[];
  /** Homography-specific diagnostics retained for parallax/replay heuristics. */
  homographyInliers?: number;
  homographyInlierRatio?: number;
  homographyMeanError?: number;
  homographyP90Error?: number;
  /** Fundamental-matrix diagnostics for non-planar scenes. */
  fundamental?: number[];
  fundamentalInliers?: number;
  fundamentalInlierRatio?: number;
  meanSampsonError?: number;
  p90SampsonError?: number;
};

const STRIDE = 6;
const DESC_BYTES = 32;
const DESC_WORDS = DESC_BYTES / 4;
const SCALE = 1.2;
const LEVELS = 8;
const CIRCLE: readonly (readonly [number, number])[] = [
  [0,-3],[1,-3],[2,-2],[3,-1],[3,0],[3,1],[2,2],[1,3],
  [0,3],[-1,3],[-2,2],[-3,1],[-3,0],[-3,-1],[-2,-2],[-1,-3],
];

type Mat3 = [number, number, number, number, number, number, number, number, number];

type NormalizedPoints = {
  points: Float64Array;
  transform: Mat3;
  inverse: Mat3;
};

function equalize(image: ImageData): Uint8Array {
  const { data, width, height } = image;
  const gray = new Uint8Array(width * height);
  const hist = new Uint32Array(256);
  for (let i = 0, p = 0; i < data.length; i += 4, p++) {
    const y = (77 * data[i] + 150 * data[i + 1] + 29 * data[i + 2] + 128) >> 8;
    gray[p] = y;
    hist[y]++;
  }
  let first = 0;
  while (first < 256 && hist[first] === 0) first++;
  let cdf = 0;
  const lut = new Uint8Array(256);
  const total = width * height;
  const cdfMin = hist[first] ?? 0;
  const denom = total - cdfMin;
  for (let i = 0; i < 256; i++) {
    cdf += hist[i];
    lut[i] = denom > 0 ? Math.round(Math.max(0, cdf - cdfMin) * 255 / denom) : i;
  }
  for (let i = 0; i < gray.length; i++) gray[i] = lut[gray[i]];
  return gray;
}

type Level = { width: number; height: number; pixels: Uint8Array };

// Keep a source canvas separate because resizing a canvas clears its bitmap.
function buildPyramid(image: ImageData, shouldEqualize: boolean): Level[] {
  const gray = shouldEqualize ? equalize(image) : (() => {
    const out = new Uint8Array(image.width * image.height);
    const d = image.data;
    for (let i = 0, p = 0; i < d.length; i += 4, p++) out[p] = (77 * d[i] + 150 * d[i + 1] + 29 * d[i + 2] + 128) >> 8;
    return out;
  })();
  const sourceCanvas: OffscreenCanvas | HTMLCanvasElement = typeof OffscreenCanvas !== "undefined"
    ? new OffscreenCanvas(image.width, image.height)
    : Object.assign(document.createElement("canvas"), { width: image.width, height: image.height });
  const sourceContext = sourceCanvas.getContext("2d", { willReadFrequently: true }) as OffscreenCanvasRenderingContext2D | CanvasRenderingContext2D | null;
  if (!sourceContext) throw new Error("Canvas 2D is unavailable");
  const rgba = new Uint8ClampedArray(image.width * image.height * 4);
  for (let i = 0, p = 0; i < rgba.length; i += 4, p++) {
    const v = gray[p]; rgba[i] = v; rgba[i + 1] = v; rgba[i + 2] = v; rgba[i + 3] = 255;
  }
  sourceContext.putImageData(new ImageData(rgba, image.width, image.height), 0, 0);
  const levels: Level[] = [];
  for (let octave = 0; octave < LEVELS; octave++) {
    const scale = SCALE ** octave;
    const width = Math.max(1, Math.round(image.width / scale));
    const height = Math.max(1, Math.round(image.height / scale));
    const levelCanvas = typeof OffscreenCanvas !== "undefined"
      ? new OffscreenCanvas(width, height)
      : Object.assign(document.createElement("canvas"), { width, height });
    const ctx = levelCanvas.getContext("2d", { willReadFrequently: true }) as OffscreenCanvasRenderingContext2D | CanvasRenderingContext2D | null;
    if (!ctx) throw new Error("Canvas 2D is unavailable");
    ctx.imageSmoothingEnabled = true;
    ctx.imageSmoothingQuality = "high";
    ctx.drawImage(sourceCanvas, 0, 0, width, height);
    const data = ctx.getImageData(0, 0, width, height).data;
    const pixels = new Uint8Array(width * height);
    for (let i = 0, p = 0; i < data.length; i += 4, p++) pixels[p] = data[i];
    levels.push({ width, height, pixels });
  }
  return levels;
}

type Corner = { x: number; y: number; response: number; angle: number; octave: number };

function extractLevel(level: Level, octave: number, threshold: number, maxCount: number): Corner[] {
  const { width: w, height: h, pixels: im } = level;
  if (w < 32 || h < 32) return [];
  const raw: Corner[] = [];
  const at = (x: number, y: number) => im[y * w + x];
  for (let y = 3; y < h - 3; y++) for (let x = 3; x < w - 3; x++) {
    const center = at(x, y), hi = center + threshold, lo = center - threshold;
    let bright = 0, dark = 0;
    for (const i of [0, 4, 8, 12]) {
      const [dx, dy] = CIRCLE[i]; const v = at(x + dx, y + dy);
      if (v > hi) bright++; else if (v < lo) dark++;
    }
    if (bright < 3 && dark < 3) continue;
    let b = 0, d = 0, maxB = 0, maxD = 0;
    for (let i = 0; i < 25; i++) {
      const [dx, dy] = CIRCLE[i & 15]; const v = at(x + dx, y + dy);
      if (v > hi) { maxB = Math.max(maxB, ++b); d = 0; }
      else if (v < lo) { maxD = Math.max(maxD, ++d); b = 0; }
      else { b = 0; d = 0; }
    }
    if (maxB < 9 && maxD < 9) continue;
    let response = 0;
    for (const [dx, dy] of CIRCLE) response += Math.max(0, Math.abs(at(x + dx, y + dy) - center) - threshold);
    raw.push({ x, y, response, angle: 0, octave });
  }
  // FAST score-based non-maximum suppression in each 3x3 neighborhood.
  const responseMap = new Int32Array(w * h);
  for (const p of raw) responseMap[p.y * w + p.x] = p.response;
  const points = raw.filter(p => {
    // ORB's 31x31 orientation/BRIEF patch must be fully inside this pyramid level.
    if (p.x < 16 || p.y < 16 || p.x >= w - 16 || p.y >= h - 16) return false;
    for (let dy = -1; dy <= 1; dy++) for (let dx = -1; dx <= 1; dx++) {
      if ((dx || dy) && responseMap[(p.y + dy) * w + p.x + dx] > p.response) return false;
    }
    return true;
  }).sort((a, b) => b.response - a.response).slice(0, maxCount);
  for (const p of points) {
    let m10 = 0, m01 = 0;
    for (let dy = -15; dy <= 15; dy++) for (let dx = -15; dx <= 15; dx++) {
      if (dx * dx + dy * dy > 225) continue;
      const v = at(p.x + dx, p.y + dy); m10 += v * dx; m01 += v * dy;
    }
    p.angle = Math.atan2(m01, m10) * 180 / Math.PI;
  }
  return points;
}

function popcount32(value: number): number {
  let v = value >>> 0;
  v -= (v >>> 1) & 0x55555555;
  v = (v & 0x33333333) + ((v >>> 2) & 0x33333333);
  return (((v + (v >>> 4)) & 0x0f0f0f0f) * 0x01010101) >>> 24;
}

function packDescriptors32(descriptors: Uint8Array, count: number): Uint32Array {
  const words = new Uint32Array(count * DESC_WORDS);
  for (let i = 0; i < count; i++) {
    const byteBase = i * DESC_BYTES;
    const wordBase = i * DESC_WORDS;
    for (let w = 0; w < DESC_WORDS; w++) {
      const p = byteBase + w * 4;
      words[wordBase + w] = (
        descriptors[p] |
        (descriptors[p + 1] << 8) |
        (descriptors[p + 2] << 16) |
        (descriptors[p + 3] << 24)
      ) >>> 0;
    }
  }
  return words;
}

function hamming32(a: Uint32Array, ai: number, b: Uint32Array, bi: number): number {
  let sum = 0;
  const ao = ai * DESC_WORDS;
  const bo = bi * DESC_WORDS;
  for (let i = 0; i < DESC_WORDS; i++) sum += popcount32(a[ao + i] ^ b[bo + i]);
  return sum;
}

function multiply3(a: Mat3, b: Mat3): Mat3 {
  const out = new Array<number>(9).fill(0) as Mat3;
  for (let r = 0; r < 3; r++) for (let c = 0; c < 3; c++) {
    out[r * 3 + c] = a[r * 3] * b[c] + a[r * 3 + 1] * b[3 + c] + a[r * 3 + 2] * b[6 + c];
  }
  return out;
}

function transpose3(a: Mat3): Mat3 {
  return [a[0], a[3], a[6], a[1], a[4], a[7], a[2], a[5], a[8]];
}

function normalizePointSet(points: Float64Array): NormalizedPoints | undefined {
  const count = points.length / 2;
  if (count < 2) return;
  let cx = 0, cy = 0;
  for (let i = 0; i < count; i++) { cx += points[i * 2]; cy += points[i * 2 + 1]; }
  cx /= count; cy /= count;
  let meanDistance = 0;
  for (let i = 0; i < count; i++) meanDistance += Math.hypot(points[i * 2] - cx, points[i * 2 + 1] - cy);
  meanDistance /= count;
  if (!Number.isFinite(meanDistance) || meanDistance < 1e-6) return;
  const scale = Math.SQRT2 / meanDistance;
  const transform: Mat3 = [scale, 0, -scale * cx, 0, scale, -scale * cy, 0, 0, 1];
  const inverse: Mat3 = [1 / scale, 0, cx, 0, 1 / scale, cy, 0, 0, 1];
  const normalized = new Float64Array(points.length);
  for (let i = 0; i < count; i++) {
    normalized[i * 2] = (points[i * 2] - cx) * scale;
    normalized[i * 2 + 1] = (points[i * 2 + 1] - cy) * scale;
  }
  return { points: normalized, transform, inverse };
}

function collectPairPoints(q: Float32Array, t: Float32Array, pairs: MatchPair[], selected?: number[]): { q: Float64Array; t: Float64Array } {
  const n = selected?.length ?? pairs.length;
  const qp = new Float64Array(n * 2);
  const tp = new Float64Array(n * 2);
  for (let z = 0; z < n; z++) {
    const m = pairs[selected ? selected[z] : z];
    const qi = m.queryIndex * STRIDE, ti = m.trainIndex * STRIDE;
    qp[z * 2] = q[qi]; qp[z * 2 + 1] = q[qi + 1];
    tp[z * 2] = t[ti]; tp[z * 2 + 1] = t[ti + 1];
  }
  return { q: qp, t: tp };
}

function has2DSpread(points: Float64Array, minArea = 8): boolean {
  const count = points.length / 2;
  if (count < 3) return false;
  let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
  for (let i = 0; i < count; i++) {
    const x = points[i * 2], y = points[i * 2 + 1];
    minX = Math.min(minX, x); minY = Math.min(minY, y);
    maxX = Math.max(maxX, x); maxY = Math.max(maxY, y);
  }
  if ((maxX - minX) * (maxY - minY) < minArea) return false;
  let maxCross = 0;
  for (let i = 1; i + 1 < count; i++) {
    const ax = points[i * 2] - points[0], ay = points[i * 2 + 1] - points[1];
    for (let j = i + 1; j < count; j++) {
      const bx = points[j * 2] - points[0], by = points[j * 2 + 1] - points[1];
      maxCross = Math.max(maxCross, Math.abs(ax * by - ay * bx));
    }
  }
  return maxCross > 1;
}

function solveLinear(a: number[][], b: number[]): number[] | undefined {
  const n = b.length;
  for (let c = 0; c < n; c++) {
    let pivot = c;
    for (let r = c + 1; r < n; r++) if (Math.abs(a[r][c]) > Math.abs(a[pivot][c])) pivot = r;
    if (Math.abs(a[pivot][c]) < 1e-10) return;
    [a[c], a[pivot]] = [a[pivot], a[c]]; [b[c], b[pivot]] = [b[pivot], b[c]];
    const d = a[c][c];
    for (let j = c; j < n; j++) a[c][j] /= d;
    b[c] /= d;
    for (let r = 0; r < n; r++) if (r !== c) {
      const f = a[r][c];
      if (Math.abs(f) < 1e-18) continue;
      for (let j = c; j < n; j++) a[r][j] -= f * a[c][j];
      b[r] -= f * b[c];
    }
  }
  return b;
}

function fitHomography(q: Float32Array, t: Float32Array, pairs: MatchPair[], selected?: number[]): number[] | undefined {
  const n = selected?.length ?? pairs.length;
  if (n < 4) return;
  const raw = collectPairPoints(q, t, pairs, selected);
  if (!has2DSpread(raw.q) || !has2DSpread(raw.t)) return;
  const nq = normalizePointSet(raw.q), nt = normalizePointSet(raw.t);
  if (!nq || !nt) return;
  const ata = Array.from({ length: 8 }, () => Array<number>(8).fill(0));
  const atb = Array<number>(8).fill(0);
  for (let z = 0; z < n; z++) {
    const x = nq.points[z * 2], y = nq.points[z * 2 + 1];
    const u = nt.points[z * 2], v = nt.points[z * 2 + 1];
    const rows: [number[], number][] = [
      [[x, y, 1, 0, 0, 0, -x * u, -y * u], u],
      [[0, 0, 0, x, y, 1, -x * v, -y * v], v],
    ];
    for (const [row, rhs] of rows) for (let r = 0; r < 8; r++) {
      atb[r] += row[r] * rhs;
      for (let c = 0; c < 8; c++) ata[r][c] += row[r] * row[c];
    }
  }
  const h = solveLinear(ata, atb);
  if (!h) return;
  const hn: Mat3 = [h[0], h[1], h[2], h[3], h[4], h[5], h[6], h[7], 1];
  const denormalized = multiply3(multiply3(nt.inverse, hn), nq.transform);
  const scale = Math.abs(denormalized[8]) > 1e-12 ? denormalized[8] : Math.hypot(...denormalized);
  if (!Number.isFinite(scale) || Math.abs(scale) < 1e-12) return;
  return denormalized.map(v => v / scale);
}

function homographyError(h: number[], q: Float32Array, t: Float32Array, p: MatchPair): number {
  const i = p.queryIndex * STRIDE, j = p.trainIndex * STRIDE, x = q[i], y = q[i + 1];
  const w = h[6] * x + h[7] * y + h[8];
  if (Math.abs(w) < 1e-12) return Infinity;
  return Math.hypot((h[0] * x + h[1] * y + h[2]) / w - t[j], (h[3] * x + h[4] * y + h[5]) / w - t[j + 1]);
}

function sampleUnique(stateRef: { value: number }, count: number, total: number): number[] {
  const selected: number[] = [];
  while (selected.length < count) {
    stateRef.value = (Math.imul(stateRef.value, 1664525) + 1013904223) >>> 0;
    const idx = stateRef.value % total;
    if (!selected.includes(idx)) selected.push(idx);
  }
  return selected;
}

function estimateHomography(q: Float32Array, t: Float32Array, pairs: MatchPair[], threshold: number, iterations: number): { h: number[]; inliers: number; mean: number; p90: number } | undefined {
  if (pairs.length < 4) return;
  const state = { value: (0x56414e43 ^ pairs.length) >>> 0 };
  let best: number[] | undefined, bestCount = 0, bestMean = Infinity;
  for (let it = 0; it < iterations; it++) {
    const selected = sampleUnique(state, 4, pairs.length);
    const h = fitHomography(q, t, pairs, selected);
    if (!h) continue;
    let count = 0, sum = 0;
    for (const p of pairs) {
      const e = homographyError(h, q, t, p);
      if (e <= threshold) { count++; sum += e; }
    }
    const mean = count ? sum / count : Infinity;
    if (count > bestCount || (count === bestCount && mean < bestMean)) { best = h; bestCount = count; bestMean = mean; }
  }
  if (!best || bestCount < 4) return;
  const ids = pairs.map((p, i) => homographyError(best!, q, t, p) <= threshold ? i : -1).filter(i => i >= 0);
  const refined = fitHomography(q, t, pairs, ids) ?? best;
  const errors = pairs.map(p => homographyError(refined, q, t, p)).filter(e => e <= threshold).sort((a, b) => a - b);
  if (errors.length < 4) return;
  return {
    h: refined,
    inliers: errors.length,
    mean: errors.reduce((a, b) => a + b, 0) / errors.length,
    p90: errors[Math.min(errors.length - 1, Math.floor(errors.length * .9))],
  };
}

function jacobiSmallestEigenvector(matrix: number[][]): number[] | undefined {
  const n = matrix.length;
  const a = matrix.map(row => [...row]);
  const v: number[][] = Array.from({ length: n }, (_, r) => Array.from({ length: n }, (_, c) => r === c ? 1 : 0));
  for (let iter = 0; iter < 80 * n; iter++) {
    let p = 0, q = 1, max = 0;
    for (let r = 0; r < n; r++) for (let c = r + 1; c < n; c++) {
      const value = Math.abs(a[r][c]);
      if (value > max) { max = value; p = r; q = c; }
    }
    if (max < 1e-10) break;
    const app = a[p][p], aqq = a[q][q], apq = a[p][q];
    const phi = 0.5 * Math.atan2(2 * apq, aqq - app);
    const c = Math.cos(phi), s = Math.sin(phi);
    for (let k = 0; k < n; k++) {
      if (k === p || k === q) continue;
      const akp = a[k][p], akq = a[k][q];
      a[k][p] = a[p][k] = c * akp - s * akq;
      a[k][q] = a[q][k] = s * akp + c * akq;
    }
    a[p][p] = c * c * app - 2 * s * c * apq + s * s * aqq;
    a[q][q] = s * s * app + 2 * s * c * apq + c * c * aqq;
    a[p][q] = a[q][p] = 0;
    for (let k = 0; k < n; k++) {
      const vkp = v[k][p], vkq = v[k][q];
      v[k][p] = c * vkp - s * vkq;
      v[k][q] = s * vkp + c * vkq;
    }
  }
  let index = 0;
  for (let i = 1; i < n; i++) if (a[i][i] < a[index][index]) index = i;
  const out = v.map(row => row[index]);
  const norm = Math.hypot(...out);
  if (!Number.isFinite(norm) || norm < 1e-12) return;
  return out.map(x => x / norm);
}

function smallestEigenVector3(symmetric: Mat3): { vector: [number, number, number]; value: number } | undefined {
  const vec = jacobiSmallestEigenvector([
    [symmetric[0], symmetric[1], symmetric[2]],
    [symmetric[3], symmetric[4], symmetric[5]],
    [symmetric[6], symmetric[7], symmetric[8]],
  ]);
  if (!vec) return;
  const [x, y, z] = vec;
  const value = x * (symmetric[0] * x + symmetric[1] * y + symmetric[2] * z)
    + y * (symmetric[3] * x + symmetric[4] * y + symmetric[5] * z)
    + z * (symmetric[6] * x + symmetric[7] * y + symmetric[8] * z);
  return { vector: [x, y, z], value };
}

function enforceRank2(f: Mat3): Mat3 {
  // Project F to rank 2 by removing the component associated with its smallest singular value.
  const ft = transpose3(f);
  const ftf = multiply3(ft, f);
  const eig = smallestEigenVector3(ftf);
  if (!eig || eig.value < 0) return f;
  const v = eig.vector;
  const fv: [number, number, number] = [
    f[0] * v[0] + f[1] * v[1] + f[2] * v[2],
    f[3] * v[0] + f[4] * v[1] + f[5] * v[2],
    f[6] * v[0] + f[7] * v[1] + f[8] * v[2],
  ];
  const sigma = Math.hypot(...fv);
  if (sigma < 1e-10) return f;
  const u: [number, number, number] = [fv[0] / sigma, fv[1] / sigma, fv[2] / sigma];
  const out = [...f] as Mat3;
  for (let r = 0; r < 3; r++) for (let c = 0; c < 3; c++) out[r * 3 + c] -= sigma * u[r] * v[c];
  return out;
}

function fitFundamental(q: Float32Array, t: Float32Array, pairs: MatchPair[], selected?: number[]): number[] | undefined {
  const n = selected?.length ?? pairs.length;
  if (n < 8) return;
  const raw = collectPairPoints(q, t, pairs, selected);
  if (!has2DSpread(raw.q, 16) || !has2DSpread(raw.t, 16)) return;
  const nq = normalizePointSet(raw.q), nt = normalizePointSet(raw.t);
  if (!nq || !nt) return;
  const ata = Array.from({ length: 9 }, () => Array<number>(9).fill(0));
  for (let i = 0; i < n; i++) {
    const x = nq.points[i * 2], y = nq.points[i * 2 + 1];
    const u = nt.points[i * 2], v = nt.points[i * 2 + 1];
    const row = [u * x, u * y, u, v * x, v * y, v, x, y, 1];
    for (let r = 0; r < 9; r++) for (let c = r; c < 9; c++) ata[r][c] += row[r] * row[c];
  }
  for (let r = 0; r < 9; r++) for (let c = 0; c < r; c++) ata[r][c] = ata[c][r];
  const vector = jacobiSmallestEigenvector(ata);
  if (!vector) return;
  let fn = enforceRank2(vector as Mat3);
  // x2^T F x1 = 0, therefore denormalize as T2^T F T1.
  fn = multiply3(multiply3(transpose3(nt.transform), fn), nq.transform);
  const norm = Math.hypot(...fn);
  if (!Number.isFinite(norm) || norm < 1e-12) return;
  return fn.map(v => v / norm);
}

function sampsonError(f: number[], q: Float32Array, t: Float32Array, p: MatchPair): number {
  const qi = p.queryIndex * STRIDE, ti = p.trainIndex * STRIDE;
  const x = q[qi], y = q[qi + 1], u = t[ti], v = t[ti + 1];
  const fx0 = f[0] * x + f[1] * y + f[2];
  const fx1 = f[3] * x + f[4] * y + f[5];
  const ftx0 = f[0] * u + f[3] * v + f[6];
  const ftx1 = f[1] * u + f[4] * v + f[7];
  const residual = u * fx0 + v * fx1 + f[6] * x + f[7] * y + f[8];
  const denom = fx0 * fx0 + fx1 * fx1 + ftx0 * ftx0 + ftx1 * ftx1;
  return denom > 1e-12 ? (residual * residual) / denom : Infinity;
}

function estimateFundamental(q: Float32Array, t: Float32Array, pairs: MatchPair[], thresholdPx: number, iterations: number): { f: number[]; inliers: number; mean: number; p90: number } | undefined {
  if (pairs.length < 8) return;
  const state = { value: (0x46554e44 ^ pairs.length) >>> 0 };
  const thresholdSq = thresholdPx * thresholdPx;
  let best: number[] | undefined, bestCount = 0, bestMean = Infinity;
  for (let it = 0; it < iterations; it++) {
    const selected = sampleUnique(state, 8, pairs.length);
    const f = fitFundamental(q, t, pairs, selected);
    if (!f) continue;
    let count = 0, sum = 0;
    for (const p of pairs) {
      const e = sampsonError(f, q, t, p);
      if (e <= thresholdSq) { count++; sum += e; }
    }
    const mean = count ? sum / count : Infinity;
    if (count > bestCount || (count === bestCount && mean < bestMean)) { best = f; bestCount = count; bestMean = mean; }
  }
  if (!best || bestCount < 8) return;
  const ids = pairs.map((p, i) => sampsonError(best!, q, t, p) <= thresholdSq ? i : -1).filter(i => i >= 0);
  const refined = fitFundamental(q, t, pairs, ids) ?? best;
  const errors = pairs.map(p => sampsonError(refined, q, t, p)).filter(e => e <= thresholdSq).sort((a, b) => a - b);
  if (errors.length < 8) return;
  return {
    f: refined,
    inliers: errors.length,
    mean: errors.reduce((a, b) => a + b, 0) / errors.length,
    p90: errors[Math.min(errors.length - 1, Math.floor(errors.length * .9))],
  };
}

export class VisualAnchorJS {
  extract(image: ImageData, options: { maxFeatures?: number; fastThreshold?: number; equalize?: boolean } = {}): FeatureSet {
    const maxFeatures = options.maxFeatures ?? 700;
    const levels = buildPyramid(image, options.equalize ?? true);
    const levelBudgets: number[] = [];
    let assigned = 0;
    for (let octave = 0; octave < LEVELS; octave++) {
      const remaining = maxFeatures - assigned;
      const desired = octave === LEVELS - 1
        ? remaining
        : Math.max(1, Math.min(remaining, Math.round(maxFeatures * (1 - 1 / SCALE) / (1 - (1 / SCALE) ** LEVELS) * (1 / SCALE) ** octave)));
      levelBudgets.push(desired);
      assigned += desired;
    }
    const points = levels.flatMap((level, octave) => extractLevel(level, octave, Math.max(5, Math.round((options.fastThreshold ?? 18) / SCALE ** octave)), levelBudgets[octave]));
    const keypoints = new Float32Array(points.length * STRIDE), descriptors = new Uint8Array(points.length * DESC_BYTES);
    for (let i = 0; i < points.length; i++) {
      const p = points[i], scale = SCALE ** p.octave, base = i * STRIDE, level = levels[p.octave];
      keypoints.set([p.x * scale, p.y * scale, 31 * scale, p.angle, p.response, p.octave], base);
      const angle = p.angle * Math.PI / 180, c = Math.cos(angle), s = Math.sin(angle), desc = i * DESC_BYTES;
      for (let bit = 0; bit < 256; bit++) {
        const [x1, y1, x2, y2] = ORB_PATTERN[bit];
        const ax = Math.round(p.x + c * x1 - s * y1), ay = Math.round(p.y + s * x1 + c * y1);
        const bx = Math.round(p.x + c * x2 - s * y2), by = Math.round(p.y + s * x2 + c * y2);
        // extractLevel already rejects keypoints whose complete BRIEF patch would leave the image.
        const av = level.pixels[ay * level.width + ax];
        const bv = level.pixels[by * level.width + bx];
        if (av < bv) descriptors[desc + (bit >>> 3)] |= 1 << (bit & 7);
      }
    }
    return { width: image.width, height: image.height, count: points.length, keypoints, descriptors };
  }

  match(query: FeatureSet, train: FeatureSet, options: {
    maxDistance?: number;
    ratio?: number;
    crossCheck?: boolean;
    ransacThreshold?: number;
    ransacIterations?: number;
    fundamentalThreshold?: number;
    fundamentalIterations?: number;
    maxReturnedMatches?: number;
  } = {}): MatchGeometry {
    const maxDistance = options.maxDistance ?? 72;
    const ratio = options.ratio ?? .78;
    const crossCheck = options.crossCheck ?? true;
    const queryWords = packDescriptors32(query.descriptors, query.count);
    const trainWords = packDescriptors32(train.descriptors, train.count);

    const queryBestDistance = new Uint16Array(query.count); queryBestDistance.fill(0xffff);
    const querySecondDistance = new Uint16Array(query.count); querySecondDistance.fill(0xffff);
    const queryBestIndex = new Int32Array(query.count); queryBestIndex.fill(-1);
    const trainBestDistance = crossCheck ? new Uint16Array(train.count) : undefined;
    const trainBestIndex = crossCheck ? new Int32Array(train.count) : undefined;
    trainBestDistance?.fill(0xffff); trainBestIndex?.fill(-1);

    // One descriptor-distance pass updates query best/second and train best simultaneously.
    for (let qi = 0; qi < query.count; qi++) {
      for (let ti = 0; ti < train.count; ti++) {
        const d = hamming32(queryWords, qi, trainWords, ti);
        if (d < queryBestDistance[qi]) {
          querySecondDistance[qi] = queryBestDistance[qi];
          queryBestDistance[qi] = d;
          queryBestIndex[qi] = ti;
        } else if (d < querySecondDistance[qi]) {
          querySecondDistance[qi] = d;
        }
        if (trainBestDistance && trainBestIndex && d < trainBestDistance[ti]) {
          trainBestDistance[ti] = d;
          trainBestIndex[ti] = qi;
        }
      }
    }

    const pairs: MatchPair[] = [];
    for (let qi = 0; qi < query.count; qi++) {
      const best = queryBestDistance[qi];
      const second = querySecondDistance[qi];
      const ti = queryBestIndex[qi];
      if (ti < 0 || best > maxDistance) continue;
      if (second !== 0xffff && best >= ratio * second) continue;
      if (trainBestIndex && trainBestIndex[ti] !== qi) continue;
      pairs.push({ queryIndex: qi, trainIndex: ti, distance: best });
    }

    const returnedPairs = pairs.slice(0, options.maxReturnedMatches ?? query.count);
    const avgDistance = pairs.length ? pairs.reduce((s, p) => s + p.distance, 0) / pairs.length : 0;
    const hFit = estimateHomography(query.keypoints, train.keypoints, pairs, options.ransacThreshold ?? 4, options.ransacIterations ?? 220);
    const fFit = estimateFundamental(query.keypoints, train.keypoints, pairs, options.fundamentalThreshold ?? 2.5, options.fundamentalIterations ?? 180);

    const hInliers = hFit?.inliers ?? 0;
    const fInliers = fFit?.inliers ?? 0;
    const useFundamental = fInliers > hInliers + Math.max(3, Math.round(pairs.length * 0.04));
    const inliers = useFundamental ? fInliers : hInliers;
    // Preserve homography reprojection error for existing scoring/UI; fall back to Sampson distance only if H fails.
    const meanError = hFit?.mean ?? Math.sqrt(fFit?.mean ?? 0);
    // Keep p90Error homography-specific because registration/replay code uses it as a non-planarity signal.
    const p90Error = hFit?.p90 ?? 0;

    return {
      matches: pairs.length,
      inliers,
      inlierRatio: pairs.length ? inliers / pairs.length : 0,
      meanError,
      p90Error,
      avgDistance,
      homography: hFit?.h ?? [],
      pairs: returnedPairs,
      homographyInliers: hInliers,
      homographyInlierRatio: pairs.length ? hInliers / pairs.length : 0,
      homographyMeanError: hFit?.mean ?? 0,
      homographyP90Error: hFit?.p90 ?? 0,
      fundamental: fFit?.f ?? [],
      fundamentalInliers: fInliers,
      fundamentalInlierRatio: pairs.length ? fInliers / pairs.length : 0,
      meanSampsonError: fFit?.mean ?? 0,
      p90SampsonError: fFit?.p90 ?? 0,
    };
  }
}

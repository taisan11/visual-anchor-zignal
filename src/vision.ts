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
  matches: number; inliers: number; inlierRatio: number; meanError: number;
  p90Error: number; avgDistance: number; homography: number[]; pairs: MatchPair[];
};

const STRIDE = 6;
const DESC_BYTES = 32;
const SCALE = 1.2;
const LEVELS = 8;
const CIRCLE: readonly (readonly [number, number])[] = [
  [0,-3],[1,-3],[2,-2],[3,-1],[3,0],[3,1],[2,2],[1,3],
  [0,3],[-1,3],[-2,2],[-3,1],[-3,0],[-3,-1],[-2,-2],[-1,-3],
];

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
    for (let dy = -1; dy <= 1; dy++) for (let dx = -1; dx <= 1; dx++) {
      if ((dx || dy) && responseMap[(p.y + dy) * w + p.x + dx] > p.response) return false;
    }
    return true;
  }).sort((a, b) => b.response - a.response).slice(0, maxCount);
  for (const p of points) {
    if (p.x < 15 || p.y < 15 || p.x >= w - 15 || p.y >= h - 15) continue;
    let m10 = 0, m01 = 0;
    for (let dy = -15; dy <= 15; dy++) for (let dx = -15; dx <= 15; dx++) {
      if (dx * dx + dy * dy > 225) continue;
      const v = at(p.x + dx, p.y + dy); m10 += v * dx; m01 += v * dy;
    }
    p.angle = Math.atan2(m01, m10) * 180 / Math.PI;
  }
  return points;
}

function hamming(a: Uint8Array, ai: number, b: Uint8Array, bi: number): number {
  let sum = 0;
  for (let i = 0; i < DESC_BYTES; i++) {
    let v = a[ai * DESC_BYTES + i] ^ b[bi * DESC_BYTES + i];
    v -= (v >>> 1) & 0x55; v = (v & 0x33) + ((v >>> 2) & 0x33);
    sum += (v + (v >>> 4)) & 0x0f;
  }
  return sum;
}

function solve8(a: number[][], b: number[]): number[] | undefined {
  for (let c = 0; c < 8; c++) {
    let pivot = c;
    for (let r = c + 1; r < 8; r++) if (Math.abs(a[r][c]) > Math.abs(a[pivot][c])) pivot = r;
    if (Math.abs(a[pivot][c]) < 1e-9) return;
    [a[c], a[pivot]] = [a[pivot], a[c]]; [b[c], b[pivot]] = [b[pivot], b[c]];
    const d = a[c][c]; for (let j = c; j < 8; j++) a[c][j] /= d; b[c] /= d;
    for (let r = 0; r < 8; r++) if (r !== c) { const f = a[r][c]; for (let j = c; j < 8; j++) a[r][j] -= f * a[c][j]; b[r] -= f * b[c]; }
  }
  return b;
}

function fitHomography(q: Float32Array, t: Float32Array, pairs: MatchPair[], selected?: number[]): number[] | undefined {
  const ata = Array.from({ length: 8 }, () => Array<number>(8).fill(0));
  const atb = Array<number>(8).fill(0);
  const n = selected?.length ?? pairs.length;
  if (n < 4) return;
  for (let z = 0; z < n; z++) {
    const m = pairs[selected ? selected[z] : z], qi = m.queryIndex * STRIDE, ti = m.trainIndex * STRIDE;
    const x = q[qi], y = q[qi + 1], u = t[ti], v = t[ti + 1];
    const rows: [number[], number][] = [[[x,y,1,0,0,0,-x*u,-y*u],u], [[0,0,0,x,y,1,-x*v,-y*v],v]];
    for (const [row, rhs] of rows) for (let r = 0; r < 8; r++) { atb[r] += row[r] * rhs; for (let c = 0; c < 8; c++) ata[r][c] += row[r] * row[c]; }
  }
  const h = solve8(ata, atb);
  return h ? [...h, 1] : undefined;
}

function error(h: number[], q: Float32Array, t: Float32Array, p: MatchPair): number {
  const i = p.queryIndex * STRIDE, j = p.trainIndex * STRIDE, x = q[i], y = q[i + 1];
  const w = h[6] * x + h[7] * y + 1;
  if (Math.abs(w) < 1e-12) return Infinity;
  return Math.hypot((h[0]*x+h[1]*y+h[2])/w-t[j], (h[3]*x+h[4]*y+h[5])/w-t[j+1]);
}

function estimate(q: Float32Array, t: Float32Array, pairs: MatchPair[], threshold: number, iterations: number): { h: number[]; inliers: number; mean: number; p90: number } | undefined {
  if (pairs.length < 4) return;
  let state = (0x56414e43 ^ pairs.length) >>> 0, best: number[] | undefined, bestCount = 0, bestMean = Infinity;
  for (let it = 0; it < iterations; it++) {
    const selected: number[] = [];
    while (selected.length < 4) {
      state = (Math.imul(state, 1664525) + 1013904223) >>> 0;
      const idx = state % pairs.length;
      if (!selected.includes(idx)) selected.push(idx);
    }
    const h = fitHomography(q, t, pairs, selected); if (!h) continue;
    let count = 0, sum = 0;
    for (const p of pairs) { const e = error(h, q, t, p); if (e <= threshold) { count++; sum += e; } }
    const mean = count ? sum / count : Infinity;
    if (count > bestCount || (count === bestCount && mean < bestMean)) { best = h; bestCount = count; bestMean = mean; }
  }
  if (!best || bestCount < 4) return;
  const ids = pairs.map((p, i) => error(best!, q, t, p) <= threshold ? i : -1).filter(i => i >= 0);
  const refined = fitHomography(q, t, pairs, ids) ?? best;
  const errors = pairs.map(p => error(refined, q, t, p)).filter(e => e <= threshold).sort((a,b) => a-b);
  if (errors.length < 4) return;
  return { h: refined, inliers: errors.length, mean: errors.reduce((a,b) => a+b, 0) / errors.length, p90: errors[Math.min(errors.length - 1, Math.floor(errors.length * .9))] };
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
        const [x1,y1,x2,y2] = ORB_PATTERN[bit];
        const ax = Math.round(p.x + c*x1 - s*y1), ay = Math.round(p.y + s*x1 + c*y1);
        const bx = Math.round(p.x + c*x2 - s*y2), by = Math.round(p.y + s*x2 + c*y2);
        const av = ax < 0 || ay < 0 || ax >= level.width || ay >= level.height ? 0 : level.pixels[ay*level.width+ax];
        const bv = bx < 0 || by < 0 || bx >= level.width || by >= level.height ? 0 : level.pixels[by*level.width+bx];
        if (av < bv) descriptors[desc + (bit >>> 3)] |= 1 << (bit & 7);
      }
    }
    return { width: image.width, height: image.height, count: points.length, keypoints, descriptors };
  }

  match(query: FeatureSet, train: FeatureSet, options: { maxDistance?: number; ratio?: number; crossCheck?: boolean; ransacThreshold?: number; ransacIterations?: number; maxReturnedMatches?: number } = {}): MatchGeometry {
    const maxDistance = options.maxDistance ?? 72, ratio = options.ratio ?? .78, crossCheck = options.crossCheck ?? true;
    const reverse = crossCheck ? new Int32Array(train.count).fill(-1) : undefined;
    if (reverse) for (let ti = 0; ti < train.count; ti++) { let best = Infinity, idx = -1; for (let qi = 0; qi < query.count; qi++) { const d = hamming(train.descriptors, ti, query.descriptors, qi); if (d < best) { best = d; idx = qi; } } reverse[ti] = idx; }
    const pairs: MatchPair[] = [];
    for (let qi = 0; qi < query.count; qi++) {
      let best = Infinity, second = Infinity, idx = -1;
      for (let ti = 0; ti < train.count; ti++) { const d = hamming(query.descriptors, qi, train.descriptors, ti); if (d < best) { second = best; best = d; idx = ti; } else if (d < second) second = d; }
      if (best > maxDistance || (second < Infinity && best >= ratio * second) || (reverse && reverse[idx] !== qi)) continue;
      pairs.push({ queryIndex: qi, trainIndex: idx, distance: best });
    }
    const returnedPairs = pairs.slice(0, options.maxReturnedMatches ?? query.count);
    const avgDistance = pairs.length ? pairs.reduce((s,p)=>s+p.distance,0) / pairs.length : 0;
    const fit = estimate(query.keypoints, train.keypoints, pairs, options.ransacThreshold ?? 4, options.ransacIterations ?? 220);
    return { matches: pairs.length, inliers: fit?.inliers ?? 0, inlierRatio: fit && pairs.length ? fit.inliers / pairs.length : 0, meanError: fit?.mean ?? 0, p90Error: fit?.p90 ?? 0, avgDistance, homography: fit?.h ?? [], pairs: returnedPairs };
  }
}

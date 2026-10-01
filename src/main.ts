//@ts-ignore
import "./styles.css";
import {
  createAnchor,
  pruneFeatureSetSpatial,
  selectRegistrationViews,
  type RegistrationCandidate,
  type VisualAnchor,
  verifyBurst,
  type VerificationResult,
} from "./anchor";
import {
  anchorBinaryFromBase64,
  anchorBinaryToBase64,
  BINARY_EXTENSION,
  BINARY_MIME,
  decodeAnchorBinary,
  encodeAnchorBinary,
} from "./binary";
import {
  orientationAngleDeg,
  RegistrationOrientationSensor,
  type OrientationSample,
} from "./orientation";
import { VisualAnchorJS, type FeatureSet } from "./vision";

const PROCESSING_WIDTH = 640;
const REGISTER_CANDIDATES = 11;
const REGISTER_VIEWS = 5;
const REGISTER_INTERVAL_MS = 240;
const REGISTER_OUTPUT_FEATURES = 520;
const VERIFY_FRAMES = 3;
const SCALE_ASSIST_FACTOR = 1.2;
const SCALE_ASSIST_MIN_SCORE = 0.12;
const STORAGE_KEY = "visual-anchor-zignal-anchor-binary-v1";

const app = document.querySelector<HTMLDivElement>("#app");
if (!app) throw new Error("#app not found");

app.innerHTML = `
  <section class="shell">
    <header>
      <div>
        <div class="eyebrow">JavaScript · Web Platform APIs</div>
        <h1>Visual Anchor</h1>
        <p>風景をQRコード代わりにする軽量なmulti-view feature anchor。</p>
      </div>
      <div class="status-stack">
        <div id="visionStatus" class="pill">JavaScript ready</div>
        <div id="orientationStatus" class="pill sensor">Orientation: 登録時のみ</div>
      </div>
    </header>

    <section class="camera-card">
      <video id="video" playsinline muted></video>
      <canvas id="canvas"></canvas>
      <div class="camera-overlay">
        <span id="guideMain">対象を中央に入れる</span>
        <span id="guideMotion">登録時は向きを保って横に動かす</span>
      </div>
    </section>

    <section class="controls">
      <button id="cameraBtn">カメラ開始</button>
      <button id="registerBtn" disabled>地点を登録</button>
      <button id="verifyBtn" disabled>この地点を照合</button>
      <button id="exportBtn" disabled>Anchor Binaryを保存</button>
      <label class="file-button">
        Anchor Binaryを読み込む
        <input id="importInput" type="file" accept="${BINARY_MIME},application/octet-stream,${BINARY_EXTENSION}" />
      </label>
    </section>

    <section class="grid">
      <article class="panel">
        <h2>Anchor</h2>
        <div id="anchorInfo" class="metric-list muted">未登録</div>
      </article>
      <article class="panel">
        <h2>照合結果</h2>
        <div id="result" class="result empty">まだ照合していません</div>
      </article>
    </section>

    <section class="panel notes">
      <h2>方式</h2>
      <p>登録時だけDeviceOrientationを使い、多めに撮った候補から「端末を回しただけ」のviewを避け、横移動による視差と幾何整合性が取りやすい5 viewを選びます。姿勢値そのものは照合入力に使いません。</p>
      <p>各viewをRGBA→grayscale→histogram equalization→ORBに変換し、保存時は座標とdescriptorを固定長のバイナリへ直接格納します。JSON/Base64はファイル形式には使いません。</p>
      <p>通常の照合で失敗し、ある程度の類似が残っている場合だけ、1 ORB octave相当の1.2× / 0.83×再サンプリングで前後移動による画角差を補助します。判定閾値自体は緩めません。</p>
      <p>照合は画像だけで、JavaScriptのHamming + Lowe ratio/cross-check → Homography/Fundamental RANSACを実行します。</p>
      <p class="warning">「平面表示の写真」検出は短いburstの視差を使うヒューリスティックで、セキュリティ上の完全なliveness証明ではありません。</p>
    </section>
  </section>
`;

const video = document.querySelector<HTMLVideoElement>("#video")!;
const canvas = document.querySelector<HTMLCanvasElement>("#canvas")!;
const ctx = canvas.getContext("2d", { willReadFrequently: true })!;
const cameraBtn = document.querySelector<HTMLButtonElement>("#cameraBtn")!;
const registerBtn = document.querySelector<HTMLButtonElement>("#registerBtn")!;
const verifyBtn = document.querySelector<HTMLButtonElement>("#verifyBtn")!;
const exportBtn = document.querySelector<HTMLButtonElement>("#exportBtn")!;
const importInput = document.querySelector<HTMLInputElement>("#importInput")!;
const visionStatus = document.querySelector<HTMLDivElement>("#visionStatus")!;
const orientationStatus = document.querySelector<HTMLDivElement>("#orientationStatus")!;
const anchorInfo = document.querySelector<HTMLDivElement>("#anchorInfo")!;
const result = document.querySelector<HTMLDivElement>("#result")!;
const guideMain = document.querySelector<HTMLSpanElement>("#guideMain")!;
const guideMotion = document.querySelector<HTMLSpanElement>("#guideMotion")!;

const vision = new VisualAnchorJS();
let stream: MediaStream | null = null;
let anchor: VisualAnchor | null = null;

function fmt(n: number, digits = 2): string {
  return Number.isFinite(n) ? n.toFixed(digits) : "-";
}

function bytes(n: number): string {
  if (n < 1024) return `${n} B`;
  if (n < 1024 ** 2) return `${(n / 1024).toFixed(1)} KiB`;
  return `${(n / 1024 ** 2).toFixed(2)} MiB`;
}

function updateAnchorInfo(): void {
  if (!anchor) {
    anchorInfo.textContent = "未登録";
    exportBtn.disabled = true;
    return;
  }
  const binarySize = encodeAnchorBinary(anchor).byteLength;
  const features = anchor.views.reduce((sum, v) => sum + v.count, 0);
  const registration = anchor.registration;
  const sensorLabel = registration.orientationAssisted ? "used for registration" : "not used / fallback";
  const candidateLabel = registration.candidateViews
    ? `${registration.candidateViews} → ${anchor.views.length}`
    : `${anchor.views.length}`;

  anchorInfo.innerHTML = `
    <div><span>format</span><strong>binary v1</strong></div>
    <div><span>candidate → views</span><strong>${candidateLabel}</strong></div>
    <div><span>features</span><strong>${features}</strong></div>
    <div><span>binary size</span><strong>${bytes(binarySize)}</strong></div>
    <div><span>registration sensor</span><strong>${sensorLabel}</strong></div>
    <div><span>orientation span</span><strong>${fmt(registration.orientationSpanDeg ?? 0, 1)}°</strong></div>
    <div><span>registration geometry</span><strong>${fmt(registration.medianPairScore * 100, 0)}%</strong></div>
    <div><span>parallax signature</span><strong>${fmt((registration.medianParallaxSignature ?? 0) * 1000, 2)}</strong></div>
    <div><span>created</span><strong>${new Date(anchor.createdAt).toLocaleString()}</strong></div>
  `;
  exportBtn.disabled = false;
}

function loadSavedAnchor(): void {
  const encoded = localStorage.getItem(STORAGE_KEY);
  if (!encoded) return;
  try {
    anchor = decodeAnchorBinary(anchorBinaryFromBase64(encoded));
    updateAnchorInfo();
  } catch {
    localStorage.removeItem(STORAGE_KEY);
  }
}

function persistAnchor(value: VisualAnchor): Uint8Array {
  const binary = encodeAnchorBinary(value);
  localStorage.setItem(STORAGE_KEY, anchorBinaryToBase64(binary));
  return binary;
}

async function startCamera(): Promise<void> {
  if (stream) return;
  stream = await navigator.mediaDevices.getUserMedia({
    video: {
      facingMode: { ideal: "environment" },
      width: { ideal: 1280 },
      height: { ideal: 720 },
    },
    audio: false,
  });
  video.srcObject = stream;
  await video.play();
  cameraBtn.textContent = "カメラON";
  cameraBtn.disabled = true;
  registerBtn.disabled = false;
  verifyBtn.disabled = false;
}

function captureFrame(): ImageData {
  if (!video.videoWidth || !video.videoHeight) throw new Error("camera is not ready");
  const scale = Math.min(1, PROCESSING_WIDTH / video.videoWidth);
  const width = Math.max(1, Math.round(video.videoWidth * scale));
  const height = Math.max(1, Math.round(video.videoHeight * scale));
  canvas.width = width;
  canvas.height = height;
  ctx.drawImage(video, 0, 0, width, height);
  return ctx.getImageData(0, 0, width, height);
}

function resizeImageData(image: ImageData, scale: number): ImageData {
  if (Math.abs(scale - 1) < 1e-6) return image;
  const source = document.createElement("canvas");
  source.width = image.width;
  source.height = image.height;
  const sourceContext = source.getContext("2d", { willReadFrequently: true });
  if (!sourceContext) throw new Error("Canvas 2D is unavailable");
  sourceContext.putImageData(image, 0, 0);

  const target = document.createElement("canvas");
  target.width = Math.max(1, Math.round(image.width * scale));
  target.height = Math.max(1, Math.round(image.height * scale));
  const targetContext = target.getContext("2d", { willReadFrequently: true });
  if (!targetContext) throw new Error("Canvas 2D is unavailable");
  targetContext.imageSmoothingEnabled = true;
  targetContext.imageSmoothingQuality = "high";
  targetContext.drawImage(source, 0, 0, target.width, target.height);
  return targetContext.getImageData(0, 0, target.width, target.height);
}

function sleep(ms: number): Promise<void> {
  return new Promise(resolve => setTimeout(resolve, ms));
}

async function captureBurstImages(count: number, intervalMs: number): Promise<ImageData[]> {
  const images: ImageData[] = [];
  for (let i = 0; i < count; i++) {
    images.push(captureFrame());
    if (i + 1 < count) await sleep(intervalMs);
  }
  return images;
}

function extractBurstAtScale(images: ImageData[], scale: number): FeatureSet[] {
  return images.map(image => {
    const scaled = resizeImageData(image, scale);
    const features = vision.extract(scaled, {
      maxFeatures: scale === 1 ? 700 : 660,
      fastThreshold: 18,
      equalize: true,
    });
    if (features.count < 60) {
      throw new Error(`特徴点が少なすぎます (${features.count})。模様のある対象を入れてください。`);
    }
    return features;
  });
}

function betterVerification(current: VerificationResult, candidate: VerificationResult): VerificationResult {
  if (candidate.ok !== current.ok) return candidate.ok ? candidate : current;
  if (candidate.score !== current.score) return candidate.score > current.score ? candidate : current;
  return candidate.geometry.inliers > current.geometry.inliers ? candidate : current;
}

function updateRegistrationGuide(reference?: OrientationSample, current?: OrientationSample): void {
  const angle = orientationAngleDeg(reference, current);
  if (angle == null) {
    guideMotion.textContent = "向きを保ったまま、ゆっくり横に移動";
    return;
  }
  if (angle > 14) {
    guideMotion.textContent = `回しすぎ (${fmt(angle, 0)}°) — 向きを戻して横移動`;
  } else if (angle > 8) {
    guideMotion.textContent = `回転 ${fmt(angle, 0)}° — これ以上回さず横へ`;
  } else {
    guideMotion.textContent = `回転 ${fmt(angle, 0)}° — その向きのまま横へ`;
  }
}

async function captureRegistrationCandidates(
  sensor: RegistrationOrientationSensor,
): Promise<RegistrationCandidate[]> {
  const candidates: RegistrationCandidate[] = [];
  let referenceOrientation: OrientationSample | undefined;

  for (let i = 0; i < REGISTER_CANDIDATES; i++) {
    const image = captureFrame();
    const features = vision.extract(image, { maxFeatures: 820, fastThreshold: 18, equalize: true });
    if (features.count < 70) {
      throw new Error(`登録用特徴点が少なすぎます (${features.count})。模様のある対象を増やしてください。`);
    }
    const orientation = sensor.sample();
    referenceOrientation ??= orientation;
    candidates.push({ features, orientation });
    updateRegistrationGuide(referenceOrientation, orientation);
    result.innerHTML = `<strong>登録候補 ${i + 1}/${REGISTER_CANDIDATES}</strong><span>${features.count} features · 端末の向きを保って横に移動</span>`;
    if (i + 1 < REGISTER_CANDIDATES) await sleep(REGISTER_INTERVAL_MS);
  }
  return candidates;
}

async function register(): Promise<void> {
  registerBtn.disabled = true;
  verifyBtn.disabled = true;
  result.className = "result empty";
  result.textContent = "登録センサーを準備中…";
  guideMain.textContent = "登録候補を多めに取得して最適なviewを選びます";

  // Sensor lifecycle exists only inside registration. verify() never starts or reads it.
  const orientationSensor = new RegistrationOrientationSensor();
  let orientationAvailable = false;
  try {
    orientationAvailable = await orientationSensor.start();
    orientationStatus.textContent = orientationAvailable
      ? "Orientation: 登録中のみON"
      : "Orientation: fallback";
    orientationStatus.classList.toggle("ready", orientationAvailable);

    result.textContent = `${REGISTER_CANDIDATES}候補を取得中… 向きを変えずにゆっくり横へ移動してください`;
    const candidates = await captureRegistrationCandidates(orientationSensor);
    orientationSensor.stop();

    result.textContent = "候補viewを幾何評価して最適化中…";
    const selection = selectRegistrationViews(vision, candidates, REGISTER_VIEWS);
    const inputFeatureCount = candidates.reduce((sum, c) => sum + c.features.count, 0);
    const optimizedFrames = selection.frames.map(frame =>
      pruneFeatureSetSpatial(frame, REGISTER_OUTPUT_FEATURES),
    );
    const outputFeatureCount = optimizedFrames.reduce((sum, frame) => sum + frame.count, 0);

    anchor = createAnchor(vision, optimizedFrames, PROCESSING_WIDTH, {
      orientationAssisted: orientationAvailable && selection.orientationAssisted,
      candidateViews: candidates.length,
      selectedCandidateIndices: selection.indices,
      orientationSpanDeg: selection.orientationSpanDeg,
      meanOrientationStepDeg: selection.meanOrientationStepDeg,
      inputFeatureCount,
      outputFeatureCount,
    });
    const binary = persistAnchor(anchor);
    updateAnchorInfo();

    orientationStatus.textContent = orientationAvailable
      ? "Orientation: 停止済み (登録時のみ使用)"
      : "Orientation: 未使用 (fallback)";
    orientationStatus.classList.remove("ready");
    result.className = "result pass";
    result.innerHTML = `
      <strong>登録完了</strong>
      <span>candidate ${candidates.length} → view ${selection.indices.map(i => i + 1).join(" / ")}</span>
      <span>${inputFeatureCount} → ${outputFeatureCount} features · ${bytes(binary.byteLength)}</span>
    `;
  } catch (error) {
    result.className = "result fail";
    result.textContent = error instanceof Error ? error.message : String(error);
  } finally {
    orientationSensor.stop();
    orientationStatus.classList.remove("ready");
    if (!orientationStatus.textContent?.includes("停止済み") && !orientationStatus.textContent?.includes("未使用")) {
      orientationStatus.textContent = "Orientation: 登録時のみ";
    }
    guideMain.textContent = "対象を中央に入れる";
    guideMotion.textContent = "登録時は向きを保って横に動かす";
    registerBtn.disabled = false;
    verifyBtn.disabled = false;
  }
}

async function verify(): Promise<void> {
  if (!anchor) {
    result.className = "result fail";
    result.textContent = "先に地点を登録するかAnchor Binaryを読み込んでください";
    return;
  }
  registerBtn.disabled = true;
  verifyBtn.disabled = true;
  result.className = "result empty";
  result.textContent = "3 frameを画像だけで照合中… 少し横に動かしてください";
  orientationStatus.textContent = "Orientation: OFF (照合では不使用)";
  try {
    // Deliberately no DeviceOrientation access here. Scale assist is also image-only.
    const images = await captureBurstImages(VERIFY_FRAMES, 230);
    let usedScale = 1;
    let verification = verifyBurst(vision, anchor, extractBurstAtScale(images, 1));

    if (!verification.ok && verification.score >= SCALE_ASSIST_MIN_SCORE) {
      result.textContent = "画角差を検出中… 1 octave分だけスケール補助を試しています";
      const fartherScale = SCALE_ASSIST_FACTOR;
      const farther = verifyBurst(vision, anchor, extractBurstAtScale(images, fartherScale));
      const afterFarther = betterVerification(verification, farther);
      if (afterFarther !== verification) {
        verification = afterFarther;
        usedScale = fartherScale;
      }

      if (!verification.ok) {
        const nearerScale = 1 / SCALE_ASSIST_FACTOR;
        const nearer = verifyBurst(vision, anchor, extractBurstAtScale(images, nearerScale));
        const afterNearer = betterVerification(verification, nearer);
        if (afterNearer !== verification) {
          verification = afterNearer;
          usedScale = nearerScale;
        }
      }
    }

    const v = verification;
    const scaleLabel = Math.abs(usedScale - 1) < 1e-6 ? "off" : `${fmt(usedScale, 2)}×`;
    result.className = `result ${v.ok ? "pass" : "fail"}`;
    result.innerHTML = `
      <div class="verdict">${v.ok ? "MATCH" : "NO MATCH"}</div>
      <div class="score">${fmt(v.score * 100, 0)}%</div>
      <div class="stats">
        <span>best view <b>${v.bestView + 1}</b></span>
        <span>matches <b>${v.geometry.matches}</b></span>
        <span>RANSAC inliers <b>${v.geometry.inliers}</b></span>
        <span>inlier ratio <b>${fmt(v.geometry.inlierRatio * 100, 0)}%</b></span>
        <span>mean geometry error <b>${fmt(v.geometry.meanError)} px</b></span>
        <span>ORB distance <b>${fmt(v.geometry.avgDistance, 1)}</b></span>
        <span>scale assist <b>${scaleLabel}</b></span>
      </div>
      <div class="risk ${v.planarReplayRisk ? "hot" : ""}">
        planar replay heuristic: <b>${v.planarReplayRisk ? "suspicious" : "not detected"}</b>
      </div>
    `;
  } catch (error) {
    result.className = "result fail";
    result.textContent = error instanceof Error ? error.message : String(error);
  } finally {
    orientationStatus.textContent = "Orientation: 登録時のみ";
    registerBtn.disabled = false;
    verifyBtn.disabled = false;
  }
}

cameraBtn.addEventListener("click", () => {
  void startCamera().catch(error => {
    result.className = "result fail";
    result.textContent = error instanceof Error ? error.message : String(error);
  });
});
registerBtn.addEventListener("click", () => void register());
verifyBtn.addEventListener("click", () => void verify());

exportBtn.addEventListener("click", () => {
  if (!anchor) return;
  const binary = encodeAnchorBinary(anchor);
  const arrayBuffer = binary.buffer.slice(binary.byteOffset, binary.byteOffset + binary.byteLength) as ArrayBuffer;
  const blob = new Blob([arrayBuffer], { type: BINARY_MIME });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = `visual-anchor-${Date.now()}${BINARY_EXTENSION}`;
  a.click();
  URL.revokeObjectURL(url);
});

importInput.addEventListener("change", async () => {
  const file = importInput.files?.[0];
  if (!file) return;
  try {
    anchor = decodeAnchorBinary(await file.arrayBuffer());
    persistAnchor(anchor);
    updateAnchorInfo();
    result.className = "result pass";
    result.textContent = "Anchor Binaryを読み込みました";
  } catch (error) {
    result.className = "result fail";
    result.textContent = error instanceof Error ? error.message : String(error);
  } finally {
    importInput.value = "";
  }
});

async function boot(): Promise<void> {
  loadSavedAnchor();
  orientationStatus.textContent = RegistrationOrientationSensor.supported
    ? "Orientation: 登録時のみ"
    : "Orientation: 非対応 / fallback";
  visionStatus.classList.add("ready");
}

void boot().catch(error => {
  visionStatus.classList.add("failed");
  result.className = "result fail";
  result.textContent = error instanceof Error ? error.message : String(error);
});

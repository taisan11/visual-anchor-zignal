# Visual Anchor — JavaScript Native

OpenCV.jsやWASMを使わず、ブラウザのカメラ画像をJavaScriptでORB特徴量化し、風景をQRコード代わりのVisual Anchorとして登録・照合するMVPです。

## Runtime

- TypeScript + Vite (追加の実行時依存なし)

## Pipeline

```text
getUserMedia
  -> 640px程度へ縮小
  -> Canvas ImageData / grayscale + histogram equalization (JavaScript)
  -> multi-scale FAST + oriented BRIEF (JavaScript)
  -> Hamming matcher
  -> Lowe ratio + mutual cross-check
  -> Homography RANSAC
  -> multi-view score                (TypeScript)
```

## 登録時のDeviceOrientation補助

`DeviceOrientationEvent` は**登録処理の間だけ**有効化します。照合ではイベントを購読せず、姿勢値も参照しません。

登録では固定5枚をそのまま保存するのではなく、11候補を取得し、それぞれに短時間の相対姿勢を添えます。候補選択時に

- Homographyの幾何整合性
- 画像内の移動量
- Homographyで説明しきれない残差
- DeviceOrientationから得た相対回転量

を見て、端末をその場で大きく回しただけの候補を減点します。これにより、向きを大きく変えず横に移動した、視差を得やすいviewを優先します。

姿勢センサーが使えない、権限が拒否された、値が届かない場合は画像幾何だけで自動的にfallbackします。

### 権限

一部ブラウザでは `DeviceOrientationEvent.requestPermission()` が必要です。登録ボタンのユーザー操作から直接要求します。絶対方位は不要なので、対応実装では `requestPermission(false)` を使い、磁気センサーを要求しません。

## Anchor V2の出力最適化

登録時のORB計算には通常どおり完全なkeypoint情報を使いますが、照合に必要なのは実際には

- keypoint `x, y`
- 32-byte ORB descriptor

だけです。

V2 Anchorでは、従来の

```text
x, y, size, angle, response, octave = 6 × float32 = 24 bytes/feature
```

を保存せず、

```text
x, y = 2 × uint16 = 4 bytes/feature
```

へ量子化します。読み込み時にJavaScriptのkeypoint配列へ展開します。座標の丸め誤差は最大約0.5pxで、標準のRANSAC閾値4pxより十分小さい範囲です。

また、選択したviewごとに特徴点を空間グリッドで均等化し、強い特徴だけ最大520点へpruneします。画面の一部だけに特徴点が集中するのを避けつつJSONサイズを下げます。

旧形式の互換コードは持たず、Anchor version 2形式だけを読み書きします。

## 重要: 照合時は姿勢センサー非依存

照合パスは以下だけです。

```text
camera frame
  -> JavaScript ORB
  -> stored ORB descriptors
  -> JavaScript Hamming matching
  -> JavaScript Homography RANSAC
  -> multi-frame image geometry
```

`verify()` / `verifyBurst()` はDeviceOrientationを開始・参照しません。そのため、登録した端末と照合端末のセンサー差、磁気環境、ブラウザの姿勢API対応状況は照合結果に影響しません。

## ブラウザAPIと互換性

Canvas 2D (`drawImage`, `getImageData`) がカメラ画像の縮小と画素取得を行い、`ImageData` / TypedArrayでピクセルと特徴量を処理します。画像ピラミッドは `OffscreenCanvas` があれば利用し、なければ通常のCanvasへフォールバックします。MDN Browser Compatibility DataによるとOffscreenCanvasはiOS Safari 16.4以降で利用可能です。

WebGPUはiOS Safari 26以降でサポートされていますが、このアプリは小さな画像バッファを段階的に処理するため、GPUへの転送とreadbackを挟むよりCanvasの最適化とJavaScript TypedArray処理を使う構成にしています。WebGPUがなくても動作します。

- [MDN: OffscreenCanvas](https://developer.mozilla.org/en-US/docs/Web/API/OffscreenCanvas)
- [MDN: CanvasRenderingContext2D.getImageData()](https://developer.mozilla.org/en-US/docs/Web/API/CanvasRenderingContext2D/getImageData)
- [MDN: WebGPU API](https://developer.mozilla.org/en-US/docs/Web/API/WebGPU_API)

## Setup

Node.jsを用意してください。

```bash
npm install
npm run dev
```

本番ビルド:

```bash
npm run build
```

## Match result

JavaScriptのmatcherは以下を返します。

```text
matches
inliers
inlier ratio
mean reprojection error
p90 reprojection error
average Hamming distance
3x3 homography
```

UI側ではこれを0〜1のgeometry scoreにまとめ、3-frame burstの中央値で地点判定します。

## 明るさへの耐性

各フレームをORBに渡す前にgrayscale histogram equalizationしています。そのため単純なRGB差分より、露出・曇天・時間帯による全体的な明るさ変化に強くなっています。

ただし昼→ほぼ真っ暗な夜、強い逆光、季節で景観そのものが大きく変わるケースではORBの限界があります。その場合はORB extractorだけXFeat等に置き換えるのが次の段階です。

## Replay detectionについて

照合時に3 frameを撮り、最初と最後のフレーム間もHomography RANSACします。十分にカメラが動いているのに全特徴点が極端にきれいな単一Homographyで説明できる場合、平面ディスプレイ/印刷物っぽいとして `planarReplayRisk` を立てます。

これは**ヒューリスティック**です。建物の壁など本当に平面的な対象でも誤検出し得るため、セキュリティ上のliveness証明には使わないでください。

## HTTPS

カメラとDevice Orientation APIはいずれもSecure Contextが必要な環境があります。`localhost` 以外のスマホから試す場合はHTTPSで配信してください。

## Project layout

```text
src/vision.ts          JavaScript ORB + matching / RANSAC
src/orb-pattern.ts     ORB learned sampling pattern
src/orientation.ts     registration-only DeviceOrientation helper
src/anchor.ts          view selection / compression / scoring
src/main.ts            camera + UI
src/styles.css
```

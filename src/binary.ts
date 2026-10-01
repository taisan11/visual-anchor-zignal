import type { VisualAnchor } from "./anchor";

const MAGIC = new Uint8Array([0x56, 0x41, 0x5a, 0x42, 0x49, 0x4e, 0x00, 0x01]); // VAZBIN\0\1
const FILE_VERSION = 1;
const ANCHOR_VERSION = 2;
const DESC_BYTES = 32;
const FLAG_ORIENTATION_ASSISTED = 1 << 0;
const ABSENT_U16 = 0xffff;
const ABSENT_U32 = 0xffffffff;

export const BINARY_MIME = "application/vnd.visual-anchor-zignal";
export const BINARY_EXTENSION = ".vaz";

class BinaryWriter {
  private buffer = new ArrayBuffer(4096);
  private view = new DataView(this.buffer);
  private bytesView = new Uint8Array(this.buffer);
  private offset = 0;

  private ensure(extra: number): void {
    const required = this.offset + extra;
    if (required <= this.buffer.byteLength) return;
    let size = this.buffer.byteLength;
    while (size < required) size *= 2;
    const next = new ArrayBuffer(size);
    new Uint8Array(next).set(this.bytesView.subarray(0, this.offset));
    this.buffer = next;
    this.view = new DataView(next);
    this.bytesView = new Uint8Array(next);
  }

  u8(value: number): void {
    this.ensure(1);
    this.view.setUint8(this.offset, value);
    this.offset += 1;
  }

  u16(value: number): void {
    this.ensure(2);
    this.view.setUint16(this.offset, value, true);
    this.offset += 2;
  }

  u32(value: number): void {
    this.ensure(4);
    this.view.setUint32(this.offset, value, true);
    this.offset += 4;
  }

  f32(value: number): void {
    this.ensure(4);
    this.view.setFloat32(this.offset, value, true);
    this.offset += 4;
  }

  f64(value: number): void {
    this.ensure(8);
    this.view.setFloat64(this.offset, value, true);
    this.offset += 8;
  }

  bytes(bytes: Uint8Array): void {
    this.ensure(bytes.length);
    this.bytesView.set(bytes, this.offset);
    this.offset += bytes.length;
  }

  finish(): Uint8Array {
    return this.bytesView.slice(0, this.offset);
  }
}

class BinaryReader {
  private readonly view: DataView;
  private offset = 0;

  constructor(private readonly data: Uint8Array) {
    this.view = new DataView(data.buffer, data.byteOffset, data.byteLength);
  }

  private require(size: number): void {
    if (this.offset + size > this.data.length) throw new Error("truncated anchor binary");
  }

  u8(): number {
    this.require(1);
    const value = this.view.getUint8(this.offset);
    this.offset += 1;
    return value;
  }

  u16(): number {
    this.require(2);
    const value = this.view.getUint16(this.offset, true);
    this.offset += 2;
    return value;
  }

  u32(): number {
    this.require(4);
    const value = this.view.getUint32(this.offset, true);
    this.offset += 4;
    return value;
  }

  f32(): number {
    this.require(4);
    const value = this.view.getFloat32(this.offset, true);
    this.offset += 4;
    return value;
  }

  f64(): number {
    this.require(8);
    const value = this.view.getFloat64(this.offset, true);
    this.offset += 8;
    return value;
  }

  bytes(size: number): Uint8Array {
    this.require(size);
    const value = this.data.slice(this.offset, this.offset + size);
    this.offset += size;
    return value;
  }

  get remaining(): number {
    return this.data.length - this.offset;
  }
}

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

const CRC_TABLE = (() => {
  const table = new Uint32Array(256);
  for (let i = 0; i < 256; i++) {
    let value = i;
    for (let bit = 0; bit < 8; bit++) {
      value = (value & 1) !== 0 ? (0xedb88320 ^ (value >>> 1)) : (value >>> 1);
    }
    table[i] = value >>> 0;
  }
  return table;
})();

function crc32(data: Uint8Array): number {
  let crc = 0xffffffff;
  for (const byte of data) crc = CRC_TABLE[(crc ^ byte) & 0xff] ^ (crc >>> 8);
  return (crc ^ 0xffffffff) >>> 0;
}

function appendChecksum(payload: Uint8Array): Uint8Array {
  const out = new Uint8Array(payload.length + 4);
  out.set(payload);
  new DataView(out.buffer).setUint32(payload.length, crc32(payload), true);
  return out;
}

function checkedU16(value: number, label: string): number {
  if (!Number.isInteger(value) || value < 0 || value > 0xffff) throw new Error(`${label} exceeds binary format limits`);
  return value;
}

function checkedU32OrAbsent(value: number | undefined): number {
  if (value == null) return ABSENT_U32;
  if (!Number.isInteger(value) || value < 0 || value >= ABSENT_U32) throw new Error("anchor count exceeds binary format limits");
  return value;
}

export function encodeAnchorBinary(anchor: VisualAnchor): Uint8Array {
  if (anchor.version !== ANCHOR_VERSION || anchor.format !== "visual-anchor-zignal") {
    throw new Error("unsupported in-memory anchor format");
  }

  const selected = anchor.registration.selectedCandidateIndices ?? [];
  const pairScores = anchor.registration.pairScores ?? [];
  const parallax = anchor.registration.parallaxSignatures ?? [];
  const writer = new BinaryWriter();
  writer.bytes(MAGIC);
  writer.u8(FILE_VERSION);
  writer.u8(ANCHOR_VERSION);
  writer.u16(checkedU16(anchor.processingWidth, "processingWidth"));
  writer.u16(checkedU16(anchor.views.length, "view count"));
  writer.u16(anchor.registration.orientationAssisted ? FLAG_ORIENTATION_ASSISTED : 0);
  writer.f64(Date.parse(anchor.createdAt));
  writer.u16(anchor.registration.candidateViews == null ? ABSENT_U16 : checkedU16(anchor.registration.candidateViews, "candidateViews"));
  writer.u16(checkedU16(selected.length, "selected index count"));
  writer.u16(checkedU16(pairScores.length, "pair score count"));
  writer.u16(checkedU16(parallax.length, "parallax count"));
  writer.u32(checkedU32OrAbsent(anchor.registration.inputFeatureCount));
  writer.u32(checkedU32OrAbsent(anchor.registration.outputFeatureCount));
  writer.f32(anchor.registration.medianPairScore ?? 0);
  writer.f32(anchor.registration.medianParallaxSignature ?? 0);
  writer.f32(anchor.registration.orientationSpanDeg ?? Number.NaN);
  writer.f32(anchor.registration.meanOrientationStepDeg ?? Number.NaN);

  for (const index of selected) writer.u16(checkedU16(index, "selected candidate index"));
  for (const score of pairScores) writer.f32(score);
  for (const value of parallax) writer.f32(value);

  for (const featureSet of anchor.views) {
    const xy = base64ToBytes(featureSet.keypointsXYU16);
    const descriptors = base64ToBytes(featureSet.descriptorsU8);
    const count = checkedU16(featureSet.count, "feature count");
    if (xy.length !== count * 4) throw new Error("invalid anchor keypoint payload");
    if (descriptors.length !== count * DESC_BYTES) throw new Error("invalid anchor descriptor payload");
    writer.u16(checkedU16(featureSet.width, "view width"));
    writer.u16(checkedU16(featureSet.height, "view height"));
    writer.u16(count);
    writer.u16(0);
    writer.bytes(xy);
    writer.bytes(descriptors);
  }

  return appendChecksum(writer.finish());
}

export function decodeAnchorBinary(input: ArrayBuffer | Uint8Array): VisualAnchor {
  const all = input instanceof Uint8Array ? input : new Uint8Array(input);
  if (all.length < MAGIC.length + 4) throw new Error("anchor binary is too small");
  const payload = all.subarray(0, all.length - 4);
  const expected = new DataView(all.buffer, all.byteOffset + all.length - 4, 4).getUint32(0, true);
  if (crc32(payload) !== expected) throw new Error("anchor binary checksum mismatch");

  const reader = new BinaryReader(payload);
  const magic = reader.bytes(MAGIC.length);
  if (!magic.every((value, i) => value === MAGIC[i])) throw new Error("unsupported anchor binary magic");
  const fileVersion = reader.u8();
  const anchorVersion = reader.u8();
  if (fileVersion !== FILE_VERSION || anchorVersion !== ANCHOR_VERSION) throw new Error("unsupported anchor binary version");

  const processingWidth = reader.u16();
  const viewCount = reader.u16();
  const flags = reader.u16();
  const createdAtMs = reader.f64();
  if (!Number.isFinite(createdAtMs)) throw new Error("invalid anchor timestamp");
  const candidateViewsRaw = reader.u16();
  const selectedCount = reader.u16();
  const pairScoreCount = reader.u16();
  const parallaxCount = reader.u16();
  const inputFeatureCountRaw = reader.u32();
  const outputFeatureCountRaw = reader.u32();
  const medianPairScore = reader.f32();
  const medianParallaxSignature = reader.f32();
  const orientationSpanDeg = reader.f32();
  const meanOrientationStepDeg = reader.f32();

  const selectedCandidateIndices = Array.from({ length: selectedCount }, () => reader.u16());
  const pairScores = Array.from({ length: pairScoreCount }, () => reader.f32());
  const parallaxSignatures = Array.from({ length: parallaxCount }, () => reader.f32());
  const views: VisualAnchor["views"] = [];

  for (let viewIndex = 0; viewIndex < viewCount; viewIndex++) {
    const width = reader.u16();
    const height = reader.u16();
    const count = reader.u16();
    reader.u16(); // reserved
    const xy = reader.bytes(count * 4);
    const descriptors = reader.bytes(count * DESC_BYTES);
    views.push({
      width,
      height,
      count,
      keypointsXYU16: bytesToBase64(xy),
      descriptorsU8: bytesToBase64(descriptors),
    });
  }

  if (reader.remaining !== 0) throw new Error("unexpected trailing anchor data");

  return {
    format: "visual-anchor-zignal",
    version: ANCHOR_VERSION,
    createdAt: new Date(createdAtMs).toISOString(),
    processingWidth,
    engine: "javascript-native",
    views,
    registration: {
      pairScores,
      medianPairScore,
      parallaxSignatures,
      medianParallaxSignature,
      orientationAssisted: (flags & FLAG_ORIENTATION_ASSISTED) !== 0,
      candidateViews: candidateViewsRaw === ABSENT_U16 ? undefined : candidateViewsRaw,
      selectedCandidateIndices,
      orientationSpanDeg: Number.isFinite(orientationSpanDeg) ? orientationSpanDeg : undefined,
      meanOrientationStepDeg: Number.isFinite(meanOrientationStepDeg) ? meanOrientationStepDeg : undefined,
      inputFeatureCount: inputFeatureCountRaw === ABSENT_U32 ? undefined : inputFeatureCountRaw,
      outputFeatureCount: outputFeatureCountRaw === ABSENT_U32 ? undefined : outputFeatureCountRaw,
    },
  };
}

export function anchorBinaryToBase64(binary: Uint8Array): string {
  return bytesToBase64(binary);
}

export function anchorBinaryFromBase64(encoded: string): Uint8Array {
  return base64ToBytes(encoded);
}

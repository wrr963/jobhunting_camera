// WebCodecs(VideoEncoder/AudioEncoder) + mp4-muxer による録画
// iPhone の MediaRecorder で映像だけが途中で止まる問題を避けるため、
// フレームと音声を自前でエンコードして MP4 を作る。
import { Muxer, ArrayBufferTarget } from './vendor/mp4-muxer.mjs';

const AAC_RATES = [96000, 88200, 64000, 48000, 44100, 32000, 24000, 22050, 16000, 12000, 11025, 8000, 7350];

// AAC-LC の AudioSpecificConfig を自前で作る(Safari が誤った値を返す不具合対策)
function aacSpecificConfig(sampleRate, channels) {
  const idx = AAC_RATES.indexOf(sampleRate);
  if (idx < 0) return null;
  return new Uint8Array([(2 << 3) | (idx >> 1), ((idx & 1) << 7) | (channels << 3)]);
}

function stripAdts(data) {
  if (data.length > 7 && data[0] === 0xff && (data[1] & 0xf6) === 0xf0) {
    return data.subarray(data[1] & 1 ? 7 : 9);
  }
  return data;
}

function avcCandidates(w, h) {
  const big = w * h > 1280 * 720;
  return big
    ? ['avc1.640028', 'avc1.4d0028', 'avc1.420028', 'avc1.640033']
    : ['avc1.42001f', 'avc1.4d001f', 'avc1.64001f', 'avc1.640028'];
}

export function webCodecsAvailable() {
  return (
    typeof window.VideoEncoder === 'function' &&
    typeof window.AudioEncoder === 'function' &&
    typeof window.VideoFrame === 'function' &&
    typeof window.AudioData === 'function' &&
    typeof window.AudioWorkletNode === 'function'
  );
}

// 使える設定を探す。見つからなければ null
export async function probeConfigs(width, height, sampleRate, videoBitrate) {
  if (!webCodecsAvailable()) return null;
  let video = null;
  for (const codec of avcCandidates(width, height)) {
    const cfg = {
      codec,
      width,
      height,
      bitrate: videoBitrate,
      framerate: 30,
      latencyMode: 'realtime',
      avc: { format: 'avc' },
    };
    try {
      const r = await VideoEncoder.isConfigSupported(cfg);
      if (r.supported) { video = cfg; break; }
    } catch { /* 次へ */ }
  }
  if (!video) return null;
  let audio = null;
  const aacCfg = { codec: 'mp4a.40.2', sampleRate, numberOfChannels: 1, bitrate: 128000 };
  try {
    if ((await AudioEncoder.isConfigSupported(aacCfg)).supported && aacSpecificConfig(sampleRate, 1)) {
      audio = { cfg: aacCfg, muxCodec: 'aac' };
    }
  } catch { /* 次へ */ }
  if (!audio) {
    const opusCfg = { codec: 'opus', sampleRate: 48000, numberOfChannels: 1, bitrate: 128000 };
    try {
      if (sampleRate === 48000 && (await AudioEncoder.isConfigSupported(opusCfg)).supported) {
        audio = { cfg: opusCfg, muxCodec: 'opus' };
      }
    } catch { /* なし */ }
  }
  if (!audio) return null;
  return { video, audio };
}

export class WebCodecsRecorder {
  constructor({ canvas, audioCtx, audioNode, configs, onError }) {
    this.canvas = canvas;
    this.audioCtx = audioCtx;
    this.audioNode = audioNode; // AudioWorkletNode
    this.configs = configs;
    this.onError = onError || (() => {});
    this.active = false;
    this.failed = false;
  }

  start() {
    const { video, audio } = this.configs;
    const sr = audio.cfg.sampleRate;
    this.target = new ArrayBufferTarget();
    this.muxer = new Muxer({
      target: this.target,
      video: { codec: 'avc', width: video.width, height: video.height },
      audio: { codec: audio.muxCodec, numberOfChannels: 1, sampleRate: sr },
      fastStart: 'in-memory',
      firstTimestampBehavior: 'offset',
    });

    const fail = (e) => {
      console.error(e);
      if (!this.failed) {
        this.failed = true;
        this.onError(e);
      }
    };

    this.videoEncoder = new VideoEncoder({
      output: (chunk, meta) => {
        try { this.muxer.addVideoChunk(chunk, meta); } catch (e) { fail(e); }
      },
      error: fail,
    });
    this.videoEncoder.configure(video);

    const asc = audio.muxCodec === 'aac' ? aacSpecificConfig(sr, 1) : null;
    this.audioEncoder = new AudioEncoder({
      output: (chunk, meta) => {
        try {
          if (asc) {
            const raw = new Uint8Array(chunk.byteLength);
            chunk.copyTo(raw);
            const data = stripAdts(raw);
            this.muxer.addAudioChunkRaw(data, chunk.type, chunk.timestamp, chunk.duration || Math.round((1024 / sr) * 1e6), {
              decoderConfig: { codec: 'mp4a.40.2', sampleRate: sr, numberOfChannels: 1, description: asc },
            });
          } else {
            this.muxer.addAudioChunk(chunk, meta);
          }
        } catch (e) { fail(e); }
      },
      error: fail,
    });
    this.audioEncoder.configure(audio.cfg);

    this.t0 = performance.now();
    this.frameCount = 0;
    this.lastFrameAt = -1e9;
    this.samples = 0;
    this.active = true;

    // AudioWorklet から届いた音声(呼び出し側が port.onmessage から渡す)
    this.onAudio = (e) => {
      if (!this.active || this.failed) return;
      const pcm = e.data;
      const ts = Math.round((this.samples / sr) * 1e6);
      this.samples += pcm.length; // 捨てた場合も時間は進める(音ズレ防止)
      if (this.audioEncoder.encodeQueueSize > 20) return;
      try {
        const ad = new AudioData({
          format: 'f32-planar',
          sampleRate: sr,
          numberOfFrames: pcm.length,
          numberOfChannels: 1,
          timestamp: ts,
          data: pcm,
        });
        this.audioEncoder.encode(ad);
        ad.close();
      } catch (err) { fail(err); }
    };
  }

  // 描画ループから毎フレーム呼ぶ(30fpsに間引く)
  captureFrame() {
    if (!this.active || this.failed) return;
    const now = performance.now();
    if (now - this.lastFrameAt < 1000 / 30 - 4) return;
    if (this.videoEncoder.encodeQueueSize > 4) return; // 詰まっていたら捨てる(時間はずれない)
    this.lastFrameAt = now;
    try {
      const frame = new VideoFrame(this.canvas, {
        timestamp: Math.max(0, Math.round((now - this.t0) * 1000)),
        duration: Math.round(1e6 / 30),
      });
      this.videoEncoder.encode(frame, { keyFrame: this.frameCount % 60 === 0 });
      frame.close();
      this.frameCount++;
    } catch (e) {
      console.error(e);
    }
  }

  async stop() {
    this.active = false;
    try { await this.videoEncoder.flush(); } catch (e) { console.warn(e); }
    try { await this.audioEncoder.flush(); } catch (e) { console.warn(e); }
    try { this.videoEncoder.close(); } catch { /* */ }
    try { this.audioEncoder.close(); } catch { /* */ }
    this.muxer.finalize();
    return new Blob([this.target.buffer], { type: 'video/mp4' });
  }
}

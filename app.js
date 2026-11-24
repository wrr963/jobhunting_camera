// カンペカメラ - カンペを読みながら録画できるWebアプリ
'use strict';
import { WebCodecsRecorder, probeConfigs, webCodecsAvailable } from './webcodecs-recorder.js';

// ---------- 定数 ----------
const MP_VERSION = '0.10.14';
const MP_BASE = `https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@${MP_VERSION}`;
const SEG_MODEL_URL =
  'https://storage.googleapis.com/mediapipe-models/image_segmenter/selfie_segmenter/float16/latest/selfie_segmenter.tflite';
const LS_SETTINGS = 'kanpe.settings.v1';
const LS_SCRIPT = 'kanpe.script.v1';

const DEFAULTS = {
  bgMode: 'none',      // none | blur | color | image
  bgColor: '#ffffff',
  blurAmount: 6,
  speed: 20,
  fontSize: 28,
  opacity: 45,
  pHeight: 40,
  autoScroll: true,
  resetOnRec: true,
  countdownOn: true,
  mirror: true,
  quality: '720',
  landSide: 'left',
  format: 'auto',      // auto | mp4 | webm   // 横画面でのカンペ位置: left | right
};

const SAMPLE_SCRIPT = `はじめまして。〇〇大学〇〇学部の〇〇です。

本日は「学生時代に力を入れたこと」についてお話しします。

（ここに原稿を入力してください。「✎ 編集」から、貼り付けや編集ができます。）

ご清聴ありがとうございました。`;

// ---------- 要素 ----------
const $ = (id) => document.getElementById(id);
const stage = $('stage');
const ctx = stage.getContext('2d');
const video = $('cam');
const prompter = $('prompter');
const pScroll = $('prompterScroll');
const pText = $('prompterText');

// ---------- 状態 ----------
const settings = loadSettings();
let script = localStorage.getItem(LS_SCRIPT);
if (script === null) script = SAMPLE_SCRIPT;

let stream = null;
let facing = 'user';
let recorder = null;
let chunks = [];
let recording = false;
let recStart = 0;
let recTimer = null;
let resultBlob = null;
let resultUrl = null;
let resultName = '';
let resultFile = null;
let wakeLock = null;
let countingDown = false;
let recordMode = '';
let wcRecorder = null;
let audioCtx = null;
let audioNode = null;
let audioSource = null;
let recordingCanvas = false;

// 合成用キャンバス
const frameCv = document.createElement('canvas');
const frameCtx = frameCv.getContext('2d');
const personCv = document.createElement('canvas');
const personCtx = personCv.getContext('2d');
const blurCv1 = document.createElement('canvas');
const blurCtx1 = blurCv1.getContext('2d');
const blurCv2 = document.createElement('canvas');
const blurCtx2 = blurCv2.getContext('2d');
const segCv = document.createElement('canvas');
const segCtx = segCv.getContext('2d', { willReadFrequently: true });
const maskCv = document.createElement('canvas');
const maskCtx = maskCv.getContext('2d');
let maskImage = null;
let prevMask = null;

// セグメンテーション
let segmenter = null;
let segLoading = null;
let segFailed = false;
let lastSegTs = 0;
let bgImage = null;

// カンペ
let pPlaying = false;
let pPos = 0;
let touching = false;
let userHoldUntil = 0;
let lastTick = performance.now();

// ---------- 設定 ----------
function loadSettings() {
  try {
    return { ...DEFAULTS, ...JSON.parse(localStorage.getItem(LS_SETTINGS) || '{}') };
  } catch {
    return { ...DEFAULTS };
  }
}
function saveSettings() {
  localStorage.setItem(LS_SETTINGS, JSON.stringify(settings));
}

// ---------- ユーティリティ ----------
let toastTimer = null;
function toast(msg, ms = 2200) {
  const t = $('toast');
  t.textContent = msg;
  t.classList.remove('hidden');
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => t.classList.add('hidden'), ms);
}
const pad = (n) => String(n).padStart(2, '0');
function fmtTime(sec) {
  return `${pad(Math.floor(sec / 60))}:${pad(Math.floor(sec % 60))}`;
}
function fileStamp() {
  const d = new Date();
  return `${d.getFullYear()}${pad(d.getMonth() + 1)}${pad(d.getDate())}-${pad(d.getHours())}${pad(d.getMinutes())}${pad(d.getSeconds())}`;
}

// ---------- キャンバスサイズ(縦横) ----------
function isLandscape() {
  return window.innerWidth > window.innerHeight;
}
function updateCanvasSize() {
  if (recording || countingDown) return; // 録画中はサイズ固定
  const hd = settings.quality === '1080';
  const long = hd ? 1920 : 1280;
  const short = hd ? 1080 : 720;
  const land = isLandscape();
  const W = land ? long : short;
  const H = land ? short : long;
  if (stage.width === W && stage.height === H) {
    layoutStage();
    return;
  }
  for (const c of [stage, frameCv, personCv]) {
    c.width = W;
    c.height = H;
  }
  // セグメンテーション入力は長辺256px程度
  const sc = 256 / Math.max(W, H);
  segCv.width = maskCv.width = Math.round(W * sc);
  segCv.height = maskCv.height = Math.round(H * sc);
  maskImage = maskCtx.createImageData(maskCv.width, maskCv.height);
  prevMask = null;
  updateBlurSize();
  layoutStage();
}
// 画面に収まるよう縦横比を保って表示(引き伸ばし防止)
function layoutStage() {
  const vw = window.innerWidth;
  const vh = window.innerHeight;
  const sc = Math.min(vw / stage.width, vh / stage.height);
  const w = Math.round(stage.width * sc);
  const h = Math.round(stage.height * sc);
  // カメラ映像の<video>もキャンバスの真下に同じ位置で置く
  for (const el of [stage, video]) {
    el.style.width = `${w}px`;
    el.style.height = `${h}px`;
    el.style.left = `${Math.round((vw - w) / 2)}px`;
    el.style.top = `${Math.round((vh - h) / 2)}px`;
  }
}
function updateBlurSize() {
  blurCv1.width = Math.max(1, Math.round(stage.width / 4));
  blurCv1.height = Math.max(1, Math.round(stage.height / 4));
  const k = 4 + settings.blurAmount * 3; // 7..34
  blurCv2.width = Math.max(1, Math.round(stage.width / k));
  blurCv2.height = Math.max(1, Math.round(stage.height / k));
}

// ---------- カメラ ----------
async function startCamera() {
  if (stream) stream.getTracks().forEach((t) => t.stop());
  const constraints = {
    video: {
      facingMode: facing,
      // 画質に合わせて要求(必要以上に大きいと端末の負荷で録画が止まることがある)
      width: { ideal: settings.quality === '1080' ? 1920 : 1280 },
      height: { ideal: settings.quality === '1080' ? 1080 : 720 },
      frameRate: { ideal: 30 },
    },
    audio: {
      echoCancellation: true,
      noiseSuppression: true,
      autoGainControl: true,
    },
  };
  stream = await navigator.mediaDevices.getUserMedia(constraints);
  srcW = srcH = 0;
  video.srcObject = stream;
  await video.play().catch(() => {});
  if (hasRVFC && !rvfcStarted) {
    rvfcStarted = true;
    video.requestVideoFrameCallback(onVideoFrame);
  }
  applyMirror();
  setupAudioGraph();
}
let rvfcStarted = false;

// タップ操作の中で同期的に作る(iOSでは操作なしに音声処理を開始できない)
function ensureAudioCtx() {
  if (audioCtx) return;
  const AC = window.AudioContext || window.webkitAudioContext;
  if (!AC) return;
  try {
    audioCtx = new AC({ sampleRate: 48000 });
  } catch {
    try {
      audioCtx = new AC();
    } catch (e) {
      console.warn(e);
    }
  }
}

// マイク音声を AudioWorklet で取り出す(WebCodecs 録画用)
async function setupAudioGraph() {
  try {
    ensureAudioCtx();
    if (!audioCtx || !window.AudioWorkletNode) return;
    if (!audioNode) {
      await audioCtx.audioWorklet.addModule('audio-worklet.js');
      audioNode = new AudioWorkletNode(audioCtx, 'capture-processor', {
        numberOfInputs: 1,
        numberOfOutputs: 1,
        channelCount: 1,
        channelCountMode: 'explicit',
      });
      // 録画中だけ音声をエンコーダへ渡す(録画前の音声は捨てる)
      audioNode.port.onmessage = (e) => {
        if (wcRecorder && wcRecorder.active) wcRecorder.onAudio(e);
      };
      const mute = audioCtx.createGain();
      mute.gain.value = 0;
      audioNode.connect(mute).connect(audioCtx.destination);
    }
    if (audioSource) audioSource.disconnect();
    const at = stream && stream.getAudioTracks()[0];
    if (!at) return;
    audioSource = audioCtx.createMediaStreamSource(new MediaStream([at]));
    audioSource.connect(audioNode);
    if (audioCtx.state !== 'running') await audioCtx.resume().catch(() => {});
  } catch (e) {
    console.warn('audio graph setup failed', e);
    audioNode = null;
  }
}

function applyMirror() {
  const m = settings.mirror && facing === 'user';
  stage.classList.toggle('mirror', m);
  video.classList.toggle('mirror', m);
}

// 背景なしのときはキャンバス合成をやめてカメラ映像を直接表示(端末の負荷を減らす)
function usesCanvas() {
  return settings.bgMode !== 'none' && !!segmenter;
}
function applyPreviewMode() {
  document.body.classList.toggle('raw-preview', !usesCanvas());
}

// ---------- 背景処理 (MediaPipe) ----------
async function ensureSegmenter() {
  if (segmenter || segFailed) return segmenter;
  if (segLoading) return segLoading;
  $('bgStatus').textContent = '背景処理を読み込み中…';
  segLoading = (async () => {
    try {
      const vision = await import(`${MP_BASE}/vision_bundle.mjs`);
      const fileset = await vision.FilesetResolver.forVisionTasks(`${MP_BASE}/wasm`);
      const make = (delegate) =>
        vision.ImageSegmenter.createFromOptions(fileset, {
          baseOptions: { modelAssetPath: SEG_MODEL_URL, delegate },
          runningMode: 'VIDEO',
          outputCategoryMask: false,
          outputConfidenceMasks: true,
        });
      try {
        segmenter = await make('GPU');
      } catch (e) {
        console.warn('GPU delegate failed, fallback to CPU', e);
        segmenter = await make('CPU');
      }
      $('bgStatus').textContent = '';
      return segmenter;
    } catch (e) {
      console.error(e);
      segFailed = true;
      $('bgStatus').textContent = '背景処理を読み込めませんでした(通信環境を確認してください)。';
      return null;
    } finally {
      segLoading = null;
    }
  })();
  return segLoading;
}

function runSegmentation() {
  segCtx.drawImage(frameCv, 0, 0, segCv.width, segCv.height);
  let ts = performance.now();
  if (ts <= lastSegTs) ts = lastSegTs + 1;
  lastSegTs = ts;
  let ok = false;
  segmenter.segmentForVideo(segCv, ts, (result) => {
    const m = result.confidenceMasks && result.confidenceMasks[0];
    if (!m) return;
    const conf = m.getAsFloat32Array();
    const n = maskCv.width * maskCv.height;
    if (conf.length !== n) return;
    if (!prevMask || prevMask.length !== n) prevMask = new Float32Array(conf);
    const px = maskImage.data;
    for (let i = 0; i < n; i++) {
      // 時間方向に少しなめらかにしてチラつきを抑える
      const c = prevMask[i] * 0.35 + conf[i] * 0.65;
      prevMask[i] = c;
      // 0.25〜0.75 をなめらかに 0〜1 へ
      let a = (c - 0.25) / 0.5;
      a = a < 0 ? 0 : a > 1 ? 1 : a;
      a = a * a * (3 - 2 * a);
      const j = i * 4;
      px[j] = px[j + 1] = px[j + 2] = 255;
      px[j + 3] = a * 255;
    }
    ok = true;
  });
  if (ok) maskCtx.putImageData(maskImage, 0, 0);
  return ok;
}

function drawCover(c, img, iw, ih, W, H) {
  const s = Math.max(W / iw, H / ih);
  const dw = iw * s;
  const dh = ih * s;
  c.drawImage(img, (W - dw) / 2, (H - dh) / 2, dw, dh);
}

function drawBackground(W, H) {
  switch (settings.bgMode) {
    case 'blur':
      blurCtx1.drawImage(frameCv, 0, 0, blurCv1.width, blurCv1.height);
      blurCtx2.drawImage(blurCv1, 0, 0, blurCv2.width, blurCv2.height);
      ctx.imageSmoothingEnabled = true;
      ctx.imageSmoothingQuality = 'high';
      ctx.drawImage(blurCv2, 0, 0, W, H);
      break;
    case 'image':
      if (bgImage) {
        drawCover(ctx, bgImage, bgImage.naturalWidth, bgImage.naturalHeight, W, H);
        break;
      }
    // 画像未選択なら単色にフォールバック
    // falls through
    case 'color':
    default:
      ctx.fillStyle = settings.bgColor;
      ctx.fillRect(0, 0, W, H);
  }
}

// ---------- 描画ループ ----------
// カメラの実フレームサイズ(回転直後に videoWidth が古いままになる端末対策)
let srcW = 0;
let srcH = 0;
let lastFrameDraw = 0;
let segCost = 0;
let segSkip = 0;
const hasRVFC = 'requestVideoFrameCallback' in HTMLVideoElement.prototype;

function sourceSize() {
  const vw = video.videoWidth;
  const vh = video.videoHeight;
  if (srcW && srcH) {
    // メタデータと videoWidth の縦横が食い違う場合はメタデータを優先
    return [srcW, srcH];
  }
  return [vw, vh];
}

function drawFrame() {
  if (!usesCanvas() && !recordingCanvas) return;
  if (!video.videoWidth || video.readyState < 2) return;
  lastFrameDraw = performance.now();
  const W = stage.width;
  const H = stage.height;
  const [iw, ih] = sourceSize();

  drawCover(frameCtx, video, iw, ih, W, H);

  if (settings.bgMode === 'none' || !segmenter || !maskImage) {
    ctx.drawImage(frameCv, 0, 0);
    return;
  }

  // 端末が遅い場合はマスク計算を間引く(前回のマスクを再利用)
  if (segSkip > 0) {
    segSkip--;
  } else {
    const t0 = performance.now();
    try {
      runSegmentation();
    } catch (e) {
      console.error(e);
    }
    segCost = segCost * 0.8 + (performance.now() - t0) * 0.2;
    segSkip = Math.min(3, Math.floor(segCost / 18));
  }

  // 人物だけを切り抜く
  personCtx.globalCompositeOperation = 'copy';
  personCtx.imageSmoothingEnabled = true;
  personCtx.imageSmoothingQuality = 'high';
  personCtx.drawImage(maskCv, 0, 0, W, H);
  personCtx.globalCompositeOperation = 'source-in';
  personCtx.drawImage(frameCv, 0, 0);
  personCtx.globalCompositeOperation = 'source-over';

  drawBackground(W, H);
  ctx.drawImage(personCv, 0, 0);
}

// 新しいカメラフレームが来るたびに描画
function onVideoFrame(_now, meta) {
  if (meta && meta.width && meta.height) {
    srcW = meta.width;
    srcH = meta.height;
  }
  video.requestVideoFrameCallback(onVideoFrame);
}

function render(now) {
  requestAnimationFrame(render);
  tickPrompter(now);
  // 毎フレーム一定間隔で描画(キャンバス録画の映像が途切れないように)
  drawFrame();
  if (wcRecorder) wcRecorder.captureFrame();
}

// ---------- カンペ ----------
function renderScript() {
  const empty = !script.trim();
  pText.textContent = empty ? '「✎ 編集」から原稿を入力・貼り付けしてください。' : script;
  pText.classList.toggle('empty', empty);
}

function speedPxPerSec() {
  // 速さは文字サイズに比例(文字を大きくしても読むペースが変わらない)
  return (settings.speed * settings.fontSize * 1.6) / 40;
}

function setPlaying(on) {
  pPlaying = on;
  $('btnPlay').textContent = on ? '⏸' : '▶';
  if (on) pPos = pScroll.scrollTop;
}

function resetPrompter() {
  pPos = 0;
  pScroll.scrollTop = 0;
}

function tickPrompter(now) {
  const dt = Math.min(0.1, (now - lastTick) / 1000);
  lastTick = now;
  if (!pPlaying) return;
  if (touching || now < userHoldUntil) {
    pPos = pScroll.scrollTop;
    return;
  }
  const max = pScroll.scrollHeight - pScroll.clientHeight;
  pPos += speedPxPerSec() * dt;
  if (pPos >= max) {
    pPos = max;
    setPlaying(false);
  }
  pScroll.scrollTop = pPos;
}

function updateSpeedLabel() {
  $('speedLabel').textContent = `速さ ${settings.speed}`;
}

// カンペの先頭/末尾がガイド線の位置に来るよう余白を表示エリアの高さから計算
function layoutPrompter() {
  const h = pScroll.clientHeight;
  pText.style.paddingTop = `${Math.round(h * 0.3)}px`;
  pText.style.paddingBottom = `${Math.round(h * 0.7)}px`;
}

function applyPrompterStyle() {
  const r = document.documentElement.style;
  r.setProperty('--p-font', `${settings.fontSize}px`);
  r.setProperty('--p-bg', String(settings.opacity / 100));
  r.setProperty('--p-height', `${settings.pHeight}%`);
  layoutPrompter();
}

// 手で触ったら一時的に自動スクロールを止める
pScroll.addEventListener('touchstart', () => { touching = true; }, { passive: true });
pScroll.addEventListener('touchend', () => {
  touching = false;
  userHoldUntil = performance.now() + 700;
}, { passive: true });
pScroll.addEventListener('touchcancel', () => { touching = false; }, { passive: true });
pScroll.addEventListener('wheel', () => { userHoldUntil = performance.now() + 700; }, { passive: true });
pScroll.addEventListener('scroll', () => {
  if (!pPlaying) pPos = pScroll.scrollTop;
});

$('btnPlay').onclick = () => setPlaying(!pPlaying);
$('btnTop').onclick = () => resetPrompter();
$('btnSlower').onclick = () => changeSpeed(-2);
$('btnFaster').onclick = () => changeSpeed(+2);
function changeSpeed(d) {
  settings.speed = Math.max(0, Math.min(100, settings.speed + d));
  saveSettings();
  updateSpeedLabel();
  syncSettingsUI();
}

// ---------- カンペ編集 ----------
const scriptInput = $('scriptInput');
function updateCharCount() {
  $('charCount').textContent = `${scriptInput.value.replace(/\s/g, '').length} 文字`;
}
$('btnEdit').onclick = () => {
  setPlaying(false);
  scriptInput.value = script;
  updateCharCount();
  $('editor').classList.remove('hidden');
};
$('btnEditDone').onclick = () => {
  $('editor').classList.add('hidden');
  scriptInput.blur();
};
scriptInput.addEventListener('input', () => {
  script = scriptInput.value;
  localStorage.setItem(LS_SCRIPT, script);
  renderScript();
  updateCharCount();
});
$('btnPaste').onclick = async () => {
  try {
    const text = await navigator.clipboard.readText();
    if (!text) return toast('クリップボードが空です');
    const s = scriptInput.selectionStart ?? scriptInput.value.length;
    const e = scriptInput.selectionEnd ?? s;
    scriptInput.setRangeText(text, s, e, 'end');
    scriptInput.dispatchEvent(new Event('input'));
    toast('貼り付けました');
  } catch {
    scriptInput.focus();
    toast('入力欄を長押しして「ペースト」を選んでください', 3000);
  }
};
$('btnCopy').onclick = async () => {
  try {
    await navigator.clipboard.writeText(scriptInput.value);
    toast('コピーしました');
  } catch {
    scriptInput.focus();
    scriptInput.select();
    const ok = document.execCommand && document.execCommand('copy');
    toast(ok ? 'コピーしました' : '全選択しました。長押しでコピーしてください', 2500);
  }
};
$('btnClear').onclick = () => {
  if (!scriptInput.value || confirm('原稿をすべて消去しますか？')) {
    scriptInput.value = '';
    scriptInput.dispatchEvent(new Event('input'));
    scriptInput.focus();
  }
};

// ---------- 設定画面 ----------
const ranges = {
  speed: (v) => v,
  fontSize: (v) => `${v}`,
  opacity: (v) => `${v}%`,
  pHeight: (v) => `${v}%`,
  blurAmount: (v) => v,
};
const checks = ['autoScroll', 'resetOnRec', 'countdownOn', 'mirror'];

function syncSettingsUI() {
  for (const [k, fmt] of Object.entries(ranges)) {
    $(k).value = settings[k];
    $(`${k}Out`).textContent = fmt(settings[k]);
  }
  for (const k of checks) $(k).checked = !!settings[k];
  for (const b of $('bgMode').children) b.classList.toggle('active', b.dataset.v === settings.bgMode);
  for (const b of $('landSide').children) b.classList.toggle('active', b.dataset.v === settings.landSide);
  for (const b of $('format').children) b.classList.toggle('active', b.dataset.v === settings.format);
  for (const b of $('quality').children) b.classList.toggle('active', b.dataset.v === settings.quality);
  $('bgColor').value = settings.bgColor;
  $('bgColorRow').classList.toggle('hidden', !(settings.bgMode === 'color' || settings.bgMode === 'image'));
  $('bgBlurRow').classList.toggle('hidden', settings.bgMode !== 'blur');
  $('bgImageRow').classList.toggle('hidden', settings.bgMode !== 'image');
}

for (const k of Object.keys(ranges)) {
  $(k).addEventListener('input', (e) => {
    settings[k] = Number(e.target.value);
    $(`${k}Out`).textContent = ranges[k](settings[k]);
    saveSettings();
    applyPrompterStyle();
    updateSpeedLabel();
    if (k === 'blurAmount') updateBlurSize();
  });
}
for (const k of checks) {
  $(k).addEventListener('change', (e) => {
    settings[k] = e.target.checked;
    saveSettings();
    if (k === 'mirror') applyMirror();
  });
}
$('bgMode').addEventListener('click', async (e) => {
  const v = e.target.dataset && e.target.dataset.v;
  if (!v) return;
  if (recording || countingDown) {
    toast('録画中は背景を変更できません');
    return;
  }
  settings.bgMode = v;
  saveSettings();
  syncSettingsUI();
  if (v !== 'none') {
    const s = await ensureSegmenter();
    if (!s) {
      settings.bgMode = 'none';
      syncSettingsUI();
    }
  }
  applyPreviewMode();
});
function applyLandSide() {
  document.body.classList.toggle('p-right', settings.landSide === 'right');
  layoutPrompter();
}
function setLandSide(v) {
  settings.landSide = v;
  saveSettings();
  syncSettingsUI();
  applyLandSide();
}
$('landSide').addEventListener('click', (e) => {
  const v = e.target.dataset && e.target.dataset.v;
  if (v) setLandSide(v);
});
$('btnSide').onclick = () => setLandSide(settings.landSide === 'right' ? 'left' : 'right');
$('quality').addEventListener('click', (e) => {
  const v = e.target.dataset && e.target.dataset.v;
  if (!v || recording) return;
  settings.quality = v;
  saveSettings();
  syncSettingsUI();
  updateCanvasSize();
  if (stream) startCamera().catch((err) => console.warn(err));
});
$('format').addEventListener('click', (e) => {
  const v = e.target.dataset && e.target.dataset.v;
  if (!v || recording) return;
  settings.format = v;
  saveSettings();
  syncSettingsUI();
});
$('swatches').addEventListener('click', (e) => {
  const c = e.target.dataset && e.target.dataset.c;
  if (!c) return;
  settings.bgColor = c;
  if (settings.bgMode === 'image' && !bgImage) settings.bgMode = 'color';
  saveSettings();
  syncSettingsUI();
});
$('bgColor').addEventListener('input', (e) => {
  settings.bgColor = e.target.value;
  saveSettings();
});
$('bgImage').addEventListener('change', (e) => {
  const f = e.target.files && e.target.files[0];
  if (!f) return;
  const img = new Image();
  img.onload = () => {
    bgImage = img;
    toast('背景画像を設定しました');
  };
  img.onerror = () => toast('画像を読み込めませんでした');
  img.src = URL.createObjectURL(f);
});

$('btnSettings').onclick = () => {
  syncSettingsUI();
  $('settings').classList.remove('hidden');
};
$('btnSettingsDone').onclick = () => $('settings').classList.add('hidden');
for (const id of ['settings', 'editor']) {
  $(id).addEventListener('click', (e) => {
    if (e.target.id === id) $(id).classList.add('hidden');
  });
}

// ---------- カメラ切替 ----------
$('btnFlip').onclick = async () => {
  if (recording) return;
  facing = facing === 'user' ? 'environment' : 'user';
  try {
    await startCamera();
  } catch (e) {
    console.error(e);
    facing = facing === 'user' ? 'environment' : 'user';
    toast('カメラを切り替えられませんでした');
    await startCamera().catch(() => {});
  }
};

// ---------- 録画 ----------
const MP4_TYPES = ['video/mp4;codecs=avc1.42E01E,mp4a.40.2', 'video/mp4;codecs=avc1,mp4a', 'video/mp4'];
const WEBM_TYPES = ['video/webm;codecs=vp8,opus', 'video/webm;codecs=vp9,opus', 'video/webm'];
function isAppleMobile() {
  return /iPhone|iPad|iPod/.test(navigator.userAgent) ||
    (navigator.platform === 'MacIntel' && navigator.maxTouchPoints > 1);
}
function pickMimeType() {
  if (!window.MediaRecorder || !MediaRecorder.isTypeSupported) return '';
  const ok = (t) => MediaRecorder.isTypeSupported(t);
  const webm = WEBM_TYPES.find(ok);
  const mp4 = MP4_TYPES.find(ok);
  if (settings.format === 'mp4') return mp4 || webm || '';
  if (settings.format === 'webm') return webm || mp4 || '';
  // 自動: iPhone/iPad(写真アプリが WebM 非対応)は MP4、
  // それ以外(Chrome/Android等)は安定している WebM
  return isAppleMobile() ? mp4 || webm || '' : webm || mp4 || '';
}

async function runCountdown() {
  const el = $('countdown');
  el.classList.remove('hidden');
  for (let i = 3; i > 0; i--) {
    el.textContent = i;
    await new Promise((r) => setTimeout(r, 1000));
  }
  el.classList.add('hidden');
}

function videoBitrate() {
  return settings.quality === '1080' ? 5_000_000 : 2_500_000;
}

async function startRecording() {
  if (!stream) return;
  // ユーザー操作の直後に音声処理を再開しておく(iOS対策)
  if (audioCtx && audioCtx.state !== 'running') audioCtx.resume().catch(() => {});
  updateCanvasSize();
  countingDown = true;
  document.body.classList.add('recording');
  $('btnRecord').classList.add('recording');
  if (settings.resetOnRec) resetPrompter();

  // 録画方式を決める: 使えるなら WebCodecs(安定)、無理なら MediaRecorder
  let configs = null;
  if (settings.format === 'auto' && audioNode && webCodecsAvailable()) {
    try {
      configs = await probeConfigs(stage.width, stage.height, audioCtx.sampleRate, videoBitrate());
    } catch (e) {
      console.warn(e);
    }
  }
  if (!configs && (!stage.captureStream || !window.MediaRecorder)) {
    toast('このブラウザは録画に対応していません', 3500);
    stopRecording();
    return;
  }

  if (settings.countdownOn) await runCountdown();
  countingDown = false;
  if (!document.body.classList.contains('recording')) return; // カウントダウン中に中止された

  if (configs) {
    recordingCanvas = true; // 毎フレーム キャンバスに描いてエンコード
    drawFrame();
    wcRecorder = new WebCodecsRecorder({
      canvas: stage,
      audioCtx,
      audioNode,
      configs,
      onError: () => {
        toast('録画中にエラーが発生したため停止しました', 3500);
        if (recording) stopRecording();
      },
    });
    try {
      wcRecorder.start();
    } catch (e) {
      console.error(e);
      wcRecorder = null;
      recordingCanvas = false;
      toast('録画を開始できませんでした', 3000);
      stopRecording();
      return;
    }
    recordMode = 'webcodecs';
  } else {
    startMediaRecorder();
    recordMode = 'mediarecorder';
  }

  recording = true;
  recStart = performance.now();
  $('recBadge').classList.remove('hidden');
  $('recTime').textContent = '00:00';
  recTimer = setInterval(() => {
    $('recTime').textContent = fmtTime((performance.now() - recStart) / 1000);
  }, 250);
  if (settings.autoScroll) setPlaying(true);
}

function startMediaRecorder() {
  // 背景なし: カメラ映像をそのまま録画 / 背景あり: 合成したキャンバスを録画
  const direct = !usesCanvas();
  recordingCanvas = !direct;
  const canvasStream = direct ? null : stage.captureStream(30);
  const videoTracks = direct ? stream.getVideoTracks() : canvasStream.getVideoTracks();
  const mixed = new MediaStream([...videoTracks, ...stream.getAudioTracks()]);
  const mimeType = pickMimeType();
  const opts = {
    videoBitsPerSecond: videoBitrate(),
    audioBitsPerSecond: 128_000,
  };
  if (mimeType) opts.mimeType = mimeType;
  try {
    recorder = new MediaRecorder(mixed, opts);
  } catch (e) {
    console.warn(e);
    recorder = new MediaRecorder(mixed);
  }
  chunks = [];
  recorder.ondataavailable = (e) => {
    if (e.data && e.data.size) chunks.push(e.data);
  };
  recorder.onstop = () => {
    if (canvasStream) canvasStream.getVideoTracks().forEach((t) => t.stop());
    const type = (recorder.mimeType || mimeType || 'video/webm').split(';')[0];
    const blob = new Blob(chunks, { type });
    const durMs = performance.now() - recStart;
    if (type.includes('webm')) {
      fixWebmDuration(blob, durMs).then(showResult, () => showResult(blob));
    } else {
      showResult(blob);
    }
  };
  recorder.start();
}

function stopRecording() {
  document.body.classList.remove('recording');
  $('btnRecord').classList.remove('recording');
  $('countdown').classList.add('hidden');
  countingDown = false;
  clearInterval(recTimer);
  $('recBadge').classList.add('hidden');
  setPlaying(false);
  if (wcRecorder) {
    const r = wcRecorder;
    wcRecorder = null;
    toast('動画を保存しています…', 10000);
    r.stop()
      .then((blob) => {
        $('toast').classList.add('hidden');
        showResult(blob);
      })
      .catch((e) => {
        console.error(e);
        toast('動画の作成に失敗しました', 3500);
      })
      .finally(() => { recordingCanvas = false; });
  } else {
    if (recorder && recorder.state !== 'inactive') recorder.stop();
    recordingCanvas = false;
  }
  recording = false;
}

$('btnRecord').onclick = () => {
  if (recording || countingDown) stopRecording();
  else startRecording();
};

// ---------- WebM に再生時間を書き込む ----------
// Chrome の WebM は Duration が無く、再生アプリでシークできない/長さが表示されないため補う
async function fixWebmDuration(blob, durationMs) {
  const headLen = Math.min(blob.size, 256 * 1024);
  const buf = new Uint8Array(await blob.slice(0, headLen).arrayBuffer());
  const readVint = (pos, keepMarker) => {
    const first = buf[pos];
    let len = 1;
    while (len <= 8 && !(first & (0x80 >> (len - 1)))) len++;
    if (len > 8) throw new Error('bad vint');
    let val = keepMarker ? first : first & (0xff >> len);
    let unknown = val === (0xff >> len) && !keepMarker;
    for (let i = 1; i < len; i++) {
      val = val * 256 + buf[pos + i];
      if (buf[pos + i] !== 0xff) unknown = false;
    }
    return { val, len, unknown };
  };
  let pos = 0;
  // EBML ヘッダ
  let id = readVint(pos, true);
  if (id.val !== 0x1a45dfa3) return blob;
  let size = readVint(pos + id.len, false);
  pos += id.len + size.len + size.val;
  // Segment
  id = readVint(pos, true);
  if (id.val !== 0x18538067) return blob;
  size = readVint(pos + id.len, false);
  if (!size.unknown) return blob; // サイズ確定済みの形式は対象外
  pos += id.len + size.len;
  while (pos < buf.length - 12) {
    id = readVint(pos, true);
    size = readVint(pos + id.len, false);
    const dataStart = pos + id.len + size.len;
    if (id.val === 0x114d9b74) return blob; // SeekHead があると位置がずれるので触らない
    if (id.val === 0x1549a966) {
      // Info
      const dataEnd = dataStart + size.val;
      if (dataEnd > buf.length) return blob;
      let p = dataStart;
      while (p < dataEnd) {
        const cid = readVint(p, true);
        const csz = readVint(p + cid.len, false);
        if (cid.val === 0x4489) return blob; // 既に Duration あり
        p += cid.len + csz.len + csz.val;
      }
      const body = buf.slice(dataStart, dataEnd);
      const dur = new Uint8Array(11);
      dur[0] = 0x44; dur[1] = 0x89; dur[2] = 0x88;
      new DataView(dur.buffer).setFloat64(3, durationMs); // TimecodeScale=1ms 前提
      const newSize = body.length + dur.length;
      const sizeBytes = new Uint8Array(8);
      sizeBytes[0] = 0x01;
      let v = newSize;
      for (let i = 7; i >= 1; i--) { sizeBytes[i] = v & 0xff; v = Math.floor(v / 256); }
      const header = buf.slice(pos, pos + id.len);
      return new Blob(
        [blob.slice(0, pos), header, sizeBytes, body, dur, blob.slice(dataEnd)],
        { type: blob.type },
      );
    }
    if (size.unknown) return blob;
    pos = dataStart + size.val;
  }
  return blob;
}

// ---------- 録画結果・保存 ----------
function showResult(blob) {
  if (resultUrl) URL.revokeObjectURL(resultUrl);
  resultBlob = blob;
  resultUrl = URL.createObjectURL(blob);
  const ext = blob.type.includes('mp4') ? 'mp4' : 'webm';
  resultName = `kanpe-${fileStamp()}.${ext}`;
  // 共有用ファイルは先に作っておく(ボタンを押してから時間をかけない)
  resultFile = new File([blob], resultName, { type: ext === 'mp4' ? 'video/mp4' : 'video/webm' });
  // 録画結果の表示中はマイクの音声処理を止める(iOSの共有シートとの干渉を避ける)
  if (audioCtx && audioCtx.state === 'running') audioCtx.suspend().catch(() => {});
  const v = $('resultVideo');
  v.src = resultUrl;
  $('resultInfo').textContent = `${(blob.size / 1024 / 1024).toFixed(1)} MB / ${ext.toUpperCase()} / ${recordMode === 'webcodecs' ? '安定モード' : '互換モード'}`;
  $('result').classList.remove('hidden');
  updateCanvasSize();
}

$('btnShare').onclick = async () => {
  if (!resultFile) return;
  // 押した直後に同期的に share を呼ぶ(iOSは操作から時間が空くと拒否する)
  if (!navigator.share || !navigator.canShare) {
    toast('この端末は共有に対応していません。「ダウンロード」を使ってください', 3500);
    return;
  }
  if (!navigator.canShare({ files: [resultFile] })) {
    toast('この形式の動画は共有できません。「ダウンロード」を使ってください', 3500);
    return;
  }
  try {
    // title などを付けると iOS で「ビデオを保存」が出ない/失敗することがあるのでファイルのみ
    await navigator.share({ files: [resultFile] });
  } catch (e) {
    if (e && e.name === 'AbortError') return; // ユーザーが閉じた
    console.warn(e);
    toast(`共有できませんでした(${(e && e.name) || e})。もう一度押すか「ダウンロード」を使ってください`, 4500);
  }
};
$('btnDownload').onclick = () => download();
function download() {
  if (!resultUrl) return;
  const a = document.createElement('a');
  a.href = resultUrl;
  a.download = resultName;
  document.body.appendChild(a);
  a.click();
  a.remove();
  toast('ダウンロードを開始しました');
}
$('btnRetake').onclick = () => {
  if (audioCtx && audioCtx.state !== 'running') audioCtx.resume().catch(() => {});
  const v = $('resultVideo');
  v.pause();
  $('result').classList.add('hidden');
};

// ---------- 画面スリープ防止 ----------
async function requestWakeLock() {
  try {
    if ('wakeLock' in navigator) wakeLock = await navigator.wakeLock.request('screen');
  } catch {
    /* 非対応 */
  }
}
document.addEventListener('visibilitychange', () => {
  if (document.visibilityState === 'visible' && stream) requestWakeLock();
});
window.addEventListener('beforeunload', (e) => {
  if (recording) {
    e.preventDefault();
    e.returnValue = '';
  }
});

// ---------- 画面の向き ----------
let resizeTimer = null;
let lastLand = isLandscape();
const isTouchDevice = window.matchMedia('(pointer: coarse)').matches;
function onResize() {
  layoutStage();
  layoutPrompter();
  clearTimeout(resizeTimer);
  resizeTimer = setTimeout(async () => {
    updateCanvasSize();
    layoutStage();
    const land = isLandscape();
    if (land === lastLand) return;
    lastLand = land;
    // スマホを回転したとき、カメラ映像の縦横が画面と合っていなければ
    // カメラを取り直して正しいサイズのフレームを得る(録画中は行わない)
    if (!isTouchDevice || !stream || recording || countingDown) return;
    await new Promise((r) => setTimeout(r, 400));
    const [w, h] = sourceSize();
    if (w && h && (w > h) !== land && !recording && !countingDown) {
      try {
        await startCamera();
      } catch (e) {
        console.warn(e);
      }
    }
  }, 150);
}
window.addEventListener('resize', onResize);
window.addEventListener('orientationchange', onResize);

// ---------- 起動 ----------
$('btnStart').onclick = async () => {
  $('startError').textContent = '';
  ensureAudioCtx();
  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
    $('startError').textContent =
      'このブラウザではカメラを使えません。HTTPSで開いているか、Safari / Chrome の最新版か確認してください。';
    return;
  }
  try {
    await startCamera();
    $('start').classList.add('hidden');
    requestWakeLock();
    if (settings.bgMode !== 'none') {
      ensureSegmenter().then((s) => {
        if (!s) {
          settings.bgMode = 'none';
          saveSettings();
        }
        applyPreviewMode();
      });
    }
  } catch (e) {
    console.error(e);
    $('startError').textContent =
      e && e.name === 'NotAllowedError'
        ? 'カメラ/マイクへのアクセスが拒否されました。ブラウザの設定から許可してください。'
        : `カメラを起動できませんでした (${(e && e.name) || e})`;
  }
};

applyPrompterStyle();
applyLandSide();
applyPreviewMode();
renderScript();
updateSpeedLabel();
syncSettingsUI();
updateCanvasSize();
requestAnimationFrame(render);

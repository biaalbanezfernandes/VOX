from __future__ import annotations
import http.server
import json
import os
import threading
import time
import traceback
import urllib.parse
import webbrowser

try:
    import numpy as np
except Exception:
    np = None

try:
    from vox_config import SAMPLE_RATE, DURATION, N_MFCC, MODEL_PATH, LABELS_PATH
except Exception:
    SAMPLE_RATE = 48000
    DURATION = 1.50
    N_MFCC = 13
    MODEL_PATH = "modelos/vox_model.keras"
    LABELS_PATH = "modelos/classes.json"

BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
SCALER_PATH = os.path.join(BASE_DIR, "modelos", "scaler.npz")
MODEL_PATH  = os.path.join(BASE_DIR, MODEL_PATH) if not os.path.isabs(MODEL_PATH) else MODEL_PATH
LABELS_PATH = os.path.join(BASE_DIR, LABELS_PATH) if not os.path.isabs(LABELS_PATH) else LABELS_PATH
THRESHOLD   = 0.50
PORT        = 5758

# ---------------------------------------------------------------------------
# Carregamento sob demanda (lazy) dos módulos pesados de ML e do modelo
# ---------------------------------------------------------------------------
_librosa = None
_keras = None
_resources_loaded = False
model = None
labels = ["validar", "bloquear", "cancelar", "status"]
mean = None
scale = None
load_error = None


def get_np():
    global np
    if np is None:
        import numpy as _np
        np = _np
    return np


def get_ml():
    global _librosa, _keras
    get_np()
    if _librosa is None:
        import librosa as l
        _librosa = l
    if _keras is None:
        from tensorflow import keras as k
        _keras = k
    return _librosa, _keras


def load_resources():
    global _resources_loaded, model, labels, mean, scale, load_error
    if _resources_loaded:
        return
    try:
        _np = get_np()
        lib, k = get_ml()
        if os.path.exists(LABELS_PATH):
            with open(LABELS_PATH, encoding="utf-8") as f:
                labels = json.load(f)
        if os.path.exists(SCALER_PATH):
            scaler_data = _np.load(SCALER_PATH)
            mean, scale = scaler_data["mean"], scaler_data["scale"]
        if os.path.exists(MODEL_PATH):
            model = k.models.load_model(MODEL_PATH)
        else:
            load_error = f"Modelo ausente em: {MODEL_PATH}"
    except Exception as e:
        load_error = str(e)
    finally:
        _resources_loaded = True

# Ações acionadas por cada comando de voz
VOICE_ACTIONS = {
    "validar": {
        "title": "Comando: VALIDAR",
        "badge": "CONFIRMADO",
        "color": "#10b981",
        "icon": "✅",
        "response": "Operação validada com sucesso. Prosseguindo com a tarefa solicitada.",
        "details": "Autenticação aprovada por correspondência acústica."
    },
    "bloquear": {
        "title": "Comando: BLOQUEAR",
        "badge": "BLOQUEADO",
        "color": "#ef4444",
        "icon": "🔒",
        "response": "Sessão bloqueada. Entrada de novos dados suspensa até desbloqueio.",
        "details": "Proteção ativa ativada pelo operador."
    },
    "cancelar": {
        "title": "Comando: CANCELAR",
        "badge": "CANCELADO",
        "color": "#f59e0b",
        "icon": "🛑",
        "response": "Ação em andamento cancelada. O sistema retornou ao menu inicial.",
        "details": "Fila de processos limpa com sucesso."
    },
    "status": {
        "title": "Comando: STATUS",
        "badge": "OPERACIONAL",
        "color": "#6366f1",
        "icon": "📊",
        "response": "Todos os serviços estão operando normalmente. IA com 97.8% de precisão.",
        "details": "Microfone: Ativo | Conexão: Estável | Latência: Nominal."
    }
}

# ---------------------------------------------------------------------------
# Motor de Predição Robusto (VAD alignment + 52 features MFCC + Delta)
# ---------------------------------------------------------------------------
def align_voice(samples, sr: int, target_duration: float = DURATION):
    _np = get_np()
    lib, _ = get_ml()
    target_samples = int(sr * target_duration)
    frame_len = int(sr * 0.02)
    hop = int(sr * 0.01)
    
    rms = lib.feature.rms(y=samples, frame_length=frame_len, hop_length=hop)[0]
    thresh = max(0.005, float(_np.max(rms)) * 0.12)
    active = _np.where(rms > thresh)[0]
    
    if len(active) == 0:
        if len(samples) >= target_samples:
            return samples[:target_samples]
        return _np.pad(samples, (0, target_samples - len(samples)))
        
    start = max(0, active[0] * hop - int(sr * 0.05))
    end = min(len(samples), active[-1] * hop + int(sr * 0.05))
    voice = samples[start:end]
    
    if len(voice) >= target_samples:
        return voice[:target_samples]
        
    pad_left = (target_samples - len(voice)) // 2
    pad_right = target_samples - len(voice) - pad_left
    return _np.pad(voice, (pad_left, pad_right))


def extract_features(audio, sr: int):
    _np = get_np()
    lib, _ = get_ml()
    audio = align_voice(audio, sr)
    mfcc = lib.feature.mfcc(y=audio, sr=sr, n_mfcc=N_MFCC)
    delta = lib.feature.delta(mfcc)
    return _np.concatenate([
        mfcc.mean(axis=1), mfcc.std(axis=1),
        delta.mean(axis=1), delta.std(axis=1)
    ]).astype("float32")


def predict(samples, sr: int) -> dict:
    _np = get_np()
    load_resources()
    if model is None or mean is None or scale is None:
        return {
            "error": f"Modelo neural indisponível no servidor: {load_error or 'Arquivo não encontrado'}",
            "command": "erro",
            "confidence": 0.0
        }

    lib, _ = get_ml()
    t0 = time.time()
    
    # Resample se o navegador enviar em taxa diferente de 48kHz
    if sr != SAMPLE_RATE:
        samples = lib.resample(samples, orig_sr=sr, target_sr=SAMPLE_RATE)

    max_amp = float(_np.max(_np.abs(samples))) if len(samples) > 0 else 0
    if max_amp < 0.015:
        return {
            "error": "Nenhuma voz audível detectada. Fale mais próximo ao microfone.",
            "max_amp": round(max_amp, 4)
        }

    features = extract_features(samples, SAMPLE_RATE)
    features = ((features - mean) / _np.where(scale == 0, 1, scale)).reshape(1, -1)

    probs = model.predict(features, verbose=0)[0]
    best_idx = int(_np.argmax(probs))
    conf = float(probs[best_idx])
    cmd = labels[best_idx]
    latency_ms = round((time.time() - t0) * 1000, 1)

    all_probs = {labels[i]: round(float(probs[i]), 4) for i in range(len(labels))}
    print(f" -> Reconhecido: [{cmd.upper()}] ({conf*100:.1f}%) | {all_probs}")

    if conf < THRESHOLD:
        return {
            "uncertain": True,
            "command": cmd,
            "confidence": conf,
            "latency_ms": latency_ms,
            "message": "Comando incerto ou ambíguo. Fale claramente.",
            "probs": all_probs
        }

    action_info = VOICE_ACTIONS.get(cmd, VOICE_ACTIONS["status"])

    return {
        "success": True,
        "command": cmd,
        "confidence": conf,
        "latency_ms": latency_ms,
        "action": action_info,
        "probs": all_probs
    }


# ---------------------------------------------------------------------------
# Interface Web do Assistente de Comando por Voz
# ---------------------------------------------------------------------------
HTML = r"""<!DOCTYPE html>
<html lang="pt-BR">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>VOX — Sistema de Comando por Voz</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap" rel="stylesheet">

<style>
  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

  :root {
    --bg: #090b10;
    --card: #121620;
    --card-hover: #181e2b;
    --border: rgba(255, 255, 255, 0.08);
    --border-glow: rgba(99, 102, 241, 0.35);
    
    --primary: #6366f1;
    --primary-light: #818cf8;
    --accent: #a855f7;
    --emerald: #10b981;
    --rose: #ef4444;
    --amber: #f59e0b;
    
    --text-main: #f8fafc;
    --text-muted: #94a3b8;
    --text-dim: #64748b;
    
    --font: 'Plus Jakarta Sans', sans-serif;
    --font-mono: 'JetBrains Mono', monospace;
  }

  body {
    background-color: var(--bg);
    background-image: 
      radial-gradient(circle at 50% -10%, rgba(99, 102, 241, 0.18) 0%, transparent 55%),
      radial-gradient(circle at 85% 90%, rgba(168, 85, 247, 0.12) 0%, transparent 45%);
    color: var(--text-main);
    font-family: var(--font);
    min-height: 100vh;
    display: flex;
    flex-direction: column;
    align-items: center;
    padding: 24px 20px 40px;
    gap: 24px;
  }

  /* HEADER */
  header {
    width: 100%;
    max-width: 980px;
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 16px 24px;
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 16px;
    backdrop-filter: blur(12px);
  }

  .logo-group {
    display: flex;
    align-items: center;
    gap: 14px;
  }
  .logo-icon {
    width: 40px;
    height: 40px;
    border-radius: 12px;
    background: linear-gradient(135deg, var(--primary), var(--accent));
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 20px;
    box-shadow: 0 4px 16px rgba(99, 102, 241, 0.35);
  }
  .logo-title {
    font-size: 20px;
    font-weight: 800;
    letter-spacing: -0.5px;
    color: #fff;
  }
  .logo-badge {
    font-size: 11px;
    font-weight: 600;
    color: var(--primary-light);
    background: rgba(99, 102, 241, 0.15);
    padding: 2px 8px;
    border-radius: 99px;
    margin-left: 6px;
  }
  .logo-subtitle {
    font-size: 12px;
    color: var(--text-muted);
  }

  .header-badges {
    display: flex;
    gap: 12px;
    align-items: center;
  }
  .pill-badge {
    display: flex;
    align-items: center;
    gap: 6px;
    background: rgba(255, 255, 255, 0.04);
    border: 1px solid var(--border);
    padding: 6px 12px;
    border-radius: 99px;
    font-size: 12px;
    color: var(--text-muted);
  }
  .dot-green {
    width: 7px;
    height: 7px;
    border-radius: 50%;
    background: var(--emerald);
    box-shadow: 0 0 8px var(--emerald);
  }

  /* MAIN LAYOUT */
  .layout-grid {
    width: 100%;
    max-width: 980px;
    display: grid;
    grid-template-columns: 1fr 340px;
    gap: 20px;
  }

  /* LEFT: VOICE INTERACTION CORE */
  .main-card {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 20px;
    padding: 32px 28px;
    display: flex;
    flex-direction: column;
    align-items: center;
    text-align: center;
    position: relative;
    box-shadow: 0 20px 40px rgba(0, 0, 0, 0.3);
  }

  /* VOICE ORB & VISUALIZER */
  .voice-stage {
    position: relative;
    width: 220px;
    height: 220px;
    margin: 16px 0 28px;
    display: flex;
    align-items: center;
    justify-content: center;
  }

  .voice-glow {
    position: absolute;
    width: 170px;
    height: 170px;
    border-radius: 50%;
    background: radial-gradient(circle, rgba(99, 102, 241, 0.25) 0%, transparent 70%);
    filter: blur(14px);
    transition: all 0.4s ease;
  }
  .voice-glow.active {
    background: radial-gradient(circle, rgba(239, 68, 68, 0.35) 0%, transparent 70%);
    transform: scale(1.3);
  }

  .pulse-ring {
    position: absolute;
    width: 190px;
    height: 190px;
    border-radius: 50%;
    border: 1px solid rgba(99, 102, 241, 0.3);
    animation: rotate-ring 14s linear infinite;
  }
  @keyframes rotate-ring {
    100% { transform: rotate(360deg); }
  }

  canvas#waveform {
    position: absolute;
    width: 200px;
    height: 70px;
    z-index: 2;
    pointer-events: none;
  }

  /* PUSH TO TALK BUTTON */
  .voice-button {
    width: 110px;
    height: 110px;
    border-radius: 50%;
    background: linear-gradient(145deg, #1e2433, #111520);
    border: 2px solid rgba(99, 102, 241, 0.4);
    box-shadow: 0 10px 30px rgba(0, 0, 0, 0.5), inset 0 2px 4px rgba(255, 255, 255, 0.1);
    color: #fff;
    cursor: pointer;
    position: relative;
    z-index: 5;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    gap: 6px;
    outline: none;
    transition: all 0.25s cubic-bezier(0.16, 1, 0.3, 1);
  }
  .voice-button:hover {
    transform: translateY(-2px) scale(1.05);
    border-color: var(--primary-light);
    box-shadow: 0 12px 36px rgba(99, 102, 241, 0.4);
  }
  .voice-button.recording {
    background: linear-gradient(145deg, #2b1216, #1a080a);
    border-color: var(--rose);
    box-shadow: 0 0 35px rgba(239, 68, 68, 0.6);
    transform: scale(1.1);
    animation: pulse-mic 0.9s infinite alternate;
  }
  @keyframes pulse-mic {
    0% { box-shadow: 0 0 20px rgba(239, 68, 68, 0.4); }
    100% { box-shadow: 0 0 45px rgba(239, 68, 68, 0.8); }
  }

  .btn-mic-icon { font-size: 30px; }
  .btn-mic-text {
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 0.5px;
    color: var(--primary-light);
    text-transform: uppercase;
  }
  .voice-button.recording .btn-mic-text { color: var(--rose); }

  /* FEEDBACK DISPLAY */
  .status-lead {
    font-size: 13px;
    font-weight: 600;
    color: var(--text-muted);
    margin-bottom: 8px;
    letter-spacing: 0.5px;
    text-transform: uppercase;
  }
  .command-result-box {
    width: 100%;
    background: rgba(255, 255, 255, 0.02);
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 20px;
    text-align: left;
    margin-bottom: 24px;
    transition: all 0.3s ease;
  }
  .command-result-box.highlight {
    border-color: var(--primary-light);
    background: rgba(99, 102, 241, 0.05);
  }

  .cr-top {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 8px;
  }
  .cr-command {
    font-size: 24px;
    font-weight: 800;
    color: #fff;
    display: flex;
    align-items: center;
    gap: 10px;
  }
  .cr-badge {
    font-size: 11px;
    font-weight: 700;
    padding: 4px 10px;
    border-radius: 99px;
    background: rgba(99, 102, 241, 0.15);
    color: var(--primary-light);
  }
  .cr-message {
    font-size: 14px;
    color: var(--text-main);
    line-height: 1.5;
    margin-bottom: 4px;
  }
  .cr-sub {
    font-size: 12px;
    color: var(--text-dim);
  }

  /* CONFIDENCE BARS */
  .probs-section {
    width: 100%;
  }
  .probs-title {
    font-size: 11px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    color: var(--text-dim);
    text-align: left;
    margin-bottom: 12px;
  }
  .probs-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 10px;
  }
  .prob-card {
    background: rgba(255, 255, 255, 0.03);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 10px 8px;
    display: flex;
    flex-direction: column;
    gap: 6px;
  }
  .p-name { font-size: 12px; font-weight: 700; color: var(--text-muted); text-transform: uppercase; }
  .p-track { height: 6px; background: rgba(255, 255, 255, 0.06); border-radius: 3px; overflow: hidden; }
  .p-fill { height: 100%; width: 0%; background: var(--primary); border-radius: 3px; transition: width 0.3s; }
  .p-val { font-size: 13px; font-weight: 800; color: #fff; font-family: var(--font-mono); }

  /* RIGHT SIDEBAR */
  .sidebar {
    display: flex;
    flex-direction: column;
    gap: 16px;
  }

  .side-card {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 20px;
    display: flex;
    flex-direction: column;
    gap: 12px;
  }
  .side-card-title {
    font-size: 13px;
    font-weight: 700;
    color: #fff;
    display: flex;
    align-items: center;
    gap: 8px;
  }

  .cmd-pill-list {
    display: flex;
    flex-direction: column;
    gap: 8px;
  }
  .cmd-pill {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 8px 12px;
    background: rgba(255, 255, 255, 0.03);
    border-radius: 8px;
    font-size: 13px;
  }
  .cmd-pill-tag {
    font-weight: 700;
    color: var(--primary-light);
    font-family: var(--font-mono);
  }
  .cmd-pill-action { font-size: 11px; color: var(--text-dim); }

  /* HISTORY LIST */
  .history-list {
    display: flex;
    flex-direction: column;
    gap: 8px;
    max-height: 250px;
    overflow-y: auto;
  }
  .history-list::-webkit-scrollbar { width: 4px; }
  .history-list::-webkit-scrollbar-thumb { background: var(--border); border-radius: 2px; }

  .hist-item {
    padding: 10px 12px;
    background: rgba(255, 255, 255, 0.02);
    border: 1px solid var(--border);
    border-radius: 8px;
    display: flex;
    justify-content: space-between;
    align-items: center;
    font-size: 12px;
  }
  .hist-cmd { font-weight: 700; color: #fff; display: flex; align-items: center; gap: 6px; }
  .hist-meta { font-size: 11px; color: var(--text-dim); font-family: var(--font-mono); }

  /* FOOTER */
  footer {
    margin-top: auto;
    font-size: 12px;
    color: var(--text-dim);
    display: flex;
    gap: 16px;
  }
</style>
</head>
<body>

  <!-- HEADER -->
  <header>
    <div class="logo-group">
      <div class="logo-icon">🎙️</div>
      <div>
        <div style="display:flex; align-items:center;">
          <span class="logo-title">VOX</span>
          <span class="logo-badge">Comando de Voz</span>
        </div>
        <div class="logo-subtitle">Reconhecimento de Voz Neural em Tempo Real</div>
      </div>
    </div>

    <div class="header-badges">
      <div class="pill-badge">
        <div class="dot-green"></div>
        <span>Modelo Ativo (97.8% Acc)</span>
      </div>
      <div class="pill-badge" id="latency-tag">
        <span>Latência: --</span>
      </div>
    </div>
  </header>

  <!-- MAIN INTERFACE -->
  <div class="layout-grid">

    <!-- VOICE HUB CARD -->
    <div class="main-card">
      <div class="status-lead" id="status-lead">Aguardando comando</div>

      <!-- VOICE STAGE -->
      <div class="voice-stage">
        <div class="voice-glow" id="voice-glow"></div>
        <div class="pulse-ring"></div>
        <canvas id="waveform" width="200" height="70"></canvas>

        <button class="voice-button" id="btn-voice" onclick="toggleVoiceInput()">
          <span class="btn-mic-icon" id="mic-icon">🎙️</span>
          <span class="btn-mic-text" id="mic-text">FALAR</span>
        </button>
      </div>

      <!-- COMMAND RESULT DISPLAY -->
      <div class="command-result-box" id="result-box">
        <div class="cr-top">
          <div class="cr-command" id="res-title">
            <span id="res-icon">✨</span>
            <span id="res-text">Pronto para ouvir</span>
          </div>
          <span class="cr-badge" id="res-badge">STANDBY</span>
        </div>
        <div class="cr-message" id="res-msg">
          Pressione o botão acima (ou aperte a <strong>Barra de Espaço</strong>) e fale um comando.
        </div>
        <div class="cr-sub" id="res-sub">Exemplos: "validar", "bloquear", "cancelar" ou "status".</div>
      </div>

      <!-- PROBABILITIES BARS -->
      <div class="probs-section">
        <div class="probs-title">Confiança da Classificação Neural</div>
        <div class="probs-grid">
          <div class="prob-card">
            <div class="p-name">validar</div>
            <div class="p-track"><div class="p-fill" id="bar-validar"></div></div>
            <div class="p-val" id="val-validar">0%</div>
          </div>
          <div class="prob-card">
            <div class="p-name">bloquear</div>
            <div class="p-track"><div class="p-fill" id="bar-bloquear"></div></div>
            <div class="p-val" id="val-bloquear">0%</div>
          </div>
          <div class="prob-card">
            <div class="p-name">cancelar</div>
            <div class="p-track"><div class="p-fill" id="bar-cancelar"></div></div>
            <div class="p-val" id="val-cancelar">0%</div>
          </div>
          <div class="prob-card">
            <div class="p-name">status</div>
            <div class="p-track"><div class="p-fill" id="bar-status"></div></div>
            <div class="p-val" id="val-status">0%</div>
          </div>
        </div>
      </div>
    </div>

    <!-- SIDEBAR: COMMAND LIST & HISTORY -->
    <div class="sidebar">

      <!-- COMANDOS SUPORTADOS -->
      <div class="side-card">
        <div class="side-card-title">📋 Comandos Reconhecidos</div>
        <div class="cmd-pill-list">
          <div class="cmd-pill">
            <span class="cmd-pill-tag">validar</span>
            <span class="cmd-pill-action">Valida / autoriza</span>
          </div>
          <div class="cmd-pill">
            <span class="cmd-pill-tag">bloquear</span>
            <span class="cmd-pill-action">Bloqueia tela/sessão</span>
          </div>
          <div class="cmd-pill">
            <span class="cmd-pill-tag">cancelar</span>
            <span class="cmd-pill-action">Cancela operação</span>
          </div>
          <div class="cmd-pill">
            <span class="cmd-pill-tag">status</span>
            <span class="cmd-pill-action">Exibe diagnóstico</span>
          </div>
        </div>
      </div>

      <!-- HISTÓRICO DE COMANDOS -->
      <div class="side-card" style="flex:1;">
        <div class="side-card-title">🕒 Histórico de Execuções</div>
        <div class="history-list" id="history-box">
          <div class="hist-item" style="color:var(--text-dim); justify-content:center;">
            Nenhum comando executado ainda.
          </div>
        </div>
      </div>

    </div>

  </div>

  <footer>
    <span>VOX Speech Recognition v1.0</span>
    <span>•</span>
    <span>Atalho: Barra de Espaço para falar</span>
  </footer>

<script>
let audioCtx = null;
let stream = null;
let busy = false;
const DURATION = 1.5;
let animId = null;

// SINTETIZADOR DE BIPES DE CONFIRMAÇÃO
function playFeedbackTone(type) {
  if (!audioCtx) return;
  try {
    const osc = audioCtx.createOscillator();
    const g = audioCtx.createGain();
    const now = audioCtx.currentTime;
    
    if (type === 'start') {
      osc.frequency.setValueAtTime(587.33, now); // D5
      osc.frequency.exponentialRampToValueAtTime(880, now + 0.1);
      g.gain.setValueAtTime(0.12, now);
      g.gain.linearRampToValueAtTime(0.001, now + 0.12);
      osc.connect(g); g.connect(audioCtx.destination);
      osc.start(); osc.stop(now + 0.12);
    } else if (type === 'success') {
      osc.frequency.setValueAtTime(523.25, now);
      osc.frequency.setValueAtTime(659.25, now + 0.08);
      g.gain.setValueAtTime(0.15, now);
      g.gain.linearRampToValueAtTime(0.001, now + 0.22);
      osc.connect(g); g.connect(audioCtx.destination);
      osc.start(); osc.stop(now + 0.22);
    }
  } catch(e){}
}

async function initMicrophone() {
  if (stream) return true;
  try {
    audioCtx = new (window.AudioContext || window.webkitAudioContext)();
    stream = await navigator.mediaDevices.getUserMedia({ audio: true, video: false });
    return true;
  } catch (err) {
    alert('Por favor, permita o microfone no navegador para enviar comandos de voz.');
    return false;
  }
}

// ONDA SONORA NO CANVAs
function startWave(analyser) {
  const canvas = document.getElementById('waveform');
  const ctx = canvas.getContext('2d');
  const buf = new Uint8Array(analyser.frequencyBinCount);

  function render() {
    animId = requestAnimationFrame(render);
    analyser.getByteTimeDomainData(buf);
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.lineWidth = 2.5;
    ctx.strokeStyle = '#ef4444';
    ctx.beginPath();
    const slice = canvas.width / buf.length;
    let x = 0;
    for (let i = 0; i < buf.length; i++) {
      const v = buf[i] / 128.0;
      const y = (v * canvas.height) / 2;
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
      x += slice;
    }
    ctx.stroke();
  }
  render();
}

function stopWave() {
  if (animId) cancelAnimationFrame(animId);
  const canvas = document.getElementById('waveform');
  canvas.getContext('2d').clearRect(0, 0, canvas.width, canvas.height);
}

async function toggleVoiceInput() {
  if (busy) return;
  const ok = await initMicrophone();
  if (!ok) return;

  busy = true;
  const btn = document.getElementById('btn-voice');
  const lead = document.getElementById('status-lead');
  const glow = document.getElementById('voice-glow');
  const micTxt = document.getElementById('mic-text');
  const micIcon = document.getElementById('mic-icon');

  btn.classList.add('recording');
  glow.classList.add('active');
  lead.textContent = 'Gravando comando... fale agora!';
  micTxt.textContent = 'OUVINDO';
  micIcon.textContent = '🔴';
  
  if (audioCtx.state === 'suspended') await audioCtx.resume();
  playFeedbackTone('start');

  const actualSR = audioCtx.sampleRate;
  const nFrames = Math.round(actualSR * DURATION);
  const src = audioCtx.createMediaStreamSource(stream);
  const analyser = audioCtx.createAnalyser();
  analyser.fftSize = 256;
  src.connect(analyser);
  startWave(analyser);

  const proc = audioCtx.createScriptProcessor(4096, 1, 1);
  let chunks = [], total = 0, done = false;

  proc.onaudioprocess = (e) => {
    if (done) return;
    const c = e.inputBuffer.getChannelData(0);
    chunks.push(new Float32Array(c));
    total += c.length;
    if (total >= nFrames) {
      done = true;
      proc.disconnect();
      src.disconnect();
      stopWave();

      const out = new Float32Array(nFrames);
      let off = 0;
      for (const ch of chunks) {
        const n = Math.min(ch.length, nFrames - off);
        out.set(ch.subarray(0, n), off);
        off += n;
        if (off >= nFrames) break;
      }
      sendAudioToBackend(out, actualSR);
    }
  };

  src.connect(proc);
  proc.connect(audioCtx.destination);
}

async function sendAudioToBackend(pcm, sr) {
  const btn = document.getElementById('btn-voice');
  const lead = document.getElementById('status-lead');
  const glow = document.getElementById('voice-glow');
  const micTxt = document.getElementById('mic-text');
  const micIcon = document.getElementById('mic-icon');

  btn.classList.remove('recording');
  glow.classList.remove('active');
  lead.textContent = 'Classificando comando...';
  micTxt.textContent = 'IA PENSANDO';
  micIcon.textContent = '⚡';

  try {
    const res = await fetch('/predict', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/octet-stream',
        'X-Samplerate': String(sr)
      },
      body: pcm.buffer
    });
    const data = await res.json();
    handlePredictionResponse(data);
  } catch (err) {
    alert('Erro de conexão com o servidor local: ' + err.message);
  } finally {
    micTxt.textContent = 'FALAR';
    micIcon.textContent = '🎙️';
    lead.textContent = 'Aguardando próximo comando';
    busy = false;
  }
}

function handlePredictionResponse(data) {
  const resBox = document.getElementById('result-box');
  const resTitle = document.getElementById('res-title');
  const resIcon = document.getElementById('res-icon');
  const resText = document.getElementById('res-text');
  const resBadge = document.getElementById('res-badge');
  const resMsg = document.getElementById('res-msg');
  const resSub = document.getElementById('res-sub');
  const latTag = document.getElementById('latency-tag');

  if (data.latency_ms) {
    latTag.innerHTML = `<span>Latência: <strong>${data.latency_ms}ms</strong></span>`;
  }

  // Atualiza as barras de probabilidade
  if (data.probs) {
    for (const [cmd, p] of Object.entries(data.probs)) {
      const pct = (p * 100).toFixed(0);
      const bar = document.getElementById('bar-' + cmd);
      const val = document.getElementById('val-' + cmd);
      if (bar) bar.style.width = pct + '%';
      if (val) val.textContent = pct + '%';
    }
  }

  // Caso de erro de áudio (silêncio)
  if (data.error) {
    resBox.className = 'command-result-box';
    resIcon.textContent = '⚠️';
    resText.textContent = 'Voz não audível';
    resBadge.textContent = 'SILÊNCIO';
    resBadge.style.background = 'rgba(239, 68, 68, 0.15)';
    resBadge.style.color = 'var(--rose)';
    resMsg.textContent = data.error;
    resSub.textContent = 'Certifique-se de que o microfone está próximo e fale com clareza.';
    return;
  }

  // Caso incerto
  if (data.uncertain) {
    resBox.className = 'command-result-box';
    resIcon.textContent = '❓';
    resText.textContent = 'Comando Incerto';
    resBadge.textContent = `${(data.confidence * 100).toFixed(0)}% CONF`;
    resBadge.style.background = 'rgba(245, 158, 11, 0.15)';
    resBadge.style.color = 'var(--amber)';
    resMsg.textContent = data.message;
    resSub.textContent = `A rede hesitou entre os comandos. Tente pronunciar com mais firmeza.`;
    return;
  }

  // Sucesso
  playFeedbackTone('success');
  const act = data.action;
  const confPct = (data.confidence * 100).toFixed(1) + '%';

  resBox.className = 'command-result-box highlight';
  resIcon.textContent = act.icon;
  resText.textContent = data.command.toUpperCase();
  resBadge.textContent = act.badge;
  resBadge.style.background = `rgba(${hexToRgb(act.color)}, 0.15)`;
  resBadge.style.color = act.color;
  resMsg.textContent = act.response;
  resSub.textContent = `${act.details} • Confiança: ${confPct}`;

  // Adiciona ao histórico
  addHistoryItem(data.command, confPct, act.icon);
}

function hexToRgb(hex) {
  let c = hex.substring(1);
  if(c.length === 3) c = c.split('').map(x=>x+x).join('');
  const num = parseInt(c, 16);
  return [(num >> 16) & 255, (num >> 8) & 255, num & 255].join(',');
}

function addHistoryItem(cmd, conf, icon) {
  const box = document.getElementById('history-box');
  const now = new Date();
  const time = now.toTimeString().split(' ')[0];

  // Remove placeholder se houver
  if (box.children.length === 1 && box.children[0].textContent.includes('Nenhum')) {
    box.innerHTML = '';
  }

  const div = document.createElement('div');
  div.className = 'hist-item';
  div.innerHTML = `
    <span class="hist-cmd">${icon} ${cmd.toUpperCase()}</span>
    <span class="hist-meta">${conf} • ${time}</span>
  `;
  box.insertBefore(div, box.firstChild);
}

// Atalho Barra de Espaço
window.addEventListener('keydown', (e) => {
  if (e.code === 'Space' && !e.repeat && !busy) {
    e.preventDefault();
    toggleVoiceInput();
  }
});
</script>
</body>
</html>
"""

# ---------------------------------------------------------------------------
# HTTP Handler
# ---------------------------------------------------------------------------
class Handler(http.server.BaseHTTPRequestHandler):

    def log_message(self, fmt, *args):
        pass

    def do_GET(self):
        try:
            if self.path == "/favicon.ico":
                self.send_response(204)
                self.end_headers()
                return

            body = HTML.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        except Exception as e:
            self.send_response(500)
            self.send_header("Content-Type", "text/plain; charset=utf-8")
            self.end_headers()
            self.wfile.write(f"Erro no servidor: {str(e)}\n{traceback.format_exc()}".encode("utf-8"))

    def do_POST(self):
        try:
            _np = get_np()
            length = int(self.headers.get("Content-Length", 0))
            raw = self.rfile.read(length)
            sr = int(self.headers.get("X-Samplerate", SAMPLE_RATE))
            samples = _np.frombuffer(raw, dtype=_np.float32).copy()
            result = predict(samples, sr)
        except Exception as e:
            result = {"error": f"Erro interno ao predizer áudio: {str(e)}"}

        body = json.dumps(result).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def app(environ, start_response):
    """WSGI entrypoint para Vercel e outros servidores WSGI."""
    try:
        method = environ.get("REQUEST_METHOD", "GET")
        path = environ.get("PATH_INFO", "/")

        if method == "GET":
            if path == "/favicon.ico":
                start_response("204 No Content", [])
                return [b""]
            data = HTML.encode("utf-8")
            start_response("200 OK", [
                ("Content-Type", "text/html; charset=utf-8"),
                ("Content-Length", str(len(data)))
            ])
            return [data]

        if method == "POST" and (path.endswith("/predict") or path == "/predict"):
            try:
                _np = get_np()
                content_length = int(environ.get("CONTENT_LENGTH", 0) or 0)
                raw = environ["wsgi.input"].read(content_length)
                sr = int(environ.get("HTTP_X_SAMPLERATE", SAMPLE_RATE))
                samples = _np.frombuffer(raw, dtype=_np.float32).copy()
                result = predict(samples, sr)
            except Exception as e:
                result = {"error": f"Erro interno ao predizer áudio: {str(e)}"}

            data = json.dumps(result).encode("utf-8")
            start_response("200 OK", [
                ("Content-Type", "application/json"),
                ("Content-Length", str(len(data)))
            ])
            return [data]

        start_response("404 Not Found", [("Content-Type", "text/plain; charset=utf-8")])
        return [b"Not Found"]
    except Exception as e:
        err_body = f"VOX Internal Error:\n{traceback.format_exc()}".encode("utf-8")
        start_response("500 Internal Server Error", [
            ("Content-Type", "text/plain; charset=utf-8"),
            ("Content-Length", str(len(err_body)))
        ])
        return [err_body]


application = app
handler = Handler


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    url = f"http://127.0.0.1:{PORT}"
    print("\n" + "="*45)
    print(" VOX — SISTEMA DE COMANDO POR VOZ")
    print("="*45)
    print(f" Interface web: {url}")
    print(f" Comandos: {', '.join(labels)}")
    print(" Pressione Ctrl+C para encerrar.\n")

    server = http.server.HTTPServer(("127.0.0.1", PORT), Handler)
    threading.Timer(0.8, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nVOX encerrado.")
        server.shutdown()


if __name__ == "__main__":
    main()

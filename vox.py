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

BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
SCALER_PATH  = os.path.join(BASE_DIR, "modelos", "scaler.npz")
WEIGHTS_PATH = os.path.join(BASE_DIR, "modelos", "weights.npz")
MODEL_PATH   = os.path.join(BASE_DIR, MODEL_PATH) if not os.path.isabs(MODEL_PATH) else MODEL_PATH
LABELS_PATH  = os.path.join(BASE_DIR, LABELS_PATH) if not os.path.isabs(LABELS_PATH) else LABELS_PATH
THRESHOLD    = 0.50
PORT         = 5758

# ---------------------------------------------------------------------------
# Carregamento sob demanda (lazy) dos módulos pesados de ML e do modelo
# ---------------------------------------------------------------------------
_librosa = None
_resources_loaded = False
model = None
model_weights = None
labels = ["validar", "bloquear", "cancelar", "status"]
mean = None
scale = None
load_error = None


_np_import_error = None

def get_np():
    global np, _np_import_error
    if np is not None:
        return np
    try:
        import numpy as _np
        np = _np
        return np
    except Exception as e:
        _np_import_error = f"{type(e).__name__}: {str(e)}"
        raise RuntimeError(f"Erro ao carregar NumPy no servidor: {_np_import_error}")


def get_librosa():
    global _librosa
    get_np()
    if _librosa is None:
        import librosa as l
        _librosa = l
    return _librosa


def load_resources():
    global _resources_loaded, model, model_weights, labels, mean, scale, load_error
    if _resources_loaded:
        return
    try:
        _np = get_np()
        get_librosa()
        if os.path.exists(LABELS_PATH):
            with open(LABELS_PATH, encoding="utf-8") as f:
                labels = json.load(f)
        if os.path.exists(SCALER_PATH):
            scaler_data = _np.load(SCALER_PATH)
            mean, scale = scaler_data["mean"], scaler_data["scale"]
        if os.path.exists(WEIGHTS_PATH):
            d = _np.load(WEIGHTS_PATH)
            model_weights = [d["w_0"], d["w_1"], d["w_2"], d["w_3"], d["w_4"], d["w_5"]]
        elif os.path.exists(MODEL_PATH):
            try:
                from tensorflow import keras as k
                model = k.models.load_model(MODEL_PATH)
            except Exception as e:
                load_error = f"Keras load error: {e}"
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
    lib = get_librosa()
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
    lib = get_librosa()
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
    if (model is None and model_weights is None) or mean is None or scale is None:
        return {
            "error": f"Modelo neural indisponível no servidor: {load_error or 'Arquivo não encontrado'}",
            "command": "erro",
            "confidence": 0.0
        }

    lib = get_librosa()
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

    if model_weights is not None:
        w0, b0, w1, b1, w2, b2 = model_weights
        h1 = _np.maximum(0, _np.dot(features, w0) + b0)
        h2 = _np.maximum(0, _np.dot(h1, w1) + b1)
        logits = _np.dot(h2, w2) + b2
        exp = _np.exp(logits - _np.max(logits))
        probs = (exp / _np.sum(exp))[0]
    else:
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

  /* ===== LOCK SCREEN OVERLAY ===== */
  #lock-overlay {
    position: fixed;
    inset: 0;
    z-index: 9999;
    display: none;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    gap: 32px;
    background: rgba(5, 5, 10, 0.97);
    backdrop-filter: blur(24px);
    animation: lockIn 0.4s cubic-bezier(0.16, 1, 0.3, 1);
  }
  #lock-overlay.visible { display: flex; }
  @keyframes lockIn {
    from { opacity: 0; transform: scale(1.04); }
    to   { opacity: 1; transform: scale(1); }
  }
  .lock-glow-ring {
    position: absolute;
    width: 280px; height: 280px;
    border-radius: 50%;
    border: 2px solid rgba(239, 68, 68, 0.4);
    animation: lockPulse 1.8s ease-in-out infinite;
  }
  .lock-glow-ring:nth-child(2) { animation-delay: 0.6s; }
  .lock-glow-ring:nth-child(3) { animation-delay: 1.2s; }
  @keyframes lockPulse {
    0%  { transform: scale(0.9); opacity: 0.7; }
    50% { transform: scale(1.2); opacity: 0.15; }
    100%{ transform: scale(0.9); opacity: 0.7; }
  }
  .lock-center {
    position: relative;
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 20px;
    text-align: center;
  }
  .lock-icon-wrap {
    width: 120px; height: 120px;
    border-radius: 50%;
    background: radial-gradient(circle, rgba(239,68,68,0.25), transparent 70%);
    display: flex; align-items: center; justify-content: center;
    font-size: 64px;
    animation: lockBounce 0.6s cubic-bezier(0.34,1.56,0.64,1);
  }
  @keyframes lockBounce {
    from { transform: scale(0.2); opacity: 0; }
    to   { transform: scale(1);   opacity: 1; }
  }
  .lock-title {
    font-size: 42px;
    font-weight: 900;
    letter-spacing: -1px;
    color: #fff;
    text-shadow: 0 0 40px rgba(239,68,68,0.6);
  }
  .lock-sub {
    font-size: 15px;
    color: var(--text-muted);
    max-width: 360px;
    line-height: 1.6;
  }
  .lock-timestamp {
    font-family: var(--font-mono);
    font-size: 13px;
    color: var(--rose);
    background: rgba(239,68,68,0.12);
    border: 1px solid rgba(239,68,68,0.3);
    padding: 6px 18px;
    border-radius: 99px;
  }
  .unlock-btn {
    margin-top: 8px;
    padding: 14px 40px;
    font-size: 15px;
    font-weight: 700;
    border-radius: 99px;
    background: linear-gradient(135deg, #ef4444, #b91c1c);
    color: #fff;
    border: none;
    cursor: pointer;
    letter-spacing: 0.5px;
    box-shadow: 0 6px 30px rgba(239,68,68,0.45);
    transition: all 0.2s;
  }
  .unlock-btn:hover { transform: translateY(-2px); box-shadow: 0 10px 40px rgba(239,68,68,0.6); }
  .unlock-btn:active { transform: scale(0.97); }

  /* ===== COMMAND STATUS BANNER ===== */
  #cmd-status-banner {
    width: 100%;
    max-width: 980px;
    border-radius: 16px;
    padding: 18px 24px;
    display: none;
    align-items: center;
    gap: 16px;
    border: 1px solid;
    transition: all 0.4s;
    animation: bannerIn 0.4s cubic-bezier(0.16, 1, 0.3, 1);
  }
  #cmd-status-banner.show { display: flex; }
  @keyframes bannerIn {
    from { opacity: 0; transform: translateY(-12px); }
    to   { opacity: 1; transform: translateY(0); }
  }
  .banner-icon { font-size: 32px; }
  .banner-info { flex: 1; }
  .banner-label { font-size: 11px; font-weight: 700; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 4px; }
  .banner-title { font-size: 20px; font-weight: 800; color: #fff; }
  .banner-detail { font-size: 12px; color: var(--text-muted); margin-top: 2px; }
  .banner-status-pill {
    padding: 8px 20px;
    border-radius: 99px;
    font-size: 13px;
    font-weight: 800;
    letter-spacing: 0.5px;
    font-family: var(--font-mono);
    display: flex; align-items: center; gap: 8px;
  }
  .status-dot {
    width: 9px; height: 9px;
    border-radius: 50%;
    animation: statusBlink 1.2s ease-in-out infinite;
  }
  @keyframes statusBlink {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.3; }
  }

  /* ===== AI CHARTS SECTION ===== */
  .charts-section {
    width: 100%;
    max-width: 980px;
    display: flex;
    flex-direction: column;
    gap: 16px;
  }
  .charts-header {
    display: flex;
    align-items: center;
    gap: 12px;
  }
  .charts-title {
    font-size: 18px;
    font-weight: 800;
    color: #fff;
    letter-spacing: -0.3px;
  }
  .charts-subtitle {
    font-size: 13px;
    color: var(--text-muted);
    margin-left: auto;
  }
  .charts-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 20px;
  }
  .chart-card {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 16px;
    overflow: hidden;
    transition: border-color 0.3s, box-shadow 0.3s;
  }
  .chart-card:hover {
    border-color: var(--border-glow);
    box-shadow: 0 8px 32px rgba(99,102,241,0.15);
  }
  .chart-card-header {
    padding: 16px 20px 12px;
    display: flex;
    align-items: center;
    gap: 10px;
    border-bottom: 1px solid var(--border);
  }
  .chart-card-icon { font-size: 20px; }
  .chart-card-title { font-size: 14px; font-weight: 700; color: #fff; }
  .chart-card-sub { font-size: 11px; color: var(--text-dim); margin-top: 2px; }
  .chart-img-wrap {
    padding: 16px;
    background: #0d1117;
  }
  .chart-img-wrap img {
    width: 100%;
    border-radius: 8px;
    display: block;
    opacity: 0.9;
    transition: opacity 0.2s;
  }
  .chart-img-wrap img:hover { opacity: 1; }
  .chart-placeholder {
    height: 200px;
    display: flex;
    align-items: center;
    justify-content: center;
    flex-direction: column;
    gap: 8px;
    color: var(--text-dim);
    font-size: 13px;
  }
  .chart-placeholder-icon { font-size: 32px; opacity: 0.4; }

  @media (max-width: 700px) {
    .charts-grid { grid-template-columns: 1fr; }
    .layout-grid { grid-template-columns: 1fr; }
  }
</style>
</head>
<body>

  <!-- LOCK SCREEN OVERLAY -->
  <div id="lock-overlay">
    <div class="lock-glow-ring"></div>
    <div class="lock-glow-ring"></div>
    <div class="lock-glow-ring"></div>
    <div class="lock-center">
      <div class="lock-icon-wrap">🔒</div>
      <div class="lock-title">SISTEMA BLOQUEADO</div>
      <div class="lock-sub">Sessão suspensa por comando de voz. Nenhuma entrada aceita até desbloqueio manual.</div>
      <div class="lock-timestamp" id="lock-ts">--:--:--</div>
      <button class="unlock-btn" onclick="unlockSystem()" id="unlock-btn">🔓 Desbloquear Sistema</button>
    </div>
  </div>

  <!-- COMMAND STATUS BANNER -->
  <div id="cmd-status-banner">
    <span class="banner-icon" id="banner-icon">✨</span>
    <div class="banner-info">
      <div class="banner-label" id="banner-label">ÚLTIMO COMANDO</div>
      <div class="banner-title" id="banner-title">—</div>
      <div class="banner-detail" id="banner-detail">—</div>
    </div>
    <div class="banner-status-pill" id="banner-pill">
      <span class="status-dot" id="banner-dot"></span>
      <span id="banner-pill-text">—</span>
    </div>
  </div>

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

  <!-- AI TRAINING CHARTS SECTION -->
  <section class="charts-section">
    <div class="charts-header">
      <span style="font-size:22px;">📈</span>
      <span class="charts-title">Resultados do Treinamento da IA</span>
      <span class="charts-subtitle">Gerado por <code>treinar_modelo.py</code> · Rede Neural MFCCs+Delta</span>
    </div>
    <div class="charts-grid">

      <!-- ACURÁCIA -->
      <div class="chart-card" id="chart-acc">
        <div class="chart-card-header">
          <span class="chart-card-icon">📉</span>
          <div>
            <div class="chart-card-title">Evolução da Acurácia</div>
            <div class="chart-card-sub">Treino vs. Validação por época</div>
          </div>
        </div>
        <div class="chart-img-wrap" id="acc-wrap">
          <div class="chart-placeholder" id="acc-placeholder">
            <span class="chart-placeholder-icon">🖼️</span>
            <span>Verificando gráfico...</span>
          </div>
        </div>
      </div>

      <!-- MATRIZ DE CONFUSÃO -->
      <div class="chart-card" id="chart-cm">
        <div class="chart-card-header">
          <span class="chart-card-icon">🔢</span>
          <div>
            <div class="chart-card-title">Matriz de Confusão</div>
            <div class="chart-card-sub">Classificação normalizada por classe</div>
          </div>
        </div>
        <div class="chart-img-wrap" id="cm-wrap">
          <div class="chart-placeholder" id="cm-placeholder">
            <span class="chart-placeholder-icon">🖼️</span>
            <span>Verificando gráfico...</span>
          </div>
        </div>
      </div>

    </div>
  </section>

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

  // Caso de erro de áudio ou servidor
  if (data.error) {
    const isSilence = data.error.toLowerCase().includes('nenhuma voz') || data.error.toLowerCase().includes('audível');
    resBox.className = 'command-result-box';
    resIcon.textContent = '⚠️';
    resText.textContent = isSilence ? 'Voz não audível' : 'Erro no Servidor';
    resBadge.textContent = isSilence ? 'SILÊNCIO' : 'ERRO IA';
    resBadge.style.background = 'rgba(239, 68, 68, 0.15)';
    resBadge.style.color = 'var(--rose)';
    resMsg.textContent = data.error;
    resSub.textContent = isSilence
      ? 'Certifique-se de que o microfone está próximo e fale com clareza.'
      : 'Ocorreu um erro no processamento do comando no servidor.';
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

  // Mostra banner de status
  showBanner(act.icon, act.title, act.badge, act.color, act.details + ' · Confiança: ' + confPct);

  // Ativa bloqueio visual se comando for 'bloquear'
  if (data.command === 'bloquear') {
    setTimeout(() => activateLock(), 500);
  }
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

// ===== LOCK SYSTEM =====
let systemLocked = false;

function activateLock() {
  systemLocked = true;
  const overlay = document.getElementById('lock-overlay');
  const ts = document.getElementById('lock-ts');
  const now = new Date();
  ts.textContent = 'Bloqueado em: ' + now.toLocaleTimeString('pt-BR') + ' · ' + now.toLocaleDateString('pt-BR');
  overlay.classList.add('visible');
  // buzz-like visual on header
  document.querySelector('header').style.borderColor = 'rgba(239,68,68,0.5)';
  document.querySelector('header').style.boxShadow = '0 0 30px rgba(239,68,68,0.2)';
  busy = true; // prevent recording while locked
}

function unlockSystem() {
  systemLocked = false;
  const overlay = document.getElementById('lock-overlay');
  overlay.classList.remove('visible');
  document.querySelector('header').style.borderColor = '';
  document.querySelector('header').style.boxShadow = '';
  busy = false;
  showBanner('🔓', 'Sistema desbloqueado manualmente.', 'DESBLOQUEADO', '#10b981', 'Pronto para receber novos comandos.');
}

// ===== COMMAND STATUS BANNER =====
let bannerTimer = null;

function showBanner(icon, title, pill, color, detail) {
  const banner = document.getElementById('cmd-status-banner');
  document.getElementById('banner-icon').textContent = icon;
  document.getElementById('banner-title').textContent = title;
  document.getElementById('banner-detail').textContent = detail;
  document.getElementById('banner-pill-text').textContent = pill;
  document.getElementById('banner-label').textContent = 'STATUS DO SISTEMA';

  const pillEl = document.getElementById('banner-pill');
  const dot = document.getElementById('banner-dot');
  pillEl.style.background = `rgba(${hexToRgb(color)}, 0.15)`;
  pillEl.style.color = color;
  dot.style.background = color;

  banner.style.background = `rgba(${hexToRgb(color)}, 0.06)`;
  banner.style.borderColor = `rgba(${hexToRgb(color)}, 0.3)`;

  banner.classList.remove('show');
  void banner.offsetWidth; // reflow to restart animation
  banner.classList.add('show');

  if (bannerTimer) clearTimeout(bannerTimer);
  if (pill !== 'BLOQUEADO') {
    bannerTimer = setTimeout(() => banner.classList.remove('show'), 8000);
  }
}


// ===== LOAD AI CHARTS =====
function loadChartImage(imgSrc, wrapId, placeholderId) {
  const wrap = document.getElementById(wrapId);
  const placeholder = document.getElementById(placeholderId);
  const img = new Image();
  img.onload = () => {
    placeholder.style.display = 'none';
    img.alt = wrapId.includes('acc') ? 'Gráfico de acurácia do modelo VOX' : 'Matriz de confusão do modelo VOX';
    wrap.appendChild(img);
  };
  img.onerror = () => {
    placeholder.innerHTML = '<span class="chart-placeholder-icon" style="font-size:28px;opacity:.4;">📊</span><span style="color:var(--text-dim);font-size:12px;">Gráfico não disponível.<br>Execute <code>python treinar_modelo.py</code> para gerá-lo.</span>';
  };
  img.src = imgSrc + '?t=' + Date.now();
}

document.addEventListener('DOMContentLoaded', () => {
  loadChartImage('/resultados/acuracia.png', 'acc-wrap', 'acc-placeholder');
  loadChartImage('/resultados/matriz_confusao.png', 'cm-wrap', 'cm-placeholder');
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

            # Serve static image files from resultados/
            if self.path.startswith("/resultados/"):
                fname = os.path.basename(self.path)
                fpath = os.path.join(BASE_DIR, "resultados", fname)
                if os.path.exists(fpath) and fname.lower().endswith(".png"):
                    with open(fpath, "rb") as f:
                        img_bytes = f.read()
                    self.send_response(200)
                    self.send_header("Content-Type", "image/png")
                    self.send_header("Content-Length", str(len(img_bytes)))
                    self.end_headers()
                    self.wfile.write(img_bytes)
                else:
                    self.send_response(404)
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
            if path.startswith("/resultados/"):
                fname = os.path.basename(path)
                fpath = os.path.join(BASE_DIR, "resultados", fname)
                if os.path.exists(fpath) and fname.lower().endswith(".png"):
                    with open(fpath, "rb") as f:
                        img_bytes = f.read()
                    start_response("200 OK", [
                        ("Content-Type", "image/png"),
                        ("Content-Length", str(len(img_bytes)))
                    ])
                    return [img_bytes]
                else:
                    start_response("404 Not Found", [])
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

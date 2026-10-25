"""Grava amostras de voz via navegador web (funciona em computadores restritos).

Execute: python gravar_amostras.py
O script abre o navegador automaticamente. Permita o uso do microfone
quando o browser pedir — isso usa a permissão do navegador, independente
das restrições do Windows para apps desktop.
"""
import base64
import http.server
import json
import os
import threading
import time
import webbrowser

import numpy as np
import soundfile as sf
from vox_config import COMMANDS, SAMPLE_RATE, DURATION

PORT = 5757
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(BASE_DIR, "dataset")

# ---------------------------------------------------------------------------
# Estado compartilhado (thread-safe via lock)
# ---------------------------------------------------------------------------
_lock = threading.Lock()

def _init_state():
    cmd_idx = 0
    folder = os.path.join(ROOT, COMMANDS[cmd_idx])
    os.makedirs(folder, exist_ok=True)
    existing = len([f for f in os.listdir(folder) if f.lower().endswith(".wav")])
    return {
        "cmd_idx": cmd_idx,
        "count":   existing,
        "target":  max(20, existing + 1),
        "done":    False,
    }

state = _init_state()

# ---------------------------------------------------------------------------
# Interface HTML embutida
# ---------------------------------------------------------------------------
HTML = r"""<!DOCTYPE html>
<html lang="pt-BR">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>VOX | Coleta de Áudio</title>
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;900&display=swap');
  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

  :root {
    --bg: #0a0a0f;
    --surface: #13131a;
    --surface2: #1c1c28;
    --accent: #6c63ff;
    --accent2: #a78bfa;
    --success: #22d3a0;
    --danger: #f87171;
    --text: #e8e8f0;
    --muted: #6b6b8a;
    --radius: 16px;
  }

  body {
    font-family: 'Inter', system-ui, sans-serif;
    background: var(--bg);
    color: var(--text);
    min-height: 100vh;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    padding: 24px;
  }

  .card {
    background: var(--surface);
    border: 1px solid rgba(108,99,255,.2);
    border-radius: var(--radius);
    padding: 40px 48px;
    width: 100%;
    max-width: 560px;
    box-shadow: 0 24px 80px rgba(0,0,0,.5);
  }

  .logo {
    font-size: 13px;
    font-weight: 600;
    letter-spacing: .15em;
    color: var(--accent2);
    text-transform: uppercase;
    margin-bottom: 32px;
  }

  .command-label { font-size: 13px; color: var(--muted); margin-bottom: 8px; }

  .command-word {
    font-size: 52px;
    font-weight: 900;
    letter-spacing: -.02em;
    background: linear-gradient(135deg, var(--accent), var(--accent2));
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 4px;
    line-height: 1.1;
  }

  .progress-wrap { margin: 24px 0; }
  .progress-info {
    display: flex;
    justify-content: space-between;
    font-size: 13px;
    color: var(--muted);
    margin-bottom: 8px;
  }
  .progress-bar {
    height: 6px;
    background: var(--surface2);
    border-radius: 99px;
    overflow: hidden;
  }
  .progress-fill {
    height: 100%;
    background: linear-gradient(90deg, var(--accent), var(--accent2));
    border-radius: 99px;
    transition: width .4s ease;
  }

  .status-area {
    min-height: 48px;
    display: flex;
    align-items: center;
    gap: 10px;
    margin: 20px 0;
    padding: 14px 18px;
    background: var(--surface2);
    border-radius: 10px;
    font-size: 14px;
    color: var(--muted);
  }
  .dot {
    width: 10px; height: 10px;
    border-radius: 50%;
    background: var(--muted);
    flex-shrink: 0;
    transition: background .3s;
  }
  .dot.recording { background: var(--danger); animation: pulse 1s infinite; }
  .dot.ok { background: var(--success); }

  @keyframes pulse {
    0%, 100% { opacity: 1; transform: scale(1); }
    50%       { opacity: .5; transform: scale(1.3); }
  }

  .countdown {
    font-size: 64px;
    font-weight: 900;
    color: var(--accent);
    text-align: center;
    display: none;
    margin: 8px 0;
    transition: opacity .3s;
  }

  .btn {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    gap: 8px;
    width: 100%;
    padding: 16px;
    border: none;
    border-radius: 12px;
    font-family: inherit;
    font-size: 16px;
    font-weight: 700;
    cursor: pointer;
    transition: all .2s;
    margin-top: 4px;
  }
  .btn-record {
    background: linear-gradient(135deg, var(--accent), var(--accent2));
    color: #fff;
    box-shadow: 0 8px 24px rgba(108,99,255,.4);
  }
  .btn-record:hover:not(:disabled) { transform: translateY(-2px); box-shadow: 0 12px 32px rgba(108,99,255,.5); }
  .btn-record:disabled { opacity: .5; cursor: not-allowed; transform: none; }

  .btn-skip {
    background: var(--surface2);
    color: var(--muted);
    margin-top: 8px;
    font-size: 14px;
    padding: 12px;
  }
  .btn-skip:hover:not(:disabled) { background: #252535; color: var(--text); }

  .done-screen {
    text-align: center;
    display: none;
  }
  .done-screen h2 { font-size: 32px; font-weight: 900; margin-bottom: 12px; color: var(--success); }
  .done-screen p  { color: var(--muted); line-height: 1.6; }

  .waveform {
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 3px;
    height: 40px;
    margin: 12px 0;
  }
  .bar {
    width: 4px;
    border-radius: 2px;
    background: var(--accent);
    height: 6px;
    transition: height .1s;
  }
</style>
</head>
<body>
<div class="card" id="main-card">
  <div class="logo">🔒 VOX Security — Coleta de Dados</div>

  <div id="recording-ui">
    <div class="command-label">Diga o comando:</div>
    <div class="command-word" id="cmd-word">...</div>

    <div class="progress-wrap">
      <div class="progress-info">
        <span id="prog-text">0 / 20 amostras</span>
        <span id="prog-pct">0%</span>
      </div>
      <div class="progress-bar"><div class="progress-fill" id="prog-fill" style="width:0%"></div></div>
    </div>

    <div class="status-area">
      <div class="dot" id="dot"></div>
      <span id="status-msg">Clique em Gravar e fale o comando claramente.</span>
    </div>

    <div class="waveform" id="waveform">
      <div class="bar" id="b1"></div><div class="bar" id="b2"></div>
      <div class="bar" id="b3"></div><div class="bar" id="b4"></div>
      <div class="bar" id="b5"></div><div class="bar" id="b6"></div>
      <div class="bar" id="b7"></div><div class="bar" id="b8"></div>
      <div class="bar" id="b9"></div><div class="bar" id="b10"></div>
      <div class="bar" id="b11"></div><div class="bar" id="b12"></div>
    </div>

    <button class="btn btn-record" id="btn-record" onclick="startRecording()">
      🎙️ Gravar (1.5s)
    </button>
    <button class="btn btn-skip" id="btn-skip" onclick="skipCommand()" disabled>
      Pular para próximo comando →
    </button>
  </div>

  <div class="done-screen" id="done-screen">
    <div style="font-size:64px;margin-bottom:16px;">✅</div>
    <h2>Coleta concluída!</h2>
    <p>Todas as amostras foram gravadas com sucesso.<br>
       Pode fechar esta janela e rodar <code>python treinar_modelo.py</code>.</p>
  </div>
</div>

<script>
let audioCtx = null;
let stream   = null;
let stateData = {};
let waveAnimId = null;
let analyser  = null;
const bars = Array.from({length: 12}, (_, i) => document.getElementById('b'+(i+1)));

async function init() {
  await refreshState();
  if (!stateData.done) {
    try {
      stream = await navigator.mediaDevices.getUserMedia({ audio: {
        sampleRate: { ideal: stateData.samplerate },
        channelCount: 1,
        echoCancellation: false,
        noiseSuppression: false,
      }, video: false });
      audioCtx = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: stateData.samplerate });
      setStatus('Microfone pronto. Clique em Gravar e diga o comando.', '');
    } catch(e) {
      setStatus('⚠️ Permita o acesso ao microfone no navegador e recarregue.', 'danger');
      document.getElementById('btn-record').disabled = true;
    }
  }
}

async function refreshState() {
  const r = await fetch('/state');
  stateData = await r.json();
  render();
}

function render() {
  if (stateData.done) {
    document.getElementById('recording-ui').style.display = 'none';
    document.getElementById('done-screen').style.display  = 'block';
    return;
  }
  const cmd = stateData.command;
  const cnt = stateData.count;
  const tgt = stateData.target;
  document.getElementById('cmd-word').textContent   = cmd ? cmd.toUpperCase() : '—';
  document.getElementById('prog-text').textContent  = cnt + ' / ' + tgt + ' amostras';
  const pct = tgt > 0 ? Math.round(cnt/tgt*100) : 0;
  document.getElementById('prog-pct').textContent   = pct + '%';
  document.getElementById('prog-fill').style.width  = pct + '%';
  if (cnt >= tgt) document.getElementById('btn-skip').disabled = false;
}

function setStatus(msg, type) {
  document.getElementById('status-msg').textContent = msg;
  const dot = document.getElementById('dot');
  dot.className = 'dot' + (type === 'recording' ? ' recording' : type === 'ok' ? ' ok' : '');
}

function startWaveAnimation() {
  function frame() {
    if (analyser) {
      const data = new Uint8Array(analyser.frequencyBinCount);
      analyser.getByteFrequencyData(data);
      const step = Math.floor(data.length / bars.length);
      bars.forEach((b, i) => {
        const v = data[i * step] / 255;
        b.style.height = (6 + v * 34) + 'px';
        b.style.opacity = 0.4 + v * 0.6;
      });
    }
    waveAnimId = requestAnimationFrame(frame);
  }
  frame();
}

function stopWaveAnimation() {
  if (waveAnimId) { cancelAnimationFrame(waveAnimId); waveAnimId = null; }
  bars.forEach(b => { b.style.height = '6px'; b.style.opacity = '1'; });
}

async function startRecording() {
  if (!audioCtx || !stream) return;
  const btn = document.getElementById('btn-record');
  btn.disabled = true;

  // Resume context if suspended
  if (audioCtx.state === 'suspended') await audioCtx.resume();

  const actualSR    = audioCtx.sampleRate;
  const nFrames     = Math.round(actualSR * stateData.duration);
  const bufferSize  = 4096;
  const srcNode     = audioCtx.createMediaStreamSource(stream);
  analyser          = audioCtx.createAnalyser();
  analyser.fftSize  = 256;
  const processor   = audioCtx.createScriptProcessor(bufferSize, 1, 1);
  let collected     = [];
  let totalFrames   = 0;
  let recording     = true;

  srcNode.connect(analyser);
  srcNode.connect(processor);
  processor.connect(audioCtx.destination);
  startWaveAnimation();

  setStatus('🔴 GRAVANDO — diga: ' + stateData.command.toUpperCase(), 'recording');

  processor.onaudioprocess = function(e) {
    if (!recording) return;
    const chunk = e.inputBuffer.getChannelData(0);
    collected.push(new Float32Array(chunk));
    totalFrames += chunk.length;
    if (totalFrames >= nFrames) {
      recording = false;
      processor.disconnect();
      srcNode.disconnect();
      stopWaveAnimation();
      analyser = null;
      finalize(collected, nFrames, actualSR);
    }
  };
}

async function finalize(chunks, nFrames, sampleRate) {
  // Concatenar e truncar
  const total = new Float32Array(nFrames);
  let offset = 0;
  for (const c of chunks) {
    const n = Math.min(c.length, nFrames - offset);
    total.set(c.subarray(0, n), offset);
    offset += n;
    if (offset >= nFrames) break;
  }

  // Enviar como ArrayBuffer
  setStatus('Salvando...', '');
  try {
    const res = await fetch('/save', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/octet-stream',
        'X-Samplerate': String(sampleRate),
      },
      body: total.buffer,
    });
    const data = await res.json();
    if (data.ok) {
      setStatus('✅ Salvo! Pode gravar a próxima.', 'ok');
      await refreshState();
    } else {
      setStatus('Erro ao salvar: ' + (data.error || '?'), '');
    }
  } catch(e) {
    setStatus('Erro de rede: ' + e.message, '');
  }

  document.getElementById('btn-record').disabled = false;
}

async function skipCommand() {
  await fetch('/skip', { method: 'POST' });
  document.getElementById('btn-skip').disabled = true;
  await refreshState();
  setStatus('Próximo comando. Clique em Gravar.', '');
}

window.addEventListener('load', init);
</script>
</body>
</html>
"""

# ---------------------------------------------------------------------------
# HTTP handler
# ---------------------------------------------------------------------------
class GravarHandler(http.server.BaseHTTPRequestHandler):

    def log_message(self, fmt, *args):
        pass  # silencia logs do servidor

    def _send_json(self, data, code=200):
        body = json.dumps(data).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            body = HTML.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        elif self.path == "/state":
            with _lock:
                if state["done"]:
                    self._send_json({"done": True})
                else:
                    cmd = COMMANDS[state["cmd_idx"]]
                    self._send_json({
                        "done":       False,
                        "command":    cmd,
                        "count":      state["count"],
                        "target":     state["target"],
                        "duration":   DURATION,
                        "samplerate": SAMPLE_RATE,
                    })
        else:
            self.send_error(404)

    def do_POST(self):
        if self.path == "/save":
            length = int(self.headers.get("Content-Length", 0))
            raw = self.rfile.read(length)
            sr = int(self.headers.get("X-Samplerate", SAMPLE_RATE))

            samples = np.frombuffer(raw, dtype=np.float32).copy()

            with _lock:
                if state["done"]:
                    self._send_json({"ok": False, "error": "collection done"})
                    return
                cmd = COMMANDS[state["cmd_idx"]]
                folder = os.path.join(ROOT, cmd)
                os.makedirs(folder, exist_ok=True)
                state["count"] += 1
                cnt = state["count"]

            filename = os.path.join(folder, f"{cmd}_{cnt:03d}.wav")
            sf.write(filename, samples, sr)
            print(f"  [{cmd}] {cnt:03d} salvo — {len(samples)/sr:.2f}s @ {sr} Hz")

            self._send_json({"ok": True})

        elif self.path == "/skip":
            with _lock:
                state["cmd_idx"] += 1
                if state["cmd_idx"] >= len(COMMANDS):
                    state["done"] = True
                else:
                    cmd = COMMANDS[state["cmd_idx"]]
                    folder = os.path.join(ROOT, cmd)
                    os.makedirs(folder, exist_ok=True)
                    existing = len([f for f in os.listdir(folder) if f.lower().endswith(".wav")])
                    state["count"]  = existing
                    state["target"] = max(20, existing + 1)
            self._send_json({"ok": True})

        else:
            self.send_error(404)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    # Inicializa estado com amostras existentes da primeira classe
    with _lock:
        cmd0   = COMMANDS[state["cmd_idx"]]
        folder = os.path.join(ROOT, cmd0)
        os.makedirs(folder, exist_ok=True)
        existing = len([f for f in os.listdir(folder) if f.lower().endswith(".wav")])
        state["count"]  = existing
        state["target"] = max(20, existing + 1)

    server = http.server.HTTPServer(("127.0.0.1", PORT), GravarHandler)
    url    = f"http://127.0.0.1:{PORT}"

    print(f"\nVOX | Coleta de áudio via navegador")
    print(f"Abrindo: {url}")
    print("Comandos:", ", ".join(COMMANDS))
    print("Quando o navegador abrir, permita o uso do microfone.")
    print("Pressione Ctrl+C aqui para encerrar.\n")

    threading.Timer(0.8, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nServidor encerrado.")
        server.shutdown()


if __name__ == "__main__":
    main()


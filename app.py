"""
app.py — Face verification gate + chat console

Overview
- Uses notebook-produced artifacts (models/gallery_mean.npy, labels.json, threshold.json)
  as the identity gallery. These were generated offline (e.g., in Colab) with your
  preferred face embedding model. At runtime we avoid installing heavy packages on Windows.
- Provides:
  - "/"      : Upload form (name + photo) and verification gate
  - "/verify": Face verification endpoint
  - "/chat"  : Simple chat UI powered by Groq API
  - "/api/*" : Chat and speech-to-text helpers

Important
- Accuracy depends on the quality and consistency of notebook embeddings. The query
  embedding here is a lightweight placeholder to keep the system running without
  heavy models. Replace `embed_query_vector()` later with the same embedding model
  used in the notebook for best results.
"""

from __future__ import annotations

import os
import time
import ssl
import smtplib
import html
import json
from email.message import EmailMessage

from flask import (
    Flask,
    request,
    redirect,
    make_response,
    jsonify,
    send_from_directory,
)
from werkzeug.utils import secure_filename

# Lightweight runtime deps (no heavy model required on Windows)
import cv2
import numpy as np
import requests


# ============================
# Application configuration
# ============================

TITLE = "Face Verify Gate"
BACKGROUND_IMG = "https://i.pinimg.com/originals/f6/7a/18/f67a1897acd0eb4c8824f214d4e48f9e.gif"

# Flask and uploads
APP_SECRET = os.getenv("APP_SECRET", "dev-secret")
UPLOAD_DIR = os.getenv("UPLOAD_DIR", "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Optional email alerts (leave empty to disable)
ALERT_EMAIL_TO = os.getenv("ALERT_EMAIL_TO", "")
ALERT_EMAIL_FROM = os.getenv("ALERT_EMAIL_FROM", "")
SMTP_HOST = os.getenv("SMTP_HOST", "smtp.gmail.com")
SMTP_PORT = int(os.getenv("SMTP_PORT", "465"))
SMTP_USER = os.getenv("SMTP_USER", "")
SMTP_PASS = os.getenv("SMTP_PASS", "")

# Groq API (demo key shown; use your own secret in production)
GROQ_API_KEY = "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"
GROQ_MODEL_CHAT = os.getenv("GROQ_MODEL_CHAT", "llama-3.1-8b-instant")
GROQ_MODEL_STT = os.getenv("GROQ_MODEL_STT", "whisper-large-v3")

# Echo mode short-circuits Groq for quick local testing
ECHO_MODE = False


# ============================
# Model artifacts (from notebook)
# ============================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
GALLERY_NPY = os.path.join(MODELS_DIR, "gallery_mean.npy")  # shape: (N, D), float32
LABELS_JSON = os.path.join(MODELS_DIR, "labels.json")       # list[str], len N
THRESH_JSON = os.path.join(MODELS_DIR, "threshold.json")    # {"cosine_threshold": float}

# In-memory state
G: np.ndarray | None = None         # (N, D) gallery templates, L2-normalized
labels: list[str] | None = None
COSINE_DIST_THRESHOLD: float = 0.35 # distance threshold; lower = stricter


# ============================
# Face detector (OpenCV Haar)
# - Portable and good enough to crop the largest face region.
# - You may later swap it with a stronger detector.
# ============================

HAAR_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
FACE_DETECTOR = cv2.CascadeClassifier(HAAR_PATH)


# ============================
# Flask app
# ============================

app = Flask(__name__)
app.config["SECRET_KEY"] = APP_SECRET
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16 MB upload cap


# ============================
# Utilities
# ============================

def send_alert_email(subject: str, body: str) -> None:
    """
    Sends a simple email via SMTP_SSL when credentials are configured.
    Silently skips if SMTP env vars are missing.
    """
    if not (SMTP_USER and SMTP_PASS and ALERT_EMAIL_TO):
        app.logger.warning("Email not configured; skipping alert.")
        return

    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = ALERT_EMAIL_FROM or SMTP_USER
    msg["To"] = ALERT_EMAIL_TO
    msg.set_content(body)

    ctx = ssl.create_default_context()
    with smtplib.SMTP_SSL(SMTP_HOST, SMTP_PORT, context=ctx) as s:
        s.login(SMTP_USER, SMTP_PASS)
        s.send_message(msg)
    app.logger.info("Alert email sent.")

def save_upload(file_storage, prefix: str = "file") -> str:
    """
    Persist an uploaded file to UPLOAD_DIR with a timestamped, sanitized filename.
    Returns the saved filesystem path.
    """
    filename = f"{int(time.time())}_{secure_filename(file_storage.filename)}"
    path = os.path.join(UPLOAD_DIR, filename)
    file_storage.save(path)
    return path

def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    """
    Cosine distance between vectors a and b in [0, 2].
    0 = identical direction, 1 = orthogonal, 2 = opposite.
    We typically operate in [0, 1] when vectors are non-negative.
    """
    return 1.0 - float(np.dot(a, b) / ((np.linalg.norm(a) * np.linalg.norm(b)) + 1e-12))


# ============================
# Query “embedding” placeholder
# - Keeps the end-to-end pipeline working on Windows without heavy installs.
# - Replace `embed_query_vector` later with the same model used in the notebook.
# ============================

def detect_and_crop_face(bgr: np.ndarray) -> np.ndarray | None:
    """
    Detect the largest frontal face and return a cropped BGR image.
    Returns None if no face is detected.
    """
    if bgr is None:
        return None

    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    faces = FACE_DETECTOR.detectMultiScale(
        gray, scaleFactor=1.2, minNeighbors=5, minSize=(60, 60)
    )
    if len(faces) == 0:
        return None

    # Select the largest bounding box
    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
    return bgr[y:y + h, x:x + w]

def embed_query_vector(bgr: np.ndarray) -> np.ndarray | None:
    """
    Build a normalized vector from the cropped face pixels.
    This is a temporary stand-in for a true neural embedding.
    Returns a float32 vector or None if face not found.
    """
    crop = detect_and_crop_face(bgr)
    if crop is None or crop.size == 0:
        return None

    # Standardize geometry to reduce variance
    face = cv2.resize(crop, (112, 112), interpolation=cv2.INTER_LINEAR)

    # Normalize to unit-length vector (L2)
    vec = face.astype("float32").ravel()
    vec = vec / (np.linalg.norm(vec) + 1e-12)
    return vec.astype("float32")


# ============================
# Artifact bootstrap
# ============================

def bootstrap_artifacts() -> None:
    """
    Load notebook-produced artifacts into memory:
      - G: (N, D) gallery templates (assumed L2-normalized)
      - labels: list of identity strings with length N
      - COSINE_DIST_THRESHOLD: float from threshold.json
    Raises FileNotFoundError if any artifact is missing.
    """
    global G, labels, COSINE_DIST_THRESHOLD

    if not os.path.exists(GALLERY_NPY):
        raise FileNotFoundError(f"Missing: {GALLERY_NPY}")
    if not os.path.exists(LABELS_JSON):
        raise FileNotFoundError(f"Missing: {LABELS_JSON}")
    if not os.path.exists(THRESH_JSON):
        raise FileNotFoundError(f"Missing: {THRESH_JSON}")

    G = np.load(GALLERY_NPY).astype("float32")

    with open(LABELS_JSON, "r", encoding="utf-8") as f:
        labels = json.load(f)

    with open(THRESH_JSON, "r", encoding="utf-8") as f:
        config = json.load(f)
        COSINE_DIST_THRESHOLD = float(config.get("cosine_threshold", COSINE_DIST_THRESHOLD))

    app.logger.info(
        "[bootstrap] gallery=%s labels=%d threshold=%.4f",
        None if G is None else tuple(G.shape),
        len(labels or []),
        COSINE_DIST_THRESHOLD,
    )


# ============================
# HTML Gate (upload form)
# ============================

def render_gate(status_msg: str = ""):
    """
    Render the landing page with a simple upload form (name + photo).
    """
    status_msg = html.escape(status_msg or "")
    html_page = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8" />
<meta name="viewport" content="width=device-width, initial-scale=1.0"/>
<title>{TITLE}</title>
<style>
  :root {{
    --bg:#0a1424; --panel:#10203c; --text:#f2f6ff; --muted:#d3e1ff;
    --accent1:#98c2ff; --accent2:#4dd9ff; --border: rgba(152,194,255,.5);
  }}
  * {{ box-sizing: border-box; }}
  html,body {{ height:100%; margin:0; font-family: Inter, system-ui, -apple-system, Segoe UI, Roboto, Arial; color:var(--text); background:var(--bg) }}
  body::before {{
    content:""; position:fixed; inset:0; z-index:0;
    background: #000 url('{BACKGROUND_IMG}?v=2') center/cover no-repeat fixed;
    filter: brightness(.95) saturate(1.05); opacity:.45;
  }}
  .wrap {{ position:relative; z-index:1; min-height:100%; display:flex; align-items:center; justify-content:center; padding: min(10vh,6rem) 1rem; }}
  .hero {{ max-width:1050px; width:100%; text-align:center; position:relative; padding:0 1rem; }}
  .chip {{ display:inline-flex; align-items:center; gap:8px; padding:8px 12px; border-radius:999px;
    color:#e7f1ff; border:1px solid var(--border); background: rgba(152,194,255,.22);
    font-size:12px; letter-spacing:.12em; text-transform:uppercase; }}
  .title {{ margin:16px auto 10px; font-weight:850; line-height:1.05; font-size: clamp(2.4rem, 6vw, 4.6rem); text-shadow: 0 0 22px rgba(152,194,255,.5); }}
  .subtitle {{ max-width:840px; margin:0 auto 28px; color:var(--muted); font-size: clamp(1.05rem, 1.6vw, 1.2rem); }}
  .card {{ margin: 26px auto 0; max-width:620px; padding:1.2rem; background: var(--panel);
    border:1px solid var(--border); border-radius:16px; box-shadow: 0 24px 70px rgba(0,0,0,.35), inset 0 0 60px rgba(152,194,255,.10); }}
  label {{ display:block; margin:.6rem 0 .35rem; color:#e1ebff; font-weight:700 }}
  input[type="text"], input[type="file"] {{
    width:100%; padding:1rem; border-radius:12px; border:1px solid rgba(255,255,255,.7);
    background: rgba(255,255,255,.2); color:#06101e; outline:none;
  }}
  input[type="text"]::placeholder {{ color:#2a4066 }}
  .cta {{ display:flex; gap:.75rem; margin-top:1rem }}
  .btn {{ flex:1; padding:1rem; border-radius:12px; border:1px solid rgba(255,255,255,.6);
    background: linear-gradient(90deg, var(--accent1), var(--accent2));
    color:#06101e; font-weight:900; letter-spacing:.3px; cursor:pointer;
    box-shadow: 0 12px 34px rgba(152,194,255,.36); }}
  .status {{ min-height:1.2rem; margin-top:.6rem; color:#1ee5ff; font-weight:600 }}
  .ticker-wrap {{ margin-top:.7rem; overflow:hidden; border-radius:10px; border:1px solid var(--border); background: rgba(255,255,255,.12); }}
  .ticker {{ display:flex; gap:40px; padding:9px 12px; color:#e7f1ff; white-space:nowrap; animation: marquee 18s linear infinite; font-size:13px; letter-spacing:.08em }}
  @keyframes marquee {{ from {{ transform: translateX(0) }} to {{ transform: translateX(-50%) }} }}
</style>
</head>
<body>
<div class="wrap">
  <div class="hero">
    <div class="chip">Operational • J.A.R.V.I.S. Security Core</div>
    <h1 class="title">J.A.R.V.I.S. verifies to protect what matters.</h1>
    <p class="subtitle">Adaptive identity verification for smooth and secure access.</p>

    <form class="card" method="POST" action="/verify" enctype="multipart/form-data" onsubmit="onSubmit()">
      <label for="name">Your name</label>
      <input id="name" name="name" type="text" placeholder="e.g., Tony Stark" required />
      <label for="photo">Your photo</label>
      <input id="photo" name="photo" type="file" accept="image/*" required />
      <div class="cta"><button class="btn" type="submit">Start verification</button></div>
      <p class="status" id="status">{status_msg}</p>
      <div class="ticker-wrap" aria-hidden="true">
        <div class="ticker">
          <span>Preparing next verification…</span>
          <span>Analyzing facial features…</span>
          <span>Matching against trusted identities…</span>
          <span>Threat Model: Low • All systems nominal</span>
          <span>Preparing next verification…</span>
        </div>
      </div>
    </form>
  </div>
</div>
<script>
function onSubmit(){{
  const s=document.getElementById('status');
  s.textContent='Scanning…';
}}
</script>
</body>
</html>"""
    resp = make_response(html_page)
    resp.headers["Content-Type"] = "text/html; charset=utf-8"
    return resp


@app.get("/")
def index():
    """Landing page with the verification gate."""
    return render_gate("")


# ============================
# Verification endpoint
# ============================

def verify_face_identity(user_name: str, image_bytes: bytes) -> dict:
    """
    Verify claimed identity by comparing a query face to a gallery template.

    Returns:
        dict with keys:
          - ok: bool             (accepted / rejected)
          - score: float         (cosine distance; lower is better)
          - threshold: float     (decision boundary used)
          - reason: str | None   (set on failure)
    """
    global G, labels, COSINE_DIST_THRESHOLD

    # Lazy-load notebook artifacts on first request
    if G is None or labels is None:
        try:
            bootstrap_artifacts()
        except Exception as e:
            return {"ok": False, "reason": f"bootstrap_failed: {e}"}

    # Identity must exist in labels
    try:
        idx = labels.index(user_name)
    except ValueError:
        return {"ok": False, "reason": "Name Not Found"}

    # Decode uploaded image from bytes
    arr = np.frombuffer(image_bytes, np.uint8)
    bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if bgr is None:
        return {"ok": False, "reason": "Invalid image"}

    # Create a lightweight query vector (replace with real embedding later)
    q = embed_query_vector(bgr)
    if q is None or not np.isfinite(q).all():
        return {"ok": False, "reason": "No face detected"}

    # Lookup gallery template for this identity
    g = G[idx].astype("float32")

    # Cosine distance decision
    dist = cosine_distance(q, g)
    accepted = dist <= COSINE_DIST_THRESHOLD

    return {
        "ok": bool(accepted),
        "score": float(dist),
        "threshold": float(COSINE_DIST_THRESHOLD),
        "reason": None if accepted else "Not within threshold",
    }


@app.post("/verify")
def verify():
    """
    Handle form submission:
    - Save upload for audit/debug
    - Run verification
    - On success: redirect to chat console
    - On failure: show gate with status and (optionally) email an alert
    """
    name = (request.form.get("name") or "").strip()
    file = request.files.get("photo")
    if not name or not file or not file.filename.strip():
        return render_gate("Please enter a name and select an image."), 400

    saved_path = save_upload(file, prefix="photo")
    with open(saved_path, "rb") as f:
        image_bytes = f.read()

    result = verify_face_identity(name, image_bytes)
    if not result.get("ok"):
        try:
            send_alert_email(
                f"[Access Denied] {name}",
                f"Denied file: {saved_path}\nReason: {result.get('reason')}",
            )
        except Exception as e:
            app.logger.error("Email error: %s", e)

        return render_gate("Access denied."), 401

    # Minimal session continuity via cookie
    resp = redirect("/chat", code=302)
    resp.set_cookie("user", name, httponly=False, samesite="Lax")
    return resp


# ============================
# Chat UI (post-verification)
# ============================

CHAT_HTML = """<!doctype html>
<html>
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>J.A.R.V.I.S. Console</title>
<style>
  :root{ --bg:#0a1424; --text:#0a1530; --accent:#98c2ff; --accent2:#4dd9ff; --border: rgba(152,194,255,.75); }
  html,body{height:100%; margin:0; font-family:Inter, system-ui, -apple-system, Segoe UI, Roboto, Arial; color:#0a1530; background:var(--bg)}
  body::before{
    content:""; position:fixed; inset:0; z-index:0;
    background:url('""" + BACKGROUND_IMG + """?v=2') center/cover fixed no-repeat;
    filter:brightness(1) saturate(1.05);
    opacity:.45;
  }
  header{position:sticky; top:0; z-index:2; padding:14px 16px; backdrop-filter: blur(8px);
    background: linear-gradient(180deg, rgba(230,241,255,.85), rgba(216,234,255,.7));
    border-bottom:1px solid var(--border); display:flex; gap:10px; align-items:center}
  .welcome{margin-left:auto; color:#0a1530; font-weight:700}
  main{max-width:980px; margin:0 auto; padding:18px; position:relative; z-index:1}
  .banner{background: linear-gradient(180deg, #eaf3ff, #d9ecff); border:1px solid var(--border); border-radius:14px; padding:12px 14px; margin-bottom:12px; color:#0a1530}
  #log{background: rgba(243,248,255,.92); border:2px solid var(--border); border-radius:14px; padding:14px; min-height:56vh; overflow:auto}
  .msg{padding:12px 14px; margin:10px 0; border-radius:12px; white-space:pre-wrap; line-height:1.5; opacity:0; transform: translateY(6px); animation: fadeUp .25s ease forwards; font-size:16px; color:#0a1530}
  .user{background: #dff0ff; border:1px solid var(--border)}
  .bot{background: #e3fff8; border:1px solid #7fe9d5}
  .sys{color:#0a1530; background:#fff7d6; border:1px dashed #ffd36f}
  @keyframes fadeUp{to{opacity:1; transform:translateY(0)}}
  .row{display:flex; gap:10px; margin-top:12px; position:relative; z-index:3}
  textarea#inp{
    flex:1; min-height:80px; padding:14px; background:#ffffff; color:#0a1530;
    border:2px solid var(--border); border-radius:12px; font-size:16px; outline:none; resize:vertical;
    position:relative; z-index:4; pointer-events:auto;
  }
  textarea#inp::placeholder{ color:#446aa1; font-weight:600 }
  textarea#inp:focus{ border-color: var(--accent); box-shadow: 0 0 0 4px rgba(152,194,255,.35) }
  button,.iconbtn{
    background: linear-gradient(90deg, var(--accent), var(--accent2));
    color:#06101e; border:none; padding:14px 18px; border-radius:12px; font-weight:900; cursor:pointer;
    box-shadow: 0 10px 28px rgba(152,194,255,.36); position:relative; overflow:hidden; font-size:15px; z-index:4;
  }
  .iconbtn{padding:14px}
  .rec{background: linear-gradient(90deg, #ff8a8a, #ffbe7d); color:#1b0e0e; box-shadow: 0 10px 28px rgba(255,138,138,.4)}
  .eq{position:absolute; inset:0; display:none; align-items:center; justify-content:center; gap:3px}
  .rec .eq{display:flex}
  .bar{width:3px; height:12px; background:#1b0e0e; opacity:.9; border-radius:2px; animation: bounce .8s ease-in-out infinite}
  .bar:nth-child(2){animation-delay:.1s}
  .bar:nth-child(3){animation-delay:.2s}
  .bar:nth-child(4){animation-delay:.3s}
  @keyframes bounce{0%,100%{transform:scaleY(.6)} 50%{transform:scaleY(1.6)}}
</style>
</head>
<body>
<header>
  <div style="font-weight:900; letter-spacing:.3px">J.A.R.V.I.S. Console</div>
  <div id="welcome" class="welcome"></div>
</header>

<main>
  <div class="banner" id="banner"></div>
  <div id="log"></div>
  <div class="row">
    <textarea id="inp" placeholder="Type a message..."></textarea>
    <button id="sendBtn">Send</button>
    <button id="recbtn" class="iconbtn">🎙️
      <div class="eq"><div class="bar"></div><div class="bar"></div><div class="bar"></div><div class="bar"></div></div>
    </button>
  </div>
</main>

<script>
const inp = document.getElementById('inp');
const sendBtn = document.getElementById('sendBtn');
const recbtn = document.getElementById('recbtn');

setTimeout(()=> inp.focus(), 100);

// Text chat
sendBtn.addEventListener('click', sendText);
inp.addEventListener('keydown', (e)=>{ if(e.key === 'Enter' && !e.shiftKey){ e.preventDefault(); sendText(); }});

// Simple welcome banner
const user = decodeURIComponent((document.cookie.match(/(?:^|; )user=([^;]+)/)?.[1] || 'guest'));
document.getElementById('welcome').textContent = 'Welcome, ' + user;
document.getElementById('banner').textContent =
  `Hello ${user}! Your identity was verified successfully. Ask me anything or use the microphone.`;

// Append messages to chat log
function appendMsg(text, who='bot'){
  const div=document.createElement('div');
  div.className='msg '+who;
  div.textContent = text;
  document.getElementById('log').appendChild(div);
  div.scrollIntoView({behavior:'smooth', block:'end'});
}
appendMsg("Systems online. How can I assist you today?", "bot");

// Send a text message to /api/chat
async function sendText(){
  const t = inp.value.trim();
  if(!t) return;
  appendMsg(t,'user');
  inp.value='';
  try{
    const r = await fetch('/api/chat', {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({message:t})});
    const j = await r.json();
    appendMsg(j.reply || j.error || '[no response]');
  }catch(e){
    appendMsg('Chat error: '+(e.message||e), 'sys');
  }
}

// Voice capture + STT
let mediaRecorder, chunks = [], recording = false;

function getSupportedMime() {
  const cand = ['audio/webm;codecs=opus','audio/webm','audio/ogg;codecs=opus','audio/ogg','audio/mp4'];
  for (const c of cand) { try { if (MediaRecorder.isTypeSupported && MediaRecorder.isTypeSupported(c)) return c; } catch(_){} }
  return '';
}

recbtn.addEventListener('click', async () => {
  const btn = recbtn;
  if(!recording){
    try{
      const stream = await navigator.mediaDevices.getUserMedia({audio:true});
      const mimeType = getSupportedMime();
      mediaRecorder = mimeType ? new MediaRecorder(stream, {mimeType}) : new MediaRecorder(stream);
      chunks = [];
      mediaRecorder.ondataavailable = e => { if(e.data && e.data.size) chunks.push(e.data); };
      mediaRecorder.onstop = async () => {
        const type = mediaRecorder.mimeType || 'audio/webm';
        const blob = new Blob(chunks, {type});
        if (blob.size < 400) { appendMsg('Recording too short. Speak 2–3 seconds, then stop.', 'sys'); btn.textContent='🎙️'; return; }
        const old = btn.textContent;
        btn.textContent = '🧠 Transcribing…';
        try { await sendAudioBlob(blob); }
        catch(e){ appendMsg('STT send error: '+(e.message||e),'sys'); }
        finally { btn.textContent = '🎙️'; }
      };
      mediaRecorder.start();
      recording=true; btn.classList.add('rec'); btn.textContent='⏹️';
    } catch(e){ appendMsg('Mic error: '+(e.message || e.name), 'sys'); }
  } else {
    try { mediaRecorder.stop(); } catch(_) {}
    recording=false; btn.classList.remove('rec'); btn.textContent='🎙️';
  }
});

async function sendAudioBlob(blob){
  const fd = new FormData();
  const ext = blob.type.includes('ogg') ? 'ogg' : (blob.type.includes('mp4') ? 'mp4' : 'webm');
  fd.append('audio', blob, `voice.${ext}`);
  let r;
  try {
    r = await fetch('/api/speech_to_text', {method:'POST', body: fd});
  } catch(e) {
    appendMsg('Network error sending audio: ' + (e.message || e), 'sys');
    return;
  }
  const raw = await r.text();
  let j;
  try{ j = JSON.parse(raw); }catch(_){ j = { error:'Invalid JSON from STT', raw: raw.slice(0,200) }; }
  if(!r.ok){ appendMsg(`STT HTTP ${r.status}: ${j.error || j.raw}`, 'sys'); return; }
  if(j.text){
    appendMsg('[voice→text] '+j.text, 'user');
    try {
      const cr = await fetch('/api/chat', {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({message:j.text, meta:{from:'voice'}})});
      const cj = await cr.json();
      appendMsg(cj.reply || cj.error || '[no response]');
    } catch(e) {
      appendMsg('Chat error: ' + (e.message || e), 'sys');
    }
  } else {
    appendMsg('STT error: ' + (j.error || 'no text'), 'sys');
  }
}
</script>
</body>
</html>
"""

@app.get("/chat")
def chat_page():
    """Return the chat console HTML."""
    resp = make_response(CHAT_HTML)
    resp.headers["Content-Type"] = "text/html; charset=utf-8"
    return resp


# ============================
# API: Chat (Groq)
# ============================

SYSTEM_PROMPT = "You are J.A.R.V.I.S., a helpful, concise assistant. Keep answers short and practical."

def call_groq_chat(messages: list[dict]) -> str:
    """
    Call Groq's Chat Completions API. Returns the assistant message text.
    Set ECHO_MODE=True to bypass the API for local testing.
    """
    if ECHO_MODE:
        return "Echo: " + messages[-1]["content"]

    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"}
    data = {"model": GROQ_MODEL_CHAT, "messages": messages, "temperature": 0.3}
    r = requests.post(url, headers=headers, json=data, timeout=90)
    r.raise_for_status()
    j = r.json()
    return j["choices"][0]["message"]["content"].strip()

@app.post("/api/chat")
def api_chat():
    """
    Chat endpoint used by the UI. Accepts JSON {message:string}.
    """
    try:
        payload = request.get_json(force=True, silent=True) or {}
        user_msg = (payload.get("message") or "").strip()
        if not user_msg:
            return jsonify(error="empty message"), 400

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ]
        reply = call_groq_chat(messages)
        return jsonify(reply=reply)
    except Exception as e:
        app.logger.exception("Chat error")
        return jsonify(error=str(e)), 500


# ============================
# API: Speech-to-Text (Groq Whisper)
# ============================

@app.post("/api/speech_to_text")
def api_speech_to_text():
    """
    Accepts multipart/form-data with 'audio' file (webm/ogg/mp4).
    Returns JSON {text:string} on success.
    """
    f = request.files.get("audio")
    if not f:
        return jsonify(error="no audio"), 400

    filename = f.filename or "voice.webm"
    lower = filename.lower()
    if lower.endswith(".ogg"):
        mime = "audio/ogg"
    elif lower.endswith(".mp4") or lower.endswith(".m4a"):
        mime = "audio/mp4"
    else:
        mime = "audio/webm"

    path = os.path.join(UPLOAD_DIR, f"voice_{int(time.time())}_{filename}")
    f.save(path)

    try:
        url = "https://api.groq.com/openai/v1/audio/transcriptions"
        headers = {"Authorization": f"Bearer {GROQ_API_KEY}"}
        with open(path, "rb") as fp:
            files = {
                "file": (os.path.basename(path), fp, mime),
                "model": (None, GROQ_MODEL_STT),
            }
            r = requests.post(url, headers=headers, files=files, timeout=180)
        if r.status_code >= 400:
            return jsonify(error=f"groq stt {r.status_code}: {r.text[:200]}"), 500
        j = r.json()
        text = j.get("text") or j.get("transcript") or ""
        if not text:
            return jsonify(error="no text from STT"), 500
        return jsonify(text=text)
    except Exception as e:
        app.logger.exception("STT error")
        return jsonify(error=str(e)), 500


# ============================
# Static uploads (debug convenience)
# ============================

@app.get("/uploads/<path:fname>")
def get_upload(fname: str):
    """Serve saved uploads for manual inspection/debugging."""
    return send_from_directory(UPLOAD_DIR, fname)


# ============================
# Entrypoint
# ============================

if __name__ == "__main__":
    print(
        "Groq chat:", GROQ_MODEL_CHAT,
        "| STT:", GROQ_MODEL_STT,
        "| ECHO_MODE:", ECHO_MODE,
        "| Key set:", bool(GROQ_API_KEY),
    )
    try:
        # Preload artifacts once (non-fatal if not present yet; will retry on /verify)
        bootstrap_artifacts()
    except Exception as e:
        print(f"[WARN] Bootstrap will retry on first verify: {e}")

    app.run(host="127.0.0.1", port=5000, debug=True)

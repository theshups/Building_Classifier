"""
app.py — Property Classifier
FastAPI inference server with embedded web UI.
No gradio = no dependency conflicts on HuggingFace Spaces.
"""

import os
import io
import json
import time
import numpy as np
from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware

os.environ["TF_CPP_MIN_LOG_LEVEL"]  = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import tensorflow as tf

# ── Paths ─────────────────────────────────────────────────────────────────
MODEL_PATH     = "models/building_classifier.h5"
TOKENIZER_PATH = "models/tokenizer.json"
IMG_SIZE       = (224, 224)

# ── Class display names ───────────────────────────────────────────────────
CLASS_DISPLAY = {
    "exterior_facade": "Exterior Facade",
    "office_interior": "Office Interior",
    "warehouse":       "Warehouse",
    "hvac_pipeline":   "HVAC Pipeline",
}
CLASS_COLORS = {
    "exterior_facade": "#3b82f6",
    "office_interior": "#22c55e",
    "warehouse":       "#f59e0b",
    "hvac_pipeline":   "#a855f7",
}

# ── Load model once at startup ────────────────────────────────────────────
model:   tf.keras.Model | None = None
classes: list[str]             = []

try:
    if os.path.exists(MODEL_PATH) and os.path.exists(TOKENIZER_PATH):
        print("Loading model...")
        model   = tf.keras.models.load_model(MODEL_PATH)
        tok     = json.loads(open(TOKENIZER_PATH, encoding="utf-8").read())
        classes = tok["class_names"]
        print(f"Model loaded successfully. Classes: {classes}")
    else:
        print(f"WARNING: Model not found at {MODEL_PATH}")
except Exception as e:
    print(f"ERROR loading model: {e}")

# ── FastAPI app ───────────────────────────────────────────────────────────
app = FastAPI(title="Property Classifier", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Embedded HTML UI ──────────────────────────────────────────────────────
HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Property Classifier</title>
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{background:#0b0f1a;color:#e2e8f0;font-family:system-ui,-apple-system,sans-serif;min-height:100vh;display:flex;flex-direction:column}
canvas{position:fixed;top:0;left:0;width:100%;height:100%;z-index:0;pointer-events:none;opacity:.3}
.app{position:relative;z-index:1;display:flex;flex-direction:column;min-height:100vh}
header{padding:16px 28px;background:rgba(11,15,26,.95);border-bottom:1px solid #1e3050;display:flex;align-items:center;gap:14px;backdrop-filter:blur(12px);position:sticky;top:0;z-index:10}
.brand{font-size:18px;font-weight:700;color:#e2e8f0}.brand b{color:#3b82f6}
.hdr-right{margin-left:auto;display:flex;align-items:center;gap:12px}
.badge{font-size:10px;padding:3px 10px;border-radius:20px;border:1px solid;letter-spacing:.5px}
.badge-green{color:#22c55e;border-color:#22c55e;background:rgba(34,197,94,.08)}
.badge-gray{color:#64748b;border-color:#1e3050}
.dot{width:7px;height:7px;border-radius:50%;display:inline-block;margin-right:5px;animation:blink 2s infinite}
@keyframes blink{0%,100%{opacity:1}50%{opacity:.3}}
main{display:grid;grid-template-columns:1fr 1fr;gap:18px;padding:20px 28px;max-width:1100px;margin:0 auto;width:100%;flex:1}
@media(max-width:820px){main{grid-template-columns:1fr}}
.col{display:flex;flex-direction:column;gap:16px}
.card{background:#111827;border:1px solid #1e3050;border-radius:12px;padding:20px}
.card-lbl{font-size:10px;letter-spacing:2px;color:#64748b;text-transform:uppercase;margin-bottom:14px}
.drop{border:2px dashed #1e3050;border-radius:10px;padding:36px 20px;text-align:center;cursor:pointer;transition:.25s;min-height:210px;display:flex;flex-direction:column;align-items:center;justify-content:center;gap:12px;position:relative;overflow:hidden}
.drop:hover,.drop.drag{border-color:#3b82f6;background:rgba(59,130,246,.04)}
.drop-ico{font-size:38px;opacity:.4}
.drop-txt{font-size:13px;color:#64748b}
.browse-btn{background:transparent;border:1px solid #3b82f6;color:#3b82f6;padding:8px 18px;border-radius:8px;font-size:11px;cursor:pointer;transition:.2s;letter-spacing:.5px}
.browse-btn:hover{background:rgba(59,130,246,.1)}
.scan{position:absolute;left:0;width:100%;height:2px;background:linear-gradient(90deg,transparent,#3b82f6,transparent);animation:scan 2s linear infinite;opacity:0}
.drop.scanning .scan{opacity:1}
@keyframes scan{0%{top:0}100%{top:100%}}
#prev-wrap{display:none;flex-direction:column;align-items:center;gap:8px}
#prev-img{max-width:100%;max-height:190px;border-radius:8px;border:1px solid #1e3050;object-fit:cover}
.prev-meta{font-size:11px;color:#64748b}
#cls-btn{width:100%;padding:12px;margin-top:12px;border-radius:9px;background:linear-gradient(135deg,rgba(59,130,246,.12),rgba(168,85,247,.08));border:1px solid #3b82f6;color:#3b82f6;font-size:13px;cursor:pointer;display:none;transition:.25s;letter-spacing:.5px}
#cls-btn:hover{background:rgba(59,130,246,.18)}
#cls-btn:disabled{opacity:.4;cursor:not-allowed}
.spin{display:inline-block;width:12px;height:12px;border:2px solid rgba(59,130,246,.3);border-top-color:#3b82f6;border-radius:50%;animation:sp .7s linear infinite;vertical-align:middle;margin-right:6px}
@keyframes sp{to{transform:rotate(360deg)}}
.err{background:rgba(239,68,68,.07);border:1px solid rgba(239,68,68,.3);border-radius:8px;padding:10px;color:#f87171;font-size:12px;display:none;margin-top:10px}
.empty{display:flex;flex-direction:column;align-items:center;justify-content:center;min-height:210px;gap:10px;opacity:.35}
.empty-ico{font-size:36px}
.empty-txt{font-size:12px;color:#64748b;letter-spacing:1px;text-transform:uppercase}
#res-content{display:none}
.top-card{border-radius:10px;padding:16px;margin-bottom:16px;border:1px solid rgba(59,130,246,.2);background:rgba(59,130,246,.05)}
.top-cls{font-size:20px;font-weight:700;letter-spacing:1px;text-transform:uppercase;margin-bottom:4px}
.top-meta{font-size:11px;color:#64748b}
.top-meta b{font-size:14px;color:#e2e8f0}
.bars-lbl{font-size:10px;letter-spacing:1.5px;color:#64748b;text-transform:uppercase;margin-bottom:12px}
.bar-row{margin-bottom:12px}
.bar-head{display:flex;justify-content:space-between;font-size:12px;margin-bottom:4px}
.bar-pct{font-weight:600}
.bar-track{height:6px;background:rgba(255,255,255,.05);border-radius:3px;overflow:hidden}
.bar-fill{height:100%;border-radius:3px;transition:width .8s cubic-bezier(.4,0,.2,1);width:0}
.hist-list{display:flex;flex-direction:column;gap:7px;max-height:180px;overflow-y:auto}
.hist-item{display:flex;justify-content:space-between;align-items:center;background:rgba(255,255,255,.02);border:1px solid #1e3050;border-radius:7px;padding:7px 12px;font-size:11px}
.hist-cls{font-weight:600;text-transform:uppercase;letter-spacing:.5px}
.hist-meta{color:#64748b}
.hist-meta b{color:#22c55e}
footer{display:flex;border-top:1px solid #1e3050;background:rgba(11,15,26,.95)}
.stat{flex:1;padding:12px;text-align:center;border-right:1px solid #1e3050}
.stat:last-child{border:none}
.stat-v{font-size:15px;font-weight:700;color:#3b82f6}
.stat-k{font-size:9px;letter-spacing:1.5px;color:#64748b;text-transform:uppercase;margin-top:2px}
::-webkit-scrollbar{width:4px}
::-webkit-scrollbar-track{background:#0b0f1a}
::-webkit-scrollbar-thumb{background:#1e3050;border-radius:2px}
</style>
</head>
<body>
<canvas id="cv"></canvas>
<div class="app">

<header>
  <div class="brand">Property <b>Classifier</b></div>
  <div class="hdr-right">
    <span class="badge badge-gray">ResNet101V2</span>
    <span class="badge badge-green" id="status-badge">
      <span class="dot" style="background:#22c55e"></span>Online
    </span>
  </div>
</header>

<main>
  <div class="col">
    <div class="card">
      <div class="card-lbl">Image Input</div>
      <div class="drop" id="drop">
        <div class="scan"></div>
        <div id="up-default">
          <div class="drop-ico">&#9654;</div>
          <div class="drop-txt">Drop a building image here</div>
          <button class="browse-btn" onclick="document.getElementById('fi').click()">Browse File</button>
          <div style="font-size:11px;color:#334155">JPEG or PNG</div>
        </div>
        <div id="prev-wrap">
          <img id="prev-img" alt="">
          <div class="prev-meta" id="prev-meta"></div>
        </div>
      </div>
      <input type="file" id="fi" accept="image/jpeg,image/png" style="display:none">
      <button id="cls-btn" onclick="classify()">&#9654; &nbsp;Classify Property</button>
      <div class="err" id="err"></div>
    </div>
    <div class="card">
      <div class="card-lbl">Prediction History</div>
      <div class="hist-list" id="hist">
        <div style="font-size:11px;color:#64748b;text-align:center;padding:14px">No predictions yet</div>
      </div>
    </div>
  </div>

  <div class="col">
    <div class="card" style="flex:1">
      <div class="card-lbl">Classification Result</div>
      <div class="empty" id="empty">
        <div class="empty-ico">&#9636;</div>
        <div class="empty-txt">Awaiting image</div>
        <div style="font-size:10px;color:#475569">Upload a property image to begin</div>
      </div>
      <div id="res-content">
        <div class="top-card" id="top-card">
          <div class="top-cls" id="top-cls">---</div>
          <div class="top-meta">
            Confidence: <b id="top-conf">0%</b>
            &nbsp;&bull;&nbsp;
            Latency: <b id="top-lat">0ms</b>
          </div>
        </div>
        <div class="bars-lbl">Probability Distribution</div>
        <div id="bars"></div>
      </div>
    </div>
  </div>
</main>

<footer>
  <div class="stat"><div class="stat-v" id="s-total">0</div><div class="stat-k">Predictions</div></div>
  <div class="stat"><div class="stat-v">4</div><div class="stat-k">Classes</div></div>
  <div class="stat"><div class="stat-v">224×224</div><div class="stat-k">Input Size</div></div>
  <div class="stat"><div class="stat-v">24,640</div><div class="stat-k">Train Params</div></div>
  <div class="stat"><div class="stat-v" id="s-lat">--ms</div><div class="stat-k">Last Latency</div></div>
</footer>
</div>

<script>
// Canvas background
const cv=document.getElementById('cv'),cx=cv.getContext('2d');
let W,H,pts=[];
function rsz(){W=cv.width=window.innerWidth;H=cv.height=window.innerHeight;pts=[];for(let x=0;x<W;x+=64)for(let y=0;y<H;y+=64)pts.push({x,y,a:Math.random()*.3+.05});}
function draw(){cx.clearRect(0,0,W,H);cx.strokeStyle='#1e3050';cx.lineWidth=.4;cx.globalAlpha=.12;for(let x=0;x<W;x+=64){cx.beginPath();cx.moveTo(x,0);cx.lineTo(x,H);cx.stroke();}for(let y=0;y<H;y+=64){cx.beginPath();cx.moveTo(0,y);cx.lineTo(W,y);cx.stroke();}const t=Date.now()/1000;pts.forEach((p,i)=>{const v=Math.sin(t*.7+i*.4)*.5+.5;cx.globalAlpha=v*p.a;cx.fillStyle='#3b82f6';cx.beginPath();cx.arc(p.x,p.y,1.2,0,Math.PI*2);cx.fill();});cx.globalAlpha=1;requestAnimationFrame(draw);}
window.addEventListener('resize',rsz);rsz();draw();

// App state
const COLORS={'exterior_facade':'#3b82f6','office_interior':'#22c55e','warehouse':'#f59e0b','hvac_pipeline':'#a855f7'};
let selFile=null,predCount=0,history=[];

// Drag & drop
const drop=document.getElementById('drop'),fi=document.getElementById('fi');
drop.addEventListener('dragover',e=>{e.preventDefault();drop.classList.add('drag')});
drop.addEventListener('dragleave',()=>drop.classList.remove('drag'));
drop.addEventListener('drop',e=>{e.preventDefault();drop.classList.remove('drag');if(e.dataTransfer.files[0])handle(e.dataTransfer.files[0])});
fi.addEventListener('change',()=>{if(fi.files[0])handle(fi.files[0])});
drop.addEventListener('click',e=>{if(e.target.tagName!=='BUTTON')fi.click()});

function handle(f){
  if(!f.type.startsWith('image/')){showErr('Please upload a JPEG or PNG image.');return;}
  selFile=f;
  const r=new FileReader();
  r.onload=e=>{
    document.getElementById('up-default').style.display='none';
    const pw=document.getElementById('prev-wrap');pw.style.display='flex';
    document.getElementById('prev-img').src=e.target.result;
    document.getElementById('prev-meta').textContent=f.name+' · '+(f.size/1024).toFixed(0)+' KB';
    document.getElementById('cls-btn').style.display='block';
    hideErr();
    drop.classList.add('scanning');
    setTimeout(()=>drop.classList.remove('scanning'),2000);
  };
  r.readAsDataURL(f);
}

async function classify(){
  if(!selFile)return;
  const btn=document.getElementById('cls-btn');
  btn.disabled=true;btn.innerHTML='<span class="spin"></span>Analysing...';
  drop.classList.add('scanning');hideErr();
  const fd=new FormData();fd.append('file',selFile);
  const t0=Date.now();
  try{
    const r=await fetch('/predict',{method:'POST',body:fd});
    if(!r.ok){const e=await r.json();throw new Error(e.detail||'Server error');}
    const d=await r.json();
    const ms=Date.now()-t0;
    showResult(d,ms);
    addHistory(d,ms);
    predCount++;
    document.getElementById('s-total').textContent=predCount;
    document.getElementById('s-lat').textContent=ms+'ms';
  }catch(e){showErr('Error: '+e.message);}
  finally{btn.disabled=false;btn.innerHTML='&#9654; &nbsp;Classify Property';drop.classList.remove('scanning');}
}

function showResult(d,ms){
  document.getElementById('empty').style.display='none';
  document.getElementById('res-content').style.display='block';
  const raw=d.predictions[0].class_raw||Object.keys(COLORS).find(k=>d.top_class.toLowerCase().includes(k.split('_')[0]))||'exterior_facade';
  const col=COLORS[raw]||'#3b82f6';
  document.getElementById('top-cls').textContent=d.top_class;
  document.getElementById('top-cls').style.color=col;
  document.getElementById('top-card').style.borderColor=col+'40';
  document.getElementById('top-card').style.background=col+'08';
  document.getElementById('top-conf').textContent=(d.confidence*100).toFixed(1)+'%';
  document.getElementById('top-lat').textContent=ms+'ms';
  const bars=document.getElementById('bars');bars.innerHTML='';
  d.predictions.forEach(p=>{
    const c=COLORS[p.class_raw]||'#3b82f6';
    const pct=(p.probability*100).toFixed(1);
    const row=document.createElement('div');row.className='bar-row';
    row.innerHTML=`<div class="bar-head"><span style="color:#e2e8f0">${p.class}</span><span class="bar-pct" style="color:${c}">${pct}%</span></div><div class="bar-track"><div class="bar-fill" style="background:${c}" data-p="${pct}"></div></div>`;
    bars.appendChild(row);
  });
  requestAnimationFrame(()=>document.querySelectorAll('.bar-fill').forEach(b=>b.style.width=b.dataset.p+'%'));
}

function addHistory(d,ms){
  history.unshift({cls:d.top_class,conf:(d.confidence*100).toFixed(1),ms});
  if(history.length>8)history.pop();
  const h=document.getElementById('hist');
  h.innerHTML=history.map(x=>`<div class="hist-item"><span class="hist-cls" style="color:#3b82f6">${x.cls}</span><span class="hist-meta">Conf: <b>${x.conf}%</b></span><span style="color:#475569">${x.ms}ms</span></div>`).join('');
}

function showErr(m){const e=document.getElementById('err');e.textContent=m;e.style.display='block';}
function hideErr(){document.getElementById('err').style.display='none';}
</script>
</body>
</html>"""


@app.get("/", response_class=HTMLResponse)
async def root():
    return HTML


@app.get("/health")
async def health():
    return JSONResponse({
        "status":       "ok",
        "model_loaded": model is not None,
        "classes":      classes,
        "input_size":   list(IMG_SIZE),
    })


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Validate
    if file.content_type not in ("image/jpeg", "image/png", "image/jpg"):
        raise HTTPException(415, f"Unsupported file type: {file.content_type}")

    if model is None:
        raise HTTPException(503, "Model not loaded. Check Space logs.")

    # Read image
    data = await file.read()
    if len(data) == 0:
        raise HTTPException(400, "Empty file received.")

    try:
        img = Image.open(io.BytesIO(data)).convert("RGB")
        img = img.resize(IMG_SIZE, Image.BILINEAR)
    except Exception as e:
        raise HTTPException(400, f"Cannot read image: {e}")

    # Inference
    arr   = np.array(img, dtype=np.float32) / 255.0
    arr   = np.expand_dims(arr, 0)
    t0    = time.time()
    probs = model.predict(arr, verbose=0)[0]
    ms    = round((time.time() - t0) * 1000, 1)

    # Build response
    top = np.argsort(probs)[::-1]
    return JSONResponse({
        "top_class":  CLASS_DISPLAY.get(classes[top[0]], classes[top[0]]),
        "confidence": float(probs[top[0]]),
        "latency_ms": ms,
        "predictions": [
            {
                "class":       CLASS_DISPLAY.get(classes[i], classes[i]),
                "class_raw":   classes[i],
                "probability": float(probs[i]),
            }
            for i in top[:4]
        ],
    })

"""AURA DRIVE v9: low-latency AI monitor + simulated safety response."""
from __future__ import annotations

import base64
import importlib.util
import io
import json
import sys
import threading
import time
from collections import deque
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
import torch
from PIL import Image, ImageDraw
from common import to_square
from roi_utils import crop_eyes, LEFT_EYE, RIGHT_EYE

ROOT = Path(__file__).resolve().parent
TEAM = ROOT / "teammate_yawn"
if not TEAM.exists():
    TEAM = ROOT / "yawm1"

st.set_page_config(page_title="AURA DRIVE | AI 운전자 모니터", page_icon="🚘", layout="wide", initial_sidebar_state="collapsed")
st.html("""<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=Noto+Sans+KR:wght@400;500;600;700;800&display=swap');
html,body,[data-testid="stApp"]{font-family:Inter,'Noto Sans KR',sans-serif;background:#090f19;color:#eaf4fa}
[data-testid="stAppViewContainer"]{background:radial-gradient(ellipse at 14% 0,#18405277,transparent 44%),#090f19}
[data-testid="stMainBlockContainer"]{max-width:1320px;padding:1.3rem 2.2rem 4rem}
[data-testid="stHeader"]{background:transparent}#MainMenu,footer{visibility:hidden}
h1,h2,h3{color:#f0f9ff;letter-spacing:-.04em}
.brand{display:flex;justify-content:space-between;align-items:center;gap:10px;margin-bottom:5px;padding:0 0 9px;border-bottom:1px solid #26384b;line-height:1.25}
.brand b{font-size:20px;letter-spacing:.17em}.brand b span{color:#88f9db}.brand small{font-size:11px;color:#89a8b9;letter-spacing:.12em}
.eyebrow{color:#7ff0d0;font-size:11px;font-weight:800;letter-spacing:.2em}.headline{font-size:clamp(24px,3.2vw,40px);font-weight:800;line-height:1.3;margin:10px 0 8px}
.subtitle{font-size:13px;color:#adc2d3;margin-bottom:18px}
.scene{height:270px;position:relative;overflow:hidden;border-radius:24px;border:1px solid #39566c;background:linear-gradient(#102e43,#15546b 52%,#06111e 53%);box-shadow:0 20px 70px #0006;margin-bottom:24px}
.scene:before{content:"";position:absolute;width:145%;height:110%;left:-22%;top:39%;background:linear-gradient(90deg,#07111f,#1d3546 49%,#07111f);clip-path:polygon(49.5% 0,50.5% 0,100% 100%,0 100%)}
.lane{position:absolute;top:44%;left:49.7%;height:70%;width:8px;background:repeating-linear-gradient(to bottom,#adf9e8 0px,#adf9e8 15px,transparent 15px,transparent 45px);transform:perspective(130px) rotateX(32deg);animation:road .7s linear infinite;opacity:.85}
@keyframes road{to{background-position:0 45px}}
.scene:after{content:"";position:absolute;bottom:-145px;right:2%;width:320px;height:320px;border:29px solid #1b2a3b;border-radius:50%;box-shadow:0 0 0 5px #65788966,inset 0 0 35px #111c2e;transform:rotate(-12deg)}
.scene-text{position:absolute;z-index:2;top:23px;left:29px;max-width:650px}.scene-title{font-size:32px;font-weight:800;letter-spacing:-.04em;color:#effffb}
.scene-caption{font-size:12px;color:#a5cfd5;line-height:1.7}.scene-pill{display:inline-block;padding:7px 10px;background:#112d39d9;border:1px solid #417575;border-radius:999px;font-size:10px;color:#aaf5df;margin-top:17px}
.glass{position:absolute;inset:0;pointer-events:none;background:linear-gradient(90deg,#08132199,transparent 70%),linear-gradient(0deg,#09111c77,transparent 47%)}
.card{background:linear-gradient(140deg,#152537,#101b2b);border:1px solid #2b3e52;border-radius:18px;padding:19px 20px;min-height:110px;margin:8px 0}
.card .label{color:#9aafc3;font-size:12px;font-weight:600}.card .num{font-size:30px;font-weight:800;margin-top:11px;letter-spacing:-.045em}.card .foot{font-size:11px;color:#83a4b5;margin-top:7px}
.alert{padding:17px 20px;background:#13382f;border:1px solid #397c6a;border-radius:16px;margin:16px 0;display:flex;gap:13px;align-items:center}
.alert.eye{background:#57282e;border-color:#f67d84;animation:pulse 1.1s ease-in-out infinite}.alert.yawn{background:#554027;border-color:#e6b66e}
.alert b{font-size:17px}.alert span{font-size:12px;color:#c6d9df}.dot{width:14px;height:14px;flex-shrink:0;border-radius:50%;background:#8cf6d5;box-shadow:0 0 18px #86ffdd}.alert.eye .dot{background:#ff8489;box-shadow:0 0 23px #ff6565}.alert.yawn .dot{background:#fccc77}
@keyframes pulse{50%{box-shadow:0 0 29px #ff666944}}
.section{font-size:19px;font-weight:800;margin:17px 0 9px}.dim{font-size:12px;color:#8faabf}
[data-testid="stRadio"] [role="radiogroup"]{gap:9px;flex-wrap:wrap}
[data-testid="stRadio"] [role="radiogroup"] label{border:1px solid #31485b;background:#142233;padding:8px 14px;border-radius:12px}
[data-testid="stRadio"] [role="radiogroup"] label:has(input:checked){background:#1b4745;border-color:#5bbaa6}
.stButton>button{border:1px solid #3b5b69;border-radius:12px;font-weight:700;min-height:42px}
.stButton>button[kind="primary"]{background:#89f5d8;color:#082b31;border:0}
.audio-controls{padding:7px 10px;background:#152637;border:1px solid #375265;border-radius:13px;display:flex;gap:7px;flex-wrap:wrap;align-items:center}
.audio-controls button{padding:6px 9px;min-height:32px;font-size:12px;line-height:1.2;background:#22483f;color:#eafff9;border:1px solid #5db79f;border-radius:9px;font-weight:800;cursor:pointer}
.audio-controls button.off{background:#25384c;border-color:#536779}.audio-hint{font-size:11px;color:#bed1dd}
[data-testid="stVideo"]{border-radius:16px;overflow:hidden}
@media(max-width:700px){[data-testid="stMainBlockContainer"]{padding:.7rem .9rem 2.5rem}.scene{height:250px;border-radius:17px}.scene-text{left:19px;top:22px}.scene-title{font-size:23px}.scene:after{width:230px;height:230px;bottom:-135px;right:-85px}.card{padding:14px;min-height:93px}.card .num{font-size:23px}.alert b{font-size:14px}}
@media(prefers-reduced-motion:reduce){.lane,.alert.eye{animation:none}}

/* V3 cockpit -- responsive windshield and car interior */
[data-testid="stAppViewContainer"]{background:radial-gradient(ellipse 78% 38% at 45% -8%,#143e50 0%,transparent 67%),#080e18}
[data-testid="stMainBlockContainer"]{max-width:1480px;padding-top:1.2rem}
[data-testid="stRadio"] [role="radiogroup"] label p{color:#dceef3!important;font-weight:750!important}
[data-testid="stRadio"] [role="radiogroup"] label:has(input:checked) p{color:#a3ffe4!important}
.cockpit{--mint:#85ffe0;position:relative;isolation:isolate;overflow:hidden;height:415px;border:1px solid #386779;border-radius:24px;background:#071622;box-shadow:inset 0 0 140px #0009,0 24px 55px #0009;margin:13px 0 17px}
.cockpit .sky{position:absolute;inset:0 0 25%;background:radial-gradient(ellipse at 48% 70%,#e7bb9162 0%,transparent 29%),linear-gradient(180deg,#0a1a32 0%,#153c5a 51%,#39617a 86%,#a27463 100%)}
.cockpit .sun{position:absolute;left:49%;top:31%;width:105px;height:105px;border-radius:50%;background:radial-gradient(circle,#f6cda1 0%,#ffcd994e 37%,transparent 70%);filter:blur(2px)}
.cockpit .mountains{position:absolute;left:0;right:0;top:41%;height:28%;background:#112d40;clip-path:polygon(0 63%,7% 50%,13% 63%,20% 41%,27% 57%,38% 34%,43% 53%,51% 40%,59% 62%,71% 28%,78% 49%,84% 35%,92% 62%,100% 51%,100% 100%,0 100%)}
.cockpit .road{position:absolute;inset:49% 0 0;background:linear-gradient(160deg,#243745 0%,#0b1620 72%);clip-path:polygon(49% 0,51% 0,100% 100%,0 100%)}
.cockpit .shoulder{position:absolute;inset:49% 0 0;clip-path:polygon(48.7% 0,49.2% 0,4% 100%,0 100%,100% 100%,96% 100%,50.8% 0,51.3% 0,100% 100%,0 100%);background:linear-gradient(180deg,#a3d2b970,#a5f1dc5b)}
.cockpit .roadline{position:absolute;top:51%;bottom:8%;width:5px;left:49.8%;background:repeating-linear-gradient(180deg,transparent 0 23px,#ddffee 23px 33px,transparent 33px 67px);animation:laneFlow 1.25s linear infinite;transform:perspective(110px) rotateX(20deg);filter:drop-shadow(0 0 4px #76efd7)}
@keyframes laneFlow{to{background-position:0 67px}}
.cockpit .car-ahead{position:absolute;left:63%;top:52%;width:34px;height:19px;background:#263644;border-radius:9px 9px 3px 3px;box-shadow:-1px 0 #1c2735,1px 0 #1c2735}
.cockpit .car-ahead:before{content:"";position:absolute;top:3px;left:7px;width:19px;height:7px;background:#5e7481;border-radius:4px 4px 1px 1px}
.cockpit .car-ahead:after{content:"";position:absolute;bottom:2px;left:3px;width:27px;height:3px;background:linear-gradient(90deg,#e27263 0 16%,transparent 17% 82%,#e27263 83%)}
.cockpit .windshield{position:absolute;inset:0;border:19px solid #111b29;border-bottom-width:0;clip-path:polygon(0 0,100% 0,100% 100%,0 100%);pointer-events:none;box-shadow:inset 0 0 30px #010912}
.cockpit .pillar-left{position:absolute;left:-16px;top:0;height:125%;width:68px;background:linear-gradient(95deg,#05090f,#192331 62%,#334050);transform:skewX(-10deg);transform-origin:top;filter:drop-shadow(15px 0 22px #050c16a5)}
.cockpit .pillar-right{position:absolute;right:-16px;top:0;height:125%;width:75px;background:linear-gradient(85deg,#334050,#192331 35%,#05090f);transform:skewX(10deg);transform-origin:top}
.cockpit .mirror{position:absolute;left:50%;top:-1px;width:145px;height:45px;transform:translateX(-50%);background:linear-gradient(#0a111c,#162534);border:6px solid #1e3040;border-radius:0 0 16px 16px;box-shadow:0 5px 17px #0008}
.cockpit .mirror:after{content:"";position:absolute;inset:7px 9px;background:linear-gradient(110deg,#4a6274,#153047 40%,#516b73);border-radius:5px}
.cockpit .cabin-glow{position:absolute;inset:0;background:linear-gradient(90deg,#06101fa3,transparent 42%,#07132266);pointer-events:none}
.cockpit .dash{position:absolute;bottom:-90px;left:-5%;width:110%;height:215px;background:radial-gradient(ellipse at 50% 5%,#2e4a5d 0%,#101c2b 37%,#050b13 73%);border-radius:52% 52% 0 0;border-top:8px solid #273c4b;box-shadow:0 -22px 55px #07101fa6,inset 0 17px 45px #000a;transform:perspective(600px) rotateX(13deg)}
.cockpit .dashboard-strip{position:absolute;bottom:8px;left:50%;transform:translateX(-50%);height:57px;width:36%;max-width:400px;min-width:220px;background:linear-gradient(#102331,#09131e);border:1px solid #365269;border-radius:25px 25px 8px 8px;display:flex;gap:19px;align-items:center;justify-content:center;color:#8affdd;font:700 11px Inter,monospace;letter-spacing:.06em;box-shadow:0 -2px 16px #6fddc121}
.cockpit .dashboard-strip strong{font-size:22px;letter-spacing:-.04em;color:#f5ffff}
.cockpit .wheel{position:absolute;right:12%;bottom:-130px;width:265px;height:265px;border:28px solid #1a2635;border-radius:50%;background:transparent;box-shadow:0 0 0 5px #4b6472,inset 0 0 0 4px #405161,0 13px 50px #000;transform:rotate(-15deg)}
.cockpit .wheel:before{content:"";position:absolute;left:-7px;right:-7px;top:88px;height:27px;background:linear-gradient(#293b4c,#101c2a);border-radius:11px}
.cockpit .wheel:after{content:"✦";position:absolute;top:67px;left:69px;width:70px;height:70px;display:grid;place-items:center;background:linear-gradient(140deg,#22384a,#080f1b);border:5px solid #405d6b;border-radius:50%;color:#96f4d8;font-size:23px}
.cockpit .hud{position:absolute;top:28px;left:36px;z-index:6;max-width:min(540px,70%)}
.cockpit .hud-kicker{font-size:10px;letter-spacing:.24em;color:#8af7dd;font-weight:800;text-shadow:0 2px 8px #07151e}
.cockpit .hud-title{font-size:clamp(26px,3vw,43px);line-height:1.23;font-weight:900;color:#f5fff9;letter-spacing:-.05em;margin:14px 0 12px;text-shadow:0 3px 14px #010b16}
.cockpit .hud-title span{color:#9effe2}
.cockpit .hud-sub{font-size:12px;line-height:1.9;color:#d1eae8;text-shadow:0 2px 6px #041320}
.cockpit .hud-tags{display:flex;gap:8px;flex-wrap:wrap;margin-top:19px}
.cockpit .hud-tags span{padding:7px 11px;border-radius:8px;font-size:10px;font-weight:800;color:#adffe9;background:#061c28c9;border:1px solid #3e998a}
.cockpit .side-status{position:absolute;z-index:6;top:28px;right:30px;min-width:154px;padding:11px 13px;background:#06141ccc;border:1px solid #497b82;border-radius:12px;box-shadow:0 10px 30px #071020}
.cockpit .side-status .led{width:7px;height:7px;display:inline-block;background:#7ef6d0;border-radius:50%;box-shadow:0 0 10px #85ffdf;margin-right:7px}
.cockpit .side-status b{font-size:11px;color:#dbfff6;letter-spacing:.1em}
.cockpit .side-status small{display:block;font-size:9px;color:#a4c5cd;margin-top:7px}
.cockpit.is-eye .side-status{border-color:#ff7878;background:#521b25ea}.cockpit.is-eye .side-status .led{background:#ff8080;box-shadow:0 0 15px #ff6969}
.cockpit.is-yawn .side-status{border-color:#ffb875;background:#4c311fe0}.cockpit.is-yawn .side-status .led{background:#ffba6c}
.cockpit.is-eye{box-shadow:inset 0 0 70px #e63d475c,0 20px 55px #000}
.cockpit .corner-tag{position:absolute;z-index:5;bottom:17px;left:35px;color:#85e6d2;opacity:.85;font-size:10px;letter-spacing:.15em}
.monitor-head{display:flex;align-items:center;justify-content:space-between;margin:14px 0;gap:8px}.monitor-head strong{font-size:16px;color:#e8fbfa}.monitor-head small{font-size:11px;color:#6cdbc1}
.metric-grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:7px;margin:8px 0}.metric-grid .card{min-height:76px;margin:0;padding:10px}
.metric-grid .card .num{font-size:22px;margin-top:5px}.metric-grid .card .foot{font-size:10px;margin-top:4px}
.audio-controls{background:linear-gradient(90deg,#143039,#101d2d);border-color:#326477;margin:0 0 15px;box-shadow:inset 0 1px 0 #9cfde010}
.audio-controls button{background:#193b38;border-color:#3a9c86}.audio-controls button:first-child{background:#9af6dc;color:#092931;border:0}
.audio-controls button.off:first-child{background:#334456;color:#d8ebf5}
@media(max-width:780px){
.cockpit{height:355px;border-radius:16px}.cockpit .hud{left:20px;top:62px;max-width:87%}.cockpit .hud-title{font-size:27px}.cockpit .hud-sub{font-size:11px}.cockpit .hud-tags{margin-top:13px}.cockpit .side-status{top:13px;right:14px;min-width:unset;padding:7px 9px}.cockpit .side-status small{display:none}.cockpit .mirror{width:95px;height:28px}.cockpit .wheel{width:200px;height:200px;border-width:21px;bottom:-128px;right:-6px}.cockpit .wheel:before{top:57px}.cockpit .wheel:after{top:40px;left:45px;width:53px;height:53px}.cockpit .dashboard-strip{bottom:8px;left:40%;width:42%;min-width:160px;height:50px;gap:9px}.cockpit .dashboard-strip strong{font-size:18px}.cockpit .corner-tag{bottom:70px;left:20px}.metric-grid{grid-template-columns:repeat(2,minmax(0,1fr))}
}
@media(max-width:450px){.cockpit{height:330px}.cockpit .hud{top:50px}.cockpit .hud-title{font-size:25px}.cockpit .hud-sub{max-width:235px}.cockpit .hud-tags span{font-size:8px;padding:6px}.cockpit .side-status{top:12px}.cockpit .dashboard-strip{left:37%}.cockpit .corner-tag{display:none}.audio-controls{padding:10px}.audio-controls button{font-size:12px;padding:10px}}
@media(prefers-reduced-motion:reduce){.cockpit .roadline{animation:none}}

</style>""")

# V6 compact cockpit: keep the original camera and status within one viewport.
st.html("""<style>
@media(min-width:781px){
  .cockpit{height:245px;margin:7px 0 12px}
  .cockpit .hud{top:36px;left:32px}
  .cockpit .hud-title{font-size:27px;line-height:1.2}
  .cockpit .hud-sub{font-size:11px;margin-top:7px}
  .cockpit .hud-tags{margin-top:10px}
  .cockpit .wheel{bottom:-170px}
  .cockpit .dashboard-strip{bottom:3px}
  .metric-grid{grid-template-columns:repeat(2,minmax(0,1fr));gap:7px;margin:8px 0}
  .metric-grid .card{min-height:76px;padding:10px}
  .metric-grid .card .num{font-size:22px}
}
@media(max-width:780px){.cockpit{height:240px}.cockpit .hud{top:45px}.cockpit .hud-title{font-size:24px}}
</style>""")



# Live landmark thumbnail caption styling and mobile wrapping
st.html("""<style>
.roi-mini-title{font-size:11px;font-weight:800;letter-spacing:.12em;color:#a1ead7;margin-top:12px;margin-bottom:7px}
.roi-mini-title span{float:right;color:#7d9aab;font-size:10px}
[data-testid="stImage"] img{border-radius:11px}
@media(max-width:780px){.roi-mini-title{font-size:10px}}
</style>""")

def model_paths():
    team = Path(st.session_state.get("team_path", str(TEAM))).expanduser().resolve()
    eye = Path(st.session_state.get("eyes_path", str(ROOT / "runs/eyes_yolo26n-cls/weights/best.pt"))).expanduser().resolve()
    yawn = team / "runs/03_yawn_resnet18/weights/03_best.pt"
    return eye, team, yawn, team / "models/face_landmarker.task"


def missing_files(paths):
    eye, team, yawn, _ = paths
    return [str(p) for p in (eye, team / "03_yawn_common.py", yawn) if not p.is_file()]


def load_team(path):
    spec = importlib.util.spec_from_file_location("teammate_yawn_common", path / "03_yawn_common.py")
    if spec is None or spec.loader is None:
        raise ImportError("팀원 03_yawn_common.py를 불러오지 못했어")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


@st.cache_resource(show_spinner="AI 모델 연결 중...")
def get_models(eye_path, team_path, yawn_path, face_path):
    from ultralytics import YOLO
    team = load_team(Path(team_path))
    device = team.choose_device("cpu")
    yawn, ckpt = team.load_checkpoint(Path(yawn_path), device)
    face = team.verify_face_model(Path(face_path), ckpt)
    eyes = YOLO(str(eye_path))
    closed_idx = next((int(k) for k, v in eyes.names.items() if str(v).lower() == "closed"), None)
    if closed_idx is None:
        raise ValueError("눈 모델에 Closed 클래스가 없음")
    return dict(eyes=eyes, closed_idx=closed_idx, yawn=yawn, team=team, ckpt=ckpt,
                face=face, device=device, trans=team.make_transform(training=False),
                eye_lock=threading.RLock(), yawn_lock=threading.RLock())


def eye_clahe(roi):
    """Match the LAB L-channel CLAHE recipe in 15_roi_ablation.py."""
    rgb = np.asarray(roi.convert("RGB"))
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    lab[:, :, 0] = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4)).apply(lab[:, :, 0])
    return Image.fromarray(cv2.cvtColor(lab, cv2.COLOR_LAB2RGB))


def mouth_variant(roi, variant):
    if variant == "Grayscale (실험)":
        return Image.fromarray(cv2.cvtColor(np.asarray(roi.convert("RGB")), cv2.COLOR_RGB2GRAY)).convert("RGB")
    return roi.convert("RGB")


def preview_bytes(image):
    if image is None:
        return None
    im = image.copy().convert("RGB")
    im.thumbnail((320, 220))
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=84)
    return buf.getvalue()


# Visualization only: these dots are never added to classifier inputs.
# The same FaceMesh result used for the eye crop provides preview landmarks.
LIP_POINTS = (61, 40, 37, 0, 267, 270, 291, 321, 314, 17, 84, 91,
              78, 81, 13, 311, 308, 402, 14, 178, 87, 82)


def landmark_preview(image, face_result, indices_groups, pad_x=0.35, pad_y=0.55):
    if not face_result or not face_result.multi_face_landmarks:
        return None
    rgb = np.asarray(image.convert("RGB"))
    h, w = rgb.shape[:2]
    lm = face_result.multi_face_landmarks[0].landmark
    groups = []
    for indices in indices_groups:
        pts = [(float(lm[i].x * w), float(lm[i].y * h)) for i in indices]
        groups.append(pts)
    points = [pt for group in groups for pt in group]
    if not points:
        return None
    xs, ys = zip(*points)
    x0, x1 = min(xs), max(xs)
    y0, y1 = min(ys), max(ys)
    bw, bh = x1-x0, y1-y0
    if bw < 4 or bh < 1:
        return None
    # Keep enough context to identify eyelids / lips while keeping preview compact.
    xpad, ypad = max(8., bw*pad_x), max(8., max(bh, bw*.18)*pad_y)
    left, top = max(0, int(x0-xpad)), max(0, int(y0-ypad))
    right, bottom = min(w, int(x1+xpad)), min(h, int(y1+ypad))
    if right-left < 10 or bottom-top < 10:
        return None
    preview = Image.fromarray(rgb[top:bottom, left:right].copy())
    draw = ImageDraw.Draw(preview)
    dot = max(1, min(3, round(preview.width/140)))
    for gi, group in enumerate(groups):
        color = "#5dffe0" if gi == 0 else "#f6c36e"
        for x, y in group:
            cx, cy = round(x-left), round(y-top)
            draw.ellipse((cx-dot,cy-dot,cx+dot,cy+dot), fill=color,
                         outline="#09212a", width=1)
    return preview


class CachedMesh:
    def __init__(self, result):
        self.result = result

    def process(self, _rgb):
        return self.result


class Monitor:
    def __init__(self):
        self.lock = threading.RLock()
        self.closed_since = None
        self.yawn_since = None
        self.closed_latched = False
        self.yawn_latched = False
        self.last_update = 0
        self.last_yawn_event = 0
        self.serial = 0
        self.yawn_count = 0
        self.events = deque(maxlen=100)
        self.sounds = deque(maxlen=20)
        self.data = dict(eye=None, yawn=None, duration=0., fps=0., mode="wait", eye_roi=None, mouth_roi=None, eye_points=None, mouth_points=None, eye_alert=False, yawn_alert=False)

    def update(self, eye, yawn, fps, cfg, eye_roi=None, mouth_roi=None, eye_points=None, mouth_points=None):
        now = time.monotonic()
        with self.lock:
            if now - self.last_update > 1.5:
                self.closed_since = self.yawn_since = None
                self.closed_latched = self.yawn_latched = False
            self.last_update = now
            if eye is not None and eye >= cfg["eye"]:
                self.closed_since = self.closed_since or now
            else:
                self.closed_since = None
                self.closed_latched = False
            duration = now - self.closed_since if self.closed_since is not None else 0.
            if duration >= cfg["eye_sec"] and not self.closed_latched:
                self.closed_latched = True
                self.event("eye", eye)
            if yawn is not None and yawn >= cfg["yawn"]:
                self.yawn_since = self.yawn_since or now
            else:
                self.yawn_since = None
                self.yawn_latched = False
            if self.yawn_since is not None and now - self.yawn_since >= cfg["yawn_sec"] and not self.yawn_latched:
                self.yawn_latched = True
                self.last_yawn_event = now
                self.yawn_count += 1
                self.event("yawn", yawn)
            mode = "eye" if duration >= cfg["eye_sec"] else "yawn" if now - self.last_yawn_event < 5 and self.last_yawn_event else "ok"
            if eye is None and yawn is None:
                mode = "wait"
            self.data = dict(eye=eye, yawn=yawn, duration=duration, fps=fps, mode=mode,
                             eye_alert=duration >= cfg["eye_sec"],
                             yawn_alert=self.yawn_latched,
                             eye_roi=preview_bytes(eye_roi), mouth_roi=preview_bytes(mouth_roi),
                             eye_points=preview_bytes(eye_points), mouth_points=preview_bytes(mouth_points))

    def event(self, kind, p):
        self.serial += 1
        self.events.appendleft(dict(시간=datetime.now().strftime("%H:%M:%S"), 감지="눈 감김" if kind == "eye" else "하품", 확률=f"{p:.1%}"))
        self.sounds.append(dict(id=self.serial, kind=kind))

    def snapshot(self):
        with self.lock:
            if time.monotonic() - self.last_update > 2:
                return dict(eye=None, yawn=None, duration=0., fps=0., mode="wait", eye_roi=None, mouth_roi=None, eye_points=None, mouth_points=None, eye_alert=False, yawn_alert=False), list(self.events), list(self.sounds)
            return dict(self.data), list(self.events), list(self.sounds)


class Engine:
    def __init__(self, m, cfg, static=False):
        import mediapipe as mp
        self.m = m
        self.cfg = cfg
        self.mesh = mp.solutions.face_mesh.FaceMesh(static_image_mode=static, max_num_faces=1,
            refine_landmarks=True, min_detection_confidence=.5, min_tracking_confidence=.5)
        self.mouth = m["team"].MouthExtractor(m["face"], m["team"].CropConfig(**m["ckpt"]["crop_config"]))
        self.lock = threading.RLock()

    def predict(self, im):
        with self.lock:
            # One FaceMesh pass for both eye inference and landmark previews.
            face_result = self.mesh.process(np.asarray(im.convert("RGB")))
            eyes = crop_eyes(im, CachedMesh(face_result))
            eye_points = landmark_preview(im, face_result, (LEFT_EYE, RIGHT_EYE), pad_x=.22, pad_y=.8)
            mouth_points = landmark_preview(im, face_result, (LIP_POINTS,), pad_x=.40, pad_y=.62)
            e = None
            if eyes:
                # Batch left/right eye in one YOLO invocation to reduce CPU overhead.
                inputs = [to_square(eye_clahe(c) if self.cfg["eye_clahe"] else c, 224) for c in eyes]
                with self.m["eye_lock"]:
                    predictions = self.m["eyes"].predict(inputs, imgsz=224, device="cpu", verbose=False)
                    p = [float(result.probs.data[self.m["closed_idx"]]) for result in predictions]
                e = sum(p) / len(p)
            res = self.mouth.extract(im)
            crop = res.get("crop") if res.get("status") == "ok" else None
            y = None
            if crop is not None:
                t = self.m["trans"](mouth_variant(crop, self.cfg["yawn_color"])).unsqueeze(0).to(self.m["device"])
                with self.m["yawn_lock"], torch.inference_mode():
                    y = float(self.m["yawn"](t).softmax(1)[0, 1])
            return dict(eye=e, yawn=y, eyes=eyes, mouth=crop, eye_processed=eye_clahe(eyes[0]) if eyes and self.cfg["eye_clahe"] else (eyes[0] if eyes else None),
                        eye_points=eye_points, mouth_points=mouth_points)

    def close(self):
        self.mesh.close()
        self.mouth.close()



class LatestFrameWorker:
    """Keep WebRTC frames fluid; inference consumes the latest frame only.

    The callback never waits for YOLO, MediaPipe, or ResNet. Old queued frames
    are discarded instead of building up seconds of video latency.
    """
    def __init__(self, engine, monitor, interval_ms=250, max_width=640):
        self.engine = engine
        self.monitor = monitor
        self.interval = max(0.08, interval_ms / 1000.)
        self.max_width = max_width
        self.lock = threading.Lock()
        self.latest = None
        self.frame_fps = 0.0
        self.last_frame_time = None
        self.last_inference_start = 0.0
        self.inference_count = 0
        self.stop_event = threading.Event()
        self.worker = threading.Thread(target=self._work, daemon=True, name="aura-ai-worker")
        self.worker.start()

    def submit(self, bgr):
        now = time.monotonic()
        h, w = bgr.shape[:2]
        if w > self.max_width:
            bgr = cv2.resize(bgr, (self.max_width, round(h * self.max_width / w)), interpolation=cv2.INTER_AREA)
        with self.lock:
            if self.last_frame_time is not None:
                instant_fps = 1. / max(0.001, now - self.last_frame_time)
                self.frame_fps = instant_fps if not self.frame_fps else .85 * self.frame_fps + .15 * instant_fps
            self.last_frame_time = now
            self.latest = bgr

    def _work(self):
        while not self.stop_event.is_set():
            start = time.monotonic()
            with self.lock:
                ready = self.latest is not None
                if ready and start - self.last_inference_start >= self.interval:
                    frame = self.latest
                    self.latest = None
                    self.last_inference_start = start
                    fps = self.frame_fps
                else:
                    frame = None
            if frame is None:
                self.stop_event.wait(0.015)
                continue
            try:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                result = self.engine.predict(Image.fromarray(rgb))
                self.monitor.update(result["eye"], result["yawn"], fps, self.engine.cfg,
                                    eye_roi=result["eye_processed"], mouth_roi=result["mouth"],
                                    eye_points=result["eye_points"], mouth_points=result["mouth_points"])
                self.inference_count += 1
            except Exception as exc:
                print(f"[AURA ASYNC INFERENCE ERROR] {type(exc).__name__}: {exc}", flush=True)

    def close(self):
        self.stop_event.set()
        self.worker.join(timeout=1.)


# =============================================================
# V5: Car atmosphere is CSS applied AROUND the WebRTC widget.
# The original video frame is never segmented, filtered, flipped,
# cropped or annotated for display. Eye/mouth ROI previews are shown
# beside the camera, never over the original feed.
# =============================================================

def cabin_frame_style():
    """Decorate the Streamlit container, not the video pixels."""
    photo = ROOT / "aura_car_interior.jpg"
    background = "linear-gradient(140deg, #152a3b, #0b1624)"
    if photo.is_file():
        uri = base64.b64encode(photo.read_bytes()).decode("ascii")
        background = (
            "linear-gradient(160deg,rgba(7,20,32,.55),rgba(7,18,27,.72)),"
            f'url("data:image/jpeg;base64,{uri}")'
        )
    st.html(f"""<style>
    .st-key-aura_camera_frame {{
        background-image: {background};
        background-position: center center;
        background-size: cover;
        border: 1px solid #42757b;
        border-radius: 16px;
        padding: 8px;
        box-shadow: 0 12px 30px #0006;
    }}
    .st-key-aura_camera_frame [data-testid="stWebRtcPlayer"] {{
        border-radius: 14px;
        overflow: hidden;
    }}
    .st-key-aura_camera_frame video {{max-height: 48vh; object-fit: contain;}}
    @media (max-width: 700px) {{
        .st-key-aura_camera_frame {{padding: 12px 8px; border-radius: 16px;}}
    }}
    </style>""")


def kpi(title, value, note):
    st.html(f'<div class="card"><div class="label">{title}</div><div class="num">{value}</div><div class="foot">{note}</div></div>')


def audio_controls():
    files = [ROOT / "눈감김.mp3", ROOT / "하품.mp3"]
    if not all(p.is_file() for p in files):
        st.warning("🔇 음성 파일이 없어. VisionProject 최상위에 눈감김.mp3, 하품.mp3 두 개를 넣어줘.")
        return
    src1, src2 = [base64.b64encode(p.read_bytes()).decode("ascii") for p in files]
    js = """<div class="audio-controls"><button id="aura-toggle" class="off">🔇 사운드 켜기</button>
<button id="aura-eye">눈 감김 소리 테스트</button><button id="aura-yawn">하품 소리 테스트</button>
<span class="audio-hint" id="aura-hint">경고는 한 이벤트에 한 번씩 재생됩니다.</span></div>
<script>(function(){
if(!window.AURA){
  let sounds={eye:new Audio('data:audio/mpeg;base64,__EYE__'),yawn:new Audio('data:audio/mpeg;base64,__YAWN__')};
  window.AURA={enabled:false,last:0,playing:null,queue:[],sounds:sounds};
  for(let [kind,a] of Object.entries(sounds))a.onended=()=>{window.AURA.playing=null;window.AURA.next();};
  window.AURA.next=function(){if(!this.enabled||this.playing||!this.queue.length)return;
    let type=this.queue.shift(),a=this.sounds[type];this.playing=type;a.currentTime=0;
    a.play().catch(()=>{this.playing=null;});};
  window.AURA.push=function(kind){if(!this.enabled||this.queue.includes(kind)||this.playing===kind)return;
    if(kind==='eye'&&this.playing==='yawn'){this.sounds.yawn.pause();this.sounds.yawn.currentTime=0;this.playing=null;}
    kind==='eye'?this.queue.unshift(kind):this.queue.push(kind);this.next();};
  window.AURA.sync=function(events){for(const e of events){if(e.id<=this.last)continue;
    this.last=e.id;this.push(e.kind);}};
}
const a=window.AURA,button=document.getElementById('aura-toggle');
function paint(){button.textContent=a.enabled?'🔊 사운드 켜짐':'🔇 사운드 켜기';button.classList.toggle('off',!a.enabled);}
button.onclick=()=>{a.enabled=!a.enabled;if(!a.enabled){a.queue=[];for(const s of Object.values(a.sounds)){s.pause();s.currentTime=0;}a.playing=null;}paint();};
document.getElementById('aura-eye').onclick=()=>{a.enabled=true;paint();a.push('eye');};
document.getElementById('aura-yawn').onclick=()=>{a.enabled=true;paint();a.push('yawn');};paint();
})();</script>"""
    st.html(js.replace("__EYE__", src1).replace("__YAWN__", src2), unsafe_allow_javascript=True)


def sound_events(events):
    st.html("<script>window.AURA?.sync(" + json.dumps(events) + ");</script>", unsafe_allow_javascript=True)


def render_response_simulation():
    st.html("""<style>
    .response-demo{overflow:hidden;border:1px solid #345365;border-radius:20px;color:#eaf5fa;
      background:radial-gradient(ellipse at 75% 0,#174354 0,transparent 45%),linear-gradient(145deg,#132739,#09121e);
      box-shadow:0 16px 42px #0005}
    .response-demo *{box-sizing:border-box}
    .response-heading{display:flex;justify-content:space-between;align-items:center;gap:12px;padding:18px 20px 8px}
    .response-heading strong{font-size:20px}
    .response-tag{padding:7px 10px;border:1px solid #9a454d;border-radius:999px;background:#3a1c24;
      color:#ffb5b8;font-size:10px;font-weight:850;letter-spacing:.08em}
    .response-note{padding:0 20px 14px;color:#a8bdc9;font-size:12px;line-height:1.55}
    .response-scene{height:310px;position:relative;overflow:hidden;margin:0 14px;border:1px solid #3b5869;
      border-radius:16px;background:linear-gradient(180deg,#19374b 0%,#214254 62%,#111d28 63%,#0a1119 100%)}
    .response-scene:before{content:"";position:absolute;left:4%;right:4%;bottom:22px;height:2px;
      background:repeating-linear-gradient(90deg,#d9e3dd 0 42px,transparent 42px 78px);opacity:.35}
    .response-scene:after{content:"";position:absolute;inset:0;background:linear-gradient(90deg,#07111a55,transparent 50%,#07111a66)}
    .scene-label{position:absolute;z-index:5;left:14px;top:14px;padding:7px 10px;border-radius:999px;
      background:#08131ddd;border:1px solid #617988;color:#e1edf0;font-size:10px;font-weight:850;letter-spacing:.08em}
    .alert-beacon{position:absolute;z-index:4;right:16px;top:14px;color:#ffd9dc;border:1px solid #a74850;
      background:#471f28d9;border-radius:10px;padding:8px 11px;font-size:11px;font-weight:850;
      animation:alert-flash 1s steps(2,end) infinite}
    .scene-road{position:absolute;left:0;right:0;bottom:0;height:35%;background:linear-gradient(180deg,#14202a,#0a1018)}
    .scene-car{position:absolute;z-index:2;left:24%;bottom:45px;width:210px;height:92px}
    .scene-car-body{position:absolute;left:0;bottom:0;width:210px;height:54px;border-radius:35px 48px 15px 15px;
      background:linear-gradient(160deg,#5b7582,#263d4c 66%,#172732);border:1px solid #8da1a7;box-shadow:0 10px 24px #0008}
    .scene-car-body:before{content:"";position:absolute;left:42px;top:-29px;width:117px;height:42px;
      border-radius:45px 48px 3px 3px;background:linear-gradient(130deg,#95b2bd,#365365);border:1px solid #98b3ba}
    .scene-wheel{position:absolute;z-index:2;bottom:-5px;width:37px;height:37px;border:7px solid #71838a;
      border-radius:50%;background:#101820}
    .scene-wheel.one{left:28px}.scene-wheel.two{right:25px}
    .scene-driver{position:absolute;z-index:3;left:87px;bottom:49px;width:48px;height:73px;transform-origin:50% 90%;
      animation:driver-slump 20s ease-in-out infinite}
    .scene-driver:before{content:"";position:absolute;left:10px;top:0;width:31px;height:34px;border-radius:48%;
      background:linear-gradient(140deg,#f0c4a4,#b77e69);border:2px solid #f5d0b6}
    .scene-driver:after{content:"";position:absolute;left:2px;bottom:0;width:46px;height:44px;border-radius:20px 20px 8px 8px;
      background:linear-gradient(120deg,#5d8793,#294756)}
    .ecg-line{position:absolute;z-index:4;left:6%;top:38%;width:48%;height:72px;opacity:0;
      animation:ecg-show 20s linear infinite;filter:drop-shadow(0 0 7px #ff384d)}
    .ecg-line:before{content:"";position:absolute;inset:0;background:#ff394d;
      clip-path:polygon(0 49%,18% 49%,23% 34%,28% 67%,33% 42%,38% 49%,52% 49%,56% 14%,61% 86%,66% 38%,71% 49%,100% 49%,100% 57%,73% 57%,66% 49%,61% 100%,56% 30%,53% 57%,38% 57%,33% 51%,28% 78%,23% 47%,19% 57%,0 57%);
      animation:ecg-pulse .9s ease-in-out infinite}
    .ambulance{position:absolute;z-index:3;left:110%;bottom:45px;width:148px;height:77px;
      animation:ambulance-arrive 20s ease-in-out infinite}
    .ambulance-body{position:absolute;inset:11px 0 10px;border-radius:12px 20px 8px 8px;
      background:linear-gradient(180deg,#fff,#d8e2e5);border:2px solid #a9bec5;box-shadow:0 7px 18px #0007}
    .ambulance-body:before{content:"119";position:absolute;left:54px;top:14px;color:#d82435;font-weight:950;font-size:17px}
    .ambulance-body:after{content:"";position:absolute;left:10px;top:14px;width:34px;height:23px;border-radius:4px;
      background:#93c4d2;border:2px solid #f8ffff}
    .siren{position:absolute;z-index:2;top:3px;left:67px;width:27px;height:10px;border-radius:5px;
      background:linear-gradient(90deg,#ff293f 0 50%,#258aff 50%);animation:siren-flash .5s steps(2,end) infinite}
    .ambulance-wheel{position:absolute;z-index:3;bottom:1px;width:23px;height:23px;border:5px solid #536571;
      border-radius:50%;background:#111b23}
    .ambulance-wheel.one{left:23px}.ambulance-wheel.two{right:18px}
    .response-progress{height:5px;margin:12px 18px 0;background:#233b4a;border-radius:9px;overflow:hidden}
    .response-progress span{display:block;width:0;height:100%;background:linear-gradient(90deg,#55e4bf,#ff4658);
      animation:progress-fill 20s linear infinite}
    .response-steps{display:grid;grid-template-columns:repeat(5,minmax(0,1fr));gap:8px;padding:13px 18px 17px}
    .response-step{padding:10px 8px;border:1px solid #2c4758;border-radius:11px;background:#0d1924;text-align:center}
    .response-step b{display:block;color:#9bead6;font-size:11px;margin-bottom:4px}
    .response-step span{color:#a7bac5;font-size:10px;line-height:1.4}
    .response-disclaimer{margin:0 18px 16px;padding:10px 12px;border:1px solid #74543b;border-radius:10px;
      background:#2b2118;color:#f5d49b;font-size:11px;line-height:1.5}
    @keyframes alert-flash{50%{background:#a72331;color:#fff;box-shadow:0 0 18px #ff384d88}}
    @keyframes siren-flash{50%{filter:brightness(1.8);box-shadow:0 0 18px #ff315dcc}}
    @keyframes driver-slump{0%,32%{transform:rotate(0) translateY(0)}45%,68%{transform:rotate(33deg) translateY(13px)}78%,100%{transform:rotate(0) translateY(0)}}
    @keyframes ecg-show{0%,5%{opacity:0}9%,74%{opacity:1}82%,100%{opacity:0}}
    @keyframes ecg-pulse{50%{filter:brightness(.55);opacity:.72}}
    @keyframes ambulance-arrive{0%,35%{left:110%}55%,70%{left:66%}100%{left:110%}}
    @keyframes progress-fill{to{width:100%}}
    @media(max-width:700px){.response-scene{height:250px}.scene-car{left:8%;transform:scale(.82);transform-origin:bottom left}
      .ecg-line{left:4%;width:50%}.ambulance{transform:scale(.82);transform-origin:bottom right}
      .response-steps{gap:4px;padding:10px 8px}.response-step{padding:8px 3px}.response-step span{font-size:8px}}
    @media(prefers-reduced-motion:reduce){
      .response-demo *{animation:none!important}
      .response-progress span{width:100%}.ecg-line{opacity:1}.ambulance{left:66%}
      .scene-driver{transform:rotate(28deg) translateY(10px)}
    }
    .response-note{padding-bottom:10px}
    .response-scene{height:350px;background:linear-gradient(180deg,#14283a 0%,#285064 49%,#526052 50%,#16232b 61%,#0b1219 100%)}
    .response-scene:before{inset:43% 0 42%;background:linear-gradient(180deg,#eacb9830,#eacb9800)}
    .response-scene:after{inset:59% 0 0;background:linear-gradient(180deg,#1b2b35,#090f16)}
    .scene-road{z-index:1;bottom:0;height:39%;background:linear-gradient(180deg,#25343a,#111a20)}
    .scene-road:before{content:"";position:absolute;left:0;right:-100%;top:48%;height:4px;
      background:repeating-linear-gradient(90deg,#d9e4dc 0 54px,transparent 54px 112px);
      animation:road-drive 24s linear infinite}
    .scene-car{left:4%;bottom:49px;animation:response-drive 24s cubic-bezier(.15,.55,.35,1) infinite}
    .scene-car-body{background:linear-gradient(160deg,#65a09b,#315962 66%,#20353d);border-color:#a6c1bd}
    .scene-car-body:after{content:"";position:absolute;right:5px;bottom:8px;width:11px;height:7px;border-radius:3px;
      background:#ff3347;box-shadow:-180px 0 #ff3347;opacity:0;animation:brake-lights 24s linear infinite}
    .scene-driver,.scene-neck,.scene-head,.scene-hair{animation:none!important}
    .scene-wheel{animation:wheel-turn .55s linear infinite}
    .scene-roadside-car{position:absolute;z-index:5;left:52%;bottom:47px;width:158px;height:86px;
      opacity:0;animation:discover-car 24s linear infinite}
    .scene-roadside-car .target-body{position:absolute;left:0;bottom:0;width:158px;height:45px;border:1px solid #9aadb2;
      border-radius:28px 34px 11px 11px;background:linear-gradient(160deg,#9aa9aa,#53636a 68%,#303b42);
      box-shadow:0 7px 17px #0009}
    .scene-roadside-car .target-body:before{content:"";position:absolute;left:32px;top:-29px;width:92px;height:37px;
      border:1px solid #93a9af;border-radius:40px 40px 3px 3px;background:linear-gradient(135deg,#9bbdc1,#405c67)}
    .scene-roadside-car .target-body:after{content:"";position:absolute;right:6px;bottom:7px;width:9px;height:7px;
      border-radius:3px;background:#ff4352;box-shadow:-130px 0 #ff4352}
    .scene-roadside-car .target-wheel{position:absolute;z-index:2;bottom:-6px;width:29px;height:29px;border:6px solid #89999c;
      border-radius:50%;background:#121a20}
    .scene-roadside-car .target-wheel.one{left:22px}.scene-roadside-car .target-wheel.two{right:22px}
    .target-alert-box{position:absolute;inset:-9px;border:2px solid #ff4657;border-radius:12px;opacity:0;
      animation:target-alert-box 24s linear infinite}
    .driver-monitor{position:absolute;z-index:8;left:14px;top:61px;width:176px;height:104px;overflow:hidden;
      border:1px solid #557486;border-radius:11px;background:linear-gradient(145deg,#294351,#101a25);
      box-shadow:0 7px 18px #0008}
    .driver-monitor-title{position:absolute;z-index:2;left:8px;top:6px;color:#c6d9e0;font-size:8px;font-weight:850;letter-spacing:.08em}
    .monitor-face{position:absolute;left:71px;top:24px;width:36px;height:43px;border:2px solid #e8c1a6;border-radius:48%;
      background:linear-gradient(140deg,#edc09f,#ae7865)}
    .monitor-face:before{content:"";position:absolute;left:-3px;top:-5px;width:39px;height:15px;border-radius:50% 50% 15% 15%;background:#26303a}
    .monitor-eye{position:absolute;left:6px;top:20px;width:22px;height:3px;border-radius:4px;background:#1d2730;
      animation:eye-detect 24s linear infinite}
    .monitor-box{position:absolute;left:62px;top:20px;width:54px;height:52px;border:2px solid #63e7c2;border-radius:7px;
      animation:eye-box 24s linear infinite}
    .monitor-scan{position:absolute;left:0;right:0;top:-4px;height:2px;background:#66f3d4;box-shadow:0 0 8px #66f3d4;
      animation:scan-sweep 2.4s linear infinite}
    .monitor-caption{position:absolute;left:7px;bottom:6px;color:#a8c0cb;font-size:7px}
    .monitor-caption b{color:#ff5966}
    .scene-phase{position:absolute;z-index:10;right:13px;top:13px;width:min(48%,320px);height:44px;
      border:1px solid #526c7c;border-radius:10px;background:#08131ddd;padding:8px 10px;
      color:#dcedf3;font-size:10px;font-weight:850}
    .scene-phase span{position:absolute;inset:8px 10px;display:flex;align-items:center;opacity:0;
      animation:phase-caption 24s linear infinite}
    .scene-phase span:nth-child(1){animation-delay:0s;color:#90f3d8}
    .scene-phase span:nth-child(2){animation-delay:4.8s;color:#ffd58b}
    .scene-phase span:nth-child(3){animation-delay:9.6s;color:#ff9ea4}
    .scene-phase span:nth-child(4){animation-delay:14.4s;color:#b9e8ff}
    .scene-phase span:nth-child(5){animation-delay:19.2s;color:#a6f2d8}
    .speed-streaks{position:absolute;z-index:4;inset:0;opacity:0;pointer-events:none;
      background:repeating-linear-gradient(0deg,transparent 0 28px,#dff7ff33 29px 31px,transparent 32px 55px);
      animation:speed-streaks 24s linear infinite}
    .brake-indicator{position:absolute;z-index:8;left:54%;bottom:136px;color:#ffd777;font-size:22px;opacity:0;
      animation:brake-indicator 24s linear infinite}
    .ambulance{left:110%;bottom:47px;width:158px;height:82px;animation:ambulance-arrive 24s cubic-bezier(.18,.75,.22,1) infinite}
    .stretcher{position:absolute;z-index:9;left:69%;bottom:48px;width:98px;height:19px;opacity:0;
      animation:patient-transfer 24s ease-in-out infinite}
    .stretcher:before{content:"";position:absolute;left:0;right:0;top:0;height:5px;border-radius:5px;background:#dce7e8}
    .stretcher:after{content:"";position:absolute;left:10px;bottom:-7px;width:75px;height:8px;
      border-left:3px solid #c5d3d4;border-right:3px solid #c5d3d4}
    .stretcher-person{position:absolute;left:25px;top:-11px;width:48px;height:12px;border-radius:8px;
      background:linear-gradient(90deg,#ddb295 0 22%,#497784 22% 100%)}
    .paramedic{position:absolute;z-index:8;left:67%;bottom:51px;width:15px;height:38px;border-radius:7px 7px 3px 3px;
      background:#eef2e5;opacity:0;animation:paramedic-assist 24s linear infinite}
    .paramedic:before{content:"";position:absolute;left:2px;top:-12px;width:12px;height:12px;border-radius:50%;background:#c89376}
    .response-step{animation:stage-active 24s linear infinite}
    .response-step:nth-child(1){animation-delay:0s}.response-step:nth-child(2){animation-delay:4.8s}
    .response-step:nth-child(3){animation-delay:9.6s}.response-step:nth-child(4){animation-delay:14.4s}
    .response-step:nth-child(5){animation-delay:19.2s}
    .response-progress span{animation:progress-fill 24s linear infinite}
    @keyframes scan-sweep{to{transform:translateY(104px)}}
    @keyframes eye-detect{0%,4%,22%,100%{transform:scaleY(1)}8%,19%{transform:scaleY(.18)}}
    @keyframes eye-box{0%,4%{border-color:#63e7c2}8%,24%{border-color:#ff4557;box-shadow:0 0 13px #ff394d99}}
    @keyframes phase-caption{0%,17%{opacity:1}19%,100%{opacity:0}}
    @keyframes road-drive{0%,25%{transform:translateX(0)}43%{transform:translateX(-28%)}55%,100%{transform:translateX(-31%)}}
    @keyframes speed-streaks{0%,22%{opacity:.5}40%{opacity:.13}52%,100%{opacity:0}}
    @keyframes response-drive{0%,15%{left:4%}23%{left:8%}34%{left:21%}44%{left:30%}52%,100%{left:30%}}
    @keyframes discover-car{0%,16%{opacity:0;transform:translateX(42px)}23%,100%{opacity:1;transform:translateX(0)}}
    @keyframes target-alert-box{0%,18%{opacity:0}23%,40%{opacity:1}48%,100%{opacity:0}}
    @keyframes brake-lights{0%,35%{opacity:0}39%,55%{opacity:1}59%,100%{opacity:0}}
    @keyframes brake-indicator{0%,37%{opacity:0;transform:scale(.7)}42%,56%{opacity:1;transform:scale(1)}62%,100%{opacity:0}}
    @keyframes ambulance-arrive{0%,58%{left:110%}69%{left:84%}73%,96%{left:78%}100%{left:110%}}
    @keyframes paramedic-assist{0%,72%{opacity:0}76%,98%{opacity:1}}
    @keyframes patient-transfer{0%,76%{opacity:0;transform:translateX(0)}80%{opacity:1}
      94%{opacity:1;transform:translateX(74px)}98%,100%{opacity:0;transform:translateX(74px)}}
    @keyframes stage-active{0%,17%{border-color:#56cbae;background:#13332f;box-shadow:0 0 16px #36d8ae22}19%,100%{border-color:#2c4758;background:#0d1924;box-shadow:none}}
    @media(max-width:700px){.driver-monitor{left:8px;top:54px;transform:scale(.75);transform-origin:top left}
      .scene-phase{right:8px;top:54px;width:48%;font-size:8px}.response-scene{height:290px}
      .scene-car{transform-origin:bottom left;scale:.76}.scene-roadside-car{left:50%;scale:.75;transform-origin:bottom left}
      .ambulance{scale:.72;transform-origin:bottom left}.stretcher{scale:.75;transform-origin:left bottom}
      .response-steps{gap:4px;padding:10px 8px}.response-step{padding:8px 3px}.response-step b{font-size:9px}.response-step span{font-size:8px}}
    @media(prefers-reduced-motion:reduce){.response-demo *{animation:none!important}
      .response-progress span{width:100%}.scene-car{left:30%}.scene-roadside-car{opacity:1}
      .ambulance{left:78%}.paramedic{opacity:1}.stretcher{opacity:1;transform:translateX(74px)}
      .scene-phase span{display:none}.scene-phase span:last-child{display:flex;opacity:1}}
    </style>
    <div class="response-demo" aria-label="자동 재생되는 가상 안전 대응 시나리오">
      <div class="response-heading"><strong>가상 안전 대응 시나리오</strong><span class="response-tag">SIMULATION ONLY · AUTO</span></div>
      <div class="response-note">차량 안에서 징후를 감지하고, 전방 차량 발견 후 감속·정차해 구급대가 이송하는 순서로 반복 재생합니다.</div>
      <div class="response-scene">
        <div class="scene-label">AURA DRIVE · DEMO SCENE</div>
        <div class="scene-phase"><span>01 · 차량 안 운전자 눈 감김 징후 감지</span>
          <span>02 · 전방 정차 차량 발견 · 접근 중</span><span>03 · 위험 회피를 위한 감속</span>
          <span>04 · 안전거리 확보 후 정차</span><span>05 · 구급차 도착 · 환자 이송</span></div>
        <div class="driver-monitor"><div class="driver-monitor-title">DRIVER MONITOR · LIVE</div>
          <div class="monitor-face"><div class="monitor-eye"></div></div><div class="monitor-box"><div class="monitor-scan"></div></div>
          <div class="monitor-caption"><b>●</b> EYE-CLOSURE ALERT · DEMO</div></div>
        <div class="scene-road"></div>
        <div class="scene-car"><div class="scene-car-body"></div><div class="scene-wheel one"></div><div class="scene-wheel two"></div><div class="scene-driver"></div></div>
        <div class="scene-roadside-car"><div class="target-alert-box"></div><div class="target-body"></div>
          <div class="target-wheel one"></div><div class="target-wheel two"></div></div>
        <div class="speed-streaks"></div><div class="brake-indicator">⚠</div>
        <div class="ambulance"><div class="siren"></div><div class="ambulance-body"></div>
          <div class="ambulance-wheel one"></div><div class="ambulance-wheel two"></div></div>
        <div class="paramedic"></div><div class="stretcher"><div class="stretcher-person"></div></div>
      </div>
      <div class="response-progress"><span></span></div>
      <div class="response-steps">
        <div class="response-step"><b>01 · 감지</b><span>차량 안 눈 감김 경고</span></div>
        <div class="response-step"><b>02 · 발견</b><span>전방 차량 확인</span></div>
        <div class="response-step"><b>03 · 감속</b><span>위험 회피·제동</span></div>
        <div class="response-step"><b>04 · 정차</b><span>안전거리 확보</span></div>
        <div class="response-step"><b>05 · 이송</b><span>구급차 도착·탑승</span></div>
      </div>
      <div class="response-disclaimer">전체 흐름은 발표용 가상 연출입니다. 이 모델이 전방 차량·의료 상황을 감지하거나 차량을 제어하지 않으며, 실제 119 신고·출동 요청도 전송하지 않습니다.</div>
    </div>""")


for key, val in {"eye_t": .25, "yawn_t": .15, "eye_s": 2., "yawn_s": 1.5, "stride": 2, "eye_clahe": True, "yawn_color": "RGB (기본)"}.items():
    st.session_state.setdefault(key, val)
cfg = dict(eye=st.session_state.eye_t, yawn=st.session_state.yawn_t,
           eye_sec=st.session_state.eye_s, yawn_sec=st.session_state.yawn_s,
           stride=st.session_state.stride, eye_clahe=st.session_state.eye_clahe, yawn_color=st.session_state.yawn_color)
paths = model_paths()
missing = missing_files(paths)

st.html('<div class="brand"><div><b>AURA <span>DRIVE</span></b><br><small>INTELLIGENT DRIVER MONITORING</small></div><small>◉ AI VISION SYSTEM</small></div>')
page = st.radio("메뉴", ["실시간 모니터", "이미지 분석", "모의 대응", "설정"], horizontal=True, label_visibility="collapsed")

if page == "실시간 모니터":
    if missing:
        st.error("모델 파일이 부족해. 설정에서 경로를 확인해줘.")
        for f in missing:
            st.code(f, language=None)
    else:
        try:
            from streamlit_webrtc import WebRtcMode, webrtc_streamer
            import av
            models = get_models(*(str(p) for p in paths))
            if "aura_m" not in st.session_state:
                st.session_state.aura_m = Monitor()
            monitor = st.session_state.aura_m
            if "aura_engine" not in st.session_state or st.session_state.get("aura_paths") != tuple(map(str, paths)):
                st.session_state.aura_engine = Engine(models, cfg)
                st.session_state.aura_paths = tuple(map(str, paths))
            engine = st.session_state.aura_engine
            engine.cfg = cfg
            if "aura_worker" not in st.session_state or st.session_state.get("aura_worker_engine") is not engine:
                old_worker = st.session_state.get("aura_worker")
                if old_worker is not None:
                    old_worker.close()
                st.session_state.aura_worker = LatestFrameWorker(engine, monitor, interval_ms=250, max_width=640)
                st.session_state.aura_worker_engine = engine
            background_worker = st.session_state.aura_worker
            a, b = st.columns([1.15, 1], gap="medium")
            with a:
                st.html('<div class="monitor-head"><strong>◉ 운전자 실시간 원본 카메라</strong><small>LIVE · ORIGINAL CAMERA</small></div>')
                cabin_frame_style()

                # Keep the inference engine and camera processing separate.
                # Background worker never accesses st.session_state directly.
                def on_frame(frame):
                    raw = frame.to_ndarray(format="bgr24")
                    try:
                        # Never block the live video on CPU model inference.
                        background_worker.submit(raw)
                    except Exception as exc:
                        print(f"[AURA FRAME ERROR] {type(exc).__name__}: {exc}", flush=True)
                    return av.VideoFrame.from_ndarray(raw, format="bgr24")

                with st.container(key="aura_camera_frame"):
                    webrtc_streamer(
                        key="aura-camera", mode=WebRtcMode.SENDRECV,
                        video_frame_callback=on_frame,
                        media_stream_constraints={"video": {"facingMode": "user", "width": {"ideal": 640}, "height": {"ideal": 480}, "frameRate": {"ideal": 30}}, "audio": False},
                        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
                        async_processing=True,
                    )
                st.caption("START → 카메라 권한 허용. 휴대폰 원격 이용 시 HTTPS가 필요할 수 있어.")

            @st.fragment(run_every="500ms")
            def live_results():
                data, events, audio = monitor.snapshot()
                mode = data["mode"]
                headline = "🚨 눈 감김 경고!" if mode == "eye" else "⚠️ 하품 감지!" if mode == "yawn" else "운전자 상태 모니터링" if mode == "ok" else "카메라 연결 대기"
                desc = "눈을 오래 감고 있어요. 안전한 장소에서 쉬세요." if mode == "eye" else "하품이 감지됐어요. 피로 상태를 점검하세요." if mode == "yawn" else "AI가 눈과 입의 상태를 분석하고 있습니다."
                st.html(f'<div class="alert {mode}"><div class="dot"></div><div><b>{headline}</b><br><span>{desc}</span></div></div>')
                sound_events(audio)
                values = [("눈 감김 점수", "판단 불가" if data["eye"] is None else f'{data["eye"]:.1%}', "YOLO26n · CLAHE" if cfg["eye_clahe"] else "YOLO26n · RGB"),
                    ("하품 점수", "판단 불가" if data["yawn"] is None else f'{data["yawn"]:.1%}', "ResNet18 · " + cfg["yawn_color"]),
                    ("연속 눈 감김", f'{data["duration"]:.1f}초', "경고 기준 시간"),
                    ("카메라 FPS", f'{data["fps"]:.1f} FPS' if data["fps"] else "—", "AI 분석과 분리 · 카메라 수신 프레임")]
                cards = "".join(f'<div class="card"><div class="label">{label}</div><div class="num">{value}</div><div class="foot">{note}</div></div>' for label, value, note in values)
                st.html('<div class="metric-grid">' + cards + '</div>')
                st.caption(f'모델 점수(보정 확률 아님) · 눈 ≥ {cfg["eye"]:.2f} / 하품 ≥ {cfg["yawn"]:.2f}')
                st.html('<div class="roi-mini-title">◈ 실시간 모델 입력 ROI <span>LIVE · EYE / MOUTH</span></div>')
                left_roi, right_roi = st.columns(2, gap="small")
                with left_roi:
                    if data.get("eye_roi"):
                        st.image(data["eye_roi"], caption="👁 눈 ROI · YOLO 입력", use_container_width=True)
                    else:
                        st.caption("👁 눈 ROI 대기 중")
                with right_roi:
                    if data.get("mouth_roi"):
                        st.image(data["mouth_roi"], caption="◉ 입 ROI · ResNet 입력", use_container_width=True)
                    else:
                        st.caption("◉ 입 ROI 대기 중")
                with st.expander("랜드마크 및 감지 이벤트", expanded=False):
                    eye_col, mouth_col = st.columns(2, gap="small")
                    with eye_col:
                        if data.get("eye_points"):
                            st.image(data["eye_points"], caption="눈꺼풀 랜드마크", use_container_width=True)
                    with mouth_col:
                        if data.get("mouth_points"):
                            st.image(data["mouth_points"], caption="입술 랜드마크", use_container_width=True)
                    if events:
                        st.dataframe(events, use_container_width=True, hide_index=True)
                    else:
                        st.caption("아직 발생한 경고가 없어.")

            with b:
                st.html('<div class="monitor-head"><strong>실시간 분석</strong><small>ROI · MODEL OUTPUT</small></div>')
                live_results()
                with st.expander("사운드 알림 설정", expanded=False):
                    audio_controls()
        except Exception as exc:
            st.error(f"실시간 모니터 초기화 실패: {type(exc).__name__}: {exc}")
            st.code("python -m pip install --upgrade --force-reinstall 'mediapipe==0.10.21' 'protobuf>=4.25.3,<5'")

elif page == "이미지 분석":
    st.html('<div class="section">📸 IMAGE ANALYSIS</div><div class="dim">사진을 촬영하거나 업로드해서 눈·하품 모델을 테스트해.</div>')
    shot = st.camera_input("카메라 촬영")
    upl = st.file_uploader("또는 사진 업로드", type=["jpg", "jpeg", "png", "webp"])
    photo = shot if shot is not None else upl
    if missing:
        st.error("모델 경로를 먼저 연결해야 해.")
    elif photo:
        try:
            raw = photo.getvalue()
            if len(raw) > 12 * 1024 * 1024:
                st.error("12MB 이하 이미지를 사용해줘.")
            else:
                im = Image.open(io.BytesIO(raw)).convert("RGB")
                if max(im.size) > 2000:
                    im.thumbnail((2000, 2000))
                st.image(im, width="stretch")
                if st.button("AI 이미지 분석하기", type="primary"):
                    m = get_models(*(str(p) for p in paths))
                    eng = Engine(m, cfg, static=True)
                    try:
                        result = eng.predict(im)
                    finally:
                        eng.close()
                    st.session_state.img_result = result
                if "img_result" in st.session_state:
                    r = st.session_state.img_result
                    c1, c2 = st.columns(2)
                    with c1:
                        kpi("눈 감김", "미검출" if r["eye"] is None else f'{r["eye"]:.1%}', "YOLO26n · CLAHE" if cfg["eye_clahe"] else "YOLO26n · RGB")
                    with c2:
                        kpi("하품", "미검출" if r["yawn"] is None else f'{r["yawn"]:.1%}', "ResNet18 · " + cfg["yawn_color"])
                    if r["mouth"] is not None:
                        st.image(r["mouth"], caption="하품 모델 입력 · 입 ROI", width=224)
                    if r["eyes"]:
                        st.image(r["eye_processed"], caption="눈 모델 실제 입력 · 전처리 적용 ROI", width=224)
        except Exception as exc:
            st.error(f"이미지 분석 오류: {type(exc).__name__}: {exc}")
elif page == "모의 대응":
    render_response_simulation()
else:
    st.html('<div class="section">⚙️ CONFIGURATION</div>')
    st.text_input("YOLO26n 눈 모델 경로", value=str(ROOT / "runs/eyes_yolo26n-cls/weights/best.pt"), key="eyes_path")
    st.text_input("팀원 프로젝트 폴더", value=str(TEAM), key="team_path")
    for name, p in [("눈 가중치", paths[0]), ("팀원 모듈", paths[1] / "03_yawn_common.py"), ("하품 가중치", paths[2]), ("눈 감김 MP3", ROOT / "눈감김.mp3"), ("하품 MP3", ROOT / "하품.mp3"), ("차량 장식 배경(선택)", ROOT / "aura_car_interior.jpg")]:
        st.write(("✅" if p.is_file() else "❌") + " " + name + " — " + str(p))
    c1, c2 = st.columns(2)
    with c1:
        st.checkbox("눈 ROI CLAHE 활성화 (개선 후보)", key="eye_clahe")
        st.slider("눈 감김 판정 임계값", .05, .95, step=.05, key="eye_t")
        st.slider("눈 감김 경고 지속시간(초)", .5, 5., step=.5, key="eye_s")
    with c2:
        st.selectbox("하품 입 ROI 전처리", ["RGB (기본)", "Grayscale (실험)"], key="yawn_color")
        st.slider("하품 판정 임계값", .05, .95, step=.05, key="yawn_t")
        st.slider("하품 지속시간(초)", .5, 5., step=.5, key="yawn_s")
    st.slider("추론 간격(프레임)", 1, 6, key="stride")
    st.caption("상단의 모의 대응 메뉴에서 자동 재생되는 가상 경고·정차·구조 시나리오를 볼 수 있습니다. 무반응/기절 판단, 차량 제어, 실제 신고를 수행하지 않습니다.")
    st.caption("🚘 차량 실내 사진은 라이브 카메라 바깥 프레임의 배경으로만 사용돼. 웹캠 원본은 변경하지 않아. 설정 변경은 카메라 재시작 후 적용하는 게 안전해.")
    st.info("눈: CLAHE + 0.25, 하품: RGB + 0.15는 외부 개발셋 기반 후보 설정입니다. 하품 0.10 및 0.50과 실시간 비교 후 최종 확정하세요.")

st.caption("AURA DRIVE v9 · 감속·창문·119는 모두 가상 시뮬레이션 · 실제 운전 안전장치로 사용하지 마세요.")

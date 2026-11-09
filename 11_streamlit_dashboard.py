"""AURA DRIVE v11 — cinematic local-only AI driver monitoring demo.

- Local OpenCV camera + localhost MJPEG stream (preview never waits for AI).
- Existing weights are NEVER retrained or modified.
- Eye: MediaPipe eye crops -> LAB CLAHE -> YOLO26n-cls.
- Yawn: original full RGB frame -> YOLO26s-cls, SAME input as experiment 31.
- Vehicle control / 119: simulation only, NOT connected to real hardware.

Run on the same Windows PC as the webcam:
    streamlit run 11_streamlit_dashboard.py --server.address 127.0.0.1
"""
from __future__ import annotations

import base64
import importlib
import io
import threading
import time
from collections import deque
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
import torch
from PIL import Image, ImageDraw, ImageOps

from common import to_square
from roi_utils import LEFT_EYE, RIGHT_EYE, crop_eyes
from aura_local_camera import LatestFrameWorker, LocalCameraStream
from aura_safety_sim import SafetySimulation
from aura_event_logic import ContinuousAlerts
from aura_audio_panel import render_audio_panel

ROOT = Path(__file__).resolve().parent
EYE_FILE = ROOT / "runs/eyes_yolo26n-cls/weights/best.pt"
YAWN_FILE = ROOT / "runs/yawn_yolo26s-cls/weights/best.pt"
CAR_FILE = ROOT / "aura_vehicle_hero.jpg"

st.set_page_config(page_title="AURA DRIVE | LOCAL VISION SYSTEM", page_icon="🚘", layout="wide", initial_sidebar_state="collapsed")


@st.cache_data(show_spinner=False)
def hero_data(path: str):
    p = Path(path)
    return base64.b64encode(p.read_bytes()).decode("ascii") if p.is_file() else ""


def base_styles():
    st.html("""<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&family=Noto+Sans+KR:wght@400;500;600;700;800;900&display=swap');
    :root {--mint:#62eacc;--ink:#080e16;--panel:#111d29;--edge:#26394a;--muted:#8fa6b7}
    html, body,[data-testid="stApp"]{font-family:Inter,'Noto Sans KR',sans-serif;color:#eff8fb;background:#07101a}
    [data-testid="stAppViewContainer"]{background:linear-gradient(155deg,#0a1721 0%,#080d16 50%,#060a11 100%)}
    [data-testid="stMainBlockContainer"]{max-width:1510px;padding:1.05rem 2.0rem 4rem}
    [data-testid="stHeader"]{background:transparent}
    [data-testid="stToolbar"],#MainMenu,footer{visibility:hidden}
    h1,h2,h3,h4{letter-spacing:-.05em;color:#f5fafb}
    [data-testid="stCaptionContainer"] p{color:#89a3b4;font-size:12px}
    [data-testid="stWidgetLabel"] p{color:#d4e6ef!important;font-weight:650!important}
    [data-testid="stRadio"] [role="radiogroup"]{display:flex;gap:7px;flex-wrap:wrap}
    [data-testid="stRadio"] [role="radiogroup"] label{border:1px solid #304457;background:#14212d;padding:8px 17px;border-radius:10px}
    [data-testid="stRadio"] [role="radiogroup"] label:has(input:checked){background:#1e4547;border-color:#62dfc5}
    .stButton button{min-height:50px;border-radius:13px;font-size:15px;font-weight:800;border:1px solid #334b5e;background:#152433;color:#f1f8ff;transition:all .2s}
    .stButton button:hover{transform:translateY(-1px);border-color:#69d9c4;color:#fff}
    .stButton button[kind="primary"]{background:#69efd0;color:#06262a;border-color:#69efd0;box-shadow:0 8px 28px #39edc026}
    .stButton button:disabled{opacity:.4;cursor:not-allowed}
    .aura-nav{display:flex;justify-content:space-between;align-items:center;border-bottom:1px solid #263748;padding:10px 3px 17px;margin-bottom:15px}
    .aura-logo{font-weight:900;letter-spacing:.24em;font-size:18px;color:#f5ffff}.aura-logo span{color:#69efd0}
    .aura-topright{font-size:11px;color:#9fb6c5;letter-spacing:.15em}
    .aura-hero{position:relative;height:355px;overflow:hidden;border:1px solid #273b49;border-radius:24px;background:#07101a center 42%/cover no-repeat;box-shadow:0 26px 65px #0008;margin:4px 0 18px}
    .aura-hero::before{content:'';position:absolute;inset:0;background:linear-gradient(90deg,#08111af2 0%,#07111b7e 35%,transparent 70%),linear-gradient(0deg,#07111b80,transparent 30%)}
    .aura-hero.is-welcome{height:min(590px,65vh);min-height:405px;background-position:center center}
    .aura-hero-copy{position:absolute;left:clamp(22px,4vw,67px);top:50%;transform:translateY(-50%);max-width:620px}
    .aura-eyebrow{font-size:11px;color:#69e8c9;font-weight:800;letter-spacing:.24em;margin-bottom:12px}
    .aura-h1{font-size:clamp(31px,4.3vw,59px);line-height:1.18;font-weight:900;letter-spacing:-.07em;color:#fff}
    .aura-h1 em{color:#70ecd0;font-style:normal}
    .aura-hero-sub{color:#d0dfe6;font-size:13px;line-height:1.95;margin-top:15px;max-width:480px}
    .aura-hero-info{margin-top:22px;display:flex;gap:9px;flex-wrap:wrap}
    .aura-hero-info b{font-size:10px;font-weight:700;letter-spacing:.05em;padding:7px 10px;background:#092835b3;border:1px solid #497c7c;border-radius:8px;color:#affce9}
    .aura-stage{border:1px solid #24394c;background:linear-gradient(160deg,#121f2d,#0a141f);border-radius:20px;padding:16px;margin:8px 0;box-shadow:0 12px 35px #0003}
    .aura-stage-title{display:flex;align-items:center;justify-content:space-between;margin:4px 0 14px}
    .aura-stage-title b{color:#effbfb;font-size:15px;letter-spacing:-.02em}
    .aura-stage-title small{font-size:10px;color:#72e3c8;letter-spacing:.17em;font-weight:750}
    .aura-frame{border-radius:16px;background:#02070c;padding:8px;border:1px solid #2f4658;min-height:200px}
    .aura-frame img{display:block;width:100%;height:auto;aspect-ratio:4/3;object-fit:contain;border-radius:10px;background:#03070d}
    .aura-idle{aspect-ratio:4/3;display:grid;place-items:center;text-align:center;background:radial-gradient(circle at 50% 30%,#17303d,#070f1a 70%);border-radius:10px;color:#d0e7ec}
    .aura-idle strong{font-size:25px;letter-spacing:-.04em}.aura-idle span{font-size:12px;line-height:2;color:#90adbb}
    .aura-status{border:1px solid #254b48;background:linear-gradient(105deg,#153b35,#112832);border-radius:14px;padding:17px 18px;margin:7px 0 14px}
    .aura-status.warn{border-color:#92554a;background:linear-gradient(120deg,#532f30,#34262d)}
    .aura-status.yawn{border-color:#987647;background:linear-gradient(120deg,#4f4229,#2c3029)}
    .aura-status b{font-size:21px;color:#efffff}.aura-status p{margin:6px 0 0;color:#bad2d4;font-size:12px}
    .aura-kpis{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:11px}
    .aura-kpi{border:1px solid #26394b;background:#0c1824;border-radius:12px;padding:16px}
    .aura-kpi span{font-size:11px;color:#9bb0c2}.aura-kpi strong{display:block;font-size:27px;color:#fff;letter-spacing:-.05em;margin-top:7px}
    .aura-kpi small{display:block;color:#69e6cc;font-size:10px;margin-top:7px}
    .aura-progress{height:7px;border-radius:7px;background:#233444;overflow:hidden;margin-top:10px}
    .aura-progress div{height:100%;background:linear-gradient(90deg,#40978b,#78f9d2);border-radius:7px}
    .aura-mini{display:grid;grid-template-columns:1fr 1fr;gap:10px;margin-top:8px}
    .aura-section-note{font-size:12px;color:#93aab6;margin:12px 0}
    .aura-footer{border-top:1px solid #293947;margin-top:36px;padding-top:13px;font-size:10px;color:#8aa6b7;letter-spacing:.06em}
    [data-testid="stImage"] img{border-radius:11px}
    [data-testid="stDataFrame"]{border-radius:11px;overflow:hidden}
    .aura-hero:not(.is-welcome){height:178px;background-position:center 57%;margin-bottom:11px}
    .aura-hero:not(.is-welcome) .aura-h1{font-size:clamp(24px,2.5vw,34px)}
    .aura-hero:not(.is-welcome) .aura-hero-sub{font-size:11px;margin-top:6px}
    .aura-hero:not(.is-welcome) .aura-hero-info{margin-top:8px}
    .aura-hero:not(.is-welcome) .aura-hero-info b{padding:4px 8px}
    .aura-stage-title{margin:8px 0 7px}
    .aura-status{margin:0 0 10px;padding:14px 16px}
    .aura-kpi{padding:12px}
    .aura-kpi strong{font-size:25px}
    .aura-progress{margin-top:7px}
    .aura-eyestate{display:flex;justify-content:space-between;gap:8px;font-size:11px;color:#abc1cc;padding:6px 2px 0}
    .aura-quiet{color:#9ab4bf;font-size:12px}
    .aura-control-caption{font-size:11px;color:#a8c6d0;border:1px solid #2a4853;border-radius:12px;padding:10px 13px;background:#10232d}
    .aura-start-banner{display:flex;justify-content:space-between;flex-wrap:wrap;align-items:center;gap:7px;background:#101f2c;border:1px solid #294c51;border-radius:13px;padding:10px 15px;margin:8px 0 12px;font-size:12px;color:#c8e1e5}
    .aura-start-banner strong{color:#79efcf}
    @media(max-width:900px){.aura-hero:not(.is-welcome){height:145px}.aura-hero:not(.is-welcome) .aura-h1{font-size:23px}.aura-hero:not(.is-welcome) .aura-hero-sub{display:none}
[data-testid="stMainBlockContainer"]{padding:1rem .9rem 3rem}.aura-hero{height:285px;border-radius:14px}.aura-hero.is-welcome{min-height:430px}.aura-hero-copy{left:18px}.aura-h1{font-size:31px}.aura-hero-sub{font-size:11px}}
    </style>""")


def hero(welcome=False):
    img = hero_data(str(CAR_FILE))
    bg = f'url("data:image/jpeg;base64,{img}")' if img else 'linear-gradient(110deg,#0b2535,#132c3c)'
    title = "AI가 운전자의<br><em>눈 감김과 하품을</em> 관찰합니다." if welcome else "운전자의 상태를<br><em>실시간으로 모니터링</em>"
    sub = ("로컬 카메라와 사전학습 분류 모델로 졸음 징후를 시각화합니다. 차량 제어와 119 신고는 연결하지 않은 시뮬레이션입니다."
           if welcome else "로컬 원본 영상은 부드럽게, AI는 별도 분석. 눈은 5초, 하품은 3초 연속 감지 후 경고합니다.")
    cls = ' is-welcome' if welcome else ''
    st.html(f'''<div class="aura-hero{cls}" style="background-image:{bg}">
      <div class="aura-hero-copy"><div class="aura-eyebrow">AURA DRIVE  /  LOCAL VISION DEMO</div>
      <div class="aura-h1">{title}</div><div class="aura-hero-sub">{sub}</div>
      <div class="aura-hero-info"><b>눈 · YOLO26n + CLAHE</b><b>하품 · YOLO26s</b><b>CPU · LOCAL STREAM</b></div>
      </div></div>''')


def eye_clahe(roi: Image.Image) -> Image.Image:
    a = np.asarray(roi.convert('RGB'))
    lab = cv2.cvtColor(a, cv2.COLOR_RGB2LAB)
    lab[:, :, 0] = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4)).apply(lab[:, :, 0])
    return Image.fromarray(cv2.cvtColor(lab, cv2.COLOR_LAB2RGB))


class CachedMesh:
    def __init__(self, result):
        self.result = result
    def process(self, _unused):
        return self.result


LIP_POINTS = (61, 40, 37, 0, 267, 270, 291, 321, 314, 17, 84, 91,
              78, 81, 13, 311, 308, 402, 14, 178, 87, 82)


def landmark_preview(image, face_result, groups):
    if not face_result or not face_result.multi_face_landmarks:
        return None
    rgb = np.asarray(image.convert('RGB'))
    h, w = rgb.shape[:2]
    lm = face_result.multi_face_landmarks[0].landmark
    pts = [[(lm[i].x * w, lm[i].y * h) for i in group] for group in groups]
    flat = [pt for group in pts for pt in group]
    if not flat:
        return None
    xs, ys = zip(*flat)
    x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
    span = max(8., x1-x0)
    pad = max(8., span * .36)
    left, right = max(0, int(x0-pad)), min(w, int(x1+pad))
    top, bottom = max(0, int(y0-span*.36)), min(h, int(y1+span*.36))
    if right-left < 10 or bottom-top < 10:
        return None
    out = Image.fromarray(rgb[top:bottom, left:right].copy())
    d = ImageDraw.Draw(out)
    for j, group in enumerate(pts):
        color = '#70f5d7' if j == 0 else '#e5b76a'
        for x,y in group:
            cx,cy=round(x-left),round(y-top)
            d.ellipse((cx-2,cy-2,cx+2,cy+2),fill=color,outline='#08222b')
    return out


def jpeg_thumb(image, size=(260,170)):
    if image is None:
        return None
    img=image.copy().convert('RGB')
    img.thumbnail(size)
    buff=io.BytesIO()
    img.save(buff,format='JPEG',quality=73)
    return buff.getvalue()


@st.cache_resource(show_spinner='기존 AI 가중치 로딩 중...')
def models_cached(eye_path: str, yawn_path: str):
    from ultralytics import YOLO
    torch.set_num_threads(min(4, max(1, torch.get_num_threads())))
    eyes=YOLO(eye_path)
    names={int(k):str(v).lower() for k,v in eyes.names.items()}
    closed_idx=next((k for k,v in names.items() if v=='closed'),None)
    if closed_idx is None:
        raise ValueError(f'눈 모델 클래스가 잘못됐어: {names}')
    # The predictor comes from the same code used for the 393-image Roboflow evaluation.
    comparator=importlib.import_module('31_yawn_external_compare')
    yawn=comparator.Predictor('yolo26s',device='cpu')
    if str(Path(yawn.file).resolve()).casefold()!=str(Path(yawn_path).resolve()).casefold():
        raise ValueError('하품 모델 파일 경로가 31번 비교 코드의 경로와 달라.')
    return dict(eyes=eyes,closed_idx=closed_idx,yawn=yawn,eye_lock=threading.RLock(),yawn_lock=threading.RLock())


def face_focus_crop(image: Image.Image, result):
    """Exploratory live-webcam crop. NOT the 31-image face baseline recipe."""
    if not result or not result.multi_face_landmarks:
        return image
    landmarks=result.multi_face_landmarks[0].landmark
    w,h=image.size
    xs=[lm.x*w for lm in landmarks]
    ys=[lm.y*h for lm in landmarks]
    x0,x1=max(0,min(xs)),min(w,max(xs))
    y0,y1=max(0,min(ys)),min(h,max(ys))
    span=max(x1-x0,y1-y0)
    pad=span*.25
    box=(max(0,int(x0-pad)),max(0,int(y0-pad)),min(w,int(x1+pad)),min(h,int(y1+pad)))
    if box[2]-box[0]<64 or box[3]-box[1]<64:
        return image
    return image.crop(box)


class FaceEngine:
    def __init__(self, models, cfg, static=False):
        import mediapipe as mp
        self.models=models
        self.cfg=cfg
        self.static=static
        self.mesh=mp.solutions.face_mesh.FaceMesh(static_image_mode=static,max_num_faces=1,
            refine_landmarks=True,min_detection_confidence=.5,min_tracking_confidence=.5)
        self.lock=threading.RLock()

    def predict(self, image):
        with self.lock:
            result=self.mesh.process(np.asarray(image.convert('RGB')))
            eyes=crop_eyes(image,CachedMesh(result))
            eye_pt=landmark_preview(image,result,(LEFT_EYE,RIGHT_EYE))
            mouth_pt=landmark_preview(image,result,(LIP_POINTS,))
            e=None
            crop_eye=None
            if eyes:
                transformed=[to_square(eye_clahe(x) if self.cfg['eye_clahe'] else x,224) for x in eyes]
                crop_eye=transformed[0]
                with self.models['eye_lock']:
                    predictions=self.models['eyes'].predict(transformed,imgsz=224,device='cpu',verbose=False)
                    vals=[float(r.probs.data[self.models['closed_idx']]) for r in predictions]
                e=sum(vals)/len(vals)
            y=None
            yawn_input=image.convert('RGB')
            if self.cfg.get('yawn_view') == 'face_focus':
                # Optional webcam experiment: focuses on the driver when the
                # camera image contains lots of interior background.
                yawn_input=face_focus_crop(yawn_input,result)
            if self.static or result.multi_face_landmarks:
                with self.models['yawn_lock']:
                    # Never use the mouth ROI: trained classifier expects face context.
                    # Baseline mode matches the 393-photo experiment's full-frame input.
                    y=self.models['yawn'].probability(yawn_input)
            return dict(eye=e,yawn=y,eye_processed=crop_eye,
                mouth=to_square(yawn_input,224),eye_points=eye_pt,mouth_points=mouth_pt)

    def close(self):
        with self.lock:
            self.mesh.close()


class Monitor:
    def __init__(self):
        self.lock=threading.RLock()
        self.alert_logic=ContinuousAlerts(max_gap=1.6)
        self.last_at=0.
        self.events=deque(maxlen=80)
        self.serial=0
        self.yawn_count=0
        self.eye_count=0
        self.safety=None
        self.data=dict(eye=None,yawn=None,duration=0.,yawn_duration=0.,fps=0.,mode='idle',
                       eye_points=None,mouth_points=None,eye_roi=None,mouth_roi=None)

    def reset(self):
        with self.lock:
            self.alert_logic.reset()
            self.last_at=0.
            self.data.update(eye=None,yawn=None,duration=0.,yawn_duration=0.,mode='idle')

    def update(self,eye,yawn,fps,cfg,eye_roi=None,mouth_roi=None,eye_points=None,mouth_points=None):
        now=time.monotonic()
        with self.lock:
            events=self.alert_logic.update(eye,yawn,cfg,now)
            self.last_at=now
            for kind in events:
                if kind=='eye':
                    self.eye_count+=1
                    self._event('eye',eye)
                else:
                    self.yawn_count+=1
                    self._event('yawn',yawn)
            eye_state=self.alert_logic.eye
            yawn_state=self.alert_logic.yawn
            mode=('eye' if eye_state.latched else 'yawn' if yawn_state.latched else
                  'pending_eye' if eye_state.since is not None else
                  'pending_yawn' if yawn_state.since is not None else
                  'ok' if eye is not None or yawn is not None else 'idle')
            self.data=dict(eye=eye,yawn=yawn,duration=eye_state.duration,
                yawn_duration=yawn_state.duration,fps=fps,mode=mode,
                eye_points=jpeg_thumb(eye_points),mouth_points=jpeg_thumb(mouth_points),
                eye_roi=jpeg_thumb(eye_roi),mouth_roi=jpeg_thumb(mouth_roi))

    def _event(self,kind,prob):
        self.serial+=1
        self.events.appendleft(dict(id=self.serial,time=datetime.now().strftime('%H:%M:%S'),kind=kind,
            label='눈 감김' if kind=='eye' else '하품',score=f'{prob:.1%}' if prob is not None else '-'))
        if kind=='eye' and self.safety is not None:
            self.safety.on_eye_warning()

    def snapshot(self):
        with self.lock:
            d=dict(self.data)
            if time.monotonic()-self.last_at>2:
                d.update(eye=None,yawn=None,duration=0.,yawn_duration=0.,mode='idle')
            return d,list(self.events),self.eye_count,self.yawn_count


def params():
    return dict(eye=float(st.session_state.eye_threshold), yawn=float(st.session_state.yawn_threshold),
                eye_seconds=float(st.session_state.eye_seconds),yawn_seconds=float(st.session_state.yawn_seconds),
                eye_clahe=bool(st.session_state.eye_clahe),
                yawn_view=st.session_state.get('yawn_view', 'full'))


def get_running_camera():
    return st.session_state.get('local_camera')


def shutdown_camera():
    cam=st.session_state.pop('local_camera',None)
    if cam is not None:
        cam.stop()
    worker=st.session_state.pop('local_worker',None)
    if worker is not None:
        worker.close()
    st.session_state.monitor.reset()
    engine=st.session_state.pop('local_engine',None)
    if engine is not None:
        engine.close()


def launch_camera():
    shutdown_camera()
    missing=[str(p) for p in (EYE_FILE,YAWN_FILE) if not p.is_file()]
    if missing:
        return '모델 가중치를 찾을 수 없어:\n'+'\n'.join(missing)
    try:
        m=models_cached(str(EYE_FILE.resolve()),str(YAWN_FILE.resolve()))
        st.session_state.monitor.reset()
        engine=FaceEngine(m,params())
        worker=LatestFrameWorker(engine,st.session_state.monitor,interval_ms=int(st.session_state.infer_interval),max_width=640)
        size=(640,480) if st.session_state.camera_resolution=='640×480 (권장)' else (1280,720)
        cam=LocalCameraStream(st.session_state.camera_index,worker,width=size[0],height=size[1],
                              jpeg_quality=65,preview_fps=20)
        if not cam.start():
            worker.close()
            engine.close()
            return cam.error
        st.session_state.local_engine=engine
        st.session_state.local_worker=worker
        st.session_state.local_camera=cam
        return ''
    except Exception as e:
        return f'{type(e).__name__}: {e}'




def probs_panel():
    @st.fragment(run_every='500ms')
    def fragment():
        monitor=st.session_state.monitor
        data,events,eyes,yawns=monitor.snapshot()
        mode=data['mode']
        eng=st.session_state.get('local_engine')
        applied=eng.cfg if eng is not None else params()
        et,yt=applied['eye_seconds'],applied['yawn_seconds']
        titles={'eye':'🚨 눈 감김 경고','yawn':'⚠️ 하품 경고',
                'pending_eye':'눈 감김 확인 중','pending_yawn':'하품 확인 중',
                'ok':'● 정상 모니터링','idle':'○ 카메라 연결 대기'}
        subtitles={'eye':f'{et:g}초 이상 연속 감지되어 경고가 발생했습니다.',
                   'yawn':f'{yt:g}초 이상 연속 감지되어 경고가 발생했습니다.',
                   'pending_eye':f'눈 감김이 이어지면 {et:g}초 후에 경고합니다.',
                   'pending_yawn':f'하품 신호가 이어지면 {yt:g}초 후에 경고합니다.',
                   'ok':'AI가 눈과 하품 상태를 계속 확인하고 있습니다.',
                   'idle':'카메라를 시작하면 이곳에 분석 결과가 나타납니다.'}
        cls='warn' if mode=='eye' else 'yawn' if mode=='yawn' else ''
        st.html(f'<div class="aura-status {cls}"><b>{titles[mode]}</b><p>{subtitles[mode]}</p></div>')
        e,y=data['eye'],data['yawn']
        def fmt(x):return f'{x*100:.1f}%' if x is not None else '—'
        v1=max(0,min(100,int(e*100))) if e is not None else 0
        v2=max(0,min(100,int(y*100))) if y is not None else 0
        ed,yd=data['duration'],data['yawn_duration']
        ep=min(100,100*ed/et) if et>0 else 0
        yp=min(100,100*yd/yt) if yt>0 else 0
        st.html(f'''<div class="aura-kpis">
          <div class="aura-kpi"><span>👁 눈 감김 분류 점수</span><strong>{fmt(e)}</strong>
            <div class="aura-progress"><div style="width:{v1}%"></div></div>
            <small>YOLO26n · 눈 ROI + CLAHE</small></div>
          <div class="aura-kpi"><span>◉ 하품 분류 점수</span><strong>{fmt(y)}</strong>
            <div class="aura-progress"><div style="width:{v2}%"></div></div>
            <small>YOLO26s · {('원본 RGB' if applied.get('yawn_view')=='full' else '얼굴 중심 실험 입력')}</small></div>
          <div class="aura-kpi"><span>눈 감김 연속 유지</span><strong>{ed:.1f} / {et:.0f}초</strong>
            <div class="aura-progress"><div style="width:{ep:.0f}%"></div></div>
            <small>{et:g}초 경과 후 경고음 1회</small></div>
          <div class="aura-kpi"><span>하품 연속 유지</span><strong>{yd:.1f} / {yt:.0f}초</strong>
            <div class="aura-progress"><div style="width:{yp:.0f}%"></div></div>
            <small>{yt:g}초 경과 후 경고음 1회</small></div>
        </div>''')
        st.html(f'<div class="aura-eyestate"><span>기준: 눈 ≥ {applied["eye"]:.2f} / 하품 ≥ {applied["yawn"]:.2f}</span><span>이번 세션 경고: 눈 {eyes}회 · 하품 {yawns}회</span></div>')
        worker=st.session_state.get('local_worker')
        cam=st.session_state.get('local_camera')
        if cam and cam.running:
            _,fps,age,error=cam.snapshot()
            st.caption(f'로컬 카메라 {fps:.1f} FPS · AI {worker.fps:.1f} FPS · 추론 {worker.elapsed_ms:.0f}ms · 화면 지연 {age*1000:.0f}ms' if worker else '카메라 연결 중')
            if worker and worker.error:st.warning('AI 처리 오류: '+worker.error)
            if error:st.warning('카메라 오류: '+error)
        st.html('<div class="aura-section-note">얼굴 랜드마크 · 눈/입 인식 상태</div>')
        a,b=st.columns(2,gap='small')
        with a:
            if data['eye_points'] is not None:st.image(data['eye_points'],caption='눈 위치 추적',width='stretch')
            else:st.caption('눈 ROI 미검출')
        with b:
            if data['mouth_points'] is not None:st.image(data['mouth_points'],caption='입 위치 추적',width='stretch')
            else:st.caption('입 ROI 미검출')
        with st.expander('분류기에 넣은 실제 이미지 · 경고 이력'):
            a,b=st.columns(2)
            with a:
                if data['eye_roi']:st.image(data['eye_roi'],caption='눈: MediaPipe ROI + CLAHE',width='stretch')
            with b:
                if data['mouth_roi']:st.image(data['mouth_roi'],caption='하품: 실제 RGB 분류 입력',width='stretch')
            if events:
                st.dataframe([dict(시간=x['time'],이벤트=x['label'],점수=x['score']) for x in events],
                             hide_index=True,width='stretch')
            else:st.caption('아직 경고가 없습니다.')
        st.caption('표시 점수는 보정된 위험 확률이 아니라 모델의 분류 출력입니다.')
    fragment()


def safety_panel():
    st.markdown('### 안전 대응 시뮬레이션')
    st.caption('가상 차량 감속·정차 및 SOS 화면만 구현. 차량 장치나 119에 연결되지 않습니다.')
    safety=st.session_state.safety
    @st.fragment(run_every='1s')
    def safety_status():
        safety.configure(st.session_state.fast_demo)
        v=safety.tick()
        names={'monitor':'대기 / 정상','stopping':'가상 감속 중','rest':'가상 정차 / 휴식',
               'ready':'운전자 확인 완료','unresponsive':'가상 무반응 대기','sos':'모의 119 알림'}
        m1,m2,m3=st.columns(3)
        m1.metric('시뮬레이션 상태',names.get(v['mode'],v['mode']))
        m2.metric('가상 속도',f"{v['speed']:.0f} km/h")
        m3.metric('눈 감김 경고 / 5분',f"{v['recent_eye_events']}회")
        if v['mode']=='stopping':st.progress(max(0.,min(1.,1-v['speed']/80.)))
        if v['mode']=='rest':st.success('가상 정차 완료 · 창문 열림(시뮬레이션)')
        if v['mode']=='unresponsive':st.warning(f"시나리오 남은 시간 {v['response_remaining']:.0f}초 · 의식 상태를 측정하는 기능이 아닙니다.")
        if v['mode']=='sos':st.error('모의 119 화면입니다. 실제 신고는 하지 않습니다.')
        with st.expander('시뮬레이션 이벤트 기록'):
            st.dataframe(v['events'],hide_index=True,width='stretch')
    safety_status()
    a,b,c,d=st.columns(4,gap='small')
    with a:
        if st.button('가상 안전 정차',width='stretch',key='sim_stop'):
            safety.start_stop_demo();st.rerun()
    with b:
        if st.button('무반응 시나리오',width='stretch',key='sim_unresponsive'):
            safety.start_unresponsive_demo();st.rerun()
    with c:
        if st.button('운전자 응답 확인',width='stretch',key='sim_ack'):
            safety.respond();st.rerun()
    with d:
        if st.button('↺ 초기화',width='stretch',key='sim_reset'):
            safety.reset();st.rerun()
    st.caption('빠른 시연: 휴식 15초·무반응 30초 / 기본: 5분·10분. 시간은 의료·안전 기준이 아니라 연출 값입니다.')


# ---- session defaults and persistent objects ----
for k,v in dict(started=False,camera_index=0,camera_resolution='640×480 (권장)',infer_interval=350,
                eye_threshold=.25,yawn_threshold=.50,eye_seconds=5.0,yawn_seconds=3.0,
                eye_clahe=True,yawn_view='full',fast_demo=True,camera_notice='').items():
    st.session_state.setdefault(k,v)
if st.session_state.get('aura_settings_revision')!='v12_5s_3s':
    st.session_state.eye_seconds=5.0
    st.session_state.yawn_seconds=3.0
    st.session_state.aura_settings_revision='v12_5s_3s'
if 'monitor' not in st.session_state:st.session_state.monitor=Monitor()
if 'safety' not in st.session_state:st.session_state.safety=SafetySimulation()
st.session_state.monitor.safety=st.session_state.safety
base_styles()

st.html('<div class="aura-nav"><div class="aura-logo">AURA <span>DRIVE</span></div><div class="aura-topright">LOCAL DEMO • YOLO26n / YOLO26s • SIMULATION ONLY</div></div>')

if not st.session_state.started:
    hero(welcome=True)
    st.html('<div class="aura-start-banner"><span><strong>로컬 전용 AI 모니터</strong> · 눈 5초 · 하품 3초 경고 · 화면 원본 유지</span><span>AI ON CPU  /  NO CLOUD</span></div>')
    left,center,right=st.columns([.45,1.3,1.2])
    with center:
        if st.button('▶  로컬 웹캠과 AI 시작',type='primary',width='stretch',key='enter_and_start'):
            st.session_state.started=True
            st.session_state.camera_notice=launch_camera()
            st.rerun()
    with right:
        if st.button('대시보드만 먼저 보기',width='stretch',key='enter_only'):
            st.session_state.started=True
            st.rerun()
    st.caption('카메라를 바로 켜거나, 연결 전에 설정·분류기를 확인할 수 있습니다. 소리는 모니터 화면에서 별도 활성화하세요.')
    st.html('<div class="aura-footer">로컬 웹캠 · 실제 차량/긴급 신고 연결 없음 · 교육용 프로토타입</div>')
    st.stop()

page=st.radio('대시보드 메뉴',['실시간 모니터','사진 분석','안전 대응 시뮬레이션','설정'],horizontal=True,label_visibility='collapsed')
if page!='실시간 모니터':
    cam=get_running_camera()
    if cam is not None and cam.running:shutdown_camera()

if page=='실시간 모니터':
    hero()
    st.html('<div class="aura-start-banner"><span><strong>LOCAL CAMERA</strong> · 웹캠 원본 + 독립 AI 추론</span><span>눈 5초  /  하품 3초  /  소리 충돌 방지</span></div>')
    top1,top2,top3,top4=st.columns([1,1,1.6,1.4],gap='small')
    with top1:
        st.number_input('카메라 장치 번호',min_value=0,max_value=9,step=1,key='camera_index')
    with top2:
        st.selectbox('카메라 화질',['640×480 (권장)','1280×720 (고화질)'],key='camera_resolution')
    active=bool((cam:=get_running_camera()) and cam.running)
    with top3:
        st.caption('시작 · 정상 종료 후에도 기존 모델 가중치는 유지됩니다.')
        if st.button('▶  로컬 웹캠 시작',type='primary',width='stretch',disabled=active,key='camera_start'):
            st.session_state.camera_notice=launch_camera()
            st.rerun()
    with top4:
        st.caption('장치 점유 해제 · 다른 앱에서 사용 가능')
        if st.button('■  웹캠 종료',width='stretch',disabled=not active,key='camera_stop'):
            shutdown_camera();st.session_state.camera_notice='카메라 연결을 종료했어.';st.rerun()
    if st.session_state.camera_notice:
        if '실패' in st.session_state.camera_notice or '없어' in st.session_state.camera_notice or 'Error' in st.session_state.camera_notice:
            st.error(st.session_state.camera_notice)
        else:st.info(st.session_state.camera_notice)
    if any(not p.exists() for p in (EYE_FILE,YAWN_FILE)):
        st.warning('가중치가 아직 없다면 runs/eyes_yolo26n-cls/weights/best.pt 및 runs/yawn_yolo26s-cls/weights/best.pt 위치를 확인해줘.')
    st.markdown('#### 음성 경고 · 눈 5초 / 하품 3초')
    cam=get_running_camera()
    render_audio_panel(ROOT,cam.events_url if cam and cam.running else None)
    c1,c2=st.columns([2,1])
    with c1:
        st.caption('경고음은 사용자가 켠 뒤에만 재생됩니다. 눈 경고 우선, 동시 재생 없음, 연속 판정당 1회.')
    with c2:
        if st.button('↺ 시뮬레이션 초기화',width='stretch',key='reset_safety'):
            st.session_state.safety.reset()
            st.rerun()
    live_col,stats_col=st.columns([1.6,1],gap='medium')
    with live_col:
        st.html('<div class="aura-stage-title"><b>◉ 원본 카메라 화면</b><small>LOCAL / MJPEG STREAM</small></div>')
        camera=get_running_camera()
        if camera is not None and camera.running:
            # The browser downloads MJPEG directly from a localhost server, NOT via Streamlit rerun.
            st.html(f'<div class="aura-frame"><img src="{camera.url}" alt="원본 로컬 웹캠 스트림"></div>')
        else:
            st.html('<div class="aura-frame"><div class="aura-idle"><div><strong>LOCAL CAMERA READY</strong><br><span>웹캠 시작 버튼으로 연결하세요.<br>클라우드 전송 없이 이 PC에서 작동합니다.</span></div></div></div>')
        st.caption('원본 영상은 얼굴을 합성하거나 잘라 표시하지 않아. AI 분석 입력만 별도로 ROI 처리해.')
    with stats_col:
        st.html('<div class="aura-stage-title"><b>AI 상태 분석</b><small>LIVE · EYE / YAWN</small></div>')
        probs_panel()
    with st.expander('실시간 경고 설정 · 필요할 때만 열기'):
        a,b,c=st.columns(3)
        with a:st.slider('눈 판정 임계값',0.05,0.95,step=.05,key='eye_threshold')
        with b:st.slider('하품 판정 임계값',0.05,0.95,step=.05,key='yawn_threshold')
        with c:st.slider('AI 프레임 분석 간격 (ms)',200,800,step=50,key='infer_interval')
        a,b=st.columns(2)
        with a:st.slider('눈 감김 경고 시간 (초)',5.,10.,step=.5,key='eye_seconds')
        with b:st.slider('하품 경고 시간 (초)',3.,8.,step=.5,key='yawn_seconds')
        st.radio('하품 웹캠 입력 방식',
            ['full','face_focus'],
            format_func=lambda v: '원본 전체 RGB (외부 평가와 동일)' if v=='full' else '얼굴 중심 ROI (웹캠 실험용 · 정확도 미검증)',
            key='yawn_view',horizontal=True)
        st.caption('웹캠에서 얼굴이 작게 나올 때만 얼굴 중심 모드를 비교해봐. 기존 31번 정확도는 원본 전체 입력 기준이야.')
        if st.button('현재 설정 적용',type='primary'):
            engine=st.session_state.get('local_engine')
            if engine:engine.cfg=params()
            worker=st.session_state.get('local_worker')
            if worker:worker.interval=st.session_state.infer_interval/1000
            st.success('새 기준을 적용했어. 기존 가중치는 그대로야.')
    with st.expander('안전 대응 시뮬레이션 바로가기'):
        safety_panel()

elif page=='사진 분석':
    st.markdown('## 이미지 1장 분석')
    st.caption('실제 모델에 입력되는 전처리를 확인하는 페이지 · 스트림링 캠과 독립')
    upload=st.file_uploader('얼굴이 나온 이미지 업로드 (JPG / PNG / WebP)',type=['jpg','jpeg','png','webp'])
    if upload:
        content=upload.getvalue()
        if len(content)>12*1024*1024:st.error('이미지는 12 MB 이하로 선택해줘.')
        else:
            im=ImageOps.exif_transpose(Image.open(io.BytesIO(content))).convert('RGB')
            if max(im.size)>1600:im.thumbnail((1600,1600))
            a,b=st.columns([1.25,1])
            with a:st.image(im,caption='원본 이미지',width='stretch')
            with b:
                if st.button('AI 분류 실행',type='primary',width='stretch'):
                    try:
                        model=models_cached(str(EYE_FILE.resolve()),str(YAWN_FILE.resolve()))
                        engine=FaceEngine(model,params(),static=True)
                        try:result=engine.predict(im)
                        finally:engine.close()
                        st.session_state.photo_result=result
                    except Exception as e:st.error(f'{type(e).__name__}: {e}')
                result=st.session_state.get('photo_result')
                if result:
                    st.metric('눈 감김 점수',f"{result['eye']:.1%}" if result['eye'] is not None else 'ROI 실패')
                    st.metric('하품 점수',f"{result['yawn']:.1%}" if result['yawn'] is not None else '얼굴 미검출')
                    if result['eye_processed']:st.image(result['eye_processed'],caption='눈 ROI + CLAHE',width=200)
                    st.caption('하품 입력은 선택한 RGB 모드에 따라 원본 전체 또는 얼굴 중심 crop을 224px 정사각 패딩')

elif page=='안전 대응 시뮬레이션':
    hero()
    st.toggle('빠른 발표 시연 (15초/30초)',key='fast_demo')
    safety_panel()

elif page=='설정':
    st.markdown('## 실행 환경 및 최종 모델')
    st.caption('실시간 추론은 CPU에서 처리. 원본 영상은 localhost에 직접 연결되며 서버 공개 배포를 전제로 하지 않습니다.')
    for label,p in [('눈 YOLO26n',EYE_FILE),('하품 YOLO26s',YAWN_FILE),('차량 히어로 이미지',CAR_FILE),('기존 비교 코드',ROOT/'31_yawn_external_compare.py')]:
        st.write(('✅' if p.exists() else '❌')+f' {label}: `{p}`')
    st.checkbox('눈 ROI CLAHE 사용',key='eye_clahe')
    st.slider('AI 분석 간격 (ms)',200,800,step=50,key='infer_interval')
    st.slider('눈 감김 판정 임계값',.05,.95,step=.05,key='eye_threshold')
    st.slider('하품 판정 임계값',.05,.95,step=.05,key='yawn_threshold')
    st.radio('하품 입력 방식', ['full','face_focus'],
             format_func=lambda v: '원본 전체 RGB (검증된 기준)' if v=='full' else '얼굴 중심 ROI (웹캠 실험)',
             key='yawn_view', horizontal=True)
    st.caption('선정 근거: 눈 CEW 개발용 평가 YOLO26n+CLAHE+0.25 (88.36%), 하품 외부 Roboflow YOLO26s full RGB+0.50 (78.37%). 실제 영상에 대한 최종 독립 정확도는 아직 측정하지 않음.')
    if st.button('현재 설정 적용',type='primary'):
        e=st.session_state.get('local_engine')
        if e:e.cfg=params()
        w=st.session_state.get('local_worker')
        if w:w.interval=st.session_state.infer_interval/1000
        st.success('저장됨. 모델 가중치는 변경하지 않았어.')
    if st.button('모니터링 종료 및 처음 화면으로',width='stretch'):
        shutdown_camera();st.session_state.started=False;st.rerun()

st.html('<div class="aura-footer">AURA DRIVE / LOCAL DEMO · THE CAMERA RUNS ONLY ON THIS PC · SIMULATED SAFETY RESPONSE · NOT FOR REAL VEHICLE CONTROL</div>')

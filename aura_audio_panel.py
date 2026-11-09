"""One browser audio player, debounced and prioritised by event id.

Audio is activated by a real browser click. Alerts are fetched from the
localhost camera server and NEVER emitted by Streamlit st.audio rerenders.
"""
from __future__ import annotations

import base64
import html
import io
import json
import math
import struct
import wave
from pathlib import Path


def _fallback_wav(kind: str) -> bytes:
    """Produce an original local alert sound if MP3 files are absent."""
    rate = 11025
    freq = 820 if kind == 'eye' else 560
    pulses = 3 if kind == 'eye' else 2
    buf = io.BytesIO()
    with wave.open(buf, 'wb') as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        for i in range(pulses):
            duration = .17 if kind == 'eye' else .26
            for j in range(int(rate * duration)):
                t = j / rate
                fade = min(1., t * 27, (duration - t) * 28)
                # Soft, non-clipping two-tone original chime.
                v = int(10000 * max(0., fade) * (
                    .78 * math.sin(2 * math.pi * (freq + i * 20) * t)
                    + .22 * math.sin(2 * math.pi * (freq * 1.5) * t)))
                w.writeframesraw(struct.pack('<h', v))
            w.writeframesraw(b'\0\0' * int(rate * .13))
    return buf.getvalue()


def _clip_src(root: Path, kind: str) -> str:
    file = root / ('눈감김.mp3' if kind == 'eye' else '하품.mp3')
    if file.is_file():
        return 'data:audio/mpeg;base64,' + base64.b64encode(file.read_bytes()).decode()
    return 'data:audio/wav;base64,' + base64.b64encode(_fallback_wav(kind)).decode()


def render_audio_panel(root: Path, events_url: str | None):
    import streamlit.components.v1 as components
    src_eye, src_yawn = _clip_src(root, 'eye'), _clip_src(root, 'yawn')
    api = json.dumps(events_url or '')
    # All updates stay inside one iframe/player. Never use st.audio(autoplay=True)
    # in a periodically rerunning Streamlit fragment.
    content = '''<!doctype html><html lang="ko"><head><meta charset="utf-8">
<style>
*{box-sizing:border-box}body{margin:0;font:13px Arial,'Malgun Gothic',sans-serif;color:#e8f5f3;background:#0b1822}
.p{display:flex;align-items:center;flex-wrap:wrap;gap:10px;background:#101f2b;border:1px solid #2e4855;border-radius:13px;padding:12px 14px}
button{cursor:pointer;border:1px solid #3e5d6a;background:#152b39;color:#e8f5f3;font-weight:750;padding:10px 14px;border-radius:10px;font-size:12px}
button:hover{border-color:#8de8d1}button.main{background:#61e6c6;color:#062a27;border:0}
#status{color:#9ebac6;font-size:11px;margin-left:auto}
.d{color:#96abbb;font-size:10px;width:100%;margin-top:-1px}
</style></head><body>
<div class="p"><button class="main" id="toggle">🔊 경고음 켜기</button>
<button id="test-eye">👁 눈 경고음 테스트</button><button id="test-yawn">◉ 하품 경고음 테스트</button>
<span id="status">오디오 꺼짐</span>
<span class="d">눈 5초 / 하품 3초 연속 감지 후 1회 재생 · 눈 우선 · 동시 재생 없음</span></div>
<script>
const URL=__EVENTS_URL__;
const audio={eye:new Audio(__EYE_SRC__),yawn:new Audio(__YAWN_SRC__)};
for(const p of Object.values(audio))p.preload='auto';
let enabled=false, lastId=null, active=null, serial=0, locked=false;
const status=document.getElementById('status');
const toggle=document.getElementById('toggle');
function paint(){toggle.textContent=enabled?'🔇 경고음 끄기':'🔊 경고음 켜기';
toggle.className=enabled?'':'main';status.textContent=active?(active==='eye'?'눈 경고음 재생 중':'하품 경고음 재생 중'):(enabled?'음성 대기':'오디오 꺼짐');}
function stopSound(){serial+=1;for(const a of Object.values(audio)){try{a.pause();a.currentTime=0}catch(e){}}active=null;}
function play(kind){
 if(!enabled)return;
 if(active){if(kind==='eye' && active==='yawn'){stopSound();} else {return;}}
 const token=++serial;const p=audio[kind];active=kind;paint();
 p.currentTime=0;
 p.onended=()=>{if(token===serial){active=null;paint();}};
 p.onerror=()=>{if(token===serial){active=null;status.textContent='오디오 재생 실패';}};
 const attempt=p.play();
 if(attempt && attempt.catch){attempt.catch(()=>{if(token===serial){active=null;status.textContent='재생 차단 · 테스트 버튼을 눌러주세요';}});}
}
toggle.onclick=()=>{
 enabled=!enabled;
 if(!enabled)stopSound();
 else {lastId=null;fetchEvents(true);}
 paint();
};
document.getElementById('test-eye').onclick=()=>{if(!enabled){enabled=true;lastId=null;fetchEvents(true)}play('eye');};
document.getElementById('test-yawn').onclick=()=>{if(!enabled){enabled=true;lastId=null;fetchEvents(true)}play('yawn');};
let fetching=false;
async function fetchEvents(onlyBaseline=false){
 if(!URL||fetching)return;
 fetching=true;
 try{
  const r=await fetch(URL,{cache:'no-store'});
  if(!r.ok)throw Error('HTTP '+r.status);
  const body=await r.json();const entries=(body.events||[]).sort((a,b)=>a.id-b.id);
  const current=entries.length?entries[entries.length-1].id:0;
  if(lastId===null||onlyBaseline){lastId=current;return;}
  const pending=entries.filter(e=>e.id>lastId);
  lastId=Math.max(lastId,current);
  if(!enabled||!pending.length)return;
  // If eye and yawn events arrive together, play eye only.
  const kind=pending.some(e=>e.kind==='eye')?'eye':'yawn';
  play(kind);
 }catch(e){if(enabled&&!active)status.textContent='카메라 연결 대기';}
 finally{fetching=false;}
}
setInterval(()=>fetchEvents(false),500);
fetchEvents(true);paint();
</script></body></html>'''
    content = (content.replace('__EVENTS_URL__', api)
                      .replace('__EYE_SRC__', json.dumps(src_eye))
                      .replace('__YAWN_SRC__', json.dumps(src_yawn)))
    components.html(content, height=112, scrolling=False)

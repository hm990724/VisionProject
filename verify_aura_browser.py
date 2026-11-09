"""Render the real dashboard in an isolated headless Edge profile (no camera)."""
import asyncio
import argparse
import base64
import json
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import time
import urllib.request
import websockets

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'reports/presentation/final_selection'


def port():
    with socket.socket() as s: s.bind(('127.0.0.1',0));return s.getsockname()[1]


async def inspect(url, camera_test=False):
    with urllib.request.urlopen(url,timeout=5) as r: pages=json.load(r)
    page=next(p for p in pages if p['type']=='page')
    async with websockets.connect(page['webSocketDebuggerUrl'],max_size=20_000_000) as ws:
        serial=0
        async def call(method,params=None):
            nonlocal serial
            serial+=1;ident=serial
            await ws.send(json.dumps({'id':ident,'method':method,'params':params or {}}))
            while True:
                msg=json.loads(await ws.recv())
                if msg.get('id')==ident:
                    if 'error' in msg:raise RuntimeError(msg['error'])
                    return msg.get('result',{})
        async def evaluate(expr):
            r=await call('Runtime.evaluate',{'expression':expr,'returnByValue':True})
            return r.get('result',{}).get('value')
        await call('Emulation.setDeviceMetricsOverride',{'width':1440,'height':1100,'deviceScaleFactor':1,'mobile':False})
        for _ in range(60):
            ready=await evaluate("[...document.querySelectorAll('button')].some(b=>b.textContent.includes('시동 걸고'))")
            if ready:break
            await asyncio.sleep(.5)
        assert ready,'Startup button did not render'
        await evaluate("[...document.querySelectorAll('button')].find(b=>b.textContent.includes('시동 걸고')).click()")
        for _ in range(90):
            ready=await evaluate("[...document.querySelectorAll('button')].some(b=>b.textContent.includes('로컬 카메라 연결'))")
            if ready:break
            await asyncio.sleep(.5)
        assert ready,'Dashboard did not render'
        await asyncio.sleep(2)
        errors=await evaluate("[...document.querySelectorAll('[data-testid=stException]')].map(e=>e.innerText)")
        assert not errors,errors
        capture=await call('Page.captureScreenshot',{'format':'png','captureBeyondViewport':False})
        (OUT/'streamlit_dashboard.png').write_bytes(base64.b64decode(capture['data']))
        report={'monitor_rendered':True,'errors':errors,'camera_activated':camera_test}
        if camera_test:
            await evaluate("[...document.querySelectorAll('button')].find(b=>b.textContent.includes('로컬 카메라 연결')).click()")
            ready=False
            for _ in range(60):
                ready=await evaluate("[...document.images].some(i=>i.src.startsWith('http://127.0.0.1:') && i.naturalWidth>0)")
                if ready:break
                await asyncio.sleep(.5)
            report['live_image_decoded']=ready
            # Never screenshot or save the user's live camera frame.
            await evaluate("[...document.querySelectorAll('button')].find(b=>b.textContent.includes('연결 종료')).click()")
            await asyncio.sleep(1)
            assert ready,'MJPEG camera image did not decode in browser'
        (OUT/'browser_verification.json').write_text(json.dumps(report,indent=2),encoding='utf8')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--camera',action='store_true');args=parser.parse_args()
    app_port,cdp_port=port(),port();processes=[]
    OUT.mkdir(parents=True,exist_ok=True)
    profile=tempfile.mkdtemp(prefix='aura-browser-')
    with (OUT/'browser_server.log').open('w',encoding='utf8') as log:
        try:
            processes.append(subprocess.Popen([sys.executable,'-m','streamlit','run','11_streamlit_dashboard.py','--server.address=127.0.0.1',f'--server.port={app_port}','--server.headless=true','--browser.gatherUsageStats=false'],cwd=ROOT,stdout=log,stderr=log,creationflags=subprocess.CREATE_NO_WINDOW))
            for _ in range(40):
                try:
                    with urllib.request.urlopen(f'http://127.0.0.1:{app_port}/_stcore/health',timeout=1) as r:
                        if r.status==200:break
                except OSError:time.sleep(.25)
            processes.append(subprocess.Popen([r'C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe','--headless=new','--disable-gpu','--no-first-run','--no-default-browser-check',f'--user-data-dir={profile}',f'--remote-debugging-port={cdp_port}',f'http://127.0.0.1:{app_port}'],stdout=log,stderr=log,creationflags=subprocess.CREATE_NO_WINDOW))
            for _ in range(60):
                try:
                    with urllib.request.urlopen(f'http://127.0.0.1:{cdp_port}/json',timeout=1) as r:
                        if r.status==200:break
                except OSError:time.sleep(.25)
            asyncio.run(inspect(f'http://127.0.0.1:{cdp_port}/json',args.camera))
            print('Dashboard rendered; screenshot saved.')
        finally:
            for p in reversed(processes):
                p.terminate()
                try:p.wait(timeout=10)
                except subprocess.TimeoutExpired:p.kill();p.wait()


if __name__=='__main__':main()

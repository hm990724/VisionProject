"""Smoke-test app navigation and optionally measure real local webcam capture.

No camera images are saved. --camera explicitly activates camera device 0.
"""
import argparse
import json
import time
from pathlib import Path
from streamlit.testing.v1 import AppTest


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--camera',action='store_true');args=parser.parse_args()
    out=Path('reports/presentation/final_selection');out.mkdir(parents=True,exist_ok=True)
    app=AppTest.from_file('11_streamlit_dashboard.py',default_timeout=60)
    app.run();assert not app.exception,list(app.exception)
    app.button(key='aura_ignition_enter').click().run()
    assert not app.exception,list(app.exception)
    assert not app.error,[e.value for e in app.error]
    assert app.session_state['yawn_t']==.5
    assert app.session_state['aura_engine'].m['yawn'].name=='yolo26s'
    report={'startup':'pass','monitor_initialization':'pass','camera_requested':args.camera,
            'yawn_model':'yolo26s','yawn_input':'full_image','yawn_threshold':.5}
    try:
        if args.camera:
            app.button(key='aura_local_camera_start').click().run()
            camera=app.session_state['aura_local_camera'];worker=app.session_state['aura_worker']
            samples=[]
            for _ in range(15):
                camera.heartbeat();time.sleep(1)
                frame,fps,age,error=camera.snapshot()
                samples.append({'frame_available':frame is not None,'capture_fps':round(fps,2),
                                'age_ms':round(age*1000,2) if age!=float('inf') else None,
                                'error':error,'inference_count':worker.inference_count,
                                'inference_ms':round(worker.elapsed_ms,2),'inference_error':worker.error})
            report['camera_samples']=samples
            report['camera_frames_received']=any(s['frame_available'] for s in samples)
            report['inference_count']=worker.inference_count
            report['camera_stopped']=camera.stop()
            app.button(key='aura_local_camera_stop').click().run()
            assert not app.exception,list(app.exception)
        app.radio[0].set_value('설정').run();assert not app.exception,list(app.exception)
        app.radio[0].set_value('이미지 분석').run();assert not app.exception,list(app.exception)
        app.radio[0].set_value('실시간 모니터').run();assert not app.exception,list(app.exception)
        report['navigation']='pass'
        buttons=[b for b in app.button if '자동 시연 시작' in b.label]
        if buttons:
            buttons[0].click().run();assert not app.exception,list(app.exception)
            report['simulation_button']='pass'
    finally:
        camera=app.session_state.filtered_state.get('aura_local_camera')
        if camera:camera.stop()
        worker=app.session_state.filtered_state.get('aura_worker')
        if worker:worker.close()
        (out/'webcam_verification.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf8')
    print(json.dumps(report,ensure_ascii=False,indent=2))


if __name__=='__main__':main()

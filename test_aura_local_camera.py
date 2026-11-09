"""Regression tests for camera ownership, stale alarms and local transport."""
import threading
import time
import unittest
import urllib.request
import numpy as np
from aura_local_camera import LocalCameraStream, LatestFrameWorker


class Sink:
    def __init__(self): self.received = 0; self.invalidated = 0
    def submit(self, *_): self.received += 1
    def invalidate(self): self.invalidated += 1


class Capture:
    def __init__(self, opened=True):
        self.opened = opened; self.threads = []; self.released = False
        self.threads.append(threading.get_ident())
    def isOpened(self): return self.opened
    def set(self, *_): pass
    def read(self):
        self.threads.append(threading.get_ident())
        time.sleep(.01)
        return True, np.zeros((48, 64, 3), dtype=np.uint8)
    def release(self):
        self.threads.append(threading.get_ident()); self.released = True


def until(predicate, seconds=3):
    end = time.monotonic()+seconds
    while time.monotonic() < end:
        if predicate(): return
        time.sleep(.01)
    raise AssertionError('Timed out')


class CameraTests(unittest.TestCase):
    def test_single_owner_and_mjpeg(self):
        caps=[]; sink=Sink()
        def factory(*_):
            cap=Capture(); caps.append(cap); return cap
        camera=LocalCameraStream(0,sink,factory)
        try:
            self.assertTrue(camera.start())
            until(lambda: camera.jpeg is not None)
            with urllib.request.urlopen(camera.url,timeout=3) as response:
                self.assertIn(b'Content-Type: image/jpeg',response.read(100))
            self.assertGreater(sink.received,0)
            self.assertTrue(camera.stop())
            self.assertTrue(caps[0].released)
            self.assertEqual(len(set(caps[0].threads)),1)
            self.assertNotEqual(caps[0].threads[0],threading.get_ident())
        finally: camera.stop()

    def test_open_failure_closes_backends(self):
        caps=[]
        def factory(*_):
            cap=Capture(False); caps.append(cap); return cap
        camera=LocalCameraStream(0,Sink(),factory)
        camera.start()
        until(lambda: not camera.running)
        self.assertTrue(camera.stop())
        self.assertEqual(len(caps),2)
        self.assertTrue(all(c.released for c in caps))
        self.assertIsNotNone(camera.error)

    def test_abandoned_session_releases_camera(self):
        camera=LocalCameraStream(0,Sink(),lambda *_: Capture(),lease_seconds=.1)
        camera.start()
        until(lambda: not camera.running)
        self.assertTrue(camera.stop())

    def test_inflight_result_does_not_alarm_after_stop(self):
        started=threading.Event(); release=threading.Event(); updates=[]
        class Engine:
            cfg={}
            def predict(self, _):
                started.set(); release.wait(2)
                return dict(eye=.99,yawn=.99,eye_processed=None,mouth=None,eye_points=None,mouth_points=None)
            def close(self): pass
        class Monitor:
            def update(self, eye, *_args, **_kwargs): updates.append(eye)
        worker=LatestFrameWorker(Engine(),Monitor())
        worker.submit(np.zeros((48,64,3),dtype=np.uint8))
        self.assertTrue(started.wait(2))
        worker.invalidate(); release.set()
        time.sleep(.15)
        self.assertTrue(worker.close())
        self.assertNotIn(.99,updates)

    def test_slow_inference_consumes_latest_not_queue(self):
        processed=[]; started=threading.Event(); release=threading.Event()
        class Engine:
            cfg={}
            def predict(self, im):
                processed.append(np.asarray(im)[0,0,0])
                if len(processed)==1: started.set(); release.wait(2)
                return dict(eye=None,yawn=None,eye_processed=None,mouth=None,eye_points=None,mouth_points=None)
            def close(self): pass
        class Monitor:
            def update(self, *_args, **_kwargs): pass
        worker=LatestFrameWorker(Engine(),Monitor(),interval_ms=80)
        try:
            worker.submit(np.zeros((8,8,3),dtype=np.uint8)); self.assertTrue(started.wait(2))
            for n in range(1,10): worker.submit(np.full((8,8,3),n,dtype=np.uint8))
            release.set(); until(lambda: len(processed)>=2)
            self.assertEqual(list(processed),[0,9])
        finally: worker.close()


if __name__=='__main__': unittest.main()

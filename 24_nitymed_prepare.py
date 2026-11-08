"""Inventory NITYMED videos, sample frames, extract identical mouth ROIs as team ResNet18.

This is an annotation aid, NOT an automatic source of ground-truth labels.
Works alongside existing VisionProject, without overwriting original data or models.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import math
import sys
from pathlib import Path

VIDEO_EXTS = {'.avi', '.mp4', '.mov', '.mkv', '.wmv', '.mpg', '.mpeg', '.webm'}
ARCHIVES = {'.zip', '.rar', '.7z'}
ROOT = Path(__file__).resolve().parent


def discover(source: Path):
    if not source.exists():
        return [], []
    videos = sorted(p for p in source.rglob('*') if p.is_file() and p.suffix.lower() in VIDEO_EXTS)
    archives = sorted(p for p in source.rglob('*') if p.is_file() and p.suffix.lower() in ARCHIVES)
    return videos, archives


def path_id(path: Path, source: Path):
    return hashlib.sha1(path.relative_to(source).as_posix().encode('utf-8')).hexdigest()[:12]


def report_inventory(source: Path):
    videos, archives = discover(source)
    print(f'[SOURCE] {source.resolve()}')
    print(f'[VIDEO COUNT] {len(videos)} | archives {len(archives)}')
    print(f'[TOTAL VIDEO SIZE] {sum(p.stat().st_size for p in videos)/1024**2:.1f} MiB')
    for p in videos[:12]:
        print('  VIDEO', p.relative_to(source), f'{p.stat().st_size/1024**2:.1f} MiB')
    for p in archives[:12]:
        print('  ARCHIVE', p.relative_to(source))
    if not videos:
        print('\n[BLOCKED] No video files were found. The Windows folder might be empty,')
        print('or videos may be in a ZIP/7z, or in a different directory.')
        print('Find local videos:')
        print('  Get-ChildItem . -Recurse -File | Where-Object { $_.Extension -in ".mp4", ".avi", ".mov", ".mkv" } | Select-Object -First 30 FullName')
    return videos


def load_team(team_folder: Path):
    source = team_folder / '03_yawn_common.py'
    weights = team_folder / 'runs/03_yawn_resnet18/weights/03_best.pt'
    for p in (source, weights):
        if not p.is_file():
            raise FileNotFoundError(f'Missing team component: {p}')
    if str(team_folder.resolve()) not in sys.path:
        sys.path.insert(0, str(team_folder.resolve()))
    spec = importlib.util.spec_from_file_location('nitymed_yawn_common', source)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    import torch
    device = mod.choose_device('cpu')
    model, ckpt = mod.load_checkpoint(weights, device)
    if list(ckpt.get('class_names', ['no_yawn', 'yawn'])) != ['no_yawn', 'yawn']:
        raise ValueError(f'Unexpected class order: {ckpt.get("class_names")}')
    face_model = mod.verify_face_model(team_folder / 'models/face_landmarker.task', ckpt)
    extractor = mod.MouthExtractor(face_model, mod.CropConfig(**ckpt['crop_config']))
    model.eval()
    return model, extractor, mod.make_transform(training=False), device


def sample_times(cap, requested_step: float, per_video: int):
    import cv2
    frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not math.isfinite(fps) or fps <= 0 or not math.isfinite(frames) or frames <= 0:
        return []
    duration = frames / fps
    t = [float(x) for x in __import__('numpy').arange(0.2, max(duration-0.1, 0.2), requested_step)]
    if not t:
        t = [duration/2]
    if per_video > 0 and len(t) > per_video:
        import numpy as np
        t = [t[int(i)] for i in np.linspace(0, len(t)-1, per_video).round().astype(int)]
    return [(tm, min(int(tm*fps), int(frames)-1)) for tm in t]


def save_montage(examples, dest: Path):
    from PIL import Image, ImageOps, ImageDraw
    if not examples:
        return
    thumb_w, thumb_h, columns = 242, 196, 4
    nrows = math.ceil(len(examples)/columns)
    canvas = Image.new('RGB',(columns*thumb_w,nrows*thumb_h),(244,247,250))
    draw = ImageDraw.Draw(canvas)
    for i, (face,roi,time,score) in enumerate(examples):
        x=(i%columns)*thumb_w; y=(i//columns)*thumb_h
        a=ImageOps.contain(face,(130,138)); b=ImageOps.contain(roi,(96,110)) if roi is not None else None
        canvas.paste(a,(x+3,y+26))
        if b is not None: canvas.paste(b,(x+140,y+38))
        draw.text((x+5,y+5),f'{time:.1f}s  p(yawn)={score:.2f}' if score is not None else f'{time:.1f}s  ROI_FAIL',fill=(25,39,56))
        draw.text((x+4,y+175),'FACE / MOUTH ROI',fill=(25,39,56))
    dest.parent.mkdir(parents=True,exist_ok=True)
    canvas.save(dest,quality=85)


def prepare(args):
    import cv2
    import torch
    from PIL import Image
    videos = report_inventory(args.source)
    if not videos:
        raise SystemExit(2)
    if args.take > 0:
        videos=videos[:args.take]
    output=args.output.resolve()
    output.mkdir(parents=True,exist_ok=True)
    manifest=output/'annotation_manifest.csv'
    if manifest.exists() and not args.overwrite:
        raise SystemExit(f'Existing manifest protected: {manifest}\nPass --overwrite only if previous labels are backed up.')
    model, extractor, transform, device=load_team(args.team_dir.resolve())
    rows=[]
    try:
        for vi, path in enumerate(videos, 1):
            video_id=path_id(path,args.source)
            cap=cv2.VideoCapture(str(path))
            if not cap.isOpened():
                print('[WARN] Cannot open',path)
                continue
            times=sample_times(cap,args.every,args.max_per_video)
            examples=[]
            try:
                for i,(sec,frame_idx) in enumerate(times):
                    cap.set(cv2.CAP_PROP_POS_FRAMES,frame_idx)
                    ok,frame=cap.read()
                    if not ok: continue
                    if args.max_side > 0:
                        h,w=frame.shape[:2]
                        if max(h,w)>args.max_side:
                            scale=args.max_side/max(h,w)
                            frame=cv2.resize(frame,(int(w*scale),int(h*scale)),interpolation=cv2.INTER_AREA)
                    face=Image.fromarray(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB))
                    base=output/'samples'/video_id
                    base.mkdir(parents=True,exist_ok=True)
                    face_path=base/f'{i:03d}_face.jpg'
                    roi_path=base/f'{i:03d}_mouth.png'
                    face.save(face_path,quality=88)
                    roi=None; prob=None; status='ERROR'; err=''
                    try:
                        extraction=extractor.extract(face)
                        status=str(extraction.get('status','unknown'))
                        roi=extraction.get('crop') if status=='ok' else None
                        if roi is not None:
                            roi.save(roi_path)
                            with torch.inference_mode():
                                prob=float(model(transform(roi).unsqueeze(0).to(device)).softmax(1)[0,1].item())
                    except Exception as exc:
                        err=f'{type(exc).__name__}: {exc}'
                    if roi is None: roi_path=None
                    # Proposed labels deliberately left empty: no assumption that a full yawning video is all-yawn.
                    rows.append(dict(video_id=video_id, video=str(path.resolve()), subject_id='',time_seconds=round(sec,3),
                                     frame_id=i,face_image=str(face_path.relative_to(output)),
                                     mouth_roi=str(roi_path.relative_to(output)) if roi_path else '',
                                     roi_status=status,p_yawn=f'{prob:.6f}' if prob is not None else '',
                                     review_hint='LOOK_YAWN' if prob is not None and prob>=.50 else 'LOOK_NO_YAWN',
                                     label='',confirmed='0',split='',error=err))
                    examples.append((face,roi,sec,prob))
            finally:
                cap.release()
            save_montage(examples,output/'contact_sheets'/f'{vi:03d}_{video_id}.jpg')
            print(f'[{vi}/{len(videos)}] {path.name}: {len(examples)} frames')
    finally:
        extractor.close()
    if not rows:
        raise SystemExit('No frames extracted; check the codecs/OpenCV and videos.')
    with manifest.open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    print('\nSaved:',manifest)
    print('IMPORTANT: Review contact_sheets. Set label=yawn or no_yawn and confirmed=1 for correct frames.')
    print('Keep ambiguous yawning start/end frames unconfirmed. Set subject_id for same person across videos.')
    print('Do NOT train on pseudo-labels, and do not treat these as an untouched test set.')


def main():
    p=argparse.ArgumentParser(description='NITYMED mouth ROI sample/label-preparation workflow')
    p.add_argument('command',choices=['scan','prepare'])
    p.add_argument('--source',type=Path,default=ROOT/'yawn_mp')
    p.add_argument('--team-dir',type=Path,default=ROOT/'teammate_yawn')
    p.add_argument('--output',type=Path,default=ROOT/'reports/nitymed_video_review')
    p.add_argument('--every',type=float,default=1.5,help='Sample one frame every N seconds')
    p.add_argument('--max-per-video',type=int,default=14)
    p.add_argument('--max-side',type=int,default=960)
    p.add_argument('--take',type=int,default=0,help='Pilot on first N videos')
    p.add_argument('--overwrite',action='store_true')
    args=p.parse_args()
    args.source=args.source.resolve()
    if args.every<=0 or args.max_per_video<1:
        p.error('Invalid sampling values')
    if args.command=='scan':
        report_inventory(args.source)
    else:
        prepare(args)
if __name__=='__main__':main()

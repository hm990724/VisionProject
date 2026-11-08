"""Conservative, reproducible hard-case classifier-head fine-tuning on CPU.

Keeps the existing YOLO26n/ResNet18 backbones, MediaPipe crops, CLAHE/RGB,
class order and decision thresholds. No external evaluation images are trained on.
Labels are visual annotations, not predictions or folder-name-derived ground truth.
"""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import importlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
import torch
from PIL import Image, ImageOps
from common import EXT, dhash, near_groups, to_square
from roi_utils import crop_eyes

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'reports/hardcase_finetune'
RUN = ROOT / 'runs/hardcase_finetune'
SEED = 42

# Index refers to sorted filenames in wrinkles_index.txt/contact sheets.
# Only clearly visible, non-yawning mouth/face images are annotated no_yawn.
# Skin-only patches and uncertain expressions remain UNKNOWN and cannot train.
WRINKLE_NEGATIVE = set(map(int, '''70 72 74 75 77 78 79 80 81 82 83 84 85 86 87
92 93 94 95 98 100 102 104 106 107 109 110 113 114 115 116 117
121 123 124 125 126 128 130 131 132 139 140 141 144 145 146 150 151
153 154 157 158 159 160 161 162 166 167 168 169 170 173 174 175 177
193 200 202 221 240 267 274 275 276 277 278 279 281 289 290 291 292 293 294 295
343 351 352 359 360 361 362 363 367 371 372 381 384 385 386 388 389 390
393 395 396 397 398 399 400 401 402 403 404 405 406 408 409 410 411 413
416 417 418 419 420 421 422 423 424 425 426 427 428 430 431 432 433 434
435 436 437 438 439 440 441 442 443 444 445 446 447 448'''.split()))


def read_csv(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open('w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def photo(path):
    with Image.open(path) as im:
        im = ImageOps.exif_transpose(im).convert('RGB')
        im.thumbnail((1280, 1280))
        return im


def prepare():
    OUT.mkdir(parents=True, exist_ok=True)
    records = []
    for condition in ('no_makeup', 'with_makeup', 'wrinkles'):
        paths = sorted((ROOT/'data/error_data'/condition).glob('*.jpg'),
                       key=lambda p: int(p.stem) if p.stem.isdigit() else p.stem)
        for i, p in enumerate(paths):
            label = ('Open' if not (condition == 'no_makeup' and i == 25) else 'UNKNOWN') if condition != 'wrinkles' else ('no_yawn' if i in WRINKLE_NEGATIVE else 'UNKNOWN')
            records.append(dict(image=str(p), condition=condition, index=i,
                task='eyes' if condition != 'wrinkles' else 'yawn', actual=label,
                annotation='visual_review_assistant',
                review_note=('visible_open_eyes' if label == 'Open' else 'visible_non_yawning_mouth' if label == 'no_yawn' else 'unobservable_or_ambiguous_excluded'),
                sha256=sha(p), dhash=str(dhash(photo(p))),
                group=f'makeup_person_{i:02d}' if condition != 'wrinkles' else f'wrinkle_{i:03d}',
                split='', external_overlap='', roi_status='', crops=''))
    # Index annotations are tied to this reviewed snapshot. Never silently reuse
    # them when files are added, renamed, reordered or replaced.
    previous=OUT/'annotations.csv'
    if previous.exists():
        old=read_csv(previous)
        expected=[(r['image'],r['sha256']) for r in old]
        observed=[(r['image'],r['sha256']) for r in records]
        if expected!=observed:
            raise RuntimeError('Input inventory changed: re-review visual labels before preparing a new experiment.')
    # Merge exact/perceptual duplicates and manually observed same-source pairs.
    groups = near_groups([int(r['dhash']) for r in records], thr=5)
    parent = list(range(len(records)))
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]; x = parent[x]
        return x
    def union(a,b): parent[find(b)] = find(a)
    by_group = {}; by_near = {}
    for i, r in enumerate(records):
        for key, mapping in ((r['group'], by_group), (groups[i], by_near)):
            if key in mapping: union(i, mapping[key])
            else: mapping[key] = i
    wr = {r['index']:i for i,r in enumerate(records) if r['condition']=='wrinkles'}
    for a,b in [(75,240),(98,116),(421,422),(426,427),(403,404)]: union(wr[a],wr[b])
    for i,r in enumerate(records): r['group'] = f'group_{find(i):04d}'
    # Check additions against every prior external source (exact bytes + dHash).
    ext_paths = sorted(set(p for base in ('data/external_face','data/external_driver','data/external_cropped')
                           for p in (ROOT/base).rglob('*') if p.is_file() and p.suffix.lower() in EXT))
    ext_cache=OUT/'external_hashes.json'
    if ext_cache.exists(): external=json.loads(ext_cache.read_text())
    else:
        external=[]
        for j,p in enumerate(ext_paths):
            try: external.append(dict(image=str(p),sha256=sha(p),dhash=dhash(photo(p))))
            except Exception as e: raise RuntimeError(f'Unreadable external image {p}') from e
            if j%1000==0: print('external hash',j,len(ext_paths),flush=True)
        ext_cache.write_text(json.dumps(external),encoding='utf-8')
    for r in records:
        matches=[e['image'] for e in external if e['sha256']==r['sha256'] or (int(e['dhash'])^int(r['dhash'])).bit_count()<=5]
        r['external_overlap']='|'.join(matches)
    blocked={r['group'] for r in records if r['external_overlap']}
    rng=random.Random(SEED)
    for task in ('eyes','yawn'):
        gs=sorted({r['group'] for r in records if r['task']==task and r['actual']!='UNKNOWN' and r['group'] not in blocked})
        rng.shuffle(gs); n=len(gs); mapping={g:'train' if i<int(n*.6) else 'val' if i<int(n*.8) else 'test' for i,g in enumerate(gs)}
        for r in records:
            if r['task']==task: r['split']=mapping.get(r['group'],'excluded') if r['actual']!='UNKNOWN' else 'excluded'
    # Reuse the exact existing extractors. Raw patches are never forced into the mouth classifier.
    ex=importlib.import_module('12_external_eval'); net,mouth,tf,dev=ex.team_yawn(ROOT/'teammate_yawn')
    del net
    with mp.solutions.face_mesh.FaceMesh(static_image_mode=True,max_num_faces=1,refine_landmarks=True) as mesh:
        for i,r in enumerate(records):
            im=photo(r['image'])
            if r['task']=='eyes': crops=crop_eyes(im,mesh); status='ok' if crops else 'no_face'
            else:
                result=mouth.extract(im); status=result['status']; crops=[result['crop']] if status=='ok' else []
            r['roi_status']=status
            paths=[]
            for j,crop in enumerate(crops):
                p=OUT/'roi'/r['condition']/f"{r['index']:04d}_{j}.png";p.parent.mkdir(parents=True,exist_ok=True);crop.save(p);paths.append(str(p))
            r['crops']='|'.join(paths)
            if i%50==0: print('added ROI',i,len(records),flush=True)
    mouth.close()
    write_csv(OUT/'annotations.csv',records)
    summary=dict(total=len(records),conditions=dict(Counter(r['condition'] for r in records)),
        labels=dict(Counter(r['actual'] for r in records)),overlap_images=sum(bool(r['external_overlap']) for r in records),
        usable=dict(Counter(r['task']+'_'+r['split'] for r in records if r['split']!='excluded' and r['roi_status']=='ok')),
        roi_success=dict(Counter(r['condition'] for r in records if r['roi_status']=='ok')),
        note='Assistant visual annotations; not independently human verified. Uncertain/skin-only samples excluded. New additions contain no confirmed positive examples. Person independence beyond known pairs is not guaranteed.')
    (OUT/'data_audit.json').write_text(json.dumps(summary,indent=2),encoding='utf-8');print(summary,flush=True)


def load_models(task):
    if task=='eyes':
        from ultralytics import YOLO
        from ultralytics.data.augment import classify_transforms
        wrapper=YOLO(str(ROOT/'runs/eyes_yolo26n-cls/weights/best.pt'))
        net=wrapper.model.float().eval(); head=net.model[-1].linear
        assert wrapper.names=={0:'Closed',1:'Open'},wrapper.names
        return net,head,classify_transforms(224),wrapper
    ex=importlib.import_module('12_external_eval')
    ex.team_yawn  # shared model module loader, without constructing another detector
    mod=importlib.import_module('20_yawn_preprocess_ablation')
    net,tf,device=mod.load_yawn_model(ROOT/'teammate_yawn')
    return net,net.fc,tf,None


def sample_records(task):
    rows=[]
    # Internal training/validation/test retain their original membership.
    base=ROOT/('data/processed/eyes' if task=='eyes' else 'teammate_yawn/data/mouth_wide_fixed/yawn')
    for split in ('train','val','test'):
        for p in sorted((base/split).rglob('*')):
            if p.is_file() and p.suffix.lower() in EXT:
                rows.append(dict(image=str(p),actual=p.parent.name,condition='general',dataset='internal',split=split,group=str(p),crops=str(p),status='ok'))
    for r in read_csv(OUT/'annotations.csv'):
        if r['task']==task and r['split']!='excluded':
            rows.append(dict(image=r['image'],actual=r['actual'],condition=r['condition'],dataset='added',split=r['split'],group=r['group'],crops=r['crops'],status=r['roi_status']))
    return rows


def external_records(task):
    rows=[]
    if task=='eyes':
        with mp.solutions.face_mesh.FaceMesh(static_image_mode=True,max_num_faces=1,refine_landmarks=True) as mesh:
            for dataset in ('cew','simuletic','webcam'):
                base=ROOT/({'cew':'data/external_face/eyes/cew','simuletic':'data/external_face/eyes/simuletic','webcam':'data/webcam_test/eyes'}[dataset])
                if dataset=='simuletic' and not base.exists():
                    sr=read_csv(ROOT/'reports/simuletic/external_predictions.csv');paths=sorted({Path(r['image']) for r in sr})
                else: paths=sorted(p for p in base.rglob('*') if p.is_file() and p.suffix.lower() in EXT and p.parent.name in ('Open','Closed'))
                for i,p in enumerate(paths):
                    crops=crop_eyes(photo(p),mesh);cp=[]
                    for j,crop in enumerate(crops):
                        dest=OUT/'roi'/dataset/f'{i:05d}_{j}.png';dest.parent.mkdir(parents=True,exist_ok=True);crop.save(dest);cp.append(str(dest))
                    rows.append(dict(image=str(p),actual=p.parent.name,condition=dataset,dataset=dataset,split='external',group=str(p),crops='|'.join(cp),status='ok' if cp else 'no_face'))
                    if i%400==0: print('external ROI',dataset,i,len(paths),flush=True)
    else:
        source=importlib.import_module('20_yawn_preprocess_ablation')
        ix=source.source_index(ROOT/'data/external_driver')
        for r in source.read_input(ROOT/'reports/yawn_roboflow/external_predictions.csv'):
            idx=ix[str(Path(r['image']).resolve()).casefold()]
            crop=ROOT/'reports/yawn_roboflow/roi/yawn'/r['condition']/r['actual']/f'{idx:05d}.png'
            if r['status']=='OK': assert crop.exists(),crop
            rows.append(dict(image=r['image'],actual=r['actual'],condition='roboflow',dataset='roboflow',split='external',group=r['image'],crops=str(crop) if r['status']=='OK' else '',status='ok' if r['status']=='OK' else 'no_face'))
    return rows


def features(task,net,head,tf,records,name):
    cache=OUT/f'{task}_{name}_features.pt'
    baseline=ROOT/('runs/eyes_yolo26n-cls/weights/best.pt' if task=='eyes' else 'teammate_yawn/runs/03_yawn_resnet18/weights/03_best.pt')
    preprocessing=ROOT/('15_roi_ablation.py' if task=='eyes' else 'teammate_yawn/03_yawn_common.py')
    signature=hashlib.sha256(json.dumps(dict(records=records,weights_sha256=sha(baseline),
        preprocessing_sha256=sha(preprocessing),format_version=2),sort_keys=True).encode()).hexdigest()
    if cache.exists():
        ck=torch.load(cache,weights_only=True)
        if ck['signature']==signature:return ck['features'],ck['owners']
    extracted=[];owners=[];batch=[];ids=[];hook_output=[]
    hook=head.register_forward_pre_hook(lambda module,args: hook_output.append(args[0].detach().cpu()))
    ab=importlib.import_module('15_roi_ablation')
    def flush():
        if not batch:return
        with torch.inference_mode():net(torch.stack(batch))
        extracted.append(hook_output.pop());owners.extend(ids);batch.clear();ids.clear()
    for i,r in enumerate(records):
        for p in filter(None,r['crops'].split('|')):
            with Image.open(p) as im: im=im.convert('RGB')
            if task=='eyes': im=to_square(ab.make_variants(im)['clahe'],224)
            batch.append(tf(im));ids.append(i)
            if len(batch)>=32:flush()
        if i%400==0:print(task,name,'features',i,len(records),flush=True)
    flush();hook.remove();x=torch.cat(extracted)
    torch.save(dict(features=x,owners=owners,signature=signature),cache)
    return x,owners


def metrics(rows,prob,threshold,task):
    positive='Closed' if task=='eyes' else 'yawn'; negative='Open' if task=='eyes' else 'no_yawn'
    tp=tn=fp=fn=unknown=0
    for r,p in zip(rows,prob):
        pred='UNKNOWN' if p is None else positive if p>=threshold else negative
        tp+=r['actual']==positive and pred==positive
        tn+=r['actual']==negative and pred==negative
        fp+=r['actual']==negative and pred==positive
        fn+=r['actual']==positive and pred!=positive
        unknown+=pred=='UNKNOWN'
    n=len(rows);pos=sum(r['actual']==positive for r in rows);neg=n-pos
    recall=tp/pos if pos else None;precision=tp/(tp+fp) if tp+fp else 0
    return dict(n=n,positive=pos,negative=neg,correct=tp+tn,unknown=unknown,tp=tp,tn=tn,fp=fp,fn_including_unknown=fn,
                accuracy=(tp+tn)/n if n else None,valid_accuracy=(tp+tn)/(n-unknown) if n>unknown else None,
                recall=recall,fpr=fp/neg if neg else None,f1=2*precision*recall/(precision+recall) if recall and precision else 0 if pos else None)


def probabilities(head,x,owners,n,task):
    with torch.no_grad():ps=head(x).softmax(1)[:,0 if task=='eyes' else 1].tolist()
    sums=defaultdict(list)
    for owner,p in zip(owners,ps):sums[owner].append(p)
    return [float(np.mean(sums[i])) if i in sums else None for i in range(n)]


def fit(task):
    random.seed(SEED);np.random.seed(SEED);torch.manual_seed(SEED);torch.set_num_threads(4)
    net,head,tf,wrapper=load_models(task);records=sample_records(task)
    write_csv(OUT/f'{task}_samples.csv',records)
    x,owners=features(task,net,head,tf,records,'development')
    labels=['Closed','Open'] if task=='eyes' else ['no_yawn','yawn']
    y=torch.tensor([labels.index(records[i]['actual']) for i in owners])
    idx={}
    for split in ('train','val'):
        for dataset in ('internal','added'):
            idx[split,dataset]=torch.tensor([j for j,i in enumerate(owners) if records[i]['split']==split and records[i]['dataset']==dataset],dtype=torch.long)
            assert len(idx[split,dataset]),(task,split,dataset)
    candidate=copy.deepcopy(head).float().requires_grad_(True);initial=copy.deepcopy(head.state_dict())
    optimizer=torch.optim.AdamW(candidate.parameters(),lr=1e-4,weight_decay=.01)
    criterion=torch.nn.CrossEntropyLoss()
    def loss_on(split):
        return sum(weight*criterion(candidate(x[ix]),y[ix]) for dataset,weight in [('internal',.8),('added',.2)] if len(ix:=idx[split,dataset]))
    with torch.no_grad():baseline_loss=float(loss_on('val'))
    history=[];best_loss=float('inf');best=None;best_epoch=0
    for epoch in range(1,41):
        candidate.train();optimizer.zero_grad();loss=loss_on('train');loss.backward();optimizer.step();candidate.eval()
        with torch.no_grad():vl=float(loss_on('val'))
        history.append(dict(epoch=epoch,train_loss=float(loss.detach()),validation_loss=vl))
        if vl<best_loss:best_loss=vl;best=copy.deepcopy(candidate.state_dict());best_epoch=epoch
    candidate.load_state_dict(best)
    write_csv(OUT/f'{task}_training.csv',history)
    # Frozen backbone, only the final linear classification layer is updated.
    head.load_state_dict(best);RUN.mkdir(parents=True,exist_ok=True)
    weight_path=RUN/f'{task}_head_finetuned.pt'
    if task=='eyes':
        # Preserve FP32 so saved inference matches the measured candidate exactly.
        checkpoint=copy.deepcopy(wrapper.ckpt)
        checkpoint['model']=copy.deepcopy(net).cpu().float()
        checkpoint['ema']=None
        checkpoint['hardcase_finetune']=dict(seed=SEED,epoch=best_epoch,scope='linear_only')
        torch.save(checkpoint,weight_path)
    else:
        ck=torch.load(ROOT/'teammate_yawn/runs/03_yawn_resnet18/weights/03_best.pt',map_location='cpu',weights_only=True)
        ck['model_state_dict']=net.state_dict();ck['hardcase_finetune']=dict(seed=SEED,epoch=best_epoch,scope='fc_only');torch.save(ck,weight_path)
    head.load_state_dict(initial)
    external_csv=OUT/f'{task}_external_samples.csv'
    external=read_csv(external_csv) if external_csv.exists() else external_records(task)
    write_csv(external_csv,external)
    xe,oe=features(task,net,head,tf,external,'external')
    all_rows=records+external;xx=torch.cat([x,xe]);oo=owners+[i+len(records) for i in oe]
    before=probabilities(head,xx,oo,len(all_rows),task)
    after=probabilities(candidate,xx,oo,len(all_rows),task)
    thresholds=[.25] if task=='eyes' else [.10,.15]
    summaries=[];predictions=[]
    groups=defaultdict(list)
    for i,r in enumerate(all_rows):
        groups[(r['dataset'],r['split'],'ALL')].append(i)
        if r['dataset']=='added':groups[(r['dataset'],r['split'],r['condition'])].append(i)
        predictions.append(dict(**r,before_probability=before[i],after_probability=after[i]))
    for (dataset,split,condition),ix in groups.items():
        for threshold in thresholds:
            for phase,ps in [('before',before),('after',after)]:
                m=metrics([all_rows[i] for i in ix],[ps[i] for i in ix],threshold,task)
                summaries.append(dict(task=task,dataset=dataset,split=split,condition=condition,threshold=threshold,phase=phase,**m))
    write_csv(OUT/f'{task}_predictions.csv',predictions);write_csv(OUT/f'{task}_summary.csv',summaries)
    config=dict(task=task,seed=SEED,method='frozen_backbone_final_linear_layer_finetune',epochs=40,best_epoch=best_epoch,
        lr=.0001,weight_decay=.01,internal_loss_weight=.8,added_loss_weight=.2,
        baseline_validation_loss=baseline_loss,best_validation_loss=best_loss,thresholds=thresholds,
        trainable_parameters=sum(p.numel() for p in candidate.parameters()),weights=str(weight_path),sha256=sha(weight_path),
        selection='minimum validation loss; external/test never used for epoch selection',
        input='MediaPipe eye ROI then CLAHE' if task=='eyes' else 'MediaPipe mouth ROI then RGB',
        limitation='new labels visually reviewed by assistant; new data has no confirmed positive examples; source identity independence incomplete')
    (OUT/f'{task}_config.json').write_text(json.dumps(config,indent=2),encoding='utf-8')
    print(json.dumps(config,indent=2),flush=True)
    for r in summaries:
        if r['split']=='external' or (r['dataset']=='added' and r['split']=='test'):print(r,flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['prepare','eyes','yawn']);args=ap.parse_args()
    torch.set_num_threads(4)
    if args.stage=='prepare':prepare()
    else:fit(args.stage)

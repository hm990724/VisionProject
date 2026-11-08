"""Conservative ResNet18 yawn fine-tune: verified NITYMED event frames + ORIGINAL train.

Does not assume unreviewed video frames are no_yawn. Preserves checkpoint. Does not deploy.
Evaluates baseline and candidate on the exact same validation/test samples (ROI failures count as wrong).

Project root needs:
 - teammate_yawn/03_yawn_common.py, teammate_yawn/runs/03_yawn_resnet18/weights/03_best.pt
 - teammate_yawn/models/face_landmarker.task (or checkpoint-verified file)
 - data/processed/yawn/train/{yawn,no_yawn} and val/test in the same layout
 - data/yawn_mp/Yawning/Female/HDTV720/P1042778_720.mp4
Optional external Roboflow ROI cache from 12_external_eval --save-roi.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import importlib.util
import json
import math
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CLASSES = ('no_yawn', 'yawn')
EXTS = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')


def intervals(text):
    result = []
    for part in text.split(','):
        if not part.strip():
            continue
        fields = part.strip().split('-')
        if len(fields) != 2:
            raise ValueError('구간 입력 예: 30.4-34.4,41.6-46,50.4-54.4')
        a, b = [float(x.strip()) for x in fields]
        if not (math.isfinite(a) and math.isfinite(b) and 0 <= a < b):
            raise ValueError('구간 시작/끝 시간 오류: '+part)
        result.append((a, b))
    if not result or any(result[i][0] < result[i-1][1] for i in range(1,len(result))):
        raise ValueError('구간이 없거나 겹침/순서 오류')
    return result


def load_team(team_dir):
    helper = team_dir/'03_yawn_common.py'
    weights = team_dir/'runs/03_yawn_resnet18/weights/03_best.pt'
    for f in (helper, weights):
        if not f.is_file():
            raise FileNotFoundError(f'팀원 모델 파일 없음: {f}')
    if str(team_dir.resolve()) not in sys.path:
        sys.path.insert(0, str(team_dir.resolve()))
    spec = importlib.util.spec_from_file_location('team_nitymed_ft30', helper)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    dev = mod.choose_device('cpu')  # CPU stable/reproducible
    net, checkpoint = mod.load_checkpoint(weights, dev)
    if list(checkpoint.get('class_names', CLASSES)) != list(CLASSES):
        raise ValueError('팀원 가중치 클래스 순서 불일치. 0=no_yawn, 1=yawn 필수')
    face_file = mod.verify_face_model(team_dir/'models/face_landmarker.task', checkpoint)
    extractor = mod.MouthExtractor(face_file, mod.CropConfig(**checkpoint['crop_config']))
    net.eval()
    return mod, net, checkpoint, weights, extractor


def sources(split_dir):
    rows=[]
    for label in CLASSES:
        folder=split_dir/label
        if not folder.is_dir():
            raise FileNotFoundError(f'클래스 폴더 없음: {folder}')
        for p in sorted(folder.rglob('*')):
            if p.suffix.lower() in EXTS and p.is_file():
                rows.append((p, CLASSES.index(label)))
    return rows


def to_roi_cache(items, extractor, cache_dir, split):
    from PIL import Image, ImageOps
    dest=cache_dir/split
    dest.mkdir(parents=True, exist_ok=True)
    out=[]; failed=0
    for idx,(p,label) in enumerate(items):
        digest=hashlib.sha256(str(p.resolve()).encode('utf8')).hexdigest()[:16]
        target=dest/CLASSES[label]/f'{digest}.png'
        if target.is_file():
            out.append((target,label))
            continue
        try:
            with Image.open(p) as source:
                im=ImageOps.exif_transpose(source).convert('RGB')
                if max(im.size)>960:
                    im.thumbnail((960,960))
            extracted=extractor.extract(im)
            roi=extracted.get('crop') if str(extracted.get('status','')).lower()=='ok' else None
            if roi is None:
                out.append((None,label));failed+=1
            else:
                target.parent.mkdir(parents=True,exist_ok=True)
                roi.convert('RGB').save(target)
                out.append((target,label))
        except Exception as exc:
            print('[ROI_WARN]',p.name,type(exc).__name__,str(exc)[:80])
            failed+=1;out.append((None,label))
        if (idx+1)%250==0:
            print(f'[ROI] {split}: {idx+1}/{len(items)}')
    print(f'[ROI] {split}: {len(out)-failed}/{len(out)} 신규추출 실패 {failed} (기존 캐시 재사용)')
    return out


def nitymed_verified(video, extractor, cache_dir, spans, every, margin):
    import cv2
    from PIL import Image
    cap=cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f'영상 열기 실패: {video}')
    fps=cap.get(cv2.CAP_PROP_FPS)
    frame_n=cap.get(cv2.CAP_PROP_FRAME_COUNT)
    if fps<=0:
        cap.release();raise RuntimeError('영상 FPS 확인 실패')
    samples=[];used=set();events=[]
    try:
        for j,(a,b) in enumerate(spans,1):
            count=0
            t=a+margin
            while t<=b-margin+1e-5:
                idx=min(int(round(t*fps)),max(0,int(frame_n)-1))
                t+=every
                if idx in used: continue
                used.add(idx)
                target=cache_dir/'nitymed_verified_yawn'/f'{video.stem}_{idx:07d}.png'
                if not target.is_file():
                    cap.set(cv2.CAP_PROP_POS_FRAMES,idx)
                    ok,frame=cap.read()
                    if not ok:continue
                    im=Image.fromarray(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB))
                    if max(im.size)>960: im.thumbnail((960,960))
                    try:
                        result=extractor.extract(im)
                        roi=result.get('crop') if str(result.get('status','')).lower()=='ok' else None
                        if roi is None: continue
                        target.parent.mkdir(parents=True,exist_ok=True)
                        roi.convert('RGB').save(target)
                    except Exception as exc:
                        print('[NITYMED ROI_WARN]',idx,type(exc).__name__,exc)
                        continue
                samples.append((target,1))
                count+=1
            events.append({'interval_s':[a,b],'event_id':j,'usable_frames':count})
    finally:
        cap.release()
    print('[NITYMED] 검수한 실제 하품 이벤트',len(spans),'유효 양성 ROI 프레임',len(samples))
    for x in events: print('  ',x)
    print('[NITYMED] 구간 밖 프레임은 no_yawn으로 자동 라벨링하지 않음')
    return samples,events


def external_saved_rois(project, external_root, csv_file):
    """Recreate EXACT 12_external_eval index; missing ROI remains UNKNOWN (not counted correct)."""
    if not external_root.is_dir() or not csv_file.is_file():
        print('[SKIP] Roboflow 원본 또는 predictions.csv가 없어 외부 비교 생략')
        return []
    original=[]
    for path in sorted((external_root/'yawn').rglob('*')):
        if path.is_file() and path.suffix.lower() in EXTS and path.parent.name in CLASSES:
            rel=path.relative_to(external_root/'yawn').parts
            condition=rel[-3] if len(rel)>=3 else 'normal'
            original.append((path,condition,path.parent.name))
    with csv_file.open(encoding='utf-8-sig',newline='') as f:
        rows=[r for r in csv.DictReader(f) if r.get('task')=='yawn' and r.get('model')=='team_resnet18']
    statuses={str(Path(r['image']).resolve()).casefold():r for r in rows}
    cache=csv_file.parent/'roi'/'yawn'
    if len(statuses)!=len(original):
        print(f'[SKIP] 외부 목록 {len(original)}개 vs CSV {len(statuses)}개가 불일치. 비교 건너뜀')
        return []
    result=[]
    for i,(p,condition,label) in enumerate(original):
        row=statuses.get(str(p.resolve()).casefold())
        if row is None or row.get('actual')!=label:
            print('[SKIP] 외부 정답 라벨 매칭 불가. 비교 건너뜀')
            return []
        roi=cache/condition/label/f'{i:05d}.png'
        # Explicit UNKNOWN for all originally unextracted images.
        if row.get('status','')!='OK':
            result.append((None,CLASSES.index(label)))
        elif not roi.is_file():
            print(f'[SKIP] 기존 외부 ROI 파일 누락: {roi}. 평가값 왜곡 방지')
            return []
        else:
            result.append((roi,CLASSES.index(label)))
    print('[EXTERNAL] Roboflow 전체 비교',len(result),'ROI OK',sum(p is not None for p,_ in result))
    return result


def metric(probs,labels,threshold):
    tp=tn=fp=fn=unk=0
    for p,y in zip(probs,labels):
        if p is None:
            unk+=1
            if y==1:fn+=1
            continue
        pos=p>=threshold
        if y==1:
            if pos:tp+=1
            else:fn+=1
        else:
            if pos:fp+=1
            else:tn+=1
    precision=tp/(tp+fp) if tp+fp else 0.0
    recall=tp/(tp+fn) if tp+fn else 0.0
    f1=2*precision*recall/(precision+recall) if precision+recall else 0.0
    n_negative=sum(y==0 for y in labels)
    return {'total':len(labels),'ROI_unknown':unk,'accuracy':round((tp+tn)/len(labels),5) if labels else 0,
            'recall':round(recall,5),'precision':round(precision,5),'f1':round(f1,5),
            'fp':fp,'fn':fn,'tp':tp,'tn':tn,'fpr':round(fp/n_negative,5) if n_negative else None}


def infer(model,items,transform,batch_size=16):
    import torch
    from PIL import Image
    scores=[None]*len(items); labels=[y for _,y in items]
    # DataLoader is optional; scan in batches; keep original order to count ROI failures.
    for offset in range(0,len(items),batch_size):
        parts=items[offset:offset+batch_size]
        xs=[];positions=[]
        for j,(p,_) in enumerate(parts):
            if p is None:continue
            with Image.open(p) as im:
                xs.append(transform(im.convert('RGB')))
            positions.append(offset+j)
        if xs:
            with torch.inference_mode():
                probs=model(torch.stack(xs)).softmax(1)[:,1].cpu().tolist()
            for i,v in zip(positions,probs): scores[i]=float(v)
    return scores,labels


def train_data_loader(original,new,transform,batch_size,share,seed):
    import torch
    from PIL import Image
    all_items=original+new
    n_old=len(original); n_new=len(new)
    class Dataset(torch.utils.data.Dataset):
        def __len__(self): return len(all_items)
        def __getitem__(self,index):
            p,y=all_items[index]
            with Image.open(p) as im: x=transform(im.convert('RGB'))
            return x,y
    weights=([ (1-share)/n_old ]*n_old)+([share/n_new]*n_new)
    generator=torch.Generator().manual_seed(seed)
    sampler=torch.utils.data.WeightedRandomSampler(weights, num_samples=len(all_items),replacement=True,generator=generator)
    return torch.utils.data.DataLoader(Dataset(),batch_size=batch_size,sampler=sampler,num_workers=0)


def json_out(file,obj):
    file.parent.mkdir(parents=True,exist_ok=True)
    file.write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding='utf8')


def run(args):
    import torch
    root=args.project.resolve()
    video=(root/args.video).resolve()
    team=(root/args.team_dir).resolve()
    original=(root/args.original_data).resolve()
    out=(root/args.output).resolve()
    candidate_path=out/'resnet18_nitymed_candidate_state.pt'
    if candidate_path.is_file() and not args.overwrite and not args.dry_run:
        raise FileExistsError(f'기존 실험 가중치 보호: {candidate_path} (다른 --output 사용 또는 --overwrite)')
    spans=intervals(args.yawns)
    for path in (video,team/'03_yawn_common.py',team/'runs/03_yawn_resnet18/weights/03_best.pt'):
        if not path.is_file(): raise FileNotFoundError(path)
    inputs={s:sources(original/s) for s in ('train','val','test')}
    for s,rows in inputs.items():
        print(f'[INPUT] {s}: {len(rows)} ({sum(y==0 for _,y in rows)} no_yawn / {sum(y==1 for _,y in rows)} yawn)')
    if args.dry_run:
        print('[DRY RUN] 준비 완료. 학습/ROI 추출 안 함. 다음 명령에서 --dry-run 제거')
        print('[NOTICE] 검증된 이벤트 3개 = 독립 영상 1개: 전체 성능 개선 보장 불가')
        return
    random.seed(args.seed); torch.manual_seed(args.seed)
    mod,model,checkpoint,base_weight,extractor=load_team(team)
    cache=out/'roi_cache'
    try:
        training_all=to_roi_cache(inputs['train'],extractor,cache,'train')
        validation=to_roi_cache(inputs['val'],extractor,cache,'val')
        testing=to_roi_cache(inputs['test'],extractor,cache,'test')
        new, events=nitymed_verified(video,extractor,cache,spans,args.every,args.margin)
    finally:
        extractor.close()
    original_train=[x for x in training_all if x[0] is not None]
    if len(original_train)<150 or any(sum(y==c for _,y in original_train)<50 for c in (0,1)):
        raise RuntimeError('원본 train에서 ROI 성공 수가 너무 작음. 팀원 학습 입력과 호환되는지 먼저 확인 필요')
    if len(new)<6:
        raise RuntimeError('NITYMED 확인된 하품 ROI가 6장 미만. 재학습 금지')
    external=external_saved_rois(root,root/args.external_data,root/args.external_csv)
    fixed=[0.10,0.15,0.50]
    transform_val=mod.make_transform(training=False)
    eval_sets={'val':validation,'test':testing}
    if external: eval_sets['roboflow_dev']=external
    baseline={}
    for name,items in eval_sets.items():
        scores,labels=infer(model,items,transform_val,args.batch)
        baseline[name]={f'{t:.2f}':metric(scores,labels,t) for t in fixed}
        print('[BASELINE]',name,'thr=.15',baseline[name]['0.15'])
    out.mkdir(parents=True,exist_ok=True)
    json_out(out/'data_audit.json',{'source_video':str(video),'events':events,
        'source_train':len(inputs['train']),'train_roi_ok':len(original_train),
        'new_verified_yawn_frames':len(new),'new_verified_yawn_events':len(spans),
        'val_count':len(validation),'test_count':len(testing),'external_count':len(external),
        'replayed_positive_share':args.new_share,
        'warning':'Only one NITYMED video; no independent video-level improvement claim.'})
    print(f'[TRAIN] 기존 ROI {len(original_train)}장 + 신규 하품 ROI {len(new)}장 (3개 이벤트 1개 영상)')
    # Freeze layers except layer4/fc. Frozen BN running statistics too.
    active=0
    for name,p in model.named_parameters():
        p.requires_grad=('layer4.' in name or 'fc.' in name)
        active+=int(p.requires_grad)
    if not active: raise RuntimeError('기존 모델이 ResNet18 layer4/fc 구조가 아님. 안전을 위해 중단')
    print('[TRAIN] unfrozen parameter tensors:',active)
    optim=torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),lr=args.lr,weight_decay=1e-4)
    loader=train_data_loader(original_train,new,mod.make_transform(training=True),args.batch,args.new_share,args.seed)
    best_score=(-1.,-1.)
    best_epoch=None; best_state=None; no_gain=0
    log=[]
    for epoch in range(1,args.epochs+1):
        model.train()
        for m in model.modules():
            if isinstance(m,torch.nn.modules.batchnorm._BatchNorm): m.eval()
        total_loss=0.;n=0
        for x,y in loader:
            optim.zero_grad(set_to_none=True)
            loss=torch.nn.functional.cross_entropy(model(x),y)
            loss.backward();optim.step()
            total_loss+=float(loss.item())*len(y);n+=len(y)
        model.eval()
        val_scores,val_labels=infer(model,validation,transform_val,args.batch)
        row=metric(val_scores,val_labels,.15)
        record={'epoch':epoch,'loss':total_loss/max(n,1),'validation_at_0.15':row}
        log.append(record)
        score=(row['f1'],-(row['fpr'] or 0.))
        print('[EPOCH]',epoch,'loss',round(record['loss'],4),'VAL F1@0.15',round(row['f1'],4),'recall',round(row['recall'],4),'fp',row['fp'])
        if score>best_score:
            best_score=score;best_epoch=epoch;no_gain=0
            best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        else:
            no_gain+=1
            if no_gain>=args.patience: break
    if best_state is None: raise RuntimeError('학습 후보 저장 실패')
    model.load_state_dict(best_state)
    model.eval()
    after={}
    for name,items in eval_sets.items():
        scores,labels=infer(model,items,transform_val,args.batch)
        after[name]={f'{t:.2f}':metric(scores,labels,t) for t in fixed}
        print('[AFTER]',name,'thr=.15',after[name]['0.15'])
    # No modification of the original team's checkpoint.
    checkpoint_hash=hashlib.sha256(base_weight.read_bytes()).hexdigest()
    candidate=out/'resnet18_nitymed_candidate_state.pt'
    torch.save({'state_dict':best_state,'class_names':list(CLASSES),'epoch':best_epoch,
                'base_checkpoint_sha256':checkpoint_hash,'train_events':events,
                'note':'EXPERIMENTAL! Do not auto-deploy. Load state_dict into team model.'},candidate)
    report={'status':'experiment_not_deployed',
            'baseline_checkpoint':str(base_weight),'baseline_sha256':checkpoint_hash,
            'candidate_checkpoint':str(candidate),'best_epoch':best_epoch,
            'fixed_thresholds':fixed,'baseline':baseline,'finetuned':after,'training':log,
            'cautions':['Only three confirmed yawns from one NITYMED subject/video; strong overfitting risk.',
                        'Original Kaggle test may not be subject-independent.',
                        'Roboflow was repeatedly used for development; not a pristine final test.',
                        'Only checkpoint choice is based on original validation @0.15; no threshold tuning on test.',
                        'ROI failures count as incorrect and are unchanged by classifier fine-tuning.',
                        'Model should be deployed only after independent multi-person/video tests.']}
    json_out(out/'comparison.json',report)
    print('\n========== SIDE BY SIDE (@0.15) ==========')
    for name in eval_sets:
        b=baseline[name]['0.15'];a=after[name]['0.15']
        print(f'{name:14} BEFORE acc={b["accuracy"]:.2%} recall={b["recall"]:.2%} F1={b["f1"]:.2%} FP={b["fp"]} | AFTER acc={a["accuracy"]:.2%} recall={a["recall"]:.2%} F1={a["f1"]:.2%} FP={a["fp"]}')
    print('[SAVED]',candidate,'\n[SAVED]',out/'comparison.json')
    if hashlib.sha256(base_weight.read_bytes()).hexdigest()!=checkpoint_hash:
        raise RuntimeError('원본 팀 모델 가중치 변경 감지! 즉시 확인')


def main():
    p=argparse.ArgumentParser(description='Fine tune ResNet18 with verified NITYMED yawns + original train; compare before/after')
    p.add_argument('--project',type=Path,default=ROOT)
    p.add_argument('--team-dir',type=Path,default=Path('teammate_yawn'))
    p.add_argument('--original-data',type=Path,default=Path('data/processed/yawn'))
    p.add_argument('--video',type=Path,default=Path('data/yawn_mp/Yawning/Female/HDTV720/P1042778_720.mp4'))
    p.add_argument('--yawns',default='30.4-34.4,41.6-46.0,50.4-54.4')
    p.add_argument('--external-data',type=Path,default=Path('data/external_driver'))
    p.add_argument('--external-csv',type=Path,default=Path('reports/yawn_roboflow/external_predictions.csv'))
    p.add_argument('--output',type=Path,default=Path('runs/nitymed_single_video_finetune'))
    p.add_argument('--every',type=float,default=0.6,help='seconds between verified-positive frames')
    p.add_argument('--margin',type=float,default=0.45,help='drop uncertain transition frames near interval boundaries')
    p.add_argument('--new-share',type=float,default=0.08,help='maximum training minibatch fraction sampled from one new video')
    p.add_argument('--epochs',type=int,default=3)
    p.add_argument('--patience',type=int,default=2)
    p.add_argument('--batch',type=int,default=16)
    p.add_argument('--lr',type=float,default=1e-5)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--dry-run',action='store_true')
    p.add_argument('--overwrite',action='store_true',help='allow replacing candidate of identical output run; original team model never overwritten')
    args=p.parse_args()
    if not 0<args.new_share<=0.20 or not 0<args.lr<=1e-4 or args.epochs<1 or args.batch<1 or args.every<=0 or args.margin<0:
        p.error('invalid training/sampling settings (new-share must be <=0.20)')
    if any(b-a<=2*args.margin for a,b in intervals(args.yawns)):
        p.error('marked yawn intervals too short for selected --margin')
    run(args)

if __name__=='__main__':
    main()

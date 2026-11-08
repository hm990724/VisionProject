"""Fine-tune team ResNet18 on HUMAN-CONFIRMED NITYMED mouth ROI images.

- Group splits by person (when provided), else video.
- Protects original checkpoint; saves to reports/nitymed_video_review/finetune.
- Compares baseline and candidate on one NITYMED held-out split.
- Not a claim of independent real-world performance.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import importlib.util
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT=Path(__file__).resolve().parent
CLASSES=['no_yawn','yawn']


def read_rows(csv_path, review_root):
    rows=list(csv.DictReader(csv_path.open(encoding='utf-8-sig',newline='')))
    valid=[]
    for r in rows:
        if str(r.get('confirmed','')).strip().lower() not in ('1','yes','true'):
            continue
        if r.get('label') not in CLASSES or r.get('roi_status')!='ok':
            continue
        im=(review_root/r['mouth_roi']).resolve()
        if not im.is_file():
            raise FileNotFoundError(f'Confirmed ROI does not exist: {im}')
        r['image_path']=str(im)
        # Same person in multiple videos should be manually entered as subject_id.
        r['group_key']=str(r.get('subject_id','')).strip() or 'VIDEO:'+r['video_id']
        valid.append(r)
    return valid


def group_split(rows, seed, max_trials=5000):
    groups=defaultdict(list)
    for r in rows: groups[r['group_key']].append(r)
    keys=list(groups)
    if len(keys)<8:
        raise RuntimeError(f'Need >=8 distinct subjects/videos to build a meaningful split; got {len(keys)}')
    rgen=random.Random(seed)
    for _ in range(max_trials):
        rgen.shuffle(keys)
        n=len(keys); nt=max(2,round(n*.15)); nv=max(2,round(n*.15))
        testkeys=set(keys[:nt]); valkeys=set(keys[nt:nt+nv]); trainkeys=set(keys[nt+nv:])
        if not trainkeys: continue
        sets={s:[r for k in ks for r in groups[k]] for s,ks in [('train',trainkeys),('val',valkeys),('test',testkeys)]}
        if all(set(r['label'] for r in rr)==set(CLASSES) and len(rr)>=4 for rr in sets.values()):
            return sets
    raise RuntimeError('Could not make all splits contain both labels. Review sample labels or supply more person/video groups.')


def load_team(team_folder):
    source=team_folder/'03_yawn_common.py'
    weights=team_folder/'runs/03_yawn_resnet18/weights/03_best.pt'
    if not source.is_file() or not weights.is_file():
        raise FileNotFoundError(f'Missing team files: {source} or {weights}')
    if str(team_folder.resolve()) not in sys.path:
        sys.path.insert(0,str(team_folder.resolve()))
    spec=importlib.util.spec_from_file_location('nitymed_finetune_team',source)
    mod=importlib.util.module_from_spec(spec)
    sys.modules[spec.name]=mod
    spec.loader.exec_module(mod)
    device=mod.choose_device('cpu')
    model,ckpt=mod.load_checkpoint(weights,device)
    if list(ckpt.get('class_names',CLASSES)) != CLASSES:
        raise ValueError(f'Invalid checkpoint labels: {ckpt.get("class_names")}')
    return mod,model,weights


def original_samples(folder):
    if folder is None:return []
    if not folder.is_dir():raise FileNotFoundError(folder)
    samples=[]
    for label in CLASSES:
        d=folder/label
        if d.is_dir():
            samples.extend((str(p.resolve()),CLASSES.index(label)) for p in sorted(d.rglob('*'))
                           if p.is_file() and p.suffix.lower() in ('.jpg','.jpeg','.png','.webp'))
    if not samples:raise RuntimeError(f'No original samples found: {folder}')
    return samples


def rows_to_samples(rows):
    return [(r['image_path'],CLASSES.index(r['label'])) for r in rows]


def make_loader(samples, mod, batch, training):
    import torch
    from PIL import Image
    class Dataset(torch.utils.data.Dataset):
        def __init__(self):
            self.s=samples; self.trans=mod.make_transform(training=training)
        def __len__(self):return len(self.s)
        def __getitem__(self,idx):
            im,label=self.s[idx]
            with Image.open(im) as image:
                x=self.trans(image.convert('RGB'))
            return x,label
    return torch.utils.data.DataLoader(Dataset(),batch_size=batch,shuffle=training,num_workers=0)


def predictions(model,loader):
    import torch
    model.eval(); scores=[];labels=[]
    with torch.inference_mode():
        for x,y in loader:
            p=model(x).softmax(1)[:,1].cpu().tolist()
            scores.extend(p); labels.extend(y.tolist())
    return scores,labels


def metrics(scores,labels,thr):
    tp=tn=fp=fn=0
    for p,y in zip(scores,labels):
        pred=int(p>=thr)
        if y==1 and pred==1:tp+=1
        if y==1 and pred==0:fn+=1
        if y==0 and pred==1:fp+=1
        if y==0 and pred==0:tn+=1
    precision=tp/(tp+fp) if (tp+fp) else 0
    recall=tp/(tp+fn) if (tp+fn) else 0
    f1=2*precision*recall/(precision+recall) if (precision+recall) else 0
    return {'accuracy':(tp+tn)/max(1,len(labels)),'precision':precision,'recall':recall,'f1':f1,
            'fpr':fp/max(1,fp+tn),'tp':tp,'tn':tn,'fp':fp,'fn':fn,'total':len(labels)}


def choose_threshold(scores,labels,max_fpr):
    choices=[.05,.1,.15,.2,.25,.3,.4,.5,.6,.7,.8,.9]
    items=[(thr,metrics(scores,labels,thr)) for thr in choices]
    admissible=[x for x in items if x[1]['fpr']<=max_fpr]
    best=max(admissible or items,key=lambda x:(x[1]['f1'],x[1]['recall'],-x[1]['fpr']))
    return best,items


def json_dump(obj,path):
    path.write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding='utf8')


def main():
    p=argparse.ArgumentParser(description='Human-reviewed NITYMED ResNet18 fine-tune and held-out video comparison')
    p.add_argument('--review-root',type=Path,default=ROOT/'reports/nitymed_video_review')
    p.add_argument('--team-dir',type=Path,default=ROOT/'teammate_yawn')
    p.add_argument('--original-train',type=Path,default=None,help='Existing ROI training folder with yawn/no_yawn subdirs; optional but recommended')
    p.add_argument('--output',type=Path,default=ROOT/'runs/nitymed_finetune_experiment')
    p.add_argument('--epochs',type=int,default=6)
    p.add_argument('--batch',type=int,default=16)
    p.add_argument('--lr',type=float,default=1e-5)
    p.add_argument('--patience',type=int,default=3)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--max-fpr',type=float,default=.15)
    p.add_argument('--dry-run',action='store_true',help='Validate data/splits; no model training')
    args=p.parse_args()
    manifest=args.review_root/'annotation_manifest.csv'
    if not manifest.is_file():p.error(f'Annotation CSV missing: {manifest}')
    rows=read_rows(manifest,args.review_root)
    counts=Counter(r['label'] for r in rows)
    print('Human-confirmed labels:',len(rows),dict(counts))
    if any(counts.get(k,0)<20 for k in CLASSES):
        p.error('Need at least 20 confirmed examples of EACH class. Never use unreviewed pseudo-labels.')
    splits=group_split(rows,args.seed)
    for s,rr in splits.items():print(s,len(rr),dict(Counter(r['label'] for r in rr)),'groups',len({r['group_key'] for r in rr}))
    all_groups=[{r['group_key'] for r in rr} for rr in splits.values()]
    assert not (all_groups[0]&all_groups[1] or all_groups[0]&all_groups[2] or all_groups[1]&all_groups[2])
    original=original_samples(args.original_train)
    print('Original-train ROI images mixed into TRAIN only:',len(original))
    if not original:print('[CAUTION] Fine-tuning without original training examples can cause catastrophic forgetting.')
    if any(not r.get('subject_id','').strip() for r in rows):
        print('[LIMITATION] Some subject IDs missing: group split is video-level, not guaranteed person-independent.')
    args.output.mkdir(parents=True,exist_ok=True)
    splitrows=[]
    for s,rr in splits.items():
        splitrows.extend(dict(video_id=r['video_id'],subject_id=r.get('subject_id',''),time_seconds=r['time_seconds'],
                              label=r['label'],image_path=r['image_path'],split=s) for r in rr)
    with (args.output/'split_manifest.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(splitrows[0]));w.writeheader();w.writerows(splitrows)
    if args.dry_run:
        print('DRY RUN complete: no weights modified or trained');return
    import torch
    torch.manual_seed(args.seed)
    mod,model,baseline_weights=load_team(args.team_dir.resolve())
    original_hash=hashlib.sha256(baseline_weights.read_bytes()).hexdigest()
    train_items=rows_to_samples(splits['train'])+original
    val_items=rows_to_samples(splits['val']); test_items=rows_to_samples(splits['test'])
    train_loader=make_loader(train_items,mod,args.batch,True)
    val_loader=make_loader(val_items,mod,args.batch,False)
    test_loader=make_loader(test_items,mod,args.batch,False)
    base_val=predictions(model,val_loader)
    base_test=predictions(model,test_loader)
    (base_thr,base_val_metric),base_candidates=choose_threshold(*base_val,args.max_fpr)
    print('BASELINE val threshold',base_thr,'F1',base_val_metric['f1'])
    print('BASELINE test',metrics(*base_test,base_thr))
    optimizer=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=1e-4)
    criterion=torch.nn.CrossEntropyLoss()
    best=-1;bad_epochs=0;best_epoch=0
    best_file=args.output/'fine_tuned_state.pt'
    for epoch in range(1,args.epochs+1):
        model.train();total_loss=0;num=0
        for x,y in train_loader:
            optimizer.zero_grad(set_to_none=True)
            loss=criterion(model(x),y)
            loss.backward()
            optimizer.step()
            total_loss+=float(loss.item())*len(y);num+=len(y)
        vs,vl=predictions(model,val_loader)
        report=metrics(vs,vl,.5)
        # Select on validation at fixed threshold; test is never used for training or selection.
        print('EPOCH',epoch,'loss',round(total_loss/max(1,num),4),'val_F1@0.5',round(report['f1'],4))
        if report['f1']>best+1e-8:
            best=report['f1'];bad_epochs=0;best_epoch=epoch
            torch.save({'state_dict':model.state_dict(),'class_names':CLASSES,'base_weights':str(baseline_weights),
                        'epoch':epoch,'validation_f1_at_half':best,'notes':'experiment: not auto-deployed'},best_file)
        else:
            bad_epochs+=1
            if bad_epochs>=args.patience:break
    best_state=torch.load(best_file,map_location='cpu',weights_only=True)
    model.load_state_dict(best_state['state_dict'])
    fine_val=predictions(model,val_loader)
    fine_test=predictions(model,test_loader)
    (ft_thr,ft_val_metric),ft_candidates=choose_threshold(*fine_val,args.max_fpr)
    result={'baseline':{'selected_threshold_from_val':base_thr,'validation':base_val_metric,
                        'test':metrics(*base_test,base_thr)},
            'finetuned':{'selected_threshold_from_val':ft_thr,'validation':ft_val_metric,
                        'test':metrics(*fine_test,ft_thr),'best_epoch':best_epoch},
            'notes':['Thresholds selected on validation only; test is a single within-NITYMED holdout.',
                     'No original image or checkpoint was overwritten.',
                     'NITYMED is nighttime material; results may not generalize to daylight.',
                     'If subject IDs are blank the split is video-level, not person-independent.',
                     'Roboflow was repeatedly used for development; it is not a pristine final test.']}
    json_dump(result,args.output/'comparison.json')
    final_hash=hashlib.sha256(baseline_weights.read_bytes()).hexdigest()
    if original_hash!=final_hash:
        raise RuntimeError('Original model checkpoint changed unexpectedly: stop and investigate')
    print('\nBASELINE test:',result['baseline']['test'])
    print('FINETUNED test:',result['finetuned']['test'])
    print('Saved:',best_file,'and comparison.json')
    print('Do NOT replace Streamlit model unless event-level and external-condition results are better.')

if __name__=='__main__':main()

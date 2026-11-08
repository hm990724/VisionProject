"""Verify reproducibility and saved checkpoints; run a replay-only control.

The control is explanatory, not used to select deployment weights or tune thresholds.
"""
import copy
import importlib
import json
from collections import defaultdict
from pathlib import Path
import numpy as np
import torch
from PIL import Image

m=importlib.import_module('22_hardcase_finetune')
ROOT,OUT=m.ROOT,m.OUT


def verify():
    torch.set_num_threads(4)
    annotations=m.read_csv(OUT/'annotations.csv')
    splits=defaultdict(set)
    for r in annotations:
        if r['split']!='excluded':splits[r['group']].add(r['split'])
        assert not (r['split']=='train' and (r['actual']=='UNKNOWN' or r['external_overlap']))
    assert all(len(v)==1 for v in splits.values()),'group leakage'
    reports={};checks=[];control_all=[]
    for task in ('eyes','yawn'):
        torch.manual_seed(42)
        net,head,tf,wrapper=m.load_models(task)
        development=m.read_csv(OUT/f'{task}_samples.csv');external=m.read_csv(OUT/f'{task}_external_samples.csv')
        records=development+external
        c=torch.load(OUT/f'{task}_development_features.pt',weights_only=True);e=torch.load(OUT/f'{task}_external_features.pt',weights_only=True)
        x=c['features'];owners=c['owners'];xx=torch.cat([x,e['features']]);oo=owners+[i+len(development) for i in e['owners']]
        labels=['Closed','Open'] if task=='eyes' else ['no_yawn','yawn'];y=torch.tensor([labels.index(development[i]['actual']) for i in owners])
        # Same initialization, 40 optimizer updates, LR, validation objective.
        # Only difference: remove additional training data from training objective.
        model=copy.deepcopy(head).requires_grad_(True)
        optimizer=torch.optim.AdamW(model.parameters(),lr=.0001,weight_decay=.01)
        train=torch.tensor([j for j,i in enumerate(owners) if development[i]['split']=='train' and development[i]['dataset']=='internal'])
        val={ds:torch.tensor([j for j,i in enumerate(owners) if development[i]['split']=='val' and development[i]['dataset']==ds]) for ds in ('internal','added')}
        ce=torch.nn.CrossEntropyLoss();best_loss=float('inf');best=None;best_epoch=0
        for epoch in range(1,41):
            optimizer.zero_grad();loss=ce(model(x[train]),y[train]);loss.backward();optimizer.step()
            with torch.no_grad():vl=float(.8*ce(model(x[val['internal']]),y[val['internal']])+.2*ce(model(x[val['added']]),y[val['added']]))
            if vl<best_loss:best_loss=vl;best=copy.deepcopy(model.state_dict());best_epoch=epoch
        model.load_state_dict(best)
        pp=m.probabilities(model,xx,oo,len(records),task)
        groups=defaultdict(list)
        for i,r in enumerate(records):groups[r['dataset'],r['split']].append(i)
        control=[]
        for (dataset,split),ix in groups.items():
            for t in ([.25] if task=='eyes' else [.1,.15]):
                control.append(dict(task=task,dataset=dataset,split=split,condition='ALL',threshold=t,phase='general_only_control',
                                    **m.metrics([records[i] for i in ix],[pp[i] for i in ix],t,task)))
        control_all.extend(control)
        torch.save(dict(state_dict=best,best_epoch=best_epoch,validation_loss=best_loss,scope='analysis_control_only'),m.RUN/f'{task}_general_only_control_head.pt')
        summaries=m.read_csv(OUT/f'{task}_summary.csv');pred=m.read_csv(OUT/f'{task}_predictions.csv')
        baseline=next(r for r in summaries if r['dataset']==('cew' if task=='eyes' else 'roboflow') and r['phase']=='before')
        assert int(baseline['correct'])==(2141 if task=='eyes' else 281)
        assert int(baseline['unknown'])==(34 if task=='eyes' else 49)
        # Check baseline probabilities against the old full evaluation, beyond rounded headline accuracy.
        old=m.read_csv(ROOT/('reports/cew/roi_ablation/predictions.csv' if task=='eyes' else 'reports/yawn_roboflow/external_predictions.csv'))
        if task=='eyes':old=[r for r in old if r['model']=='yolo26n' and r['variant']=='clahe']
        else:old=[r for r in old if r['model']=='team_resnet18']
        ix={r['image'].casefold():r for r in old};errors=[]
        for r in pred:
            if r['image'].casefold() in ix and r['before_probability']:
                previous=ix[r['image'].casefold()].get('closed_probability' if task=='eyes' else 'positive_probability','')
                if previous:errors.append(abs(float(previous)-float(r['before_probability'])))
        assert max(errors)<2e-4,(task,max(errors))
        # Reload the actual standalone output file, not the in-memory trained head.
        if task=='eyes':
            from ultralytics import YOLO
            loaded=YOLO(str(m.RUN/'eyes_head_finetuned.pt')).model.float().eval()
        else:
            ck=torch.load(m.RUN/'yawn_head_finetuned.pt',map_location='cpu',weights_only=True)
            loaded=copy.deepcopy(net);loaded.load_state_dict(ck['model_state_dict']);loaded.eval()
        reload_errors=[];ab=importlib.import_module('15_roi_ablation')
        eligible=[r for r in pred if r['split']=='external' and r['crops']]
        for r in [eligible[0],eligible[len(eligible)//2],eligible[-1]]:
            batch=[]
            for path in r['crops'].split('|'):
                im=Image.open(path).convert('RGB')
                if task=='eyes':im=m.to_square(ab.make_variants(im)['clahe'])
                batch.append(tf(im))
            with torch.no_grad():
                output=loaded(torch.stack(batch))
                probs=(output[0] if isinstance(output,tuple) else output) if task=='eyes' else output.softmax(1)
                value=float(probs[:,0 if task=='eyes' else 1].mean())
            reload_errors.append(abs(value-float(r['after_probability'])))
        # YOLO native checkpoint saves FP16 model weights; allow small quantization drift.
        assert max(reload_errors)<.001,(task,reload_errors)
        checks.append(dict(task=task,baseline_correct=int(baseline['correct']),historical_probability_max_error=max(errors),
                           saved_model_probability_max_error=max(reload_errors),control_best_epoch=best_epoch))
        reports[task]=summaries
    m.write_csv(OUT/'general_only_control_summary.csv',control_all)
    (OUT/'verification.json').write_text(json.dumps(dict(group_disjoint=True,baseline_and_saved_models_checked=True,checks=checks),indent=2),encoding='utf-8')
    def get(task,ds,phase,split='external',t=None):
        return next(r for r in reports[task] if r['dataset']==ds and r['split']==split and r['phase']==phase and r['condition']=='ALL' and (t is None or abs(float(r['threshold'])-t)<1e-6))
    def pct(v):return f'{float(v)*100:.2f}%'
    rows=[]
    for task,ds,split,t,name in [('eyes','cew','external',.25,'눈 CEW'),('eyes','simuletic','external',.25,'눈 Simuletic'),('eyes','webcam','external',.25,'직접 촬영 눈 / 이번 설정'),('eyes','added','test',.25,'추가 눈 테스트 / Open만'),('yawn','roboflow','external',.1,'하품 Roboflow / 0.10'),('yawn','roboflow','external',.15,'하품 Roboflow / 0.15'),('yawn','added','test',.1,'추가 주름 테스트 / 비하품만')]:
        b=get(task,ds,'before',split,t);a=get(task,ds,'after',split,t)
        rows.append(f"| {name} | {b['n']} | {pct(b['accuracy'])} | {pct(a['accuracy'])} | {(float(a['accuracy'])-float(b['accuracy']))*100:+.2f}%p |")
    controls=[]
    for task,ds,t in [('eyes','cew',.25),('yawn','roboflow',.1)]:
        c=next(r for r in control_all if r['task']==task and r['dataset']==ds and r['threshold']==t)
        controls.append(f"- {task} / {ds}: 일반 데이터만 같은 조건으로 추가 학습한 대조군 정확도 {pct(c['accuracy'])}, 재현율 {pct(c['recall'])} (검증 손실로 선택).")
    report='''# 오류 데이터 추가 학습 실험 결과

실험 기준일: 2026-11-08 (Asia/Seoul)

## 결론

기존 YOLO26n/ResNet18의 특징 추출부를 고정하고 마지막 분류층을 실제 추가 학습했다. 눈 CEW는 88.36% → 88.90% (+0.54%p)로 소폭 개선됐으나 Simuletic은 97.90% → 97.20%로 하락했다. 하품은 0.10에서 71.50% → 67.94% (-3.56%p), 재현율 66.67% → 51.46%로 악화됐다. 하품 후보는 채택하지 않는 것이 타당하다. 기존 앱과 기본 가중치는 교체하지 않았다. 눈 후보도 독립 검증 전 최종 채택하지 않는다.

## 동일 조건 전후 비교

전체 정확도에는 ROI 실패(UNKNOWN)를 오답으로 포함한다. 테스트용 추가 눈/주름 사진은 모두 음성 클래스이므로 이 부분의 정답률은 이진 분류 전체 성능이나 양성 재현율을 뜻하지 않는다.

| 평가 | N | 학습 전 | 학습 후 | 차이 |
|---|---:|---:|---:|---:|
'''+ '\n'.join(rows)+'''

눈 CEW Closed 재현율: 90.69% → 91.11%, Open 오탐: 166 → 158장. 하품 Roboflow 0.10 오탐: 28 → 16장, 하품 놓침: 57 → 83장. 이 중 ROI 실패 하품 22장은 그대로다. 비하품만 보강한 학습에서 경고 민감도가 줄어든 양상이며, 주름이 오분류의 직접 원인이었다는 증명은 아니다.

## 데이터와 라벨

- 화장 전/후 각각 26장, 주름 459장: 총 511장.
- 육안 보조 판독: Open 51장, no_yawn 154장, UNKNOWN 306장. 화장·주름 폴더명을 정답으로 사용하지 않았다. 불확실하거나 입이 없는 피부 패치는 제외했다.
- 라벨은 AI 보조 판독이며 독립적인 사람 검수가 완료된 정답셋이 아니다. `annotations.csv`에서 이미지별 판독과 제외 사유를 확인할 수 있다.
- 피부 주름 패치 1장이 MRL 패치와 dHash 유사 후보로 검출됐다. 실제 동일 원본인지 확정한 것은 아니며 보수적으로 제외했다.
- 화장 전후 같은 번호는 같은 인물 그룹으로 묶고, dHash 유사 그룹과 확인한 중복 쌍도 분리되지 않게 했다. 알 수 없는 동일 인물까지 완전히 분리했다고 주장하지 않는다.
- ROI 성공 기준 학습/검증/테스트: 눈 29/10/12장, 주름 입 81/26/25장. 주름 테스트 전체는 30장이고 그 중 5장은 ROI 실패다. 정확도 분모는 30장이다.
- 추가 데이터에 확정된 Closed/Yawn 양성 표본은 없었다. 기존 일반 데이터의 두 클래스를 함께 사용했다.

## 학습 방식

같은 MediaPipe ROI를 사용했다. 눈은 ROI → CLAHE → YOLO26n, 하품은 ROI → RGB → ResNet18이다. 기존 분할은 유지했다. 캐시한 고정 특징으로 마지막 선형 분류층만 AdamW, LR 1e-4, weight decay 0.01, 최대 40회 전체 배치 업데이트했다. 일반 데이터 손실 80%, 추가 조건 데이터 손실 20%를 사용했다. 데이터 증강 및 모델 전체 미세조정은 이번 실험에서 수행하지 않았다.

검증 손실 최소 시점: 눈 40회, 하품 36회. 눈 2,562개, 하품 1,026개의 분류층 파라미터만 변경했다. 테스트 및 외부 데이터를 이용해 업데이트 횟수나 임계값을 고르지 않았다. 외부 결과를 본 뒤 수치가 좋아지는 임계값으로 재선정하지 않았다.

## 일반 데이터만 추가 학습한 대조군

추가 조건 데이터의 영향을 구분하기 위해, 같은 초기 분류층과 업데이트 수·학습률·검증 선택 기준으로 일반 데이터만 학습한 대조군을 별도로 계산했다. 결과 설명용 대조군이며 새 배포 모델을 고르는 데 사용하지 않았다.

'''+ '\n'.join(controls)+'''

## 과거 결과와 이번 결과를 구분

과거 직접 촬영 65장 결과는 얼굴 전체 입력 35/65=53.85%, 눈 ROI 적용 후 55/65=84.62%, ROI 성공분 55/60=91.67%다. 이번에는 외부 평가 공통의 최대 이미지 크기 1280, CLAHE, 임계값 0.25를 사용해 58/65=89.23%와 ROI 실패 4장이 나왔다. 과거와 이번은 입력 크기/전처리/임계값 조건이 다르므로, 과거 84.62%와 이번 89.23%의 차이를 추가 학습 효과로 계산하면 안 된다. 이번 동일 조건 전후는 89.23% → 89.23%다.

CEW 2,423장과 Roboflow 393장은 이전 설정 선택에도 사용된 데이터다. 이번 학습에는 넣지 않았지만 완전히 새로운 최종 독립 평가셋이라고 부르지 않는다. MRL 4,000장의 과거 실험은 학습 원자료 중복 가능성이 있으며 이번 후보 가중치로 재평가한 결과가 아니다.

## 검증과 재현

`verification.json`에 그룹 분리, 과거 전체 정답 수 재현(눈 2141/2423, 하품 281/393), 과거 확률과의 최대 차이, 저장된 모델을 재로딩한 확률 확인을 기록했다. 요약은 `*_summary.csv`, 이미지별 전후 확률은 `*_predictions.csv`, 학습 손실은 `*_training.csv`, 모델 해시는 `*_config.json`에 있다.

```powershell
.venv/Scripts/python.exe 22_hardcase_finetune.py prepare
.venv/Scripts/python.exe 22_hardcase_finetune.py eyes
.venv/Scripts/python.exe 22_hardcase_finetune.py yawn
.venv/Scripts/python.exe 23_verify_hardcase_results.py
.venv/Scripts/python.exe build_aura_revalidation.py
```

기존 가중치와 실시간 앱은 변경하지 않았다. 새 가중치는 `runs/hardcase_finetune/eyes_head_finetuned.pt`, `yawn_head_finetuned.pt`에 있다. 실제 영상 경고 이벤트, EAR/MAR 결합, 새 인물 검증은 아직 수행하지 않았다.
'''
    (OUT/'실험결과.md').write_text(report,encoding='utf-8')
    print(json.dumps(checks,indent=2));print('\n'.join(controls))


if __name__=='__main__':verify()

"""Audit saved per-image results and record the final development candidate.

Does not train, tune thresholds or overwrite earlier experiments. Requires a
separate 31 recheck of the three face-input models at the fixed 0.50 threshold.
"""
from collections import defaultdict
import csv
import hashlib
import importlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'reports/final_model_selection'
MODELS=('yolo26n','yolo26s','mobilenet_v3_small')


def read(p):
    with p.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    compare=importlib.import_module('31_yawn_external_compare')
    saved=ROOT/'reports/yawn_external_model_compare'
    rows=read(saved/'predictions.csv');summary=read(saved/'summary.csv')
    groups=defaultdict(list)
    for r in rows:
        if r['model'] not in MODELS or r['input']!='face':continue
        groups[r['model']].append({**r,'probability':float(r['probability']) if r['probability'] else None,
                                  'latency_ms':float(r['latency_ms']) if r['latency_ms'] else None})
    entries=compare.image_index(ROOT/'data/external_driver')
    expected={compare.path_key(p):label for p,_,label in entries}
    assert len(expected)==393
    verified=[]
    for name in MODELS:
        group=groups[name]
        assert len(group)==len(expected)
        assert {compare.path_key(r['image']):r['actual'] for r in group}==expected
        metrics=compare.metrics(group,.5)
        original=next(r for r in summary if r['model']==name and r['input']=='face' and float(r['threshold'])==.5)
        rerun=next(r for r in read(OUT/'recheck/summary.csv') if r['model']==name and r['input']=='face')
        for key in ('tp','tn','fp','fn','total','roi_fail','errors'):
            assert metrics[key]==int(original[key])==int(rerun[key]),(name,key)
        for key in ('accuracy','recall','precision','f1','fpr'):
            assert abs(metrics[key]-float(original[key]))<1e-6,(name,key)
        verified.append(dict(model=name,input='face',threshold=.5,**metrics,
                             weights=str(compare.model_file(name).relative_to(ROOT)),weights_sha256=sha(compare.model_file(name))))
    # Input and threshold are held fixed. Do not optimize a new threshold on test images.
    chosen=max(verified,key=lambda r:(r['accuracy'],r['recall']))
    assert chosen['model']=='yolo26s'
    OUT.mkdir(parents=True,exist_ok=True)
    with (OUT/'three_models_verified.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(verified[0]));writer.writeheader();writer.writerows(verified)
    config={
        'status':'selected_development_candidate_not_independent_final_test',
        'eyes':{'model':'yolo26n','input':'MediaPipe eye ROI + CLAHE + 224 square padding',
                'weights':'runs/eyes_yolo26n-cls/weights/best.pt','threshold':.25,
                'duration_seconds':2.,'accuracy':2141/2423,'recall':1081/1192,'dataset':'CEW','n':2423},
        'yawn':{'model':chosen['model'],'input':'full original image (31 face mode), RGB + 224 square padding',
                'weights':chosen['weights'],'weights_sha256':chosen['weights_sha256'],
                'threshold':.5,'duration_seconds':1.5,'accuracy':chosen['accuracy'],
                'recall':chosen['recall'],'precision':chosen['precision'],'f1':chosen['f1'],
                'tp':chosen['tp'],'tn':chosen['tn'],'fp':chosen['fp'],'fn':chosen['fn'],
                'dataset':'Roboflow development set','n':393},
        'selection_reason':'Highest accuracy and yawn recall among the three face-input models at fixed threshold 0.50. Not highest F1 across all threshold sweeps.',
        'historical_reference':'team_resnet18 mouth results in 31 reuse 12 CSV; not one of the three newly inferred face models.',
        'limitations':['Roboflow was already used for model/threshold exploration.',
                       'Full-frame webcam context can differ from external face images.',
                       'Live face-visibility gate and duration rules have not been scored as a full system.',
                       'No joint eye+yawn accuracy, event recall or real-car safety claim.'],
        'sources_sha256':{str(p.relative_to(ROOT)):sha(p) for p in [saved/'predictions.csv',saved/'summary.csv',OUT/'recheck/summary.csv']}}
    (OUT/'selected_models.json').write_text(json.dumps(config,ensure_ascii=False,indent=2),encoding='utf8')
    lines=['# 최종 후보 선정 기록','',
           '31번의 원본 예측 CSV를 이미지별로 다시 집계하고, 현재 저장된 세 모델로 393장을 재추론해 혼동행렬의 정수 값이 일치하는지 확인했습니다.','',
           '| 모델 | 입력 | 임계값 | 정확도 | 하품 재현율 | 정밀도 | F1 | 오탐 / 놓침 |',
           '|---|---|---:|---:|---:|---:|---:|---:|']
    for r in verified:
        lines.append(f"| {r['model']} | 얼굴 이미지 전체 | 0.50 | {r['accuracy']:.2%} | {r['recall']:.2%} | {r['precision']:.2%} | {r['f1']:.2%} | {r['fp']} / {r['fn']} |")
    lines+=['','선정: **YOLO26s + 얼굴 이미지 전체 + Yawn 임계값 0.50**. 위 세 모델의 같은 입력·임계값 조건에서 정확도와 재현율이 가장 높습니다. TP 129, TN 179, FP 43, FN 42로 308/393장을 맞혔습니다.',
            '', '0.15의 YOLO26s F1은 75.51%로 0.50의 75.22%보다 약간 높습니다. 그러나 오탐이 43→106장으로 늘고 정확도가 78.37→72.26%로 낮아져, 이번 데모는 기본 임계값 0.50을 선택했습니다. 선택은 이미 사용한 개발셋의 결과에 근거하므로 새 인물·환경의 평가가 필요합니다.',
            '', '팀 ResNet18 입 ROI 결과는 역사적 참고입니다. 31번에서 새로 학습하거나 재추론한 세 얼굴 모델 중 하나로 세지 않습니다. 얼굴 입력은 추가 얼굴 검출 크롭이 아니라 **원본 이미지 전체**입니다.',
            '', '눈은 기존 YOLO26n + MediaPipe 눈 ROI + CLAHE + Closed 0.25를 유지합니다. 추가 눈 분류층 후보는 모든 평가셋에서 좋아지지 않았고, 통합 검증이 완료되지 않아 기본 가중치를 교체하지 않았습니다.',
            '', '## 확인된 파일과 수행 범위','',
            '| 코드 | 확인된 결과 | 판단 |','|---|---|---|',
            '| 24 | reports/nitymed_full_review/annotation_manifest.csv, samples | 프레임 및 입 ROI 준비 기록 있음 |',
            '| 25 | runs/nitymed_finetune_experiment/comparison.json | 현재 경로에 없음: 완료 성능으로 인용하지 않음 |',
            '| 28 | reports/nitymed_single_video/P1042778_720 | 단일 영상 예측·시각화 기록 있음 |',
            '| 29 | reports/nitymed_temporal_compare | 시간 조건 후보 비교 있음: 독립 이벤트 정답 성능 아님 |',
            '| 30 | runs/nitymed_single_video_finetune/comparison.json | 현재 경로에 없음: 완료 성능으로 인용하지 않음 |',
            '| 31 | reports/yawn_external_model_compare | 세 얼굴 모델과 입 입력 대조 결과 있음 |',
            '| 32 | reports/final_model_selection | 원본 재집계·가중치 재검증·최종 후보 선정 |',
            '', '외부 정지 이미지 지표를 라이브 전체 시스템 정확도로 부르거나, 다른 입력·모델의 비교를 동일 모델 재학습 향상으로 부르지 않습니다.']
    # Refuse to falsely claim missing completion artifacts if they now exist.
    assert not (ROOT/'runs/nitymed_finetune_experiment/comparison.json').exists()
    assert not (ROOT/'runs/nitymed_single_video_finetune/comparison.json').exists()
    (OUT/'선정근거.md').write_text('\n'.join(lines)+'\n',encoding='utf8')
    print(json.dumps(config['yawn'],ensure_ascii=False,indent=2))


if __name__=='__main__':main()

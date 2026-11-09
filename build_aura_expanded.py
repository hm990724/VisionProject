"""26-slide evidence revision of the compact deck, preserving earlier outputs."""
from pathlib import Path

ROOT=Path(__file__).resolve().parent
source=(ROOT/'build_aura_compact.py').read_text(encoding='utf-8')
source=source.replace("OUT=ROOT/'reports/presentation/compact_20slides'", "OUT=ROOT/'reports/presentation/revised_26slides'")
source=source.replace('full.TOTAL=20','full.TOTAL=26').replace('20장 근거','26장 근거')

def before(marker,block):
    global source
    assert source.count(marker)==1,marker
    source=source.replace(marker,block+'\n'+marker)

before("slide('눈: 구조와 계산량", '''
slide('데이터의 역할을 학습·검증·추가 실험으로 분리했습니다','외부 데이터를 사용했다는 말이 곧 외부 데이터로 재학습했다는 뜻은 아닙니다.',note='데이터는 역할에 따라 구분해야 합니다. 기존 이미지로 모델을 학습하고 내부 검증으로 가중치를 선택했습니다. 직접 촬영한 눈 사진은 입력 영역이 맞는지를 확인하는 데 사용했습니다. CEW와 Roboflow 등 외부 이미지는 성능 확인과 설정 선택에 활용했습니다. 따라서 외부 평가 결과는 미사용 최종 시험 성능과 구분합니다. 화장과 주름 조건의 추가 이미지는 소규모 보강 실험이며 최종 모델 교체 근거로 충분하지 않았습니다. 추가 하품 영상은 준비와 분석 경로가 있지만 최종 YOLO26s가 이 영상으로 재학습됐다고 설명하지 않습니다.',source='data/processed/manifest.csv · reports/final_model_selection/선정근거.md · 31_yawn_external_compare.py')
table(['자료','역할','최종 발표에서의 의미'],[['기존 이미지','학습 / 내부 검증 / 내부 테스트','best 가중치를 만든 데이터'],['직접 촬영 눈 65장','전체 얼굴 ↔ 눈 ROI 비교','입력 일치의 필요성 확인'],['CEW 등 / Roboflow 393장','외부 평가·설정 및 후보 선택','촬영 조건 변화에서 성능 확인'],['화장·주름 / 추가 하품 영상','보강 실험·분석 준비','최종 하품 재학습 성과로 주장하지 않음']],[440,460,532],rowh=85)
d.callout('최종 하품: 기존 데이터로 학습한 YOLO26s를 외부 데이터로 평가·선정했습니다.',y=750)
''')

before("slide('외부 눈 평가에서는", '''
slide('눈 모델의 선택에는 점수 외에 크기와 처리량도 고려했습니다','동일 내부 테스트에서 100%였기 때문에 추가 비교 기준이 필요했습니다.',ids=[5,7],note='내부 217장에서는 세 눈 모델이 모두 100%여서 정확도만으로 모델을 선택할 수 없었습니다. YOLO26n은 약 3.04MB로 ResNet18과 MobileNetV3-Small보다 작았습니다. 확인한 CPU 분류 추론 중앙값도 약 10.10ms로 작았습니다. 이 시간은 분류 단계의 측정치이며 카메라 캡처와 MediaPipe를 포함한 전체 시스템 지연이 아닙니다. 모델 크기와 처리량은 실시간 데모 후보를 좁히는 근거로 사용하고, 외부 평가로 실제 조건 변화에 대한 성능을 추가 확인했습니다.')
table(['눈 모델','내부 정확도 / 재현율','가중치 크기','CPU 분류 중앙값'],[['YOLO26n','100% / 100%','3.04 MB','10.10 ms'],['ResNet18','100% / 100%','42.71 MB','23.34 ms'],['MobileNetV3-Small','100% / 100%','5.91 MB','11.63 ms']],[400,430,280,322],rowh=86)
point(625,'선정 방향','경량 후보 선택 → 직접 촬영 검증 → 외부 평가 → 전처리·임계값 조정')
d.callout('처리 시간은 해당 환경의 분류 추론 측정치이며, 웹캠 전체 FPS가 아닙니다.',y=750)
''')

before("slide('하품: 학습 입력이", '''
slide('하품의 핵심 질문: 내부 점수가 외부에서도 유지되는가','새로운 데이터를 학습에 넣기 전에, 기존 모델의 일반화를 먼저 확인했습니다.',note='하품 부분은 단순히 정확도가 낮다는 이야기보다 어떤 질문을 검증했는지가 중요합니다. 기존 데이터에서 학습한 모델이 새로운 사진에서도 하품을 잡는지 확인했습니다. 외부 평가는 Roboflow 393장이고 하품 171장과 비하품 222장으로 구성됩니다. 추가 데이터로 YOLO를 재학습한 실험이 아니라 저장된 가중치를 동일 이미지에 적용한 비교입니다. 초기 입 ROI 모델의 한계를 보고 전체 이미지 모델 세 개를 같은 조건에서 다시 평가했습니다. 따라서 이번 성과는 추가 학습의 개선이 아니라 외부 평가를 통한 입력 경로와 최종 후보의 선정입니다.',source='31_yawn_external_compare.py · reports/yawn_external_model_compare/README_results.txt · predictions.csv')
d.flow([('기존 데이터 학습','내부 검증으로 best 저장'),('새 사진으로 평가','Roboflow 393장'),('동일 조건 비교','입력 / 임계값 고정'),('최종 후보 선정','YOLO26s / 전체 이미지')],265)
table(['외부 이미지','장수','저장 위치'],[['하품','171','data/external_driver/yawn/normal/yawn'],['비하품','222','data/external_driver/yawn/normal/no_yawn']],[270,180,982],y=485,rowh=79,size=24)
txt(95,740,1400,'하품 외부 규모는 독립 데이터 확보와 프로젝트 범위에 따른 선택이며, ROI 방식 때문이 아닙니다.',25,TEAL,True)
''')

before("slide('초기 하품 모델은", '''
slide('epoch 곡선은 내부 검증, 외부 평가는 저장된 best의 결과입니다','학습 로그와 외부 평가 수치를 혼동하지 않도록 분리했습니다.',note='YOLO26n 학습 로그는 11개 epoch이고 최고 내부 검증 정확도는 97.71%입니다. YOLO26s는 20개 epoch이며 최고 내부 검증 정확도는 98.62%입니다. YOLO26s에서 이 최고값은 13번째와 16번째 epoch에 나타납니다. 이 값은 내부 테스트 정확도 98.60%와도 분모가 다릅니다. 외부 평가는 저장된 best 가중치를 한 번 적용해 계산했기 때문에 외부 데이터의 epoch별 학습 곡선은 없습니다. YOLO26s 외부 정확도 78.37%는 해당 학습 곡선과 별개의 평가 결과입니다.',source='runs/yawn_yolo26n-cls/results.csv · runs/yawn_yolo26s-cls/results.csv · reports/final_model_selection/three_models_verified.csv')
fig('yawn_training_validation.png',85,245,730,365)
table(['모델','학습 epoch','최고 내부 검증'],[['YOLO26n','11','97.71%'],['YOLO26s','20','98.62%']],[260,200,270],x=800,y=270,rowh=91,size=23)
txt(95,645,1400,'학습·검증 로그: runs/yawn_yolo26s-cls/results.csv  →  weights/best.pt',27,TEAL,True)
d.callout('외부 393장은 학습하지 않았으므로, 외부 epoch–정확도 곡선은 없습니다.',y=750)
''')

before("slide('최종 시스템은", '''
slide('하품에서 얻은 성과는 외부 검증을 통한 후보 선정입니다','재학습 효과와 모델·입력 경로를 바꾼 효과를 구분합니다.',note='초기 입 ROI ResNet의 기본 설정에서는 정확도 65.65%, 재현율 44.44%였습니다. 임계값을 조정하면 71.50%와 66.67%로 높아졌습니다. 최종 전체 이미지 YOLO26s는 78.37%와 75.44%였습니다. 초기 설정 대비 정확도 차이는 12.72퍼센트포인트이지만 모델과 입력 경로를 함께 바꿨기 때문에 YOLO 구조만의 효과 또는 추가 학습의 효과라고 해석하면 안 됩니다. 더 공정한 구조 비교는 같은 전체 이미지와 0.50 기준의 YOLO26n, YOLO26s, MobileNet 비교입니다. YOLO26s는 YOLO26n보다 하품 33장을 더 잡았지만 비하품 오탐이 25장 늘었습니다. 이 교환 관계까지 설명하고 최종 개발 후보로 선택했습니다.',source='reports/yawn_roboflow/threshold_analysis/yawn_threshold_sweep.csv · reports/final_model_selection/three_models_verified.csv')
table(['하품 평가 경로','정확도','재현율','해석'],[['입 ROI ResNet / 0.50','65.65%','44.44%','초기 외부 평가'],['입 ROI ResNet / 0.10','71.50%','66.67%','같은 모델의 확률 기준 조정'],['전체 이미지 YOLO26s / 0.50','78.37%','75.44%','모델·입력 경로 변경 후 최종 후보']],[520,240,240,432],rowh=86,size=24)
point(635,'같은 입력 비교','YOLO26n → YOLO26s: 하품 감지 96 → 129장 / 비하품 오탐 18 → 43장')
d.callout('단순 재학습 개선이 아닙니다. 외부 평가를 통해 실제로 더 나은 후보를 골랐습니다.',y=750)
''')

before("slide('다음 단계는", '''
slide('하품의 남은 약점은 다음 실험의 질문으로 연결합니다','확인된 오류와 아직 검증하지 않은 개선 가설을 분리합니다.',note='하품 모델의 약점은 감출 것이 아니라 다음 실험으로 연결해야 합니다. 최종 후보는 외부 하품 171장 중 42장을 놓치고 비하품 222장 중 43장을 하품으로 분류했습니다. 이미지 라벨은 한 장의 모양만 설명하므로 지속 시간과 반복 경고의 품질을 입증하지 못합니다. 전체 이미지 분류에서는 얼굴 크기나 배경 같은 조건이 영향을 줄 가능성도 별도로 검증해야 합니다. 다만 이 조건들이 실제 오류 원인이라는 인과 분석은 수행하지 않았습니다. 이를 확인하기 위해 사람과 영상을 분리한 이벤트 평가, 웃음과 말하기 같은 비하품 조건 보강, 프레이밍 변화 시험을 제안합니다. 모두 다음 실험 계획이며 현재 개선 성과로 표시하지 않습니다.',source='reports/final_model_selection/three_models_verified.csv · 후속 실험 설계: 미실행 개선 가설')
table(['확인된 한계','필요한 다음 실험','확인할 지표'],[['하품 42장 미검출','약한 하품·측면·가림 조건별 라벨 검수','조건별 재현율'],['비하품 43장 오탐','말하기·웃음·입 벌림 비하품 보강','정밀도 / 오탐률'],['정지 이미지 평가','미사용 사람·영상으로 이벤트 평가','이벤트 재현율 / 시간당 오경고'],['실제 웹캠 환경 미검증','얼굴 크기·조명·배경 변화 시험','조건별 성능 / 지연 / 안정성']],[360,700,372],rowh=85,size=24)
d.callout('이미지 분류 성능은 확인했습니다. 시간 판단·실제 주행의 효과는 추가 검증 과제입니다.',y=750)
''')

if __name__ == '__main__':
    exec(compile(source,str(ROOT/'build_aura_compact.py'),'exec'),{'__name__':'__main__','__file__':str(ROOT/'build_aura_compact.py')})

# AURA DRIVE

AI 기반 운전자 졸음 징후 감지와 **가상 안전 대응 시뮬레이션**을 위한 연구·발표 프로토타입입니다. 실제 자동차 제어, 실제 119 신고, 실신·발작 진단은 수행하지 않습니다.

## 실행

Windows PowerShell에서 프로젝트 루트 기준:

```powershell
py -3.10 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
streamlit run 11_streamlit_dashboard.py
```

기존 `.venv`를 쓰는 경우 활성화 후 `pip install -r requirements.txt`를 실행하면 됩니다. 앱 첫 화면에서 `시동 걸고 시스템 입장`을 누른 뒤, 실시간 모니터에서 로컬 카메라 번호를 선택하고 `로컬 카메라 연결`을 누르세요. 기본 장치는 보통 0번이며, USB 카메라 번호는 장치에 따라 다를 수 있습니다. 카메라 캡처는 Streamlit 서버가 실행되는 PC의 OpenCV에서 처리되므로, 이 프로젝트는 해당 PC에서 로컬 실행해야 합니다. 연결 종료 버튼으로 카메라를 해제할 수 있습니다.

기본 모델 경로:

- 눈: `runs/eyes_yolo26n-cls/weights/best.pt`
- 하품: `teammate_yawn/runs/03_yawn_resnet18/weights/03_best.pt`
- MediaPipe 얼굴 랜드마크: `teammate_yawn/models/face_landmarker.task`

파일이 없으면 앱의 설정 화면에서 경로를 확인하세요. 체크포인트와 데이터셋은 저장소에 임의로 추가하지 않습니다.

## 시뮬레이터

실시간 모니터 아래 독립 패널에서 `자동 시연 시작`을 누르면 모의 눈 감김 경고 2회, 가상 감속, 정차·휴식, 응답 대기, 모의 SOS 화면 순서로 진행됩니다. 빠른 발표 모드는 휴식 15초와 응답 대기 30초로 시간을 압축합니다. 일반 시나리오의 5분/10분은 제품 안전 기준이 아닌 데모 값입니다.

실제 감지 경로에서는 2초 이상 눈 감김이 경고 이벤트가 되고, 최근 5분 내 2회 경고 시 가상 감속을 시작합니다. 시연 버튼이 입력하는 경고는 명시적으로 모의 이벤트이며 모델 예측이나 실제 운전자 상태가 아닙니다. 실제 응급상황에서는 데모 타이머를 기다리지 말고 즉시 도움을 요청해야 합니다.

## 모델 후보 및 평가 범위

- 눈 후보: MediaPipe Eye ROI → CLAHE → YOLO26n-cls, `P(Closed) >= 0.25`. CEW 개발 평가에서 accuracy 88.36%, Closed Recall 90.69%, Open FPR 13.48%를 기록했습니다. threshold 0.25는 Closed Recall 90% 이상이라는 우선 조건에 맞춘 후보이며, 최종 독립 검증 결과가 아닙니다.
- 하품 후보: MediaPipe Mouth ROI → RGB → 팀 ResNet18, threshold 0.10/0.15 비교. Roboflow 외부 393장 전체 정확도는 65.65%, yawn recall은 threshold 0.50에서 44.44%였습니다. 외부 일반화는 해결되지 않았습니다.
- 외부 데이터는 설정 선택에도 사용됐으며, 사람 단위 독립 holdout은 없습니다. MRL은 학습 자료 중복 가능성이 있습니다. 숫자는 해당 데이터·전처리·임계값 조건에 한정해 해석해야 합니다.

## 발표자료 재생성

로컬 `reports`의 평가 CSV와 오류 ROI 사진을 사용해 13장 PPTX와 동일 구성 PDF를 생성합니다.

```powershell
python build_aura_presentation.py
```

출력:

- `reports/presentation/AURA_DRIVE_발표자료_완성본.pptx`
- `reports/presentation/AURA_DRIVE_발표자료_완성본.pdf`
- `reports/presentation/AURA_DRIVE_slide_overview.jpg`

PPTX는 글꼴·레이아웃이 고정되도록 슬라이드별 이미지로 구성되어 텍스트 개별 편집은 지원하지 않습니다. 내용 수정은 `build_aura_presentation.py`에서 반영한 뒤 다시 생성하세요.

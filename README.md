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

기존 `.venv`를 쓰는 경우 활성화 후 `pip install -r requirements.txt`를 실행하면 됩니다. 웹캠 사용은 브라우저의 카메라 권한이 필요합니다. 휴대폰에서 원격 접속할 때는 HTTPS가 필요할 수 있습니다.

기본 모델 경로:

- 눈: `runs/eyes_yolo26n-cls/weights/best.pt`
- 하품: `teammate_yawn/runs/03_yawn_resnet18/weights/03_best.pt`
- MediaPipe 얼굴 랜드마크: `teammate_yawn/models/face_landmarker.task`

파일이 없으면 앱의 설정 화면에서 경로를 확인하세요. 체크포인트와 데이터셋은 저장소에 임의로 추가하지 않습니다.

## 시뮬레이터

상단 메뉴의 **모의 대응**을 선택하면 눈 감김 경고 → 가상 감속 → 정차 가정 → 무반응 상황 연출 → 119 차량 도착 장면이 자동 반복됩니다. 메뉴를 선택하는 것 외에 시작 버튼은 필요하지 않습니다. 빨간 경고선과 구급차는 설명용 그래픽이며, 실제 기절/의식 상태를 AI가 감지하는 것이 아닙니다. 실제 차량 제어·창문 작동·119 신고는 수행하지 않습니다.

실시간 모니터에서는 원본 웹캠과 눈·입 모델 입력 ROI, 점수 및 경고를 나란히 표시합니다. ROI는 카메라 프레임에 덧그리지 않고 별도 분석 패널에서 확인합니다. 실제 감지 경로에서는 2초 이상 눈 감김이 경고 이벤트로 기록됩니다. 그래픽 시나리오는 이 이벤트와 연결되지 않으며 실시간 카메라 결과와 독립적입니다. 실제 응급상황에서는 시연 화면을 기다리지 말고 즉시 도움을 요청해야 합니다.

## 모델 후보 및 평가 범위

- 눈 후보: MediaPipe Eye ROI → CLAHE → YOLO26n-cls, `P(Closed) >= 0.25`. CEW 개발 평가에서 accuracy 88.36%, Closed Recall 90.69%, Open FPR 13.48%를 기록했습니다. threshold 0.25는 Closed Recall 90% 이상이라는 우선 조건에 맞춘 후보이며, 최종 독립 검증 결과가 아닙니다.
- 하품 후보: MediaPipe Mouth ROI → RGB → 팀 ResNet18, threshold 0.10/0.15 비교. Roboflow 외부 393장 전체 정확도는 65.65%, yawn recall은 threshold 0.50에서 44.44%였습니다. 외부 일반화는 해결되지 않았습니다.
- 외부 데이터는 설정 선택에도 사용됐으며, 사람 단위 독립 holdout은 없습니다. MRL은 학습 자료 중복 가능성이 있습니다. 숫자는 해당 데이터·전처리·임계값 조건에 한정해 해석해야 합니다.

## 발표자료 재생성

로컬 `reports`의 평가 CSV와 오류 ROI 사진을 사용해 17장 PPTX와 동일 구성 PDF를 생성합니다.

```powershell
python build_aura_presentation.py
```

출력:

- `reports/presentation/AURA_DRIVE_발표자료_완성본.pptx`
- `reports/presentation/AURA_DRIVE_발표자료_완성본.pdf`
- `reports/presentation/AURA_DRIVE_slide_overview.jpg`

PPTX는 글꼴·레이아웃이 고정되도록 슬라이드별 이미지로 구성되어 텍스트 개별 편집은 지원하지 않습니다. 내용 수정은 `build_aura_presentation.py`에서 반영한 뒤 다시 생성하세요.
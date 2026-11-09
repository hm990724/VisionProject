AURA DRIVE v12 | 로컬 캠 & 경고음 안정화
========================================

변경 사항
- 시작 화면에서 '▶ 로컬 웹캠과 AI 시작' 한 번에 진입 + 카메라 켜기
- 카메라 원본 화면 / AI 분류 패널 / 연속 지속시간 막대 / 이벤트 횟수
- 눈 감김: 분류 점수가 임계값 이상으로 5초 이상 연속 유지될 때만 경고 1회
- 하품: 분류 점수가 임계값 이상으로 3초 이상 연속 유지될 때만 경고 1회
- 일정 시간 신호가 끊기거나 임계값 아래로 내려가면 지속시간을 초기화
- 소리: Streamlit autoplay 오디오를 제거. 브라우저 오디오 플레이어 1개만 사용
- 눈 소리 우선, 하품과 동시 재생 방지. 동일 이벤트 재생 1회
- 소리는 모니터 화면의 '경고음 켜기' 버튼 클릭 후 활성화 (브라우저 자동재생 제한)
- MP3가 있으면 기존 '눈감김.mp3' / '하품.mp3' 사용. 없으면 기본 알림음 재생
- 입력 방식: 기존 결과에 대응하는 하품 '원본 전체 RGB'가 기본값
- '얼굴 중심 ROI' 실험 옵션을 추가했지만 별도 검증 없는 탐색적 모드임
- 웹캠 원본 MJPEG 전송과 AI 추론 스레드를 분리 (CPU 과부하 시 추론이 더 느릴 수 있음)

설치
1) VisionProject\11_streamlit_dashboard.py 백업 (예: 11_streamlit_dashboard_backup.py)
2) ZIP에서 아래 파일만 프로젝트 최상위에 복사/덮어쓰기:
   - 11_streamlit_dashboard.py
   - aura_local_camera.py
   - aura_event_logic.py
   - aura_audio_panel.py
   - aura_vehicle_hero.jpg
   - Start-Aura-Local.ps1
3) 기존 프로젝트에서 반드시 있어야 하는 파일:
   - common.py
   - roi_utils.py
   - aura_safety_sim.py
   - 31_yawn_external_compare.py
   - runs/eyes_yolo26n-cls/weights/best.pt
   - runs/yawn_yolo26s-cls/weights/best.pt
   - .venv 와 streamlit, ultralytics, mediapipe, opencv-python, numpy, torch, Pillow

Windows PowerShell 실행:
  cd C:\Users\lookm\Documents\GitHub\VisionProject
  powershell -ExecutionPolicy Bypass -File .\Start-Aura-Local.ps1

또는:
  .\.venv\Scripts\python.exe -m streamlit run 11_streamlit_dashboard.py --server.address 127.0.0.1 --server.port 8501 --server.fileWatcherType none

브라우저: http://127.0.0.1:8501

중요한 조작
- 시작 화면: '로컬 웹캠과 AI 시작'
- 경고음: 영상 화면 위 '경고음 켜기' 클릭. '눈/하품 알림 테스트'로 확인
- 일시 중지: '웹캠 종료' 버튼으로 실제 장치 점유 해제
- 각 경고는 5초/3초의 연속 양성 추론에 따라 발동하며, 단순 1장 양성인 경우 울리지 않음
- 전처리/임계값 변경 시 '현재 설정 적용' 버튼 클릭 필요
- 하품 얼굴 중심 옵션은 모델 재학습 없음. 새 성능을 검증한 결과는 아님
- 카메라 FPS와 AI FPS는 다름. 실제 Windows 장치 FPS는 아직 직접 측정하지 않음

왜 웹캠에서 정확도가 낮아 보일 수 있나?
- 외부 검증은 정지 사진에서의 정확도; 웹캠 영상 FPS나 이벤트 인식 정확도와는 다름
- 카메라에 얼굴이 작게 찍히면 눈 ROI 정보가 너무 적어짐
- YOLO26s는 입 ROI가 아닌 원본 전체 RGB로 평가했고, 실시간 웹캠 배경 비율이 다르면 점수도 변할 수 있음
- 얼굴 중심 모드를 켜서 같은 조명/거리에서 오탐/미탐을 직접 비교할 수 있지만, 성능이 개선된다고 보장할 수 없음
- 눈은 기본 CLAHE + 0.25, 하품은 기본 RGB + 0.50 유지

실제 차량 제어·119 자동 신고 기능은 없습니다. 시뮬레이션에 한정합니다.

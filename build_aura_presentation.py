"""Build the AURA DRIVE presentation deck and its matching PDF."""
from __future__ import annotations

from io import BytesIO
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps
from pptx import Presentation
from pptx.util import Inches


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "reports" / "presentation"
W, H = 1600, 900
BG = "#09131F"
PANEL = "#132436"
PANEL_2 = "#1A3043"
TEXT = "#F2F7FA"
MUTED = "#A7BBC9"
MINT = "#63DCC5"
AMBER = "#F4BD69"
CORAL = "#F17D72"
BLUE = "#85BCE7"
FONT = r"C:\Windows\Fonts\malgun.ttf"
FONT_BOLD = r"C:\Windows\Fonts\malgunbd.ttf"


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(FONT_BOLD if bold else FONT, size)


def wrap_lines(draw: ImageDraw.ImageDraw, text: str, use_font, max_width: int) -> list[str]:
    lines: list[str] = []
    for paragraph in text.split("\n"):
        line = ""
        for char in paragraph:
            candidate = line + char
            if line and draw.textbbox((0, 0), candidate, font=use_font)[2] > max_width:
                lines.append(line)
                line = char
            else:
                line = candidate
        lines.append(line)
    return lines


def paragraph(draw, xy, text, use_font, fill=TEXT, max_width=700, line_gap=10):
    x, y = xy
    lines = wrap_lines(draw, text, use_font, max_width)
    line_height = use_font.size + line_gap
    for line in lines:
        draw.text((x, y), line, font=use_font, fill=fill)
        y += line_height
    return y


def rounded(draw, box, fill=PANEL, outline=None, radius=20, width=1):
    draw.rounded_rectangle(box, radius=radius, fill=fill, outline=outline, width=width)


def base(title: str, kicker: str, page: int):
    image = Image.new("RGB", (W, H), BG)
    draw = ImageDraw.Draw(image)
    draw.text((84, 48), kicker.upper(), font=font(17, True), fill=MINT)
    draw.text((84, 82), title, font=font(43, True), fill=TEXT)
    draw.line((84, 148, 1516, 148), fill="#294257", width=2)
    draw.text((84, 855), "AURA DRIVE  ·  2026.11.06  ·  연구/시연용 프로토타입", font=font(15), fill=MUTED)
    draw.text((1490, 855), f"{page:02d}", font=font(16, True), fill=MUTED)
    return image, draw


def card(draw, box, title, body=None, accent=MINT, body_size=23):
    rounded(draw, box, PANEL, "#294257", 18, 2)
    x1, y1, x2, _ = box
    draw.rounded_rectangle((x1, y1, x1 + 8, y1 + 62), radius=4, fill=accent)
    draw.text((x1 + 28, y1 + 19), title, font=font(24, True), fill=TEXT)
    if body:
        paragraph(draw, (x1 + 28, y1 + 68), body, font(body_size), MUTED, x2 - x1 - 56, 9)


def metric(draw, box, value, label, detail="", color=MINT):
    rounded(draw, box, PANEL_2, "#35536A", 18, 2)
    x1, y1, x2, _ = box
    draw.text((x1 + 22, y1 + 17), label, font=font(20, True), fill=MUTED)
    draw.text((x1 + 22, y1 + 55), value, font=font(43, True), fill=color)
    if detail:
        paragraph(draw, (x1 + 22, y1 + 112), detail, font(17), MUTED, x2 - x1 - 44, 6)


def fit_image(image: Image.Image, box, radius=16, background="#0B1723"):
    x1, y1, x2, y2 = box
    target_w, target_h = x2 - x1, y2 - y1
    fitted = ImageOps.fit(image.convert("RGB"), (target_w, target_h), method=Image.Resampling.LANCZOS)
    mask = Image.new("L", fitted.size, 0)
    ImageDraw.Draw(mask).rounded_rectangle((0, 0, target_w - 1, target_h - 1), radius=radius, fill=255)
    return fitted, mask


def paste_image(canvas, path_or_image, box, radius=16):
    source = path_or_image if isinstance(path_or_image, Image.Image) else Image.open(path_or_image)
    fitted, mask = fit_image(source, box, radius)
    canvas.paste(fitted, (box[0], box[1]), mask)


def crop(path: str, box) -> Image.Image:
    return Image.open(ROOT / path).convert("RGB").crop(box)


def source(draw, text):
    draw.text((84, 821), text, font=font(14), fill="#829AAA")


def slide_cover():
    image = ImageOps.fit(Image.open(ROOT / "aura_car_interior.jpg").convert("RGB"), (W, H))
    image = Image.blend(image, Image.new("RGB", image.size, "#06101B"), 0.52)
    draw = ImageDraw.Draw(image)
    draw.rounded_rectangle((82, 88, 585, 137), radius=20, fill="#102A38")
    draw.text((108, 99), "AI DRIVER AWARENESS  /  PROJECT REPORT", font=font(16, True), fill=MINT)
    draw.text((82, 250), "AURA DRIVE", font=font(86, True), fill=TEXT)
    draw.text((87, 365), "졸음 징후 감지에서 안전 대응 시뮬레이션까지", font=font(36, True), fill=TEXT)
    paragraph(draw, (90, 445), "내부 점수가 높아도 실제 입력은 달랐습니다.\n외부 평가와 실패 이미지로 확인한 모델의 한계,\n그리고 그 근거로 정리한 실시간 데모를 소개합니다.", font(25), "#C2D2DE", 1000, 14)
    draw.rounded_rectangle((88, 648, 860, 710), radius=15, fill="#102436", outline="#466176", width=2)
    draw.text((112, 665), "눈  YOLO26n + CLAHE 후보     ·     하품  ResNet18 후보", font=font(20, True), fill=MINT)
    draw.text((90, 818), "프로젝트 종합 발표  ·  2026년 11월 6일 기준", font=font(19), fill="#CDDAE3")
    draw.text((90, 856), "시연/연구 프로토타입 · 실제 차량 제어 및 실제 긴급 신고 없음", font=font(16, True), fill=AMBER)
    return image


def slide_internal_external():
    image, d = base("내부 100%가 남긴 질문", "01 / Why external evaluation", 2)
    metric(d, (84, 190, 445, 360), "100%", "눈 내부 Test", "YOLO26n · ResNet18 · MobileNetV3-Small", MINT)
    metric(d, (466, 190, 827, 360), "99.06%", "하품 내부 Test", "팀 ResNet18 · 210 / 212 (ROI 성공분)", MINT)
    metric(d, (848, 190, 1209, 360), "79.12%", "CEW 외부 눈", "YOLO26n baseline · 전체 2,423장", AMBER)
    metric(d, (1230, 190, 1516, 360), "65.65%", "Roboflow 외부 하품", "ResNet18 · 전체 393장", CORAL)
    card(d, (84, 400, 750, 690), "높은 내부 점수만으로 과적합을 확정할 수는 없음", "다만 내부 Test가 완벽에 가까운데 외부 얼굴에서 성능이 떨어졌습니다. 사람 ID 기반 분할이 없고, 촬영 조건·ROI 분포도 다릅니다. 그래서 외부 평가와 실패 사례를 추가해 일반화 문제를 확인했습니다.", AMBER, 22)
    card(d, (782, 400, 1516, 690), "핵심 질문을 바꿨습니다", "‘몇 % 정확한가?’에서 ‘어떤 입력에서 실패하는가?’, ‘ROI 미검출은 몇 장인가?’, ‘임계값을 바꾸면 놓침과 오탐이 어떻게 바뀌는가?’로 평가 기준을 확장했습니다.", MINT, 22)
    d.text((90, 735), "내부 성능은 해당 분할의 결과이며, 인물 독립 검증 또는 실제 운전 성능을 보증하지 않습니다.", font=font(21, True), fill=TEXT)
    source(d, "근거: reports/model_comparison.csv · reports/cew/external_summary.csv · reports/yawn_roboflow/external_summary.csv")
    return image


def slide_data():
    image, d = base("데이터를 먼저 감사하고, 그룹 단위로 나눴습니다", "02 / Data & split", 3)
    metric(d, (84, 190, 390, 345), "2,900", "원본 이미지", "눈/하품 클래스 합계", MINT)
    card(d, (420, 190, 985, 440), "원본 클래스 수", "Closed 726   ·   Open 726\nno_yawn 725   ·   yawn 723\n완전 중복 제거 및 dHash 유사 그룹 단위 분할", BLUE, 24)
    card(d, (1015, 190, 1516, 440), "분할 결과", "눈  1,017 / 218 / 217\n하품  1,016 / 218 / 214\nTrain / Validation / Test · seed 42", MINT, 24)
    card(d, (84, 475, 750, 700), "왜 이 작업을 했나", "파일명과 유사 이미지가 서로 다른 split에 걸칠 위험을 줄이기 위해 MD5 및 dHash 그룹 기준으로 분할했습니다. 224×224 정사각형 입력으로 통일했습니다.", MINT, 21)
    card(d, (782, 475, 1516, 700), "남은 한계", "인물 ID가 없어 동일 인물의 완전한 분리를 보장하지 않습니다. 따라서 내부 Test 100%는 ‘새로운 운전자에게도 100%’라는 뜻이 아닙니다.", CORAL, 21)
    d.text((90, 750), "외부 평가: CEW 얼굴 2,423장 · MRL 잘린 눈 패치 4,000장 · Simuletic DMS 143장 · Roboflow 얼굴 393장", font=font(21, True), fill=TEXT)
    source(d, "근거: 02_prepare.py · data/processed/manifest.csv · reports/*/external_summary.csv")
    return image


def slide_epochs():
    image, d = base("설정 Epoch와 실제 종료 Epoch는 다릅니다", "03 / Training recipe", 4)
    card(d, (84, 185, 770, 450), "눈 모델 공통 설정", "YOLO26n: pretrained yolo26n-cls.pt\nCNN: ImageNet 사전학습 가중치\n최대 25 epoch · batch 16 · 입력 224×224\nseed 42 · CPU · patience 7\nCNN AdamW: lr 1e-4, weight_decay 5e-4\nYOLO optimizer: Ultralytics auto 설정", MINT, 21)
    card(d, (800, 185, 1516, 450), "하품 ResNet18 설정", "최대 25 epoch · batch 16 · 입력 224×224\nAdamW · lr 1e-4 · weight_decay 1e-4\n2 epoch warmup · patience 7 · seed 42\nCPU · workers 0 · RGB 입 ROI\n정규화: ImageNet mean / std", BLUE, 21)
    d.text((88, 490), "실제 기록", font=font(26, True), fill=AMBER)
    rows = [
        ("눈 YOLO26n", "13 epoch 기록 · val top-1 첫 100%: epoch 6", "best.pt 메타데이터 epoch=-1 → best epoch 단정 보류"),
        ("눈 ResNet18", "10 epoch 종료 · 첫 val 100%: epoch 3", "로그 기준, checkpoint 선택 시점은 별도 구분"),
        ("눈 MobileNetV3-Small", "9 epoch 종료 · 첫 val 100%: epoch 2", "로그 기준, checkpoint 선택 시점은 별도 구분"),
        ("하품 ResNet18", "17 epoch 종료 · 최저 val loss: epoch 10", "val loss 0.0890 → 종료 시 0.1621"),
    ]
    y = 535
    for name, actual, caveat in rows:
        rounded(d, (84, y, 1516, y + 60), PANEL, "#294257", 12, 1)
        d.text((103, y + 12), name, font=font(19, True), fill=TEXT)
        d.text((440, y + 12), actual, font=font(18, True), fill=MINT)
        d.text((935, y + 14), caveat, font=font(15), fill=MUTED)
        y += 68
    source(d, "근거: runs/eyes_yolo26n-cls/args.yaml, results.csv · runs/eyes_resnet18/history.csv · teammate_yawn/runs/03_yawn_resnet18/*")
    return image


def slide_external_sets():
    image, d = base("서로 다른 데이터셋으로 무엇을 확인했나", "04 / External evaluation", 5)
    cards = [
        ((84, 190, 780, 435), "CEW · 얼굴 전체", "2,423장 · 눈 ROI 2,389 성공 / 34 실패\nYOLO baseline 79.12%\nROI 검출과 분류를 함께 본 평가", MINT),
        ((820, 190, 1516, 435), "MRL · 잘린 눈 패치", "4,000장 · ROI 검출 없이 분류 평가\nYOLO 98.80% · ResNet18 99.95%\n학습 자료와 중복 가능성 있어 독립성 주의", BLUE),
        ((84, 465, 780, 710), "Simuletic DMS · 운전자 영상", "평가 143장: Closed 102 / Open 41\nROI 140 성공 / 3 실패\nDrowsy/Microsleep 25장은 별도 보류", AMBER),
        ((820, 465, 1516, 710), "Roboflow · 하품 외부 얼굴", "393장: yawn 171 / no_yawn 222\n입 ROI 344 성공 / 49 실패\n전체 정확도 65.65% · ROI 성공분 75.00%", CORAL),
    ]
    for box, title, body, accent in cards:
        card(d, box, title, body, accent, 21)
    d.text((89, 750), "외부 평가는 개발·설정 선택에도 사용된 자료입니다. 최종 미사용 holdout 성능으로 부르지 않습니다.", font=font(20, True), fill=TEXT)
    source(d, "근거: reports/cew/external_summary.csv · reports/mrl/external_summary.csv · reports/simuletic/external_summary.csv · reports/yawn_roboflow/external_summary.csv")
    return image


def slide_eye_errors():
    image, d = base("눈 오류 사례: 사진은 원인을 확정하지 않지만, 질문을 만듭니다", "05 / Eye error audit", 6)
    eye_makeup = crop("reports/cew/error_audit/both_fail_Open.png", (320, 14 * 220, 590, 15 * 220))
    eye_low_quality = crop("reports/cew/error_audit/both_fail_Open.png", (320, 9 * 220, 590, 10 * 220))
    paste_image(image, eye_makeup, (84, 190, 510, 440))
    paste_image(image, eye_low_quality, (540, 190, 966, 440))
    d = ImageDraw.Draw(image)
    d.text((98, 458), "열린 눈 → 감김 오분류 예시 ROI", font=font(18, True), fill=MINT)
    d.text((553, 458), "저해상도·경계가 불분명한 ROI 예시", font=font(18, True), fill=AMBER)
    metric(d, (1010, 190, 1516, 335), "432장", "Open → Closed", "YOLO26n · CEW", CORAL)
    metric(d, (1010, 355, 1516, 500), "40장", "Closed → Open", "YOLO26n · CEW", AMBER)
    card(d, (84, 525, 1516, 710), "육안 관찰에서 세운 가설", "낮은 화질, 고개 방향, 눈썹·속눈썹·화장으로 강조된 눈 경계가 분류를 어렵게 했을 가능성. 사진만으로 모델의 인과적 주의 영역을 증명한 것은 아닙니다. ROI 실패 34장은 분류 오류와 별도로 기록했습니다.", BLUE, 21)
    d.text((89, 755), "그래서 다음 단계는 ‘새 모델 추가’가 아니라 CLAHE 및 threshold sweep으로 trade-off를 측정하는 것이었습니다.", font=font(19, True), fill=TEXT)
    source(d, "근거: reports/cew/error_audit/both_fail_Open.png · model_error_breakdown.csv · external_summary.csv")
    return image


def slide_eye_threshold():
    image, d = base("눈 threshold 0.25: 정확도 최고점이 아니라 재현율 조건으로 골랐습니다", "06 / Eye preprocessing & threshold", 7)
    values = [(0.25, 88.36, 90.69, 13.48), (0.40, 90.18, 86.66, 6.01), (0.50, 88.90, 81.12, 3.17)]
    labels = ["CLAHE · t=.25", "CLAHE · t=.40", "CLAHE · t=.50"]
    chart_left, chart_top, chart_right, chart_bottom = 110, 270, 950, 640
    d.line((chart_left, chart_bottom, chart_right, chart_bottom), fill="#5A7182", width=2)
    for i, (t, accuracy, recall, fpr) in enumerate(values):
        x = 225 + i * 275
        acc_h = int(accuracy * 2.5)
        rec_h = int(recall * 2.5)
        d.rounded_rectangle((x, chart_bottom - acc_h, x + 72, chart_bottom), radius=10, fill=MINT)
        d.rounded_rectangle((x + 82, chart_bottom - rec_h, x + 154, chart_bottom), radius=10, fill=AMBER)
        d.text((x - 3, chart_bottom + 16), labels[i], font=font(16, True), fill=TEXT)
        d.text((x - 2, chart_bottom - acc_h - 34), f"{accuracy:.2f}%", font=font(16, True), fill=MINT)
        d.text((x + 78, chart_bottom - rec_h - 34), f"{recall:.2f}%", font=font(16, True), fill=AMBER)
        d.text((x, 205), f"Open FPR {fpr:.2f}%", font=font(16), fill=MUTED)
    d.text((117, 680), "초록: Accuracy     ·     노랑: Closed Recall     ·     각 조건의 Open FPR은 막대 위", font=font(17), fill=MUTED)
    card(d, (1010, 190, 1516, 470), "선택 근거", "프로젝트 우선 조건을 Closed Recall ≥ 90%로 둠. CLAHE + 0.25는 90.69%를 유지하면서 baseline t=.50 대비 Open FPR을 35.09% → 13.48%로 낮췄습니다.", MINT, 20)
    card(d, (1010, 500, 1516, 710), "왜 0.40이 아닌가", "0.40은 정확도 최고 90.18%였지만 Closed Recall 86.66%로 우선 조건 미달. 0.50은 FPR이 더 낮지만 Closed Recall 81.12%까지 하락합니다.", AMBER, 20)
    d.text((88, 755), "통합 후보: MediaPipe 눈 ROI → CLAHE → YOLO26n-cls → P(Closed) ≥ 0.25. CEW에서 선택한 개발 후보이며 최종 독립 검증은 아닙니다.", font=font(17, True), fill=TEXT)
    source(d, "근거: reports/cew/threshold_analysis/threshold_sweep.csv · reports/cew/roi_ablation/summary.csv · reports/simuletic/external_summary.csv")
    return image


def slide_yawn_external():
    image, d = base("하품: 내부 99.06%와 외부 65.65% 사이의 간극", "07 / Yawn generalization", 8)
    metric(d, (84, 190, 430, 350), "99.06%", "내부 ROI 성공분", "210 / 212", MINT)
    metric(d, (452, 190, 798, 350), "98.13%", "내부 전체 파이프라인", "210 / 214", MINT)
    metric(d, (820, 190, 1166, 350), "65.65%", "Roboflow 전체 393장", "258 / 393", CORAL)
    metric(d, (1188, 190, 1516, 350), "44.44%", "외부 Yawn Recall", "76 / 171 · threshold .50", AMBER)
    card(d, (84, 390, 780, 650), "어디서 성능이 빠졌나", "입 ROI 성공 344/393 (87.53%). ROI 성공분 정확도 75.00%. 하품 73장을 no_yawn으로 놓쳤고, 비하품 13장을 yawn으로 오탐했습니다. ROI 실패 49장도 전체 파이프라인 성능에서 제외하지 않았습니다.", CORAL, 20)
    card(d, (820, 390, 1516, 650), "왜 ResNet18을 계속 후보로 두나", "팀 내부 비교에서 ResNet18이 99.06% (210/212)로 높았고, 기존 MediaPipe 입 ROI → RGB → ImageNet 정규화 경로가 준비돼 있습니다. 하지만 외부 점수는 낮아 ‘완성 모델’이 아니라 추가 조정·검증이 필요한 통합 후보입니다.", BLUE, 20)
    d.text((90, 705), "관찰 가설: 손으로 입 가림, 낮은 화질, 목·턱 주름이나 입술 대비가 오분류와 함께 나타났습니다. 원인 인과는 미확정.", font=font(18, True), fill=TEXT)
    source(d, "근거: reports/yawn_roboflow/external_summary.csv · error_audit/false_alarm_page_01.jpg · missed_yawn_page_01.jpg · 팀 결과 teammate_yawn/runs/03_yawn_resnet18/")
    return image


def slide_yawn_cases():
    image, d = base("하품 오류 사진: ROI가 있어도 문맥은 사라집니다", "08 / Yawn error examples", 9)
    false_roi = crop("reports/yawn_roboflow/error_audit/false_alarm_page_01.jpg", (330, 2 * 254, 570, 3 * 254))
    missed_roi = crop("reports/yawn_roboflow/error_audit/missed_yawn_page_01.jpg", (330, 0, 570, 254))
    variants = Image.open(ROOT / "reports/yawn_roboflow/preprocess_ablation/example_variants.jpg").convert("RGB")
    mouth_variants = variants.crop((0, 0, 950, 180))
    paste_image(image, false_roi, (84, 190, 500, 470))
    paste_image(image, missed_roi, (530, 190, 946, 470))
    paste_image(image, mouth_variants, (976, 190, 1516, 470))
    d = ImageDraw.Draw(image)
    d.text((97, 485), "비하품 → 하품 오탐 · Mouth ROI", font=font(17, True), fill=CORAL)
    d.text((543, 485), "하품 → 정상 미탐 · Mouth ROI", font=font(17, True), fill=AMBER)
    d.text((990, 485), "같은 입 패치, 전처리별 점수 변화", font=font(17, True), fill=MINT)
    card(d, (84, 535, 1516, 715), "발표할 때의 해석", "입 주변의 주름·치아·손가림·ROI 경계는 혼동 후보로 관찰됐습니다. 전처리 비교에서 grayscale은 일부 recall을 올렸지만 false positive도 늘었습니다. 이미지 몇 장은 설명용 사례이지, 모델 내부가 특정 주름이나 색에 ‘의존한다’는 증거는 아닙니다.", BLUE, 20)
    source(d, "근거: reports/yawn_roboflow/error_audit/* · reports/yawn_roboflow/preprocess_ablation/example_variants.jpg")
    return image


def slide_yawn_threshold():
    image, d = base("하품 threshold: 0.10은 더 민감하고, 0.15는 오탐을 조금 줄입니다", "09 / Yawn threshold & preprocessing", 10)
    rows = [
        ("RGB · .50", "65.65%", "44.44%", "85.39%", "13"),
        ("RGB · .10", "71.50%", "66.67%", "80.28%", "28"),
        ("RGB · .15", "71.50%", "64.33%", "82.09%", "24"),
        ("Gray · .50", "67.18%", "52.05%", "81.65%", "20"),
        ("Gray · .10", "61.83%", "67.84%", "63.04%", "68"),
    ]
    headers = ["조건", "전체 정확도", "Yawn Recall", "Precision", "FP"]
    xs = [100, 440, 690, 980, 1285]
    d.rounded_rectangle((84, 190, 1516, 248), radius=12, fill="#19364A")
    for x, label in zip(xs, headers):
        d.text((x, 207), label, font=font(18, True), fill=MINT)
    y = 260
    for idx, row in enumerate(rows):
        fill = "#15293A" if idx % 2 == 0 else "#102131"
        d.rounded_rectangle((84, y, 1516, y + 62), radius=10, fill=fill)
        for x, value in zip(xs, row):
            d.text((x, y + 17), value, font=font(20, idx in (1, 2)), fill=TEXT if idx not in (1, 2) else (MINT if idx == 1 else AMBER))
        y += 70
    card(d, (84, 640, 780, 770), "현재 후보", "RGB + threshold .10 / .15 조절 가능. .15은 .10과 같은 정확도, recall은 2.34pp 낮고 FP는 4건 적었습니다.", MINT, 18)
    card(d, (820, 640, 1516, 770), "주의", "이 수치는 Roboflow 개발용 외부셋에서 선택했습니다. 실제 웹캠 이벤트 기준의 검증과 별도 holdout 평가가 남아 있습니다.", CORAL, 18)
    source(d, "근거: reports/yawn_roboflow/threshold_analysis/yawn_threshold_sweep.csv · preprocess_ablation/summary.csv")
    return image


def slide_realtime():
    image, d = base("실시간 화면은 원본 영상과 AI 분석을 분리합니다", "10 / Runtime architecture", 11)
    steps = [
        ("원본 WebRTC", "카메라 프레임 즉시 반환", BLUE),
        ("최신 프레임 worker", "오래된 프레임 skip", MINT),
        ("MediaPipe ROI", "눈 2개 · 입 1개", AMBER),
        ("개별 모델", "눈 YOLO · 입 ResNet", MINT),
        ("경고 / 시뮬레이터", "별도 상태·음성", CORAL),
    ]
    x = 84
    for i, (title, body, accent) in enumerate(steps):
        box = (x, 205, x + 264, 345)
        rounded(d, box, PANEL, "#35536A", 16, 2)
        d.rounded_rectangle((x, 205, x + 264, 218), radius=6, fill=accent)
        paragraph(d, (x + 18, 240), title, font(21, True), TEXT, 228, 5)
        paragraph(d, (x + 18, 292), body, font(16), MUTED, 228, 4)
        if i < len(steps) - 1:
            d.line((x + 267, 275, x + 290, 275), fill=MINT, width=3)
        x += 286
    metric(d, (84, 390, 420, 540), "250 ms", "AI 분석 주기", "최신 프레임 우선", MINT)
    metric(d, (442, 390, 778, 540), "2.0 s", "눈 감김 지속 경고", "시연 로직", AMBER)
    metric(d, (800, 390, 1136, 540), "1.5 s", "하품 지속 경고", "시연 로직", AMBER)
    metric(d, (1158, 390, 1516, 540), "11.4 ms", "기존 비교 스크립트 눈 YOLO median", "프레임 전체 FPS와 다른 측정치", BLUE)
    card(d, (84, 580, 1516, 745), "웹캠 A/B 실측 기록 · 경고 횟수 비교이며 정확도 측정은 아님", "11/06 13:26 · 616회 / 52.7초: baseline 3회, CLAHE 2회 (GT 없음)  ·  13:31 · 245회 / 24.1초: 수동 눈 감김 3회, 양쪽 경고 0회  ·  13:34 · 177회: 수동 눈 감김 2회, 양쪽 경고 각 2회. 수동 라벨 시작·종료 시점이 부정확해 webcam accuracy로 환산하지 않았고, 당시 FPS 표시도 신뢰하지 않았습니다.", BLUE, 19)
    source(d, "근거: 11_streamlit_dashboard.py · 17_webcam_eye_ab_test.py · reports/webcam_eye_ab/*.csv · reports/model_comparison.csv")
    return image


def slide_safety():
    image, d = base("한 번 시작하면 다음 단계가 자동 진행되는 시뮬레이터", "11 / Safety response demo", 12)
    states = [
        ("눈 감김 경고", "실시간: 2초 지속\n5분 내 2회 누적", AMBER),
        ("가상 감속", "화면 속 속도만\n80 → 0 km/h", CORAL),
        ("정차 · 휴식", "창문은 화면 연출\n자동 재출발 없음", MINT),
        ("응답 대기", "실제 의식 감지 아님\n데모 타이머", BLUE),
        ("모의 SOS", "화면에만 표시\n실제 신고 없음", CORAL),
    ]
    x = 84
    for i, (title, body, accent) in enumerate(states):
        box = (x, 225, x + 260, 440)
        rounded(d, box, PANEL, "#35536A", 18, 2)
        d.ellipse((x + 20, 248, x + 62, 290), fill=accent)
        d.text((x + 78, 250), f"0{i + 1}", font=font(20, True), fill=accent)
        paragraph(d, (x + 20, 315), title, font(24, True), TEXT, 220, 6)
        paragraph(d, (x + 20, 360), body, font(17), MUTED, 220, 5)
        if i < len(states) - 1:
            d.line((x + 262, 333, x + 283, 333), fill=MINT, width=3)
        x += 286
    card(d, (84, 485, 780, 695), "발표 시연", "‘자동 시연 시작’ 한 번으로 모의 경고 2회 → 감속 → 휴식 → 응답 대기 → 타이머 종료 후 모의 SOS. 빠른 모드: 휴식 15초·응답 대기 30초.", MINT, 20)
    card(d, (820, 485, 1516, 695), "일반 시나리오", "최근 5분 내 실제 경고 2회면 가상 감속. 휴식 5분, 무반응 10분 값은 제품/의료 기준이 아닌 데모 숫자입니다. 실제 응급상황은 즉시 도움을 요청해야 합니다.", AMBER, 20)
    d.text((90, 735), "절대 범위: 실제 자동차 제어 없음 · 실제 119 연결 없음 · 실신/발작 진단 없음 · 움직임 모델 추가 없음", font=font(19, True), fill=TEXT)
    source(d, "근거: aura_safety_sim.py · 11_streamlit_dashboard.py · 자동 흐름은 화면 내 시뮬레이션")
    return image


def slide_close():
    image, d = base("최종 후보는 ‘완성 답’이 아니라, 근거가 있는 다음 실험입니다", "12 / Takeaways & limits", 13)
    card(d, (84, 190, 780, 390), "눈 감김 후보", "MediaPipe Eye ROI → CLAHE → YOLO26n-cls → P(Closed) ≥ 0.25\nCEW: accuracy 88.36% · Closed Recall 90.69% · Open FPR 13.48%\n선택 이유: Closed Recall 90% 조건을 만족하며 baseline 대비 오탐을 줄임. 개발셋 선택 결과.", MINT, 21)
    card(d, (820, 190, 1516, 390), "하품 후보", "MediaPipe Mouth ROI → RGB → ResNet18 → P(yawn) .10 / .15\nRoboflow에서 .10: recall 66.67%, FP 28\n.15: recall 64.33%, FP 24\n외부 전체 정확도 65.65%라 추가 검증 필요.", AMBER, 21)
    card(d, (84, 430, 780, 690), "발표에서 분명히 말할 한계", "인물 독립 holdout 부재 · 외부셋으로 설정 선택 · MRL 중복 가능성 · 하품 외부 Recall 낮음 · 가림/선글라스/ROI 실패는 UNKNOWN 필요 · 웹캠 A/B는 프레임 정확도 아님", CORAL, 20)
    card(d, (820, 430, 1516, 690), "이번 프로젝트의 기여", "완벽한 내부 점수만 제시하지 않고 외부 데이터와 오류 이미지를 확인했습니다. 분류와 ROI 성공률을 분리하고, threshold 선택의 trade-off 및 안전 시뮬레이션의 범위를 명시했습니다.", BLUE, 20)
    d.text((90, 748), "AURA DRIVE는 졸음 징후 연구·시연 프로토타입이며 의료기기나 차량 안전 시스템이 아닙니다.", font=font(20, True), fill=TEXT)
    source(d, "수치 출처는 각 슬라이드 하단의 로컬 reports CSV 및 학습 로그 경로를 참조")
    return image


def make_preview(images):
    thumb_w, thumb_h = 400, 225
    cols = 3
    rows = (len(images) + cols - 1) // cols
    montage = Image.new("RGB", (cols * thumb_w, rows * thumb_h), "#06101A")
    for index, image in enumerate(images):
        thumb = image.resize((thumb_w, thumb_h), Image.Resampling.LANCZOS)
        montage.paste(thumb, ((index % cols) * thumb_w, (index // cols) * thumb_h))
    montage.save(OUT / "AURA_DRIVE_slide_overview.jpg", quality=88, optimize=True)


def build():
    OUT.mkdir(parents=True, exist_ok=True)
    images = [
        slide_cover(), slide_internal_external(), slide_data(), slide_epochs(),
        slide_external_sets(), slide_eye_errors(), slide_eye_threshold(),
        slide_yawn_external(), slide_yawn_cases(), slide_yawn_threshold(),
        slide_realtime(), slide_safety(), slide_close(),
    ]
    deck = Presentation()
    deck.slide_width = Inches(13.333333)
    deck.slide_height = Inches(7.5)
    for image in images:
        slide = deck.slides.add_slide(deck.slide_layouts[6])
        buffer = BytesIO()
        image.save(buffer, format="JPEG", quality=91, optimize=True)
        buffer.seek(0)
        slide.shapes.add_picture(buffer, 0, 0, width=deck.slide_width, height=deck.slide_height)
    pptx_path = OUT / "AURA_DRIVE_발표자료_완성본.pptx"
    pdf_path = OUT / "AURA_DRIVE_발표자료_완성본.pdf"
    deck.save(pptx_path)
    pdf_images = [image.convert("RGB") for image in images]
    pdf_images[0].save(pdf_path, "PDF", save_all=True, append_images=pdf_images[1:], resolution=150.0)
    make_preview(images)
    print(f"Created {len(images)} slides")
    print(pptx_path)
    print(pdf_path)
    print(OUT / "AURA_DRIVE_slide_overview.jpg")


if __name__ == "__main__":
    build()
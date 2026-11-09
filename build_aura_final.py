"""33-slide evidence-based talk, editable PPTX, slide PDF and speaker PDF.

No video is embedded. All figures are exported separately for reuse.
"""
from __future__ import annotations
import csv
import json
import hashlib
from collections import Counter
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import build_aura_revalidation as base
from build_aura_revalidation import Deck, read, pct, ROOT, INK, TEAL, RED, BLUE, GRAY, NAVY, WHITE, LIGHT, FONT, BOLD

OUT=ROOT/'reports/presentation/final_37slides'
TOTAL=37
ASSETS=OUT/'figures'
TESLA='https://www.tesla.com/ownersmanual/model3/en_nl/GUID-11E01B0B-DC74-4B8D-B979-DA37EC6F9093.html'
LIMITS='https://www.tesla.com/ownersmanual/model3/en_lt/GUID-BB4CE0C1-D4D7-4B31-B780-85C3EFCFFF12.html'
NIR='https://www.iosb.fraunhofer.de/content/dam/iosb/iosbtest/documents/projekte/anonymos/White%20Paper%20-%20In%20Cabin%20Monitoring.pdf'
plt.rcParams.update({'font.family':'Malgun Gothic','axes.unicode_minus':False,'font.size':13,
                     'axes.spines.top':False,'axes.spines.right':False,'axes.edgecolor':'#CCD5DA'})


class FinalDeck(Deck):
    def text(self,x,y,w,text,size=28,color=INK,bold=False):
        if y==848 and str(text).endswith(' / 33'):
            text=str(text).replace(' / 33',f' / {TOTAL}')
        if y==866:text=str(text).replace('2026.11.08','2026.11.09')
        return super().text(x,y,w,text,size,color,bold)


def find(rows, **conditions):
    matches=[r for r in rows if all(str(r.get(k))==str(v) for k,v in conditions.items())]
    assert len(matches)==1,(conditions,len(matches))
    return matches[0]


def dump_csv(name, rows):
    with (OUT/name).open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def export_plot(name, labels, series, ylabel='비율 (%)'):
    fig,ax=plt.subplots(figsize=(11,4.2),layout='constrained')
    colors=[TEAL,BLUE,RED];width=.72/len(series)
    for j,(title,values) in enumerate(series):
        x=[i+(j-(len(series)-1)/2)*width for i in range(len(labels))]
        bars=ax.bar(x,values,width,label=title,color=colors[j])
        ax.bar_label(bars,fmt='%.2f',padding=3,fontsize=11)
    ax.set_xticks(range(len(labels)),labels);ax.set_ylim(0,112);ax.set_ylabel(ylabel)
    ax.set_yticks([0,25,50,75,100]);ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)
    ax.legend(loc='upper center',bbox_to_anchor=(.5,1.15),ncols=len(series),frameon=False)
    p=ASSETS/(name+'.png');fig.savefig(p,dpi=170);fig.savefig(ASSETS/(name+'.svg'));plt.close(fig)
    return p


def training_plot():
    fig,axes=plt.subplots(1,2,figsize=(12,4),layout='constrained')
    paths=[ROOT/'runs/eyes_yolo26n-cls/results.csv',ROOT/'teammate_yawn/runs/03_yawn_resnet18/03_results.csv']
    for ax,p,title in zip(axes,paths,['눈 · YOLO26n','하품 · 입 ROI ResNet18']):
        rows=read(p);x=[int(r['epoch']) for r in rows]
        train='train/loss' if 'train/loss' in rows[0] else 'train_loss'
        val='val/loss' if 'val/loss' in rows[0] else 'val_loss'
        ax.plot(x,[float(r[train]) for r in rows],label='학습 손실',color=BLUE)
        ax.plot(x,[float(r[val]) for r in rows],label='검증 손실',color=TEAL)
        ax.set(title=title,xlabel='에포크',ylabel='손실');ax.legend(frameon=False);ax.grid(alpha=.15)
    p=ASSETS/'training_curves.png';fig.savefig(p,dpi=170);fig.savefig(ASSETS/'training_curves.svg');plt.close(fig)
    return p


def save(d):
    assert len(d.images)==TOTAL,len(d.images)
    assert not d.overflow,d.overflow
    d.prs.save(OUT/f'AURA_DRIVE_상세발표_{TOTAL}장.pptx')
    d.images[0].save(OUT/f'AURA_DRIVE_상세발표_{TOTAL}장.pdf','PDF',save_all=True,append_images=d.images[1:],resolution=144)
    for i,im in enumerate(d.images): im.save(OUT/f'slide_{i+1:02d}.png')
    for page,start in enumerate(range(0,TOTAL,12)):
        im=Image.new('RGB',(1600,900),'#DCE5E9')
        for j,slide in enumerate(d.images[start:start+12]): im.paste(slide.resize((400,225)),((j%4)*400,(j//4)*225))
        im.save(OUT/f'overview_{page+1}.jpg',quality=93)
    (OUT/'발표대본.md').write_text('\n\n'.join(f"## {n['slide']:02d}. {n['title']}\n\n{n['notes']}\n\n근거: {n['source']}" for n in d.notes),encoding='utf8')
    (OUT/'slide_manifest.json').write_text(json.dumps(d.notes,ensure_ascii=False,indent=2),encoding='utf8')
    # One slide per A4 speaker page: readable Korean, actual slide thumbnail, full notes.
    pages=[]
    for i,n in enumerate(d.notes):
        im=Image.new('RGB',(1240,1754),'white');draw=ImageDraw.Draw(im)
        draw.text((78,60),f"AURA DRIVE  |  발표대본  {i+1:02d}/{TOTAL}",font=ImageFont.truetype(BOLD,28),fill=TEAL)
        im.paste(d.images[i].resize((1080,608)),(80,120))
        y=775
        def wrap(text, size, color=INK):
            nonlocal y
            font=ImageFont.truetype(FONT,size);line=''
            for ch in text:
                if ch=='\n' or (line and draw.textlength(line+ch,font=font)>1070):
                    draw.text((84,y),line,font=font,fill=color);y+=int(size*1.65);line=''
                if ch!='\n':line+=ch
            if line:draw.text((84,y),line,font=font,fill=color);y+=int(size*1.65)
        wrap(n['notes'],25);y+=30;wrap('근거: '+n['source'],19,GRAY)
        assert y<1680,(i+1,y)
        draw.text((84,1690),'발표 권장: 22~27분 · 실험 수치의 분모와 조건을 함께 설명',font=ImageFont.truetype(FONT,19),fill=GRAY)
        if i in (0,27,30): im.save(OUT/f'speaker_preview_{i+1:02d}.png')
        pages.append(im)
    pages[0].save(OUT/'AURA_DRIVE_발표대본.pdf','PDF',save_all=True,append_images=pages[1:],resolution=150)


def build():
    OUT.mkdir(parents=True,exist_ok=True);ASSETS.mkdir(exist_ok=True)
    eye_verified=json.loads((ROOT/'reports/presentation/final_review/eyes_baseline_verified.json').read_text())
    (OUT/'eyes_baseline_verified.json').write_text(json.dumps(eye_verified,indent=2),encoding='utf8')
    selection=json.loads((ROOT/'reports/final_model_selection/selected_models.json').read_text(encoding='utf8'))
    face_verified=read(ROOT/'reports/final_model_selection/three_models_verified.csv')
    eye_thr=read(ROOT/'reports/cew/threshold_analysis/threshold_sweep.csv')
    eye_extra=read(ROOT/'reports/hardcase_finetune/eyes_summary.csv')
    ythr=read(ROOT/'reports/yawn_roboflow/threshold_analysis/yawn_threshold_sweep.csv')
    compare=read(ROOT/'reports/yawn_external_model_compare/summary.csv')
    ep=read(ROOT/'reports/hardcase_finetune/eyes_predictions.csv')
    annotations=read(ROOT/'reports/hardcase_finetune/annotations.csv')
    audit=read(ROOT/'reports/cew/error_audit/pairwise_images.csv')
    yerrors=read(ROOT/'reports/yawn_roboflow/error_audit/all_cases.csv')
    eye_settings=[find(eye_thr,model='yolo26n',variant=v,threshold=t) for v,t in [('baseline','0.500'),('clahe','0.500'),('clahe','0.250')]]
    y_settings=[next(r for r in ythr if abs(float(r['threshold'])-t)<1e-8) for t in [.5,.15,.1]]
    # Verify headline figures against integer confusion counts, not rounded prose.
    verified=[]
    for r in eye_settings:
        verified.append(dict(task='eyes',setting=r['variant']+'@'+r['threshold'],n=int(r['total']),correct=int(r['tp'])+int(r['tn']),tp=int(r['tp']),positive=int(r['closed_total']),fp=int(r['fp']),unknown=int(r['roi_fail']),accuracy=(int(r['tp'])+int(r['tn']))/int(r['total']),recall=int(r['tp'])/int(r['closed_total'])))
    for r in y_settings:
        verified.append(dict(task='yawn',setting='RGB@'+r['threshold'],n=int(r['total']),correct=int(r['tp'])+int(r['tn']),tp=int(r['tp']),positive=171,fp=int(r['fp']),unknown=int(r['unknown']),accuracy=(int(r['tp'])+int(r['tn']))/int(r['total']),recall=int(r['tp'])/171))
    for r in compare:
        assert abs(float(r['accuracy'])-(int(r['tp'])+int(r['tn']))/int(r['total']))<1e-6
        assert abs(float(r['recall'])-int(r['tp'])/int(r['positive_count']))<1e-6
    dump_csv('headline_metrics_verified.csv',verified)
    d=FinalDeck()
    def slide(title, sub, source, note, dark=False): d.slide(title,sub,source,note,dark)
    def table(headers, rows, widths, size=25, rowh=78):
        d.table(headers,rows,widths,size=size,rowh=rowh)
        name=f'table_{len(d.images):02d}'
        d.im.crop((84,234,1516,234+rowh*(len(rows)+1))).save(ASSETS/(name+'.png'))
        with (ASSETS/(name+'.csv')).open('w',encoding='utf-8-sig',newline='') as f:
            writer=csv.writer(f);writer.writerow(headers);writer.writerows(rows)
    def items(rows):
        for i,(title,body) in enumerate(rows):
            y=244+i*150;d.text(90,y,120,f'{i+1:02d}',45,TEAL,True);d.text(220,y,1260,title,33,INK,True);d.text(220,y+53,1260,body,26,GRAY)

    # 01
    slide('AURA DRIVE','이미지 분류에서 출발한 운전자 상태 모니터링', '로컬 학습·평가 기록 검증 · 2026.11.08',
          '저희는 눈 감김과 하품이라는 두 가지 시각적 행동을 분류하고, 지속 시간에 따라 경고하는 데모를 만들었습니다. 프로젝트의 핵심은 높은 내부 정확도를 그대로 믿지 않고, 실제 촬영 사진과 외부 데이터로 검증했다는 점입니다. 데이터 수집과 전처리, 세 가지 모델의 선택 이유, 오류 분석과 개선 과정을 눈 모델부터 설명하겠습니다. 마지막에는 Streamlit 구현 범위와 야간·선글라스·응급 대응의 한계를 말씀드리겠습니다.',True)
    d.pic(ROOT/'aura_car_interior.jpg',795,235,720,455)
    d.text(90,275,690,'데이터를 준비하고,\n모델을 비교하고,\n현실의 입력으로 검증하다.',51,WHITE,True)
    d.text(90,600,660,'눈 감김 · 하품 분류\n전처리 / 외부 평가 / 로컬 웹캠 데모',28,'#BFD7DF')
    # 02
    slide('눈과 입을 두 개의 이진 분류 문제로 나눴습니다','졸음의 시각적 징후를 관찰하는 프로젝트', 'common.py · 03_yawn_common.py · 클래스 폴더',
          '눈은 Open과 Closed, 입은 no_yawn과 yawn으로 나눴습니다. 두 부위는 동시에 다른 상태일 수 있으므로 네 클래스를 하나의 모델로 묶기보다 각각 분류하도록 구성했습니다. 눈 감김과 하품은 카메라로 관찰할 수 있고 화면·소리 경고로 연결하기 쉬워 선택했습니다. 다만 눈을 감았다고 모두 졸음인 것은 아니며, 하품만으로 의식 상태나 질병을 진단할 수도 없습니다. 여기서 구현한 것은 행동 분류와 경고 시나리오입니다.')
    table(['입력 부위','클래스','양성 기준','활용'],[['눈','Open / Closed','Closed','지속적인 눈 감김 경고'],['입','no_yawn / yawn','yawn','하품 행동 알림']],[260,430,270,472],rowh=110)
    d.text(100,570,1360,'관찰 가능한 행동 → 상태 분류 → 지속 시간 확인 → 경고',37,TEAL,True)
    d.callout('졸음·기절·발작을 직접 진단하는 모델은 아닙니다.')
    # 03
    slide('출발점은 네 종류의 이미지 2,900장이었습니다','원본 폴더 라벨을 확인하고 작업별로 다시 분할', 'data/raw · data/processed/manifest.csv · 01_audit_data.py',
          '로컬 원본에는 Closed 726장, Open 726장, 비하품 725장, 하품 723장이 있습니다. 학습 전에 파일을 열 수 있는지, 라벨과 폴더가 맞는지, 중복은 없는지 확인했습니다. 기존 train과 test 폴더 이름을 그대로 신뢰하지 않고 정제 후 다시 나눴습니다. 공개 수집 이미지와 직접 촬영 사진, 이후 외부 평가 자료는 역할을 구분했습니다. 원본 다운로드 페이지와 이용 조건이 완전히 보존되지 않은 자료는 출처 추적의 한계로 남겼으며 임의의 출처 주소를 붙이지 않았습니다.')
    table(['클래스','원본 수','작업'],[['Closed','726','눈'],['Open','726','눈'],['no_yawn','725','하품'],['yawn','723','하품']],[470,330,632],rowh=81)
    d.callout('학습 자료 · 직접 촬영 검증 · 외부 평가 자료를 분리해 관리했습니다.')
    # 04
    slide('중복 정제와 입력 규격 통일을 학습보다 먼저 했습니다','원본 확인 → 유사 이미지 그룹 → 분할 → 224×224 패딩', '02_prepare.py · common.py · data/processed/manifest.csv',
          '완전 중복은 해시로 제거하고, 동일 이미지에 상충하는 라벨이 있으면 제외하도록 했습니다. 유사 이미지는 dHash 해밍 거리 5 이하를 기준으로 그룹화해 같은 그룹이 분할을 넘지 않게 했습니다. 이후 약 70 대 15 대 15로 학습·검증·테스트를 나누고 종횡비를 유지한 224 곱하기 224 패딩을 적용했습니다. 눈은 1017·218·217장, 하품은 1016·218·214장입니다. 이는 유사 이미지 단위 분리이지, 모든 사람의 신원이 완전히 분리됐음을 보장하는 것은 아닙니다.')
    d.flow([('읽기·라벨 검사','손상·오라벨 확인'),('중복 정제','해시 / dHash'),('그룹 분할','약 70 / 15 / 15'),('입력 규격','224×224 패딩')],565)
    table(['작업','학습','검증','테스트'],[['눈','1,017','218','217'],['하품 원본','1,016','218','214']],[520,300,300,312],rowh=70)
    # 05
    slide('눈 모델은 속도와 구조가 다른 세 가지로 비교했습니다','사전학습된 분류 모델을 두 클래스에 맞춰 전이학습', '03_train.py · 08_train_cnn.py · 저장된 체크포인트',
          'YOLO26n-cls는 작은 분류 모델로 로컬 CPU에서도 사용할 수 있는 경량 후보였습니다. ResNet18은 잔차 연결을 쓰는 대표적인 CNN으로 비교 기준을 만들기 좋았습니다. MobileNetV3-Small은 경량 연산을 목표로 한 구조여서 크기와 지연을 비교하기 위해 선택했습니다. 여기서 YOLO는 물체 검출 박스를 학습하는 모델이 아니라 분류 버전입니다. 얼굴과 눈의 위치는 MediaPipe가 찾습니다. 사전학습 가중치를 활용한 전이학습이며 별도의 상용 클라우드 판별 API를 호출한 것은 아닙니다.')
    table(['모델','선택 이유','입력과 출력'],[['YOLO26n-cls','경량 분류 · CPU 데모 후보','224×224 → Open / Closed'],['ResNet18','잔차 CNN · 비교 기준','동일 클래스·분할'],['MobileNetV3-Small','작은 모델 · 연산량 비교','동일 클래스·분할']],[390,590,452],rowh=103)
    d.callout('ROI 검출은 MediaPipe, 상태 분류는 YOLO/CNN이 담당합니다.')
    # 06
    slide('학습 손실과 검증 손실을 함께 확인했습니다','사전학습 가중치 활용 · 학습용 증강 · 검증 기반 모델 저장', 'runs/eyes_yolo26n-cls/results.csv · teammate_yawn/runs/03_yawn_resnet18/03_results.csv',
          '그래프는 실제 저장된 에포크별 손실입니다. 눈 YOLO 학습 설정은 최대 25에포크, 배치 16, seed 42, 조기 종료 patience 7입니다. 눈 CNN은 학습 이미지에 좌우 반전과 작은 회전을 적용하고 ImageNet 정규화를 사용했습니다. 하품 ResNet은 입 ROI에 좌우 반전과 약한 색 증강을 사용합니다. 손실이 내려갔다는 사실은 학습이 진행됐다는 근거지만, 새 얼굴과 조명에서 잘 동작한다는 증거까지 되지는 않습니다. 그래서 테스트 다음에 실제 입력과 외부 검증을 추가했습니다.')
    d.pic(training_plot(),90,240,1420,425)
    d.callout('학습 곡선이 좋아도 새로운 촬영 조건에서의 성능은 별도로 확인해야 합니다.')
    # 07
    slide('내부 217장에서는 세 눈 모델이 모두 100%였습니다','저장된 모델을 같은 테스트셋으로 재평가한 결과', 'eyes_baseline_verified.json · 09_compare_models_retry.test_model · CPU 4 threads',
          '같은 내부 테스트 217장으로 세 모델을 다시 확인했습니다. Closed 102장과 Open 115장을 모두 맞혀 정확도와 Closed 재현율이 각각 100%였습니다. 이 결과만으로 과적합이라고 단정할 수는 없습니다. 데이터가 쉬웠거나 유사한 촬영 환경이 반복됐거나 학습과 평가의 분포가 비슷했을 가능성도 있습니다. 모델 간 차이가 드러나지 않았으므로 파일 크기와 처리 시간을 함께 보고, 외부 얼굴에서도 같은 결과가 나오는지 확인했습니다. 표의 지연은 단일 분류 처리이며 전체 웹캠 지연이 아닙니다.')
    table(['모델','정확도','Closed 재현율','크기','CPU 중앙 지연'],[[r['model'],pct(r['accuracy']),pct(r['recall']),f"{r['model_mb']:.1f} MB",f"{r['latency_ms_median']:.1f} ms"] for r in eye_verified],[370,245,300,230,287],rowh=90,size=23)
    d.callout('100%는 내부 분할의 결과입니다. 일반화 성능을 확인할 질문이 남았습니다.')
    # 08
    shot=sorted((ROOT/'data/webcam_test/eyes/Open').glob('*.jpg'))[0]
    roi=ROOT/'reports/eye_roi_examples'/('Open_'+shot.stem+'_eye1.jpg')
    slide('직접 찍은 얼굴에서는 입력 부위부터 맞춰야 했습니다','같은 65장 · 얼굴 전체 → MediaPipe 눈 ROI', 'reports/eyes_yolo26n-cls_extra_confusion.csv · reports/eyes_roi_predictions.csv',
          '눈 패치로 학습한 모델에 얼굴 전체를 넣었더니 65장 중 35장만 맞혀 53.85%였습니다. MediaPipe로 눈 영역을 잘라 넣은 뒤에는 전체 65장 중 55장을 맞혀 84.62%로 올랐습니다. 눈 ROI가 추출된 60장만 보면 55장 정답으로 91.67%입니다. 추출 실패 5장을 제외한 조건부 수치이므로 84.62%와 섞어서 말하지 않습니다. 이 실험은 모델을 새로 학습하기 전에 학습 입력과 실제 입력의 부위와 크기를 맞추는 것이 중요하다는 근거였습니다.')
    d.pic(shot,90,250,380,340);d.pic(roi,510,290,260,220)
    d.text(810,270,690,'53.85% → 84.62%',49,TEAL,True)
    d.text(815,365,650,'얼굴 전체 35/65 → 눈 ROI 55/65',27,INK)
    d.text(815,465,650,'ROI 성공분 91.67%',45,INK,True)
    d.text(815,540,650,'55/60 · 추출 실패 5장 제외',28,GRAY)
    d.callout('이 결과는 과거 ROI 적용 실험이며, 추가 학습 전후 실험과 구분합니다.')
    # 09
    slide('외부 데이터를 가져오자 모델 차이가 나타났습니다','출처·촬영 조건·입력 형태를 구분해 평가', 'reports/{cew,mrl,simuletic}/external_summary.csv',
          '외부 눈 평가는 CEW 2423장, MRL 4000장, Simuletic 143장으로 나눠 수행했습니다. MRL은 이미 잘린 눈 패치이며 학습 원자료와 중복 가능성이 있어 강한 독립 검증으로 해석하지 않았습니다. CEW는 얼굴에서 눈을 추출하는 조건으로, 내부 테스트보다 현실적인 입력 차이를 보여줬습니다. 기본 전처리에서 YOLO 정확도는 79.12%, ResNet은 81.18%, MobileNet은 81.55%였습니다. YOLO의 Closed 재현율은 높았지만 Open을 Closed로 보는 오탐이 많아 후속 전처리 실험을 진행했습니다.')
    table(['평가셋','규모 / 입력','해석'],[['CEW','2,423 / 얼굴 → 눈 ROI','전처리·임계값 개발에도 사용'],['MRL','4,000 / 눈 패치','학습 원자료 중복 가능'],['Simuletic','143 / 운전석 얼굴','작은 표본 · 조건 제한']],[300,550,582],rowh=80)
    d.text(95,600,1390,'CEW 기본 정확도  YOLO 79.12% · ResNet 81.18% · MobileNet 81.55%',30,TEAL,True)
    d.callout('내부 100%와 외부 결과의 차이는 일반화 점검이 필요하다는 신호였습니다.')
    # 10
    chosen=[]
    for name in ['Alina_Kabaeva','Anjelica_Huston','Abid_Hamid']:
        matches=[r for r in audit if name in r['image'] and r['actual']=='Open' and r['yolo_pred']=='Closed']
        if matches:chosen.append(matches[0])
    for r in audit:
        if len(chosen)==3:break
        if r['actual']=='Open' and r['yolo_pred']=='Closed' and r not in chosen:chosen.append(r)
    slide('오류 사진에서 눈 주변의 경계와 색을 살펴봤습니다','정답 Open → 예측 Closed인 실제 CEW 사례', 'reports/cew/error_audit/pairwise_images.csv',
          '이 사진들은 CEW 오류 감사 기록에 남아 있는 실제 오탐입니다. 정답은 Open인데 기존 YOLO가 Closed로 분류했습니다. 눈 화장, 짙은 눈 주변 색, 낮은 해상도와 흐릿한 경계가 오분류와 함께 관찰됐습니다. 다만 사진만 보고 아이라인이나 주름이 직접 원인이라고 단정할 수는 없습니다. 이러한 관찰을 전처리와 조건별 데이터 보강을 시도하는 가설로 사용했습니다. 오류를 설명하려면 전체 얼굴뿐 아니라 실제 모델에 들어간 눈 ROI도 확인해야 합니다.')
    for j,r in enumerate(chosen):
        x=84+j*485;d.pic(r['image'],x,240,440,365);d.text(x,630,440,'Open → Closed',29,RED,True)
    d.callout('화장·흐림·가림은 관찰한 조건입니다. 인과적 오류 원인은 아직 확인하지 못했습니다.')
    # 11
    r=next(r for r in ep if r['dataset']=='cew' and r['actual']=='Open' and r['crops'] and float(r['before_probability'])>=.25)
    raw=Image.open(r['crops'].split('|')[0]).convert('RGB')
    import importlib
    clahe=importlib.import_module('15_roi_ablation').make_variants(raw)['clahe']
    raw.save(ASSETS/'eye_roi_raw.png');clahe.save(ASSETS/'eye_roi_clahe.png')
    slide('눈 ROI의 국소 대비를 CLAHE로 조정했습니다','눈꺼풀 주변 밝기 경계를 보완하는 전처리 실험', '15_roi_ablation.py · LAB L채널 · clipLimit 2.0 · tileGridSize 4×4',
          '왼쪽과 오른쪽은 같은 눈 ROI의 전처리 전후입니다. RGB 이미지를 LAB 색 공간으로 바꾼 뒤 밝기인 L채널에 CLAHE를 적용하고 다시 RGB로 되돌렸습니다. 극단적인 대비 증폭을 제한하면서 국소 경계를 강조하려는 목적입니다. 이후 224 곱하기 224로 패딩해 기존 분류기에 넣었습니다. 이 단계에서는 모델 가중치를 바꾸지 않았습니다. 전체 CEW 지표가 좋아졌다는 사실과 특정 화장 유형의 오류가 해결됐다는 주장은 서로 다른 내용입니다.')
    d.pic(raw,110,280,600,280);d.pic(clahe,880,280,600,280)
    d.text(110,590,600,'동일 눈 ROI · 원본',32,INK,True);d.text(880,590,600,'CLAHE 적용',32,TEAL,True)
    d.callout('같은 가중치로 입력만 바꿔 전처리 효과를 확인했습니다.')
    # 12
    slide('정확도만 높이기보다 눈 감김을 놓치지 않도록 조정했습니다','CLAHE 이후 Closed 임계값 0.50 → 0.25', 'reports/cew/threshold_analysis/threshold_sweep.csv · CEW 2,423장',
          '기존 RGB의 정확도는 79.12%이고 Closed 재현율은 94.21%였지만 Open 오탐률이 35.09%로 높았습니다. CLAHE만 적용하면 정확도는 88.90%로 높아지지만 Closed 재현율이 81.12%로 떨어졌습니다. 그래서 눈 감김을 더 잡는 방향으로 확률 기준을 0.25로 낮췄습니다. 정확도는 88.36%, Closed 재현율은 90.69%, Open 오탐률은 13.48%였습니다. 가중치 재학습이 아니라 판정 민감도 조정입니다. CEW를 설정 선택에도 썼으므로 이 결과는 독립 최종 시험 성능이 아닙니다.')
    table(['전처리 / 임계값','정확도','Closed 재현율','Open 오탐률'],[[label,pct(r['accuracy']),pct(r['closed_recall']),pct(r['open_false_alarm_rate'])] for label,r in zip(['RGB / 0.50','CLAHE / 0.50','CLAHE / 0.25'],eye_settings)],[500,280,340,312],rowh=95)
    d.callout('임계값은 분류 기준입니다. 지속 시간 2초와도 서로 다른 설정입니다.')
    export_plot('eye_preprocessing',[ 'RGB\n0.50','CLAHE\n0.50','CLAHE\n0.25'],[('정확도',[float(r['accuracy'])*100 for r in eye_settings]),('Closed 재현율',[float(r['closed_recall'])*100 for r in eye_settings])])
    # 13
    slide('화장 조건을 확인하기 위해 별도 이미지를 모았습니다','화장 전·후 폴더명은 조건이며, 눈 정답 라벨과 다릅니다.', 'data/error_data · reports/hardcase_finetune/annotations.csv',
          '추가 폴더에는 화장 전 26장, 화장 후 26장, 주름 조건 459장이 있었습니다. 이 폴더명을 곧바로 Open이나 하품 라벨로 쓰지 않고 이미지별로 눈과 입을 확인했습니다. 눈은 보이는 Open 51장만 라벨이 확정됐고 불확실한 사진은 제외했습니다. 화장 전후 같은 사람의 사진은 같은 분할에 넣었습니다. 이 판독은 AI 보조 육안 검토이며 독립적인 사람 검수까지 완료된 정답은 아닙니다. Closed 사례가 없어 화장한 눈의 감김까지 해결했는지 평가하기에는 부족합니다.')
    for j,condition in enumerate(['no_makeup','with_makeup']):
        rr=next(r for r in annotations if r['condition']==condition and r['index']=='0')
        d.pic(rr['image'],100+j*735,250,630,390)
        d.text(100+j*735,660,630,'화장 전 · 같은 번호의 사진' if j==0 else '화장 후 · 같은 번호의 사진',29,INK,True)
    # 14
    eb=find(eye_extra,dataset='cew',condition='ALL',split='external',phase='before');ea=find(eye_extra,dataset='cew',condition='ALL',split='external',phase='after')
    slide('추가 눈 데이터로는 분류층만 작게 재학습했습니다','기존 데이터와 혼합 · 동일 전처리와 임계값에서 비교', 'reports/hardcase_finetune/eyes_config.json · eyes_summary.csv',
          '전처리 실험 다음에는 실제 가중치를 바꾸는 별도 실험도 수행했습니다. 특징을 추출하는 본체는 고정하고 마지막 분류층만 학습했습니다. 기존 일반 데이터 손실 80%, 새 데이터 손실 20%를 사용했고 학습률은 0.0001, 최대 40회 업데이트였습니다. 검증 손실로 후보를 선택한 뒤 동일한 CEW와 임계값 0.25로 비교했습니다. 정확도는 88.36%에서 88.90%, Closed 재현율은 90.69%에서 91.11%로 변했습니다. 소폭 변화이며 특정 화장 조건의 해결이나 실차 성능으로 일반화하지 않았습니다.')
    table(['CEW 2,423장','기존 가중치','추가 학습 후보'],[['정확도',pct(eb['accuracy']),pct(ea['accuracy'])],['Closed 재현율',pct(eb['recall']),pct(ea['recall'])],['Open 오탐 수',eb['fp'],ea['fp']]], [620,400,412],rowh=98)
    d.callout('전처리·임계값 변경과 가중치 추가 학습은 별개의 실험입니다.')
    # 15
    slide('눈 모델의 개선 범위는 평가셋마다 달랐습니다','추가 학습 후보와 기존 모델을 같은 조건에서 재검증', 'reports/hardcase_finetune/eyes_summary.csv · threshold 0.25',
          'CEW에서는 정답이 13장 늘었지만, Simuletic은 140장 정답에서 139장으로 바뀌었습니다. 직접 촬영 사진은 이번 전처리 조건에서 58장 정답으로 변화가 없었습니다. 직접 촬영의 이번 89.23%는 앞에서 본 과거 ROI 실험 84.62%와 입력 크기·CLAHE·임계값 조건이 달라 구분해야 합니다. 새 화장 조건 테스트는 Open 12장 모두 기존부터 맞혔으므로 개선 근거가 되지 않습니다. 따라서 앱은 검증해 사용하던 기존 눈 가중치를 유지하고 추가 학습 파일은 별도 후보로 보관합니다.')
    rows=[]
    for ds,label in [('cew','CEW · 2,423장'),('simuletic','Simuletic · 143장'),('webcam','직접 촬영 · 65장')]:
        b=find(eye_extra,dataset=ds,condition='ALL',split='external',phase='before');a=find(eye_extra,dataset=ds,condition='ALL',split='external',phase='after')
        rows.append([label,pct(b['accuracy']),pct(a['accuracy'])])
    table(['동일 조건 비교','기존 가중치','추가 학습 후보'],rows,[620,400,412],rowh=100)
    d.callout('추가 학습 후보는 앱에 자동으로 교체하지 않았습니다.')
    # 16
    slide('눈 모델에서 얻은 결론은 입력 일치와 검증의 중요성입니다','내부 점수 → 직접 촬영 → 외부 평가 → 오류 관찰 → 개선', '눈 실험 종합 · 06 / 15 / 16 / 22 / 23번 코드',
          '눈 모델은 내부 100%라는 숫자에서 출발했지만, 얼굴 전체 입력과 외부 얼굴에서는 다른 결과가 나왔습니다. 첫 번째 해결은 모델 교체가 아니라 눈 ROI를 사용해 입력을 맞추는 일이었습니다. 두 번째는 밝기 대비와 판정 민감도를 조정하는 일이었습니다. 세 번째 추가 학습에서는 소폭 변화가 있었지만 모든 조건에서 개선되지는 않았습니다. 이 과정을 통해 데이터 조건과 전처리, 임계값, 모델 가중치를 각각 구분해서 평가해야 한다는 점을 확인했습니다. 다음은 같은 질문을 하품 모델에 적용한 과정입니다.')
    items([('입력 부위를 맞춘다','눈 패치 학습 모델에는 얼굴 전체가 아닌 눈 ROI'),('놓침과 오탐을 같이 본다','정확도 + Closed 재현율 + Open 오탐률'),('관찰에서 검증으로 넘어간다','화장·안경·가림 조건별 라벨과 새 인물 테스트 필요')])
    # 17
    slide('하품은 전체 이미지와 입 ROI 두 경로를 비교했습니다','no_yawn / yawn · 학습한 입력 형태에 맞춰 모델을 평가', '03_train.py · 08_train_cnn.py · teammate_yawn/03_yawn_common.py',
          '하품 실험에는 두 경로가 있습니다. 원본 얼굴 이미지 전체를 쓰는 YOLO26n, YOLO26s, MobileNetV3-Small과, MediaPipe로 입 ROI를 잘라 쓰는 팀 ResNet18입니다. 전체 입력은 RGB 이미지를 종횡비를 유지하며 224 곱하기 224로 패딩했습니다. 입 ROI 경로는 입 너비 배율 1.8, 높이 배율 1.35, 중심 보정 0.08로 주변 문맥을 남겼습니다. 모델이 학습한 입력 형태를 유지해야 하므로 두 경로를 같은 전처리라고 부르지 않습니다. 입 ROI ResNet을 먼저 데모에 연결했지만, 최종 선택은 뒤에서 보여드릴 31번 외부 비교를 근거로 다시 결정했습니다.')
    yy=next(r for r in yerrors if r['actual']=='yawn' and r['status']=='OK' and r['roi_image'])
    d.pic(yy['image'],90,240,610,360);d.pic(yy['roi_image'],875,265,430,300)
    d.text(90,650,650,'전체 이미지 → YOLO / MobileNet',28,INK,True);d.text(855,650,650,'입 ROI → 팀 ResNet18',28,TEAL,True)
    # 18
    team_stats=[]
    for label,path in [('ResNet18','03_yawn_resnet18/03_evaluation/03_summary.json'),('MobileNetV3-Small','02_yawn_mobilenet_v3_small/02_evaluation/02_summary.json'),('EfficientNet-B0','yawn_efficientnet_b0/evaluation/summary.json')]:
        stats=json.loads((ROOT/'teammate_yawn/runs'/path).read_text(encoding='utf8'))['test'];team_stats.append((label,stats))
    internal=read(ROOT/'reports/model_comparison.csv')
    internal=[find(internal,task='yawn',model=m) for m in ['yolo26n','yolo26s','mobilenet_v3_small']]
    slide('하품의 전체 이미지 모델 세 개를 같은 분할에서 비교했습니다','내부 테스트 214장 · 하품 116장 / 비하품 98장', 'reports/model_comparison.csv · 09_compare_models_retry.py',
          '하품 전체 이미지 경로의 세 모델은 YOLO26n, YOLO26s, MobileNetV3-Small입니다. YOLO26n은 작은 모델의 속도를, YOLO26s는 같은 계열에서 용량을 늘렸을 때의 표현력을, MobileNet은 다른 경량 CNN 구조를 비교하려고 선택했습니다. 내부 정확도는 각각 95.33%, 98.60%, 95.79%였으며 하품 재현율은 91.38%, 97.41%, 92.24%였습니다. 원본 전체 이미지 분할의 214장 결과로, 팀의 입 ROI 성공 212장과 분모가 다릅니다. 내부에서 YOLO26s가 높았다는 것만으로 최종 선택하지 않고, 같은 세 모델을 31번에서 외부 데이터로 재평가했습니다.')
    table(['전체 이미지 모델','정확도 / 214장','하품 재현율 / 116장'],[[r['model'],pct(r['accuracy']),pct(r['recall'])] for r in internal],[570,400,462],rowh=100)
    d.callout('경량 YOLO26n / 용량을 늘린 YOLO26s / 다른 경량 구조 MobileNetV3-Small')
    # Additional detail: actual training logs, not illustrative curves.
    slide('하품 모델의 검증 곡선에서도 차이를 확인했습니다','실제 에포크 기록 · 테스트셋이 아닌 검증셋으로 학습 상태 확인',
          'runs/yawn_yolo26s-cls/results.csv · runs/yawn_mobilenet_v3_small/history.csv',
          '이 그래프는 하품 YOLO26s와 MobileNetV3의 실제 저장된 학습 기록입니다. YOLO26s는 최대 25에포크 설정에서 20에포크까지 기록돼 있고, 최고 검증 정확도 98.624%가 처음 나온 에포크는 13입니다. MobileNetV3도 20에포크 기록이며 최고 검증 정확도 96.33%가 처음 나온 시점은 13입니다. 내부 테스트의 YOLO26s 98.60%와 검증 정확도 98.62%는 서로 다른 분모이므로 혼용하지 않습니다. 검증 곡선은 학습 진행과 모델 저장의 근거이고, 외부 성능은 뒤의 독립 출처 이미지로 별도 확인했습니다. 이 그래프만으로 과적합이나 새 운전자 성능을 단정하지 않았습니다.')
    fig,axes=plt.subplots(1,2,figsize=(12,4),layout='constrained')
    for ax,path,title,key in [(axes[0],ROOT/'runs/yawn_yolo26s-cls/results.csv','YOLO26s · 검증 정확도','metrics/accuracy_top1'),(axes[1],ROOT/'runs/yawn_mobilenet_v3_small/history.csv','MobileNetV3 · 검증 정확도','val_accuracy')]:
        rr=read(path);values=[float(r[key])*100 for r in rr];epochs=[int(r['epoch']) for r in rr]
        ax.plot(epochs,values,color=TEAL,marker='o',markersize=4)
        best=values.index(max(values));ax.scatter([epochs[best]],[values[best]],color=RED,zorder=3)
        ax.annotate(f'{epochs[best]} epoch / {values[best]:.2f}%',(epochs[best],values[best]),xytext=(5,12),textcoords='offset points',fontsize=11)
        ax.set(title=title,xlabel='에포크',ylabel='검증 정확도 (%)',ylim=(70,102));ax.grid(alpha=.15)
    plot=ASSETS/'yawn_training_validation.png';fig.savefig(plot,dpi=170);fig.savefig(plot.with_suffix('.svg'));plt.close(fig)
    d.pic(plot,90,245,1420,425);d.callout('검증 정확도와 내부 테스트 정확도는 서로 다른 분모의 지표입니다.')
    slide('같은 세 모델도 내부와 외부에서 결과가 달랐습니다','원본 전체 이미지 입력 · 내부 214장 / 외부 393장',
          'reports/model_comparison.csv · reports/final_model_selection/three_models_verified.csv',
          '내부 테스트와 외부 평가의 차이를 세 모델에 대해 나란히 비교했습니다. YOLO26n은 내부 95.33%에서 외부 76.34%, YOLO26s는 98.60%에서 78.37%, MobileNetV3는 95.79%에서 62.34%였습니다. 같은 모델이라도 새로운 출처의 인물과 배경, 화질에서 성능이 낮아졌습니다. 이 차이는 일반화 점검의 필요성을 보여주지만 과적합 하나로 원인을 확정하는 근거는 아닙니다. 이 외부 데이터는 이후 후보 선정에도 사용했으므로 완전히 미사용인 최종 홀드아웃은 아닙니다. 다음 검증에는 새로운 인물과 촬영 조건을 남겨야 합니다.')
    external=[find(face_verified,model=r['model']) for r in internal]
    d.pic(export_plot('yawn_internal_external',['YOLO26n','YOLO26s','MobileNetV3'],[('내부 정확도',[float(r['accuracy'])*100 for r in internal]),('외부 정확도',[float(r['accuracy'])*100 for r in external])]),90,240,1420,450)
    d.callout('내부 점수만으로 선택하지 않고, 외부 정확도와 재현율을 함께 확인했습니다.')
    # 19
    slide('먼저 연결했던 입 ROI ResNet은 외부에서 성능이 낮았습니다','이전 경로 · Roboflow 393장 · 하품 171장 / 비하품 222장', 'reports/yawn_roboflow/external_summary.csv · threshold_analysis/yawn_threshold_sweep.csv',
          '이전에 데모에 연결했던 팀 입 ROI ResNet은 내부 전체 214장 중 210장을 맞혀 98.13%, ROI 성공 212장에서는 99.06%였습니다. 그러나 별도 출처의 Roboflow 393장에서는 임계값 0.50 기준 정확도 65.65%, 재현율 44.44%였습니다. 하품 171장 중 76장만 잡았습니다. 전체 입 ROI 실패는 49장이고 그중 실제 하품은 22장입니다. ROI 성공분 정확도 75%와 전체 정답률 65.65%를 구분했습니다. 하품 평가가 눈보다 작은 이유는 독립 자료 확보와 프로젝트 범위입니다. 이 결과를 계기로 오류와 전처리를 점검하고 다른 모델까지 외부 비교를 확대했습니다.')
    d.pic(export_plot('yawn_external_gap',['내부 전체\n214장','외부 전체\n393장'],[('정확도',[210/214*100,258/393*100])]),90,240,1000,425)
    d.text(1150,280,350,'44.44%',56,RED,True);d.text(1150,365,350,'외부 하품 재현율\n76 / 171',29,INK)
    d.callout('외부 데이터에서는 ROI 실패와 분류 오류가 함께 발생했습니다.')
    # 20
    fails=[r for r in yerrors if r['actual']=='yawn' and r['status']=='OK' and r['positive_probability'] and float(r['positive_probability'])<.1]
    selected=[fails[i] for i in [0,len(fails)//2,-1]]
    slide('미검출 사진의 얼굴과 실제 입 입력을 함께 확인했습니다','이전 ResNet 경로 · 실제 하품인데 점수가 0.10보다 낮은 사례', 'reports/yawn_roboflow/error_audit/all_cases.csv · ROI cache',
          '이 사례들은 실제 하품으로 라벨된 이미지지만 모델 점수가 0.10보다 낮아 놓친 사진입니다. 위에는 얼굴, 아래에는 실제 분류기에 입력한 입 ROI를 배치했습니다. 입 벌림 정도, 색감, 턱과 목 주변 선, 가림과 촬영 각도 등을 관찰 항목으로 잡았습니다. 주름이나 입술 색이 원인이라는 결론은 아직 내릴 수 없습니다. 색 전처리를 바꾼 전체 비교와 별도로 조건별 정답이 있는 데이터가 필요하기 때문입니다. 이 오류 분석을 토대로 입력 방식과 모델 후보를 추가로 비교했습니다.')
    for j,r in enumerate(selected):
        x=85+j*485;d.pic(r['image'],x,230,430,260);d.pic(r['roi_image'],x+85,505,260,160)
        d.text(x,682,450,f"정답 yawn · 점수 {float(r['positive_probability']):.3f}",25,RED,True)
    # 21
    slide('하품 임계값을 낮추자 재현율이 올라갔습니다','가중치는 동일 · Yawn 점수 0.50 → 0.15 / 0.10', 'reports/yawn_roboflow/threshold_analysis/yawn_threshold_sweep.csv',
          '이 표는 이전 입 ROI ResNet의 임계값 실험입니다. 0.50에서는 정확도 65.65%, 하품 재현율 44.44%였습니다. 0.15로 낮추면 정확도 71.50%, 재현율 64.33%이고, 0.10에서는 정확도는 같지만 재현율이 66.67%입니다. 대신 비하품 오탐이 13장에서 24장 또는 28장으로 늘어납니다. 임계값은 놓침과 오탐을 교환하는 설정입니다. 0.10에서도 하품 171장 중 57장을 놓쳤고 그중 22장은 ROI 실패였습니다. 이 0.10과 0.15는 당시 ResNet의 후보 설정이며, 최종 YOLO26s 선택에 그대로 복사하지 않았습니다.')
    table(['Yawn 임계값','정확도','하품 재현율','오탐','놓침'],[[r['threshold'],pct(r['accuracy']),pct(r['yawn_recall']),r['fp'],r['fn_including_unknown']] for r in y_settings],[340,300,330,230,232],rowh=95)
    d.callout('71.50%를 실제 차량에 충분한 정확도로 주장하지 않습니다.')
    # 22
    pp=read(ROOT/'reports/yawn_roboflow/preprocess_ablation/predictions.csv')
    color_rows=[]
    for variant in ['baseline','grayscale']:
        rr=[r for r in pp if r['variant']==variant];n=len(rr)
        tp=sum(r['actual']=='yawn' and bool(r['yawn_probability']) and float(r['yawn_probability'])>=.1 for r in rr)
        tn=sum(r['actual']=='no_yawn' and bool(r['yawn_probability']) and float(r['yawn_probability'])<.1 for r in rr)
        fp=sum(r['actual']=='no_yawn' and bool(r['yawn_probability']) and float(r['yawn_probability'])>=.1 for r in rr)
        color_rows.append(dict(variant=variant,n=n,tp=tp,tn=tn,fp=fp,accuracy=(tp+tn)/n,recall=tp/171))
    dump_csv('yawn_color_verified.csv',color_rows)
    slide('색을 없애는 것만으로 하품 분류가 좋아지지는 않았습니다','RGB와 흑백을 같은 임계값 0.10에서 다시 집계', 'reports/yawn_roboflow/preprocess_ablation/predictions.csv → yawn_color_verified.csv',
          '입술 색이나 피부 주름이 영향을 주는지 살펴보기 위해 흑백 입력도 비교했습니다. 같은 ROI와 임계값 0.10에서 RGB 정확도는 71.50%, 재현율은 66.67%이고, 흑백은 정확도 61.83%, 재현율 67.84%였습니다. 흑백의 재현율이 약간 높아도 비하품 오탐이 28장에서 68장으로 늘어 전체 균형이 나빴습니다. 따라서 현재 입 ROI 경로는 RGB를 유지합니다. 이 결과 역시 색이 모든 오류의 원인임을 증명하지 않으며, 특정 입술색 조건을 해결했다고 말할 근거도 아닙니다.')
    d.pic(export_plot('yawn_color',['RGB','흑백'],[('정확도',[r['accuracy']*100 for r in color_rows]),('하품 재현율',[r['recall']*100 for r in color_rows])]),90,230,1420,450)
    d.callout('눈에는 CLAHE, 입에는 RGB · 부위별 실험 결과에 따라 전처리를 선택했습니다.')
    # 23
    candidates=[find(face_verified,model=m) for m in ['yolo26n','yolo26s','mobilenet_v3_small']]
    slide('31번에서 세 모델을 같은 외부 이미지로 다시 돌렸습니다','원본 이미지 전체 · Roboflow 393장 · 임계값 0.50 고정', '31_yawn_external_compare.py · final_model_selection/three_models_verified.csv',
          '31번의 주 비교는 얼굴 이미지 전체 입력 YOLO26n, YOLO26s, MobileNetV3-Small 세 모델입니다. 같은 393장, 같은 RGB 224 패딩, 같은 임계값 0.50에서 비교했습니다. 정확도는 76.34%, 78.37%, 62.34%이고 재현율은 56.14%, 75.44%, 29.82%였습니다. YOLO26s가 이 조건의 정확도와 재현율에서 가장 높았습니다. 이번 정리에서는 저장된 이미지별 예측을 다시 집계하고 현재 가중치로 재추론해 TP·TN·FP·FN이 모두 일치하는지 확인했습니다. 31번 파일에 함께 있는 팀 ResNet 결과는 이전 CSV를 가져온 참고값이며 이 세 모델에 포함하지 않았습니다.')
    table(['모델','정확도','하품 재현율','F1','오탐 / 놓침'],[[r['model'],pct(r['accuracy']),pct(r['recall']),pct(r['f1']),r['fp']+' / '+r['fn']] for r in candidates],[390,250,290,250,252],rowh=100,size=24)
    d.callout('같은 입력·분모·임계값 비교에서 YOLO26s를 최종 개발 후보로 선정했습니다.')
    export_plot('yawn_final_three_models',['YOLO26n','YOLO26s','MobileNetV3'],[('정확도',[float(r['accuracy'])*100 for r in candidates]),('하품 재현율',[float(r['recall'])*100 for r in candidates])])
    # 24
    slide('최종 하품 후보는 YOLO26s + 전체 이미지 + 0.50입니다','재현율만 높이기보다 오탐과 전체 정확도를 함께 고려', 'final_model_selection/selected_models.json · 선정근거.md',
          'YOLO26s의 임계값 0.50에서는 308장 정답, 실제 하품 탐지 129장, 비하품 오탐 43장, 하품 놓침 42장이었습니다. 0.15로 낮추면 하품 재현율은 98.25%로 높아지지만 오탐이 106장으로 늘고 정확도는 72.26%로 낮아집니다. F1은 75.51%로 0.50의 75.22%보다 조금 높으므로 모든 지표의 최고점이라고 주장하지 않았습니다. 데모는 기본 임계값 0.50에서 정확도와 재현율의 균형을 선택했습니다. 입력은 31번 face 모드와 같은 원본 이미지 전체입니다. 새로운 얼굴 박스를 검출하거나 입만 잘라 넣은 결과가 아닙니다.')
    points=[find(compare,model='yolo26s',input='face',threshold=str(t)) for t in [.5,.15,.1]]
    table(['Yawn 임계값','정확도','재현율','F1','비하품 오탐'],[[r['threshold'],pct(r['accuracy']),pct(r['recall']),pct(r['f1']),r['fp']] for r in points],[350,270,290,260,262],rowh=95,size=25)
    d.callout('선정 조건: 원본 전체 RGB → 224×224 패딩 → YOLO26s → Yawn ≥ 0.50')
    slide('최종 후보가 맞힌 수와 놓친 수를 정수로 확인했습니다','YOLO26s · 전체 이미지 · 임계값 0.50 · Roboflow 393장',
          'reports/final_model_selection/selected_models.json · 재추론 혼동행렬 일치 확인',
          '최종 하품 후보의 예측을 네 칸으로 나누면 실제 비하품 222장 중 179장은 정상, 43장은 하품으로 오탐했습니다. 실제 하품 171장 중 129장을 감지했고 42장은 놓쳤습니다. 정확도는 정답 308장을 전체 393장으로 나눈 78.37%입니다. 하품 재현율은 129 나누기 171로 75.44%, 정밀도는 하품 경고 172건 중 129건이 맞은 75%입니다. F1은 75.22%입니다. 얼굴 전체 분류 경로는 외부 평가에서 입 ROI 실패로 이미지를 제외하지 않았습니다. 그러나 라이브의 얼굴 확인과 지속 시간 조건까지 이 표로 검증한 것은 아닙니다.')
    d.text(245,240,650,'예측 비하품             예측 하품',27,GRAY,True)
    for y,label,a,b in [(305,'실제 비하품',179,43),(505,'실제 하품',42,129)]:
        d.text(90,y+65,180,label,25,INK,True)
        for x,value,correct in [(285,a,y==305),(610,b,y==505)]:
            d.rect(x,y,290,165,LIGHT if correct else '#F9E9E6')
            d.text(x+83,y+43,190,str(value),61,TEAL if correct else RED,True)
    d.text(1000,315,500,'정확도  308 / 393\n78.37%',38,INK,True)
    d.text(1000,500,500,'하품 재현율  129 / 171\n75.44%',36,TEAL,True)
    d.callout('오탐 43장과 놓침 42장이 남아 있습니다. 실제 차량 적용에 충분하다고 주장하지 않습니다.')
    final_predictions=read(ROOT/'reports/yawn_external_model_compare/predictions.csv')
    final_predictions=[r for r in final_predictions if r['model']=='yolo26s' and r['input']=='face']
    fp=sorted([r for r in final_predictions if r['actual']=='no_yawn' and float(r['probability'])>=.5],key=lambda r:float(r['probability']),reverse=True)
    fn=sorted([r for r in final_predictions if r['actual']=='yawn' and float(r['probability'])<.5],key=lambda r:float(r['probability']))
    slide('선정한 YOLO26s에도 어려운 이미지가 남아 있습니다','최종 후보의 실제 오탐과 미탐 · 데이터셋 라벨 기준',
          'reports/yawn_external_model_compare/predictions.csv · yolo26s / face / 0.50',
          '이번 사진은 이전 ResNet의 입 ROI 오류가 아니라 최종 선정한 YOLO26s 전체 이미지 경로의 오류입니다. 왼쪽 두 장은 데이터셋 정답이 비하품인데 하품으로 판단한 오탐이고, 오른쪽 두 장은 실제 하품 라벨인데 놓친 미탐입니다. 얼굴 자세와 입 모양, 가림, 조명과 촬영 맥락을 확인할 수 있습니다. 모델이 특정 주름이나 색 때문에 틀렸다고 단정하지는 않습니다. 이 사례는 앞으로 추가할 어려운 양성·음성 데이터와 영상 이벤트 정답을 구체화하는 데 사용합니다. 후보를 선정했다고 남은 오류까지 해결된 것은 아닙니다.')
    for j,r in enumerate(fp[:2]+fn[:2]):
        x=84+j*363;d.pic(r['image'],x,255,340,330)
        d.text(x,610,340,'비하품 → 하품' if j<2 else '하품 → 비하품',25,RED,True)
        d.text(x,660,340,f"Yawn 점수 {float(r['probability']):.3f}",23,GRAY)
    # 25
    videos=list((ROOT/'data/yawn_mp').rglob('*.mp4'))
    yawn_videos=[p for p in videos if 'Yawning' in p.parts]
    microsleep_videos=[p for p in videos if 'Microsleep' in p.parts]
    manifest=read(ROOT/'reports/nitymed_full_review/annotation_manifest.csv')
    slide('하품 데이터 보강을 위한 전처리와 학습 경로를 준비했습니다','24번: 프레임·입 ROI 추출 / 25번: 확인된 라벨로 추가 학습', '24_nitymed_prepare.py · 25_nitymed_finetune.py · reports/nitymed_full_review',
          f'현재 폴더에는 Yawning {len(yawn_videos)}개와 Microsleep {len(microsleep_videos)}개, 총 {len(videos)}개 영상이 있습니다. 모두를 하품 영상으로 세지 않았습니다. 24번 코드는 프레임을 추출해 기존 팀 모델과 같은 방식으로 입 ROI를 만듭니다. 25번 코드는 확인된 하품·비하품 라벨만 읽고 그룹을 나눈 뒤 추가 학습하는 경로입니다. 현재 폴더에서 25번의 학습 완료 comparison.json과 최종 가중치는 확인되지 않아 정확도 향상 수치로 넣지 않았습니다. 영상 폴더 이름이나 모델 예측을 프레임의 정답으로 쓰면 안 됩니다. 인물 ID와 이벤트 정답을 검수한 뒤 학습하고 새 사람으로 평가하는 것이 다음 단계입니다. 발표에는 영상 파일을 넣지 않았습니다.')
    items([('확보·전처리',f'Yawning {len(yawn_videos)}개 / Microsleep {len(microsleep_videos)}개 · 샘플 기록 {len(manifest)}행'),('정답과 그룹 검수','하품 / 말하기 / 웃기 / 불확실 · 같은 사람은 같은 분할'),('학습·재평가 경로','25번 코드 준비 · 완료된 성능 개선 수치는 미확인')])
    # 26
    slide('최종 후보에 맞춰 눈과 하품의 입력 경로를 나눴습니다','눈은 ROI, 하품은 원본 이미지 전체를 사용합니다.', '11_streamlit_dashboard.py · roi_utils.py · 31_yawn_external_compare.py',
          '최종 경로는 이전 입 ROI ResNet 경로와 다릅니다. 눈은 원본에서 MediaPipe 눈 ROI를 추출하고 CLAHE와 패딩을 거쳐 YOLO26n에 넣습니다. 하품은 원본 이미지 전체를 RGB 224 곱하기 224로 패딩해 YOLO26s에 넣습니다. 31번 face 모드는 별도의 얼굴 검출 크롭이 아니라 원본 전체 입력입니다. 앱도 같은 분류 전처리를 재사용하도록 수정했습니다. 라이브에서는 얼굴이 보이는지 확인한 뒤 점수와 지속 시간으로 경고하며, 얼굴 미검출이나 오래된 결과는 판단을 보류합니다. 이 얼굴 확인과 지속 시간을 포함한 통합 성능은 외부 정지 이미지 정확도와 별도로 평가해야 합니다.')
    d.text(95,250,1380,'원본 웹캠 RGB 프레임',36,TEAL,True)
    d.rect(95,325,1400,125,LIGHT);d.text(125,350,1330,'눈: MediaPipe 눈 ROI → CLAHE·패딩 → YOLO26n',32,INK,True)
    d.rect(95,480,1400,125,LIGHT);d.text(125,505,1330,'하품: 원본 전체 → RGB·패딩 → YOLO26s',32,INK,True)
    d.text(100,645,1380,'얼굴 확인 · Closed ≥ 0.25 / 2초 · Yawn ≥ 0.50 / 1.5초 → 경고',29,TEAL,True)
    d.callout('입 랜드마크는 표시용입니다. 최종 하품 분류기에 입 ROI를 넣지 않습니다.')
    # 27
    slide('최종 개발 후보와 앱 기본 모델을 일치시켰습니다','외부 정지 이미지 성능과 통합 웹캠 성능은 구분합니다.', 'reports/final_model_selection/selected_models.json · 11_streamlit_dashboard.py',
          '선정 결과를 앱에 반영해 눈은 기존 YOLO26n과 CLAHE, 하품은 전체 이미지 YOLO26s를 기본으로 연결했습니다. 눈 임계값은 0.25, 하품은 0.50입니다. 입 ROI ResNet은 이전 비교 경로로 남기고 기존 가중치와 실험 자료는 보존했습니다. 눈 CEW 정확도는 88.36%, 하품 Roboflow 정확도는 78.37%입니다. 두 값은 서로 다른 정지 이미지 평가셋에서 얻은 값이며 평균내서 시스템 정확도라고 할 수 없습니다. 외부 자료는 후보 선택에도 쓰였으므로 최종 개발 후보이며, 새로운 인물의 영상에서 이벤트 재현율과 오경고를 검증해야 합니다.')
    table(['구분','현재 상태','평가 범위'],[['눈 ROI YOLO26n + CLAHE','선정 · 앱 기본','CEW 88.36% / 임계값 0.25'],['하품 전체 YOLO26s','선정 · 앱 기본','Roboflow 78.37% / 임계값 0.50'],['하품 입 ROI ResNet18','이전 비교 경로','기존 결과와 가중치 보존'],['통합 영상 이벤트 성능','정량 검증 필요','별도 영상 정답으로 평가']],[530,370,532],rowh=87,size=24)
    # 28
    slide('웹캠 화면과 AI 분석을 분리해 대기열을 줄였습니다','로컬 OpenCV 캡처 · 최신 프레임 하나만 분석', 'aura_local_camera.py · 11_streamlit_dashboard.py · test_aura_local_camera.py',
          '현재 웹캠 화면은 로컬 JPEG 스트림으로 전송하고, AI 작업은 별도 스레드에서 최신 대기 프레임을 처리합니다. 카메라 표시가 모델 추론 완료를 기다리는 구조를 피하려는 목적입니다. 기본 분석 간격은 350ms이고 화면 전송 상한은 20FPS로 설정돼 있습니다. 이 수치는 설정값이며 실제 카메라 FPS나 처리 지연을 보장하지 않습니다. 현재 앱의 모델·임계값과 사진 입력 확률이 31번 기록에 맞는지 따로 확인했습니다. 이전 앱 버전의 단기 웹캠 측정치를 현재 버전 성능으로 인용하지 않았습니다.')
    d.rect(90,315,390,290,NAVY);d.text(125,360,320,'로컬 웹캠',42,WHITE,True);d.text(125,440,320,'OpenCV 캡처\n최신 프레임 공유',29,'#BFD7DF')
    d.line(480,370,680,370,TEAL,5);d.line(480,555,680,555,TEAL,5)
    d.rect(680,270,820,180,LIGHT);d.text(715,300,750,'원본 화면 표시',35,INK,True);d.text(715,365,750,'로컬 스트림 · 최대 전송 20 FPS',26,GRAY)
    d.rect(680,500,820,180,LIGHT);d.text(715,530,750,'최신 프레임 → 입력별 전처리 → AI',32,INK,True);d.text(715,595,750,'기본 분석 간격 350 ms · 별도 추론 스레드',26,GRAY)
    check=OUT/'webcam_verification.json'
    if check.is_file():
        measured=json.loads(check.read_text(encoding='utf8'))
        samples=[r['capture_fps'] for r in measured.get('camera_samples',[]) if r['frame_available']]
        if samples:
            msg=f"로컬 장치 15초 점검: 수신 평균 {sum(samples)/len(samples):.1f} FPS · 처리 루프 {measured['inference_count']}회"
            d.callout(msg)
            d.notes[-1]['notes']+=f" 실제 15초 점검의 연결 후 평균 수신은 {sum(samples)/len(samples):.1f}FPS, 처리 루프는 {measured['inference_count']}회였습니다. 장시간 안정성이나 감지 정확도를 측정한 시험은 아닙니다."
            d.notes[-1]['source']+=' · webcam_verification.json'
            d.s.notes_slide.notes_text_frame.text=d.notes[-1]['notes']+'\n\n근거: '+d.notes[-1]['source']
    # 29
    slide('응급 대응은 진단이 아닌 시뮬레이션으로 구현했습니다','반복 눈 감김 → 가상 감속·정차 → 응답 확인 → 모의 SOS', 'aura_safety_sim.py · Streamlit 시연',
          '나이와 지병을 입력받아 기절이나 발작을 판단하는 확장도 생각했지만, 현재 눈·입 분류 모델에는 의학적 정답과 검증 근거가 없습니다. 사람마다 나타나는 양상이 다르고 단순 무반응이나 눈 감김을 응급 질환으로 단정하면 잘못된 경보와 신고로 이어질 수 있습니다. 그래서 화면 속 가상 감속, 정차, 응답 확인, 모의 SOS까지만 구현했습니다. 실제 차량 제어와 실제 119 신고는 하지 않습니다. 입력 정보만으로 의식 상실을 측정하는 기능이 있다고 발표하지 않으며, 시간 기준도 시연용으로 설명합니다.')
    d.flow([('반복 눈 감김','경고 2회 / 5분'),('가상 감속','화면 속 속도'),('정차·휴식','시연용 타이머'),('응답 확인','버튼 시나리오'),('모의 SOS','화면·로그')],280)
    d.text(100,565,1370,'연령·지병 입력만으로 기절·발작을 진단할 수는 없습니다.',35,RED,True)
    d.callout('실제 신고·자동차 제어 없이 대응 흐름만 시연합니다.')
    # 30
    slide('야간과 선글라스는 별도 검증이 필요한 한계입니다','일반 RGB 카메라에서 보이지 않는 정보를 임계값으로 복원할 수는 없습니다.', '프로젝트 장비·데이터 범위 · Tesla 공식 제한사항 참고',
          '현재 카메라는 일반 RGB 웹캠이며 적외선 조명과 센서를 갖춘 야간 실차 검증은 수행하지 않았습니다. 공개 영상 일부를 처리했다고 해서 현재 장비의 야간 성능까지 검증한 것은 아닙니다. 선글라스도 착용 여부 라벨만으로 가려진 눈의 Open과 Closed를 알 수는 없습니다. 렌즈 뒤 눈 정보가 사라지면 확률 임계값을 낮춰도 복원되지 않습니다. 이런 경우는 관측 한계로 다뤄야 합니다. 현재 코드가 선글라스를 자동 식별하는 별도 모델을 갖춘 것은 아니며 향후 착용·가림 감지와 판단 보류 정책이 필요합니다.')
    table(['조건','이번 프로젝트의 범위','필요한 검증'],[['야간·역광','RGB 웹캠 · 야간 실차 미검증','NIR 조명/센서 · 조도별 평가'],['선글라스','착용 라벨만으로 눈 상태 불명','가림 감지 · 렌즈별 눈 상태 정답'],['다양한 운전자','인물 독립성이 충분하지 않음','새 인물 · 자세·안경·화장 분리 평가']],[300,560,572],rowh=106,size=25)
    # 31
    slide('상용 사례는 가능성과 함께 남은 한계를 보여줍니다','Tesla 운전자 주의 모니터링 · NIR 센서 확장 가능성', 'Tesla Model 3 공식 매뉴얼 / Fraunhofer IOSB 백서 · 상세 링크는 발표대본',
          'Tesla의 현행 일부 지역 Model 3 매뉴얼은 선글라스를 쓴 경우에도 운전자 주의 모니터링이 활성화된다고 설명합니다. 다만 같은 회사의 주의 분산 경고 제한사항에는 안경과 선글라스, 화장, 극단적인 조명, 적외선 조명 가림 등이 영향을 줄 수 있다고 명시돼 있습니다. 따라서 테슬라가 모든 선글라스 뒤 눈 감김을 완벽히 판별한다고 설명하면 안 됩니다. Fraunhofer 자료는 NIR과 RGB-IR, 센서 융합을 차량 실내 모니터링의 기술 방향으로 소개합니다. 우리 프로젝트에서는 향후 장비와 데이터 확장 가능성으로 제시합니다. 상세 원문 링크는 같은 폴더의 참고자료.md와 PPT의 출처 링크에 있습니다.')
    items([('Tesla 공식 문서','일부 구성에서 선글라스 착용 중 주의 모니터링 활성'),('같이 공개된 제한사항','안경·화장·조명·적외선 조명 가림에 따른 성능 영향'),('우리의 발전 가능성','NIR / RGB-IR + 조건별 데이터 + 시간·센서 정보 결합')])
    for x,label,url in [(100,'Tesla 주의 모니터링',TESLA),(590,'Tesla 제한사항',LIMITS),(1080,'Fraunhofer 백서',NIR)]:
        d.text(x,735,430,label+' ↗',22,TEAL)
        d.s.shapes[-1].text_frame.paragraphs[0].runs[0].hyperlink.address=url
    # 32
    slide('다음 개선은 새 데이터와 독립 검증을 묶어서 진행합니다','조건별 양성·음성 데이터 → 사람 단위 분리 → 이벤트 평가', '프로젝트 후속 실험 계획 · 아직 검증된 성능 수치가 아님',
          '눈은 화장한 Open뿐 아니라 Closed, 안경과 가림 조건까지 균형 있게 확보해야 합니다. 하품은 약한 하품과 측면 하품, 말하기·웃기 같은 비하품을 함께 모아야 합니다. 같은 사람의 사진과 영상은 하나의 분할에 묶고 새로운 평가 자료는 모델과 임계값을 선택하기 전에 분리합니다. 이후 EAR·MAR 같은 랜드마크 지표와 입이 열리고 닫히는 지속 시간을 보조 정보로 비교할 수 있습니다. 마지막으로 정확도뿐 아니라 하품 이벤트 재현율, 시간당 오경고와 경고 지연을 측정하겠습니다. 이러한 후속 기능은 구현된 성과와 구분합니다.')
    items([('데이터 보강','화장한 Closed · 약한 하품 · 말하기·웃기 · 조명과 가림'),('후보의 재검증','새 운전자 분리 · 동일 입력 조건 · 고정된 평가 절차'),('실용성 평가','놓친 이벤트 · 시간당 오경고 · 경고 지연 · 장치 안정성')])
    # 33
    slide('높은 내부 점수에서, 검증 가능한 비전 시스템으로','수집·전처리·학습·외부 검증·시연을 하나의 과정으로 연결했습니다.', '검증 CSV / 그래프 원본 / PPT / 발표대본 제공',
          '이 프로젝트는 눈과 하품을 각각 분류하는 데서 시작했습니다. 내부 점수만 보지 않고 직접 촬영 사진과 외부 데이터에서 실패 조건을 확인했습니다. 눈은 입력 부위를 맞추고 CLAHE와 임계값을 조정해 외부 성능을 개선했습니다. 하품은 낮은 외부 성능을 계기로 전처리와 판정 기준, 다른 모델과 입력 조합까지 비교했습니다. 현재 데모와 실험 후보를 구분했고, 야간·선글라스·응급 대응의 한계도 남겼습니다. 실용성은 가장 높은 숫자 하나가 아니라 새로운 사람에게 얼마나 덜 놓치고 덜 잘못 경고하는지로 평가하겠습니다.',True)
    d.text(90,270,1400,'눈  88.36%     하품  78.37%',58,'#74D7CA',True)
    d.text(95,365,1360,'현재 경로의 외부 정지 이미지 평가 · 통합 실차 성능 아님',29,'#BFD7DF')
    d.text(95,480,1370,'최종 후보: 눈 ROI YOLO26n + 전체 이미지 YOLO26s',36,WHITE,True)
    d.text(95,600,1380,'핵심 성과: 입력을 맞추고, 오류를 관찰하고, 조건을 기록하며 비교했다.',32,WHITE)
    sources={'accessed':'2026-11-08','tesla_attention':TESLA,'tesla_limitations':LIMITS,'nir_whitepaper':NIR,
             'source_note':'Regional/configuration differences apply; no claim of universal sunglasses penetration.'}
    (OUT/'external_sources.json').write_text(json.dumps(sources,ensure_ascii=False,indent=2),encoding='utf8')
    (OUT/'참고자료.md').write_text(f'# 외부 참고자료\n\n확인일: 2026-11-08\n\n- [Tesla 운전자 주의 모니터링]({TESLA}): 선글라스 착용 시 활성화 설명. 지역·차량 구성·소프트웨어 조건에 따름.\n- [Tesla 주의 분산 경고의 제한사항]({LIMITS}): 안경·화장·조명 및 적외선 조명 가림의 영향.\n- [Fraunhofer IOSB 실내 모니터링 백서]({NIR}): NIR, RGB-IR, 센서 융합 기술.\n\n현재 프로젝트가 위 시스템과 같은 센서·검증 수준을 갖췄다는 의미가 아닙니다.\n',encoding='utf8')
    save(d)
    print('Saved',OUT)


if __name__=='__main__':build()

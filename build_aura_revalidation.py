"""Editable 33-slide Korean technical deck, PDF, previews and speaker notes.

All result slides read the actual hard-case experiment CSV files. The drawing
primitives produce editable PowerPoint objects and a matching raster preview.
"""
from __future__ import annotations
import csv
import json
import math
from io import BytesIO
from pathlib import Path
from collections import Counter

from PIL import Image, ImageOps, ImageFont, ImageDraw
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE, MSO_CONNECTOR

ROOT=Path(__file__).resolve().parent
EXP=ROOT/'reports/hardcase_finetune'
OUT=ROOT/'reports/presentation/revalidation_20261108'
W,H=1600,900
NAVY='#102536'; INK='#162D3C'; TEAL='#008B86'; RED='#C05047'; BLUE='#477BBA'
GRAY='#647783'; LIGHT='#E9EFF1'; WHITE='#FFFFFF'; AMBER='#B57724'
FONT='C:/Windows/Fonts/malgun.ttf'; BOLD='C:/Windows/Fonts/malgunbd.ttf'

def read(path):
    with Path(path).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))

def pct(x):return f'{float(x)*100:.2f}%'
def delta(a,b):return f'{(float(b)-float(a))*100:+.2f}%p'
def rgb(s):return RGBColor.from_string(s.lstrip('#'))
def u(x):return Inches(x/120)


class Deck:
    def __init__(self):
        self.prs=Presentation();self.prs.slide_width=u(W);self.prs.slide_height=u(H)
        self.images=[];self.notes=[];self.overflow=[]
    def slide(self,title,sub='',source='',note='',dark=False):
        self.s=self.prs.slides.add_slide(self.prs.slide_layouts[6])
        self.bg=NAVY if dark else WHITE;self.fg=WHITE if dark else INK
        self.s.background.fill.solid();self.s.background.fill.fore_color.rgb=rgb(self.bg)
        self.im=Image.new('RGB',(W,H),self.bg);self.d=ImageDraw.Draw(self.im)
        self.images.append(self.im);self.notes.append(dict(slide=len(self.images),title=title,notes=note,source=source))
        self.s.notes_slide.notes_text_frame.text=note+'\n\n근거: '+source
        self.text(84,38,1200,'AURA DRIVE   /   실험과 검증',18,TEAL if not dark else '#74D7CA',True)
        self.text(84,89,1430,title,49,self.fg,True)
        if sub:self.text(87,160,1420,sub,24,GRAY if not dark else '#BED0DA')
        self.line(84,824,1516,824,LIGHT if not dark else '#456070',1)
        self.text(84,839,1300,source,14,GRAY if not dark else '#A9BEC9')
        self.text(84,866,1200,'2026.11.08  ·  연구·시연 프로토타입',13,GRAY if not dark else '#A9BEC9')
        self.text(1455,848,70,f'{len(self.images):02d} / 33',17,GRAY if not dark else '#A9BEC9')
    def text(self,x,y,w,text,size=28,color=INK,bold=False):
        font=ImageFont.truetype(BOLD if bold else FONT,size)
        lines=[]
        for para in str(text).split('\n'):
            line=''
            for ch in para:
                if line and self.d.textlength(line+ch,font=font)>w-4:lines.append(line);line=ch
                else:line+=ch
            lines.append(line)
        lh=int(size*1.37);height=len(lines)*lh+8
        if y<824 and y+height>815:self.overflow.append((len(self.images),str(text)[:70],y+height))
        for i,line in enumerate(lines):self.d.text((x,y+i*lh),line,font=font,fill=color)
        box=self.s.shapes.add_textbox(u(x),u(y),u(w+10),u(height))
        tf=box.text_frame;tf.clear();tf.word_wrap=False
        tf.margin_left=tf.margin_right=tf.margin_top=tf.margin_bottom=0
        for i,line in enumerate(lines):
            p=tf.paragraphs[0] if i==0 else tf.add_paragraph();p.text=line
            p.font.name='맑은 고딕';p.font.size=Pt(size*.6);p.font.bold=bold;p.font.color.rgb=rgb(color)
            p.space_before=p.space_after=Pt(0);p.line_spacing=Pt(lh*.6)
        return y+height
    def rect(self,x,y,w,h,color):
        self.d.rectangle((x,y,x+w,y+h),fill=color)
        sh=self.s.shapes.add_shape(MSO_SHAPE.RECTANGLE,u(x),u(y),u(w),u(h));sh.fill.solid();sh.fill.fore_color.rgb=rgb(color);sh.line.fill.background()
    def line(self,x1,y1,x2,y2,color=LIGHT,width=2):
        self.d.line((x1,y1,x2,y2),fill=color,width=width)
        sh=self.s.shapes.add_connector(MSO_CONNECTOR.STRAIGHT,u(x1),u(y1),u(x2),u(y2));sh.line.color.rgb=rgb(color);sh.line.width=Pt(width*.6)
    def pic(self,path,x,y,w,h):
        im=path.copy() if isinstance(path,Image.Image) else Image.open(path).convert('RGB')
        im=ImageOps.contain(im,(int(w),int(h)),Image.Resampling.LANCZOS)
        xx=x+(w-im.width)/2;yy=y+(h-im.height)/2
        self.im.paste(im,(int(xx),int(yy)))
        buf=BytesIO();im.save(buf,format='PNG');buf.seek(0)
        self.s.shapes.add_picture(buf,u(xx),u(yy),width=u(im.width),height=u(im.height))
    def callout(self,text,y=742,color=TEAL):
        self.rect(84,y,7,52,color);self.text(108,y+4,1385,text,28,color,True)
    def table(self,headers,rows,widths,x=84,y=234,rowh=71,size=25):
        self.rect(x,y,sum(widths),rowh,NAVY)
        xx=x
        for title,w in zip(headers,widths):self.text(xx+16,y+17,w-25,title,size,WHITE,True);xx+=w
        for i,row in enumerate(rows):
            yy=y+(i+1)*rowh
            if i%2==0:self.rect(x,yy,sum(widths),rowh,'#F2F5F6')
            xx=x
            for item,w in zip(row,widths):self.text(xx+16,yy+16,w-25,str(item),size,INK);xx+=w
    def bars(self,labels,values,x=100,y=260,w=1000,rowh=110,colors=None,suffix='%'):
        for i,(label,value) in enumerate(zip(labels,values)):
            yy=y+i*rowh;self.text(x,yy,320,label,26,INK,True)
            bx=x+335;bw=w-470
            self.rect(bx,yy+5,bw,37,LIGHT);self.rect(bx,yy+5,bw*value/100,37,(colors or [TEAL]*len(values))[i])
            self.text(bx+bw+22,yy-2,150,f'{value:.2f}{suffix}',31,INK,True)
    def flow(self,items,y=300):
        n=len(items);gap=22;bw=(1432-(n-1)*gap)/n
        for i,(a,b) in enumerate(items):
            x=84+i*(bw+gap);self.rect(x,y,bw,5,TEAL);self.text(x,y+25,bw,a,28,INK,True);self.text(x,y+78,bw,b,23,GRAY)
            if i<n-1:self.text(x+bw+3,y+30,gap,'›',29,TEAL,True)
    def save(self):
        assert len(self.images)==33,len(self.images)
        if self.overflow:raise ValueError(f'Text overflows: {self.overflow}')
        OUT.mkdir(parents=True,exist_ok=True)
        self.prs.save(OUT/'AURA_DRIVE_추가학습_재검증_33장.pptx')
        self.images[0].save(OUT/'AURA_DRIVE_추가학습_재검증_33장.pdf','PDF',save_all=True,append_images=self.images[1:],resolution=144)
        for i,im in enumerate(self.images):im.save(OUT/f'slide_{i+1:02d}.png')
        for page,start in enumerate(range(0,33,12)):
            thumb=Image.new('RGB',(1600,900),'#DCE5E9')
            for j,im in enumerate(self.images[start:start+12]):thumb.paste(im.resize((400,225)),((j%4)*400,(j//4)*225))
            thumb.save(OUT/f'overview_{page+1}.jpg',quality=92)
        (OUT/'발표자_대본.md').write_text('\n\n'.join(f"## {r['slide']:02d}. {r['title']}\n\n{r['notes']}\n\n근거: `{r['source']}`" for r in self.notes),encoding='utf-8')
        (OUT/'slide_manifest.json').write_text(json.dumps(self.notes,ensure_ascii=False,indent=2),encoding='utf-8')
        print('Created editable PPTX, PDF, 33 previews and speaker notes:',OUT)


def build():
    eye=read(EXP/'eyes_summary.csv');yawn=read(EXP/'yawn_summary.csv')
    ep=read(EXP/'eyes_predictions.csv');yp=read(EXP/'yawn_predictions.csv')
    annotations=read(EXP/'annotations.csv');audit=json.loads((EXP/'data_audit.json').read_text())
    ec=json.loads((EXP/'eyes_config.json').read_text());yc=json.loads((EXP/'yawn_config.json').read_text())
    controls=read(EXP/'general_only_control_summary.csv')
    def result(rows,dataset,phase,split='external',condition='ALL',threshold=None):
        return next(r for r in rows if r['dataset']==dataset and r['phase']==phase and r['split']==split and r['condition']==condition and (threshold is None or abs(float(r['threshold'])-threshold)<1e-8))
    eb=result(eye,'cew','before');ea=result(eye,'cew','after')
    yb=result(yawn,'roboflow','before',threshold=.1);ya=result(yawn,'roboflow','after',threshold=.1)
    yb15=result(yawn,'roboflow','before',threshold=.15);ya15=result(yawn,'roboflow','after',threshold=.15)
    wr=[r for r in annotations if r['condition']=='wrinkles']
    getwr=lambda i:Path(next(r['image'] for r in wr if int(r['index'])==i))
    d=Deck()
    # 01
    d.slide('AURA DRIVE','오류 데이터 추가 학습과 외부 재검증',source='기존 33장 자료와 로컬 실험 로그를 바탕으로 전면 재구성',dark=True,
        note='이번 발표는 내부 테스트 수치에서 시작해 실제 얼굴 입력과 외부 데이터의 실패를 확인하고, 사용자가 제공한 화장·주름 데이터를 추가 학습한 결과까지 보여줍니다. 개선된 지표와 악화된 지표를 함께 제시합니다. 실제 차량용 안전 성능을 검증한 발표는 아닙니다.')
    d.pic(ROOT/'aura_car_interior.jpg',780,240,740,460)
    d.text(88,285,660,'내부 점수에서\n실제 실패로,\n실패에서 재검증으로.',61,WHITE,True)
    d.text(90,626,620,'눈 YOLO26n  ·  하품 ResNet18\n실험 이미지 / 고정 임계값 비교 / 시스템 구조',26,'#C3D8DF')
    #02
    d.slide('추가 학습의 효과는 정확도와 놓침을 함께 봅니다','동일 데이터 · 동일 전처리 · 동일 임계값으로 다시 측정',source='reports/hardcase_finetune/{eyes,yawn}_summary.csv',
        note=f"눈 CEW 정확도는 {pct(eb['accuracy'])}에서 {pct(ea['accuracy'])}로 {delta(eb['accuracy'],ea['accuracy'])} 변했습니다. 하품 Roboflow 정확도는 {pct(yb['accuracy'])}에서 {pct(ya['accuracy'])}로 {delta(yb['accuracy'],ya['accuracy'])} 변했습니다. 하품 재현율은 {pct(yb['recall'])}에서 {pct(ya['recall'])}입니다. 두 모델 모두 차량 적용에 충분하다고 주장하지 않습니다.")
    for x,title,b,a in [(84,'눈 · CEW 2,423장',eb,ea),(830,'하품 · Roboflow 393장',yb,ya)]:
        d.text(x,250,675,title,29,INK,True);d.text(x,310,675,f"{pct(b['accuracy'])} → {pct(a['accuracy'])}",49,TEAL,True)
        d.text(x,405,675,f"정확도 변화  {delta(b['accuracy'],a['accuracy'])}\n재현율  {pct(b['recall'])} → {pct(a['recall'])}\n오탐  {b['fp']} → {a['fp']}장",29,GRAY)
    d.callout('추가 학습의 채택 여부는 재현율·오탐·관측 실패를 함께 판단합니다.')
    #03
    d.slide('원본 영상에서 필요한 부위만 모델에 전달합니다','분류 앞에는 ROI 추출, 분류 뒤에는 임계값과 지속 시간 판단',source='roi_utils.py · teammate_yawn/03_yawn_common.py · 11_streamlit_dashboard.py',
        note='처리 순서는 원본 웹캠, MediaPipe 눈과 입 ROI 추출, ROI별 전처리, AI 분류, 임계값과 지속 시간 판단, 경고입니다. 눈 ROI는 CLAHE, 입 ROI는 RGB를 유지합니다. 전처리와 판정 기준은 서로 다른 단계입니다. 원본에 그린 랜드마크는 모델 입력으로 사용하지 않습니다.')
    d.flow([('원본 웹캠','RGB 프레임'),('MediaPipe','눈·입 ROI 추출'),('ROI 전처리','눈 CLAHE\n입 RGB'),('AI 분류','눈 YOLO26n\n입 ResNet18'),('상태 판단','임계값\n지속 시간'),('경고','화면·음향\n모의 대응')],280)
    d.text(110,555,1350,'눈: P(Closed) ≥ 0.25     하품: P(Yawn) ≥ 0.10 또는 0.15',34,TEAL,True)
    d.text(110,632,1340,'ROI 실패 시 UNKNOWN · 보이지 않는 상태를 정상으로 바꾸지 않음',29,GRAY)
    #04
    d.slide('출발 데이터와 평가 데이터의 역할을 구분합니다','기존 학습 분할을 유지하고, 새 데이터는 별도로 그룹 분할',source='data/processed/manifest.csv · teammate_yawn/data/mouth_wide_fixed/extraction_summary.csv',
        note='기존 원본은 Closed 726장, Open 726장, no_yawn 725장, yawn 723장으로 2,900장입니다. 눈 분할은 1017/218/217이고 하품 원본 분할은 1016/218/214입니다. 하품 ROI 추출 성공분은 1006/216/212입니다. 기존 유사 이미지 그룹 분할은 인물 단위 완전 독립을 보장하지 않습니다.')
    d.table(['데이터','학습','검증','테스트','의미'],[
        ['눈 원본','1,017','218','217','기존 분할 유지'],['하품 원본','1,016','218','214','ROI 실패 포함'],['하품 ROI 성공','1,006','216','212','분류 모델 입력']], [330,190,190,190,532],rowh=87)
    d.callout('유사 이미지 그룹 분리는 적용했지만, 모든 인물의 독립성은 보장하지 못합니다.')
    #05
    d.slide('내부 100%만으로 모델의 현실 성능을 고를 수 없습니다','눈 내부 Test 217장에서는 세 모델이 같은 정답률',source='reports/model_comparison.csv',
        note='눈 내부 테스트에서는 YOLO26n, ResNet18, MobileNetV3 모두 100%였습니다. 기존 비교 스크립트에서 측정한 중앙 지연은 약 11.4,25.7,14.7ms입니다. 이 값은 전체 웹캠 지연이나 초당 프레임 수가 아닙니다. 내부 분할의 높은 성능이 외부 일반화로 이어지는지 따로 확인해야 합니다.')
    d.table(['눈 모델','내부 정확도','기존 측정 지연','선택 시 고려'],[
        ['YOLO26n','100.00%','11.4 ms','경량 후보'],['MobileNetV3-Small','100.00%','14.7 ms','비교 모델'],['ResNet18','100.00%','25.7 ms','비교 모델']], [430,260,310,432],rowh=90)
    d.callout('다음 질문: 새로운 얼굴에서도 같은 결과가 나오는가?')
    #06
    shots=sorted((ROOT/'data/webcam_test/eyes/Open').glob('*.jpg'))
    shot=shots[0];roi=next((ROOT/'reports/eye_roi_examples').glob('Open*eye1.jpg'))
    d.slide('직접 촬영한 사진이 ROI의 필요성을 보여줬습니다','같은 65장 · 얼굴 전체 입력과 눈 ROI 입력의 차이',source='reports/eyes_yolo26n-cls_extra_confusion.csv · reports/eyes_roi_predictions.csv',
        note='직접 촬영한 65장에서는 얼굴 전체를 눈 분류기에 넣었을 때 35장 정답으로 53.85%였습니다. MediaPipe 눈 ROI 적용 후에는 전체 기준 55/65, 즉 84.62%였습니다. ROI 추출이 성공한 60장 중에는 55장 정답으로 91.67%입니다. 마지막 숫자는 실패 5장을 제외한 조건부 지표이므로 전체 정확도와 구분합니다. 추가 학습 결과와 다른 과거의 ROI 실험입니다.')
    d.pic(shot,84,245,340,315);d.text(120,578,320,'직접 촬영한 얼굴',23,GRAY)
    d.pic(roi,455,290,260,210);d.text(475,578,300,'추출한 눈 ROI',23,GRAY)
    d.text(790,250,720,'53.85% → 84.62%',56,TEAL,True)
    d.text(795,350,700,'얼굴 전체 35/65 → ROI 적용 55/65',28,INK)
    d.text(795,450,700,'91.67%',70,INK,True)
    d.text(795,550,700,'ROI 성공 이미지에 한정: 55/60\n전체 65장 중 ROI 실패 5장',28,GRAY)
    d.callout('학습 입력과 실제 입력의 부위·크기를 맞추는 것이 먼저였습니다.')
    #07
    d.slide('외부 검증은 출처와 입력 조건이 다른 네 가지 평가입니다','눈과 하품 모두 ROI 방식 사용 · 자료 확보 범위는 서로 다름',source='reports/{cew,mrl,simuletic,yawn_roboflow}/external_summary.csv',
        note='눈은 CEW 2423장, 이미 잘린 눈 패치 MRL 4000장, Simuletic 운전석 143장을 평가했습니다. 하품은 별도 Roboflow 393장을 평가했습니다. 하품 평가가 더 작은 이유는 ROI 방식이 아니라 독립 데이터 확보와 프로젝트 범위의 선택입니다. MRL은 기존 학습 원자료와의 중복 가능성이 있습니다. 모든 데이터가 인물 독립 최종 시험셋은 아닙니다.')
    d.table(['평가','규모','입력','해석 범위'],[
        ['CEW · 눈','2,423장','얼굴 → 눈 ROI','설정 선택에도 사용'],['MRL · 눈','4,000장','이미 잘린 눈 패치','학습 원자료 중복 가능'],['Simuletic · 눈','143장','운전석 얼굴 → ROI','제한된 촬영 조건'],['Roboflow · 하품','393장','얼굴 → 입 ROI','별도 출처 · 범위 제한']], [300,220,420,492],rowh=81)
    d.callout('하품 평가 규모의 이유를 “입 ROI 방식 때문”으로 설명하지 않습니다.')
    #08
    auditrows=read(ROOT/'reports/cew/error_audit/pairwise_images.csv')
    names=['Alina_Kabaeva','Anjelica_Huston','Abid_Hamid']
    chosen=[]
    for name in names:
        matches=[r for r in auditrows if name in r['image'] and r['actual']=='Open' and r['yolo_pred']=='Closed']
        if matches:chosen.append(matches[0])
    while len(chosen)<3:chosen.append(next(r for r in auditrows if r['actual']=='Open' and r['yolo_pred']=='Closed' and r not in chosen))
    d.slide('CEW 오류 사진에서 실제로 무엇이 보였는가','아래는 정답 Open인데 기존 모델이 Closed로 분류한 실제 사례',source='reports/cew/error_audit/pairwise_images.csv · both_fail_Open.png',
        note='기존 오류 감사 파일에서 Open을 Closed로 오분류한 실제 이미지를 가져왔습니다. 눈 화장, 흐릿한 눈꺼풀 경계, 눈썹이나 안경 같은 요소가 오분류와 함께 관찰됩니다. 이를 모델이 화장이나 주름을 직접 보고 틀렸다는 원인 증명으로 해석하지 않습니다. 원인 확인에는 조건별 라벨과 통제 실험이 필요합니다.')
    for j,r in enumerate(chosen):
        x=84+j*485;d.pic(r['image'],x,235,440,360);d.text(x,618,440,'정답 Open → 예측 Closed',25,RED,True)
        d.text(x,661,440,['화장·눈꺼풀 경계 관찰','짙은 눈 주변·저해상도','그림자·눈꺼풀 경계'][j],24,GRAY)
    d.callout('관찰한 특징과 오류의 원인은 구분해야 합니다.',color=RED)
    #09
    import importlib
    ab=importlib.import_module('15_roi_ablation')
    cew_pred=next(r for r in ep if r['dataset']=='cew' and r['actual']=='Open' and r['crops'] and float(r['before_probability'])>=.25)
    raw=Image.open(cew_pred['crops'].split('|')[0]);enh=ab.make_variants(raw)['clahe']
    d.slide('CLAHE는 눈 ROI의 밝기 대비를 조정합니다','원본 얼굴 → 눈 ROI 추출 → 밝기 보정 → 눈 분류',source='15_roi_ablation.py · 16_threshold_analysis.py',
        note='CLAHE는 LAB 색 공간에서 밝기 채널의 국소 대비를 보정합니다. 왼쪽과 오른쪽은 같은 눈 ROI입니다. 과거 실험의 개선은 전처리 및 판정 임계값 변경 결과이며 모델 가중치 변경 결과가 아닙니다. 특정 화장 유형의 오류가 해결됐다는 별도 검증은 당시 없었습니다.')
    d.pic(raw,110,280,600,280);d.pic(enh,880,280,600,280)
    d.text(110,590,600,'기존 눈 ROI',32,INK,True);d.text(880,590,600,'CLAHE 적용 ROI',32,TEAL,True)
    d.text(110,660,1360,'LAB 밝기 채널 · clipLimit 2.0 · tileGridSize 4×4',27,GRAY)
    d.callout('이 단계에서 바꾼 것은 입력과 판정 기준이며, 모델 가중치는 같았습니다.')
    #10
    d.slide('눈 임계값을 낮추면 민감도와 오탐이 함께 움직입니다','과거 CEW 2,423장 실험 · Closed 확률을 기준으로 판정',source='reports/cew/threshold_analysis/threshold_sweep.csv',
        note='기존 RGB/0.50에서 전체 정확도 79.12%, Closed 재현율94.21%, Open 오탐률35.09%였습니다. CLAHE/0.25에서는 정확도88.36%,재현율90.69%,오탐률13.48%였습니다. CLAHE를 적용한 뒤 0.50에서0.25로 임계값을 낮추면서 재현율을 회복한 것입니다. 과거의 임계값 실험과 이번의 실제 가중치 추가 학습을 구분합니다.')
    d.table(['기존 설정','전체 정확도','Closed 재현율','Open 오탐률'],[
        ['기존 입력 · 0.50','79.12%','94.21%','35.09%'],['CLAHE · 0.50',pct(next(r['accuracy'] for r in read(ROOT/'reports/cew/threshold_analysis/threshold_sweep.csv') if r['model']=='yolo26n' and r['variant']=='clahe' and float(r['threshold'])==.5)),'81.12%','3.17%'],['CLAHE · 0.25','88.36%','90.69%','13.48%']], [490,314,314,314],rowh=93)
    d.callout('0.25는 가중치를 고친 값이 아니라 Closed 확률의 판정 기준입니다.')
    #11
    d.slide('추가 화장 데이터는 26쌍의 전·후 사진입니다','같은 사람의 화장 전·후를 서로 다른 분할에 넣지 않음',source='data/error_data/{no_makeup,with_makeup} · annotations.csv',
        note='화장 전후 각26장으로 같은 번호의 사진은 같은 인물 쌍으로 관찰됩니다. 두 사진은 반드시 같은 학습,검증,테스트 그룹에 배치했습니다. 이미지별 눈 상태를 검토하여51장을 Open으로 표시했고 반사로 눈 상태가 불확실한1장은 제외했습니다. 이 데이터는 눈 감김 양성 사례를 보강하지 못하며 화장만의 인과 효과를 엄밀히 측정한 자료도 아닙니다.')
    for j,i in enumerate([1,10,20]):
        x=84+j*485
        d.pic(ROOT/f'data/error_data/no_makeup/{i}.jpg',x,250,210,330)
        d.pic(ROOT/f'data/error_data/with_makeup/{i}.jpg',x+225,250,210,330)
        d.text(x,606,440,f'쌍 {i:02d}  ·  화장 전 / 화장 후',25,GRAY)
    d.callout('조건 라벨(화장)과 정답 라벨(Open/Closed)은 별개입니다.')
    #12
    d.slide('주름 459장을 모두 비하품 학습 데이터로 쓸 수는 없습니다','피부 질감만 있는 사진에서는 하품 여부를 판단할 수 없음',source='data/error_data/wrinkles · reports/hardcase_finetune/annotations.csv',
        note='주름 폴더에는 피부만 확대된 패치와 실제 얼굴이 섞여 있습니다. 피부 패치는 입 상태 정답을 붙일 근거가 없어 UNKNOWN으로 제외했습니다. 보이는 입 상태를 기준으로 비하품154장을 표시했지만 일부는 기존 MediaPipe 입 추출에 실패합니다. 억지로 피부 패치를 입 ROI처럼 넣으면 실시간 입력과 다른 학습을 하게 됩니다.')
    for j,i in enumerate([0,45,384,428]):d.pic(getwr(i),84+j*365,260,330,330)
    d.text(84,624,690,'피부만 보임 → 정답 불명 · 학습 제외',28,RED,True)
    d.text(814,624,700,'얼굴·입이 보임 → 정답과 ROI 검토',28,TEAL,True)
    d.callout('주름 존재는 하품 여부의 정답이 아닙니다.')
    #13
    d.slide('새 데이터도 학습·검증·테스트를 분리했습니다','알려진 인물 쌍 + 유사 이미지 그룹 · seed 42',source='reports/hardcase_finetune/data_audit.json · annotations.csv',
        note='새 데이터는 그룹을 기준으로60/20/20에 가깝게 나눴습니다. 표는 정답이 있고 ROI도 성공한 이미지 수입니다. 전체511장 중 Open51,비하품154,UNKNOWN306장입니다. 외부 이미지와 유사한1장은 그룹 전체를 학습에서 제외했습니다. 정답은 AI 보조 육안 판독으로 독립적인 사람 검수는 아직 없습니다. 모든 사진의 인물 독립성까지 보장하지는 못합니다.')
    d.table(['ROI 성공 추가 데이터','학습','검증','테스트'],[
        ['화장 전·후 눈',audit['usable']['eyes_train'],audit['usable']['eyes_val'],audit['usable']['eyes_test']],
        ['주름 조건 비하품',audit['usable']['yawn_train'],audit['usable']['yawn_val'],audit['usable']['yawn_test']]], [700,244,244,244],rowh=97)
    d.text(90,594,1410,f"511장 검토 · 정답 불명 306장 제외 · 외부 유사 이미지 {audit['overlap_images']}장 제외",31,INK,True)
    d.text(90,666,1410,'정답은 AI 보조 육안 판독 · 독립적인 사람 검수 전',25,GRAY)
    d.callout('새로운 양성(눈 감김·하품) 보강은 없음 · 소규모 조건 평가입니다.',color=AMBER)
    #14
    d.slide('이번에는 분류층 가중치를 실제로 추가 학습했습니다','기존 모델 구조와 ROI 방식 유지 · CPU에서 마지막 선형층만 갱신',source='22_hardcase_finetune.py · {eyes,yawn}_config.json · {eyes,yawn}_training.csv',
        note=f"YOLO26n과 ResNet18의 특징 추출부를 고정하고 마지막 선형 분류층만 추가 학습했습니다. 기존 일반 데이터 손실80%,추가 데이터20%를 사용했습니다. AdamW 학습률0.0001,40회 업데이트 후 검증 손실 최소 시점을 선택했습니다. 눈은{ec['best_epoch']}회,하품은{yc['best_epoch']}회입니다. 외부 및 테스트 결과를 보고 모델이나 임계값을 선택하지 않았습니다. 모델 전체 미세조정과는 범위가 다릅니다.")
    d.flow([('기존 특징 추출부','YOLO26n / ResNet18\n가중치 고정'),('분류층 추가 학습','기존 일반 데이터 80%\n추가 조건 데이터 20%'),('검증셋으로 선택',f"눈 {ec['best_epoch']}회 · 하품 {yc['best_epoch']}회\n검증 손실 최소"),('외부 재평가','눈 0.25 고정\n하품 0.10 / 0.15 고정')],260)
    d.text(100,570,1390,'AdamW · LR 1e-4 · 40회 업데이트 · RGB/CLAHE 및 클래스 순서 유지',29,GRAY)
    ctrl_eye=next(r for r in controls if r['task']=='eyes' and r['dataset']=='cew')
    ctrl_yawn=next(r for r in controls if r['task']=='yawn' and r['dataset']=='roboflow' and float(r['threshold'])==.1)
    d.text(100,636,1390,f"일반 데이터만 추가 학습한 대조군: 눈 {pct(ctrl_eye['accuracy'])} · 하품 {pct(ctrl_yawn['accuracy'])}\n추가 조건 데이터 혼합: 눈 {pct(ea['accuracy'])} · 하품 {pct(ya['accuracy'])}",25,GRAY)
    d.callout('과거: 전처리·임계값 조정  /  이번: 분류층 가중치 추가 학습')
    #15
    d.slide('눈 모델: 외부 정확도와 재현율의 변화','CEW 전체 2,423장 · ROI 실패를 오답에 포함 · 임계값 0.25',source='reports/hardcase_finetune/eyes_summary.csv',
        note=f"같은 CEW2423장을 재평가한 정확도는 {pct(eb['accuracy'])}→{pct(ea['accuracy'])},Closed 재현율은 {pct(eb['recall'])}→{pct(ea['recall'])},Open 오탐률은 {pct(eb['fpr'])}→{pct(ea['fpr'])}입니다. 정확도가 올랐어도 재현율 하락이 있으면 함께 말해야 합니다. 이 결과가 특정 화장 유형 해결이나 새로운 운전자에 대한 확정 성능을 뜻하지 않습니다.")
    d.bars(['정확도 · 학습 전','정확도 · 학습 후','재현율 · 학습 전','재현율 · 학습 후'],[float(eb['accuracy'])*100,float(ea['accuracy'])*100,float(eb['recall'])*100,float(ea['recall'])*100],w=1370,rowh=102,colors=[GRAY,TEAL,GRAY,TEAL])
    d.callout(f"정확도 {delta(eb['accuracy'],ea['accuracy'])}  ·  Open 오탐 {eb['fp']} → {ea['fp']}장")
    #16
    def changes(preds,task,ds,t):
        pos='Closed' if task=='eyes' else 'yawn'
        fixed=[];regressed=[];still=[]
        for r in preds:
            if r['dataset']!=ds or not r['before_probability']:continue
            truth=r['actual']==pos;b=float(r['before_probability'])>=t;a=float(r['after_probability'])>=t
            if b!=truth and a==truth:fixed.append(r)
            if b==truth and a!=truth:regressed.append(r)
            if b!=truth and a!=truth:still.append(r)
        return fixed,regressed,still
    def example(r,x,y,w,probname):
        d.pic(r['image'],x,y,w*.56,285)
        if r['crops']:d.pic(r['crops'].split('|')[0],x+w*.59,y+50,w*.39,180)
        d.text(x,y+306,w,f"정답 {r['actual']} · {probname}",24,INK,True)
        d.text(x,y+350,w,f"{float(r['before_probability']):.3f} → {float(r['after_probability']):.3f}",36,TEAL,True)
    ef,er,es=changes(ep,'eyes','cew',.25)
    d.slide('눈: 고친 사례와 새로 틀린 사례를 함께 공개합니다','좋아진 사례만 선택하면 실제 개선 폭을 과장하게 됩니다.',source='reports/hardcase_finetune/eyes_predictions.csv · roi/cew',
        note=f'CEW에서 오답에서 정답으로 바뀐 사진은{len(ef)}장,정답에서 오답으로 바뀐 사진은{len(er)}장입니다. 대표 사진의 원본,실제 ROI,학습 전후 Closed 확률을 함께 표시했습니다. 얼굴의 어떤 특징이 직접 원인인지 확정한 결과는 아닙니다.')
    example((ef or es)[0],84,240,675,'P(Closed)');example((er or es)[0],830,240,675,'P(Closed)')
    d.text(84,689,675,f'오답 → 정답  {len(ef)}장',27,TEAL,True);d.text(830,689,675,f'정답 → 오답  {len(er)}장',27,RED,True)
    #17
    etb=result(eye,'added','before','test');eta=result(eye,'added','after','test')
    d.slide('화장 조건 테스트는 “눈 뜸 오탐”만 확인합니다','추가 데이터의 테스트 분할 · 눈 감김 재현율은 계산할 수 없음',source='reports/hardcase_finetune/eyes_summary.csv · annotations.csv',
        note=f"학습하지 않은 추가 눈 테스트는 {etb['n']}장이고 모두 Open입니다. 전후 정답률은{pct(etb['accuracy'])}→{pct(eta['accuracy'])}입니다. 눈 감김 샘플이 없으므로 이 숫자를 두 클래스 전체 정확도나Closed 재현율 개선으로 주장하지 않습니다. 같은 사람 전후쌍을 묶었지만 표본이 작고 라벨은 보조 육안 검토입니다.")
    table=[]
    for cond,label in [('no_makeup','화장 전'),('with_makeup','화장 후')]:
        b=result(eye,'added','before','test',cond);a=result(eye,'added','after','test',cond)
        table.append([label,b['n'],f"{b['fp']} → {a['fp']}장",f"{pct(b['accuracy'])} → {pct(a['accuracy'])}"])
    d.table(['조건','테스트 수','Open 오탐','Open 정답률'],table,[320,240,330,542],rowh=100)
    d.text(90,600,1410,'작은 표본 + 음성 클래스만 존재 → 화장 오류 해결을 일반화할 수 없음',31,RED,True)
    d.callout('정답률이 높아져도 눈을 감았을 때 놓치지 않는지는 별도 문제입니다.')
    #18
    d.slide('하품은 내부와 외부 사이에 큰 차이가 있었습니다','과거 평가: 내부 ROI 성공분 212장 / 외부 원본 393장',source='teammate_yawn/runs/03_yawn_resnet18 · reports/yawn_roboflow/external_summary.csv',
        note='기존 하품ResNet18은 내부ROI성공분212장 중210장 정답99.06%였으나 외부393장에서는258장정답65.65%였습니다. 외부393장 중입ROI성공344,실패49장이며,성공분에서도86장 오분류했습니다. 서로 다른분모라는 점과ROI실패만이 문제는 아니라는 점을 강조합니다. 임계값0.50 기준입니다.')
    d.bars(['내부 ROI 성공분','외부 전체','외부 하품 재현율'],[99.06,65.65,44.44],w=1370,rowh=135,colors=[BLUE,TEAL,RED])
    d.callout('외부 393장 = 하품 171장 + 비하품 222장 · ROI 실패 49장')
    #19
    d.slide('흑백 전처리로 바꾸는 것이 답은 아니었습니다','기존 가중치 · 하품 임계값 0.10 · 같은 Roboflow 393장',source='reports/yawn_roboflow/combined_threshold/combined_threshold_sweep.csv',
        note='기존RGB0.10은정확도71.50%,재현율66.67%,F1 72.84%입니다. 흑백0.10은정확도61.83%,재현율67.84%,F1 65.35%입니다. 재현율만 소폭 증가했지만오탐68장으로늘었습니다. 이데이터에서는RGB유지가더균형적이었습니다. 이번추가학습도RGB를유지했습니다.')
    d.table(['전처리 / 0.10','정확도','하품 재현율','F1','오탐'],[
        ['RGB 유지','71.50%','66.67%','72.84%','28장'],['흑백 변환','61.83%','67.84%','65.35%','68장']], [430,245,270,245,242],rowh=110)
    d.callout('이번 추가 학습에서는 기존 RGB 입력을 유지했습니다.')
    #20
    d.slide('하품 0.10과 0.15는 놓침·오탐의 선택입니다','과거 임계값 실험 · 가중치 변경 없이 판정 민감도 조정',source='reports/yawn_roboflow/combined_threshold/candidates.csv',
        note='0.50에서는재현율44.44%였고0.10에서는66.67%,0.15에서는64.33%였습니다. 0.10과0.15의전체정확도는둘다71.50%지만오탐과미탐구성은다릅니다.0.10에서도실제하품171장중57장을놓쳤고그중22장은ROI실패입니다. 이수준을실차적용에충분하다고말할수없습니다.')
    d.table(['RGB 임계값','정확도','재현율','오탐','하품 놓침'],[
        ['0.50','65.65%','44.44%','13장','95장'],['0.10','71.50%','66.67%','28장','57장'],['0.15','71.50%','64.33%','24장','61장']], [350,270,270,250,292],rowh=92)
    d.callout('0.10에서도 하품 171장 중 57장 미검출 · 이 중 ROI 실패 22장',color=RED)
    #21
    oldy=read(ROOT/'reports/yawn_roboflow/error_audit/all_cases.csv')
    ycandidates=[r for r in oldy if r['actual']=='yawn' and r['predicted']=='no_yawn' and r['roi_image']]
    d.slide('입이 보이는데도 놓치는 사례가 남았습니다','실제 외부 오류 이미지 · 확률은 기존 임계값 0.50 평가 로그',source='reports/yawn_roboflow/error_audit/all_cases.csv · missed_yawn_page_01.jpg',
        note='실제하품오류감사에서입이보이지만비하품으로분류한이미지를가져왔습니다. 원본과입ROI에입술색,턱과목주변의선,가림과흐림이함께보입니다. 목주름때문이라고단정하지않고가능한실패조건으로기록합니다. 단일정지사진만으로하품과말하기의시간적차이를검증할수없습니다.')
    for j,r in enumerate(ycandidates[:3]):
        x=84+j*485;d.pic(r['image'],x,245,430,245);d.pic(r['roi_image'],x+80,505,260,155)
        d.text(x,690,440,f"Yawn → no_yawn · P={float(r['positive_probability']):.3f}",23,RED,True)
    #22
    usable=[r for r in annotations if r['task']=='yawn' and r['split']=='train' and r['roi_status']=='ok']
    d.slide('주름 조건 학습도 실제 입 ROI로 제한했습니다','피부 패치를 직접 넣지 않고 기존 MediaPipe 입 추출을 통과한 사진 사용',source='reports/hardcase_finetune/annotations.csv · roi/wrinkles',
        note='비하품으로판독한사진중입ROI가성공한학습분81장을추가했습니다. 여기에는턱과입주변주름이있는비하품표정이포함됩니다. 정지사진의보이는상태를기준으로라벨을붙였으며모든주름유형의효과를검증한것은아닙니다. 새하품양성표본은없으므로비하품쪽으로결정경계가움직일위험도있습니다.')
    for j,r in enumerate(usable[:3]):
        x=84+j*485;d.pic(r['image'],x,240,430,300);d.pic(r['crops'].split('|')[0],x+95,560,240,130)
    d.callout('추가 학습 입 ROI 81장 · 모두 비하품 · 기존 하품 양성 데이터와 혼합')
    #23
    d.slide('하품 모델: 오탐은 줄었지만 놓침이 늘었습니다','Roboflow 전체 393장 · RGB 유지 · 두 임계값 모두 고정 비교',source='reports/hardcase_finetune/yawn_summary.csv',
        note=f"임계값0.10에서정확도{pct(yb['accuracy'])}→{pct(ya['accuracy'])},재현율{pct(yb['recall'])}→{pct(ya['recall'])},오탐{yb['fp']}→{ya['fp']}장입니다.0.15에서는정확도{pct(yb15['accuracy'])}→{pct(ya15['accuracy'])},재현율{pct(yb15['recall'])}→{pct(ya15['recall'])}입니다. 비하품데이터추가가오탐감소와하품놓침에어떤영향을줬는지함께판단합니다.")
    d.table(['고정 임계값','정확도 전 → 후','재현율 전 → 후','오탐 전 → 후'],[
        ['RGB · 0.10',f"{pct(yb['accuracy'])} → {pct(ya['accuracy'])}",f"{pct(yb['recall'])} → {pct(ya['recall'])}",f"{yb['fp']} → {ya['fp']}장"],
        ['RGB · 0.15',f"{pct(yb15['accuracy'])} → {pct(ya15['accuracy'])}",f"{pct(yb15['recall'])} → {pct(ya15['recall'])}",f"{yb15['fp']} → {ya15['fp']}장"]], [300,410,410,312],rowh=110)
    d.text(90,615,1400,f"0.10 기준 하품 놓침 {yb['fn_including_unknown']} → {ya['fn_including_unknown']}장 / 실제 하품 171장",34,RED,True)
    d.callout('하품 추가 학습 후보는 채택하지 않고 기존 모델을 유지합니다.',color=RED)
    #24
    yf,yr,ys=changes(yp,'yawn','roboflow',.1)
    d.slide('하품: 바뀐 사진을 확률과 함께 추적합니다','평가 사진은 추가 학습에 넣지 않았습니다.',source='reports/hardcase_finetune/yawn_predictions.csv',
        note=f'임계값0.10에서오답→정답은{len(yf)}장,정답→오답은{len(yr)}장입니다. 대표사진의원본과ROI,하품확률을제시했습니다. 주름이나입술색이직접원인이었다는주장은하지않습니다. 임계값을사후조정하지않은상태의변화입니다.')
    example((yf or ys)[0],84,240,675,'P(Yawn)');example((yr or ys)[0],830,240,675,'P(Yawn)')
    d.text(84,689,675,f'오답 → 정답  {len(yf)}장',27,TEAL,True);d.text(830,689,675,f'정답 → 오답  {len(yr)}장',27,RED,True)
    #25
    d.slide('성능 변화는 평가 범위를 붙여서 보고합니다','이번 재평가와 과거 실험의 수치를 섞지 않음',source='reports/hardcase_finetune/{eyes,yawn}_summary.csv',
        note='표는이번실험에서동일전처리와동일임계값으로재측정한전후비교입니다. 내부눈도CLAHE와0.25가적용되어과거RGB기본판정100%와조건이다릅니다. Simuletic과직접촬영65장,추가테스트조건을함께확인했습니다. MRL4000장의과거수치를이번모델재평가처럼표시하지않았습니다.')
    rows=[]
    for name,src,ds,split,t in [('눈 CEW',eye,'cew','external',.25),('눈 Simuletic',eye,'simuletic','external',.25),('눈 직접 촬영',eye,'webcam','external',.25),('눈 내부 Test',eye,'internal','test',.25),('하품 외부',yawn,'roboflow','external',.1),('하품 내부 ROI Test',yawn,'internal','test',.1)]:
        b=result(src,ds,'before',split,threshold=t);a=result(src,ds,'after',split,threshold=t)
        rows.append([name,b['n'],pct(b['accuracy']),pct(a['accuracy']),delta(b['accuracy'],a['accuracy'])])
    d.table(['평가','N','학습 전','학습 후','변화'],rows,[440,160,275,275,282],y=224,rowh=72,size=24)
    #26
    yt=result(yawn,'added','after','test',threshold=.1)
    d.slide('ROI 실패는 분류층 학습만으로 해결되지 않습니다','검출할 부위가 없으면 확률 대신 UNKNOWN으로 기록',source='reports/hardcase_finetune/*_summary.csv · data_audit.json',
        note=f"기존외부평가에서CEW눈ROI실패는{eb['unknown']}장,Roboflow입ROI실패는{yb['unknown']}장입니다. 이번분류층추가학습은검출기를바꾸지않기때문에이실패를고치지못합니다. 전체정확도에는UNKNOWN을오답에포함하고,분류가능한사진만의정확도는별도로구분합니다. 새주름테스트도ROI실패를분모에서감추지않습니다.")
    d.text(100,270,650,eb['unknown']+'장',90,AMBER,True);d.text(100,405,650,'CEW 눈 ROI 실패 / 2,423장',31,INK)
    d.text(860,270,650,yb['unknown']+'장',90,RED,True);d.text(860,405,650,'Roboflow 입 ROI 실패 / 393장',31,INK)
    d.text(100,555,1390,f"추가 주름 테스트: 전체 {yt['n']}장 중 UNKNOWN {yt['unknown']}장\n피부 패치·고개 숙임·가림은 데이터와 검출 단계에서 따로 다뤄야 함",31,GRAY)
    d.callout('UNKNOWN을 정상 판정으로 바꾸면 실제 놓침을 숨기게 됩니다.',color=RED)
    #27
    d.slide('정지사진 정확도와 영상 경고 성능은 다른 지표입니다','기존 웹캠 A/B 기록은 경고 횟수 확인 · 프레임 정확도로 환산하지 않음',source='reports/webcam_eye_ab/*.csv · 17_webcam_eye_ab_test.py',
        note='기존웹캠기록에서수동눈감김2회에기존과CLAHE방식모두경고2회가있었습니다. 다른기록에서는수동눈감김3회에도양쪽경고가0회였습니다. 수동라벨시작종료오차때문에프레임정확도로환산하지않았습니다. 이번추가학습모델을라이브웹캠에서시험했다는주장도하지않습니다. 다음검증은독립된하품이벤트놓침과시간당오경고,경고지연입니다.')
    d.flow([('정지사진','Open / Closed\nYawn / no_yawn'),('연속 프레임','확률의 변화\n관측 실패·가림'),('행동 이벤트','지속 시간\n시작과 종료'),('경고 품질','놓친 이벤트\n오경고·지연')],270)
    d.text(100,585,1400,'현재 기본 지속 기준: 눈 2.0초 · 하품 1.5초',34,TEAL,True)
    d.callout('이번에 검증한 범위는 정지사진 전후 비교입니다.')
    #28
    d.slide('카메라 표시와 AI 분석의 속도를 분리합니다','원본 화면은 유지하고, 분석은 최신 프레임을 우선 처리',source='11_streamlit_dashboard.py · LatestFrameWorker',
        note='현재대시보드에는카메라표시와최신프레임분석을분리하는구조가있습니다. AI분석주기는250ms설정이며전체FPS측정결과가아닙니다. 오래된결과와ROI실패를구분해야합니다. 추가학습후라이브성능은이번정지사진실험에서측정하지않았습니다.')
    d.rect(100,280,340,145,LIGHT);d.text(125,313,290,'카메라 캡처',34,INK,True)
    d.line(440,350,660,350,TEAL,4);d.line(550,350,550,600,TEAL,4)
    d.rect(660,280,820,145,LIGHT);d.text(695,306,740,'원본 프레임 표시',35,INK,True);d.text(695,365,740,'모델 추론 완료를 기다리지 않는 표시 경로',25,GRAY)
    d.line(550,600,660,600,TEAL,4);d.rect(660,510,820,180,NAVY)
    d.text(695,534,740,'최신 프레임 → ROI → 분류 → 결과 갱신',31,WHITE,True)
    d.text(695,600,740,'분석 주기 250 ms 설정 · 지연 결과 / UNKNOWN 구분',24,'#C3D8DF')
    #29
    d.slide('반복 경고 이후의 대응은 화면 속 시뮬레이션입니다','5분 내 장시간 눈 감김 경고 2회 → 가상 감속과 정차',source='aura_safety_sim.py · 11_streamlit_dashboard.py',
        note='실시간눈감김경고가5분내2회쌓이면화면속가상감속과정차단계로이어집니다. 이후휴식,응답확인,모의SOS가순서대로표시됩니다. 실제브레이크나창문제어,실제119신고는없습니다. 일반시나리오의휴식5분과무반응10분은의료나차량안전기준이아닌시연값입니다.')
    d.flow([('반복 눈 감김','경고 이벤트 누적'),('가상 감속','화면 속 80 → 0'),('정차·휴식','재확인 안내'),('응답 확인','버튼 / 시연 신호'),('모의 SOS','화면·로그 표시')],290)
    d.text(100,575,1370,'실제 차량 제어 · 실제 긴급 신고 · 의식 상실 진단 기능은 없습니다.',34,RED,True)
    d.callout('분류 모델의 성능과 대응 시나리오의 구현은 별도로 평가합니다.')
    #30
    d.slide('이번 결과로 주장할 수 있는 범위를 정리합니다','화장·주름 오류를 “해결했다”는 표현은 아직 이릅니다.',source='실험 설계 · 라벨 검토 · 전후 평가 로그',
        note='이번실험은고정된평가셋에서분류층추가학습의영향을측정한것입니다. 추가데이터는음성클래스만있고정답이AI보조판독이며표본이작습니다. CEW와Roboflow는이전설정선택에이미사용되어완전히새로운최종평가셋이아닙니다. 화장이나주름의인과적오류원인,실제차량안전성,미사용운전자일반화는입증하지못했습니다. 기존배포모델을자동으로교체하지않습니다.')
    d.table(['확인한 것','아직 확인하지 못한 것'],[
        ['고정 데이터·임계값에서의 전후 차이','미사용 운전자·조명에서의 성능'],['원본·ROI·확률로 추적한 개별 오류','화장·주름이 직접 원인인지'],['음성 사례 보강의 오탐·미탐 변화','새 하품 양성·눈 감김 조건의 개선'],['정지사진 분류층 학습 결과','실제 영상 이벤트와 차량 안전성']], [716,716],rowh=90,size=25)
    #31
    d.slide('다음 실험은 놓치는 양성과 시간 정보를 보강합니다','이번 추가 학습 결과를 다음 검증 설계로 연결',source='미구현 후속 실험 제안 · 이번 성능 검증 범위에 포함하지 않음',
        note='다음단계는새로운사람을분리한눈감김과하품양성데이터를확보하는것입니다. 약한하품과측면하품,말하기와웃기같은어려운음성도함께필요합니다. EAR/MAR는랜드마크거리와입벌어짐지속시간을보조지표로시험할수있지만랜드마크가틀리면실패합니다. 현재구현되거나성능향상이검증된기능으로발표하지않습니다.')
    for y,num,title,body in [(245,'01','실패 조건의 양성 데이터','화장한 눈의 Closed · 약한 하품 · 측면 하품'),(385,'02','혼동 행동을 함께 평가','말하기 · 웃기 · 입 가리기 · 큰 입 벌림'),(525,'03','시간에 따른 행동 분석','EAR/MAR 보조 지표 + 지속 시간 + 영상 이벤트')]:
        d.text(90,y,100,num,50,TEAL,True);d.text(210,y,1280,title,33,INK,True);d.text(210,y+57,1280,body,27,GRAY)
    d.callout('새 평가셋은 모델과 임계값을 고르기 전에 분리합니다.')
    #32
    d.slide('결과를 다시 확인할 수 있도록 파일을 남겼습니다','이미지별 정답 · 분할 · 확률 · 모델 가중치 · 발표 원본',source='22_hardcase_finetune.py · build_aura_revalidation.py',
        note='annotations.csv에는각사진의정답,검토메모,그룹,분할,ROI상태가있습니다. predictions.csv에는원본경로와전후확률이,summary.csv에는전체분모를포함한지표가있습니다. 학습설정과가중치해시도보관했습니다. 기존가중치를덮어쓰지않고별도후보파일로저장했습니다. PPT텍스트와도형은편집할수있고동일구성PDF와발표대본도제공합니다.')
    d.table(['파일','확인할 내용'],[
        ['annotations.csv','511장 판독·그룹·분할·ROI·제외 사유'],['eyes/yawn_predictions.csv','이미지별 학습 전·후 확률'],['eyes/yawn_summary.csv','정확도·재현율·오탐·UNKNOWN'],['eyes/yawn_config.json','학습 조건·선택 시점·가중치 SHA-256'],['runs/hardcase_finetune/*.pt','기존 모델과 분리한 추가 학습 후보']], [700,732],rowh=80,size=25)
    #33
    d.slide('개선은 “더 높은 숫자”보다 “덜 놓치는 판단”으로 평가합니다','AURA DRIVE · 오류 데이터로 학습하고, 같은 조건에서 다시 검증',source='최종 결론: reports/hardcase_finetune/실험결과.md',dark=True,
        note=f"이번눈정확도변화는{delta(eb['accuracy'],ea['accuracy'])},하품정확도변화는{delta(yb['accuracy'],ya['accuracy'])}였습니다. 하지만정확도만으로성공을판단하지않습니다. 오탐과재현율,ROI실패를함께봐야합니다. 이번핵심은실패사진을근거로데이터를선별하고실제분류층을학습한뒤향상과퇴행을모두기록했다는것입니다. 다음목표는독립데이터와영상이벤트검증입니다.")
    d.text(90,270,1400,f"눈 정확도  {delta(eb['accuracy'],ea['accuracy'])}\n하품 정확도  {delta(yb['accuracy'],ya['accuracy'])}",64,'#74D7CA',True)
    d.text(90,500,1390,'오탐 감소와 재현율 변화를 함께 공개했습니다.\n화장·주름 오류 해결과 실차 성능은 아직 검증 과제입니다.',37,WHITE)
    d.text(90,680,1390,'다음 단계: 새로운 양성 데이터 + 인물 분리 + 영상 이벤트 평가',30,'#B9CFD9')
    # Speaker notes use complete Korean sentences, independently readable from slides.
    polished={
        18:'기존 하품 ResNet18은 내부 ROI 성공분 212장 중 210장을 맞혀 99.06%였습니다. 외부 393장에서는 258장 정답으로 65.65%였습니다. 두 수치는 분모의 조건이 다르므로 직접적인 모델 성능 차이로만 읽으면 안 됩니다. 외부에서는 ROI 실패 49장이 있었고, ROI 성공분에서도 86장을 틀렸습니다. 즉 검출 실패와 분류 실패가 함께 존재했습니다.',
        19:'기존 RGB 입력과 흑백 입력을 임계값 0.10에서 비교했습니다. 흑백은 하품 재현율이 67.84%로 조금 높았지만, 오탐 68장과 정확도 61.83%로 전체 균형이 나빴습니다. RGB는 정확도 71.50%, 재현율 66.67%, 오탐 28장이었습니다. 이번 추가 학습에서는 이 결과에 따라 기존 RGB 입력을 유지했습니다.',
        20:'과거 임계값 조정은 가중치 변경 없이 하품 확률의 판정 기준만 바꾼 실험입니다. 0.10과 0.15는 정확도가 같지만 놓침과 오탐의 수가 다릅니다. 0.10에서도 하품 171장 중 57장을 놓쳤으며 이 중 22장은 입 ROI 추출 실패입니다. 71.50%라는 정확도를 실제 차량 적용에 충분한 수준이라고 주장하지 않습니다.',
        21:'오류 감사 폴더의 실제 하품 미검출 이미지를 원본과 입 ROI로 나란히 보여줍니다. 입술 색, 턱·목 주변의 선, 손 가림, 흐림과 함께 오분류가 관찰됩니다. 주름이나 화장이 직접적인 원인이었다고 단정할 근거는 없습니다. 현재 자료는 정지사진이며, 입이 벌어졌다 닫히는 시간적 움직임은 포함하지 않습니다.',
        22:'비하품으로 판독한 주름 조건 이미지 중 기존 MediaPipe 입 추출에 성공한 학습분 81장을 추가했습니다. 피부만 있는 패치를 입처럼 강제로 넣지 않았습니다. 새 데이터에는 확정된 하품 양성이 없으므로, 기존 양성 데이터와 섞더라도 비하품 쪽으로 판정이 기울 수 있습니다. 이 가능성을 외부 재현율로 확인했습니다.',
        23:f"하품 임계값 0.10에서 정확도는 {pct(yb['accuracy'])}에서 {pct(ya['accuracy'])}로 하락했습니다. 오탐은 {yb['fp']}장에서 {ya['fp']}장으로 줄었지만 재현율도 {pct(yb['recall'])}에서 {pct(ya['recall'])}로 떨어졌습니다. 실제 하품 놓침은 {yb['fn_including_unknown']}장에서 {ya['fn_including_unknown']}장으로 늘었습니다. 따라서 이번 하품 추가 학습 후보는 채택하지 않고 기존 모델을 유지합니다. 0.15에서도 성능이 개선되지 않았습니다.",
        24:f'임계값 0.10에서 오답에서 정답으로 바뀐 사진은 {len(yf)}장, 정답에서 오답으로 바뀐 사진은 {len(yr)}장입니다. 대표 사진의 원본, 입 ROI와 전후 확률을 함께 제시했습니다. 잘된 사진만 골라 보여주지 않고 새로 놓친 하품도 함께 공개합니다. 외부 결과를 보고 임계값을 다시 조정하지 않았습니다.',
        25:'이 표는 이번 실험에서 같은 전처리와 임계값으로 재평가한 전후 비교입니다. 눈 CEW는 좋아졌지만 Simuletic은 1장을 더 틀렸고, 직접 촬영 사진은 변화가 없었습니다. 직접 촬영의 과거 ROI 실험과 이번 실험은 이미지 크기, CLAHE와 임계값 조건이 달라 숫자를 섞으면 안 됩니다. 이번 직접 촬영 결과는 동일 조건에서 89.23%에서 89.23%로 유지됐습니다. MRL의 과거 결과를 이번 후보의 재평가 결과로 표시하지 않았습니다.',
        26:'CEW의 눈 ROI 실패 34장과 Roboflow의 입 ROI 실패 49장은 분류층 추가 학습으로 달라지지 않았습니다. 분류할 부위가 없는 경우는 UNKNOWN으로 남기고 전체 정확도 분모에 포함했습니다. 새로운 주름 테스트도 전체 30장 중 ROI 실패 5장을 제외하지 않은 정답률을 보고했습니다. 검출 성공분만의 성능은 별도로 구분해야 합니다.',
        27:'과거 웹캠 A/B 기록에는 수동 눈 감김 2회에 양쪽 방식이 모두 2회 경고한 사례와, 수동 눈 감김 3회에 양쪽 모두 경고하지 않은 사례가 있습니다. 라벨 입력 시점에 오차가 있어 이를 프레임 정확도로 환산하지 않았습니다. 이번 추가 학습 후보의 라이브 웹캠 성능은 측정하지 않았습니다. 이후에는 놓친 이벤트, 시간당 오경고와 경고 지연을 평가해야 합니다.',
        28:'현재 대시보드는 카메라 표시와 최신 프레임의 AI 분석을 분리합니다. 분석 주기 250ms는 설정값이며 전체 웹캠 FPS를 측정한 값은 아닙니다. 오래된 결과와 ROI 실패를 구분해 표시하는 구조가 중요합니다. 분류층의 추가 학습이 실시간 지연을 개선했다는 실측 근거는 이번 실험에 없습니다.',
        29:'5분 안에 장시간 눈 감김 경고가 두 번 쌓이면 화면 속 가상 감속과 정차로 이어집니다. 휴식, 응답 확인, 모의 SOS는 시연용 흐름입니다. 실제 자동차의 브레이크나 창문을 제어하거나 실제 긴급 신고를 보내지 않습니다. 휴식 5분과 무반응 10분도 의료·차량 안전 기준이 아닌 시나리오 값입니다.',
        30:'이번 실험으로 확인한 것은 정해진 데이터와 고정 임계값에서 분류층 추가 학습이 만든 변화입니다. 추가 데이터는 음성 클래스만 있으며 라벨은 AI 보조 육안 판독입니다. CEW와 Roboflow는 과거 설정 선택에 이미 사용되었기 때문에 완전히 새로운 최종 시험셋이 아닙니다. 화장·주름의 인과적 오류 원인, 새 운전자에 대한 성능과 실차 안전성은 입증하지 못했습니다.',
        31:'다음 실험은 새로운 인물의 눈 감김과 하품 양성 사례를 보강해야 합니다. 약한 하품, 측면 하품과 함께 말하기·웃기·입 가리기처럼 혼동하기 쉬운 비하품도 필요합니다. EAR와 MAR, 지속 시간은 보조 지표로 실험할 수 있지만 랜드마크 자체가 틀리면 실패합니다. 이 기능은 이번에 구현하거나 성능 향상을 검증한 기능으로 소개하지 않습니다.',
        32:'이미지별 정답과 분할은 annotations.csv, 학습 전후 확률은 predictions.csv, 집계 지표는 summary.csv에 저장했습니다. 설정, 선택 시점과 모델 해시도 남겼으며 실제 저장한 가중치를 다시 읽어 확률을 확인했습니다. 기존 모델과 앱은 교체하지 않았습니다. 새 PPT의 텍스트와 도형은 편집할 수 있고, 같은 구성의 PDF와 별도 발표 대본도 제공합니다.',
        33:f"눈 CEW 정확도는 {delta(eb['accuracy'],ea['accuracy'])}, 하품 정확도는 {delta(yb['accuracy'],ya['accuracy'])} 변했습니다. 눈의 소폭 개선을 모든 환경의 해결로 일반화하지 않았고, 하품은 악화된 결과를 그대로 기록해 채택하지 않았습니다. 이번 작업의 핵심은 실패 사진을 선별해 실제로 학습하고, 같은 조건에서 개선과 퇴행을 함께 검증한 것입니다. 다음 단계는 양성 데이터 보강과 인물 독립 영상 이벤트 검증입니다."
    }
    for number,note in polished.items():
        d.notes[number-1]['notes']=note
        d.prs.slides[number-1].notes_slide.notes_text_frame.text=note+'\n\n근거: '+d.notes[number-1]['source']
    d.notes[13]['notes']+=f" 일반 데이터만 학습한 대조군은 눈 {pct(ctrl_eye['accuracy'])}, 하품 {pct(ctrl_yawn['accuracy'])}였습니다. 이는 추가 데이터의 영향을 구분하기 위한 결과 설명용 실험이며, 외부 결과에 맞춰 새 모델을 선택하는 데 사용하지 않았습니다."
    d.prs.slides[13].notes_slide.notes_text_frame.text=d.notes[13]['notes']+'\n\n근거: '+d.notes[13]['source']+' · general_only_control_summary.csv'
    d.save()


if __name__=='__main__':build()

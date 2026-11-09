"""Check the current dashboard and selected full-image inference without a camera."""
import ast
import csv
import importlib
import json
from pathlib import Path
from streamlit.testing.v1 import AppTest

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'reports/presentation/final_selection'


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    app=AppTest.from_file(str(ROOT/'11_streamlit_dashboard.py'),default_timeout=60).run()
    assert not app.exception,list(app.exception)
    app.button(key='enter').click().run()
    assert not app.exception,list(app.exception)
    assert app.session_state['yawn_threshold']==.5
    for page in ['설정','사진 분석','안전 대응 시뮬레이션','실시간 모니터']:
        app.radio[0].set_value(page).run()
        assert not app.exception,(page,list(app.exception))
    source=ROOT/'11_streamlit_dashboard.py'
    tree=ast.parse(source.read_text(encoding='utf8'));body=[]
    for node in tree.body:
        if isinstance(node,(ast.Import,ast.ImportFrom,ast.FunctionDef,ast.ClassDef)):
            if isinstance(node,ast.FunctionDef):node.decorator_list=[]
            body.append(node)
        elif isinstance(node,ast.Assign) and all(isinstance(t,ast.Name) and t.id.isupper() for t in node.targets):
            body.append(node)
    ns={'__file__':str(source),'__name__':'aura_inference_check'}
    exec(compile(ast.Module(body=body,type_ignores=[]),str(source),'exec'),ns)
    models=ns['models_cached'](str(ns['EYE_FILE']),str(ns['YAWN_FILE']))
    assert models['yawn'].name=='yolo26s'
    engine=ns['FaceEngine'](models,dict(eye_clahe=True),static=True)
    comparator=importlib.import_module('31_yawn_external_compare')
    with (ROOT/'reports/yawn_external_model_compare/predictions.csv').open(encoding='utf-8-sig') as f:
        rows=[r for r in csv.DictReader(f) if r['model']=='yolo26s' and r['input']=='face']
    picked=[]
    for actual in ('yawn','no_yawn'):
        for positive in (False,True):
            picked.append(next(r for r in rows if r['actual']==actual and (float(r['probability'])>=.5)==positive))
    results=[]
    try:
        for r in picked:
            result=engine.predict(comparator.read_rgb(Path(r['image'])))
            difference=abs(result['yawn']-float(r['probability']))
            assert difference<.0001,difference
            assert result['mouth'].size==(224,224)
            results.append({'image':r['image'],'expected':float(r['probability']),
                            'actual':result['yawn'],'difference':difference})
    finally:engine.close()
    report={'navigation':'pass','camera_activated':False,'yawn_model':'yolo26s',
            'yawn_input':'full_image','threshold':.5,'engine_vs_saved_31':results,
            'camera_performance':'not measured for current dashboard version'}
    (OUT/'app_verification.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf8')
    print('PASS: navigation, selected model/threshold, 4 TP/TN/FP/FN image probabilities match experiment 31.')


if __name__=='__main__':main()

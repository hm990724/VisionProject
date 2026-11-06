"""Evaluate face photos with MediaPipe eye ROIs and an existing YOLO classifier.

python 06_evaluate_eye_roi.py --data data/webcam_test/eyes --weights runs/eyes_yolo26n-cls/weights/best.pt
"""
import argparse
import csv
from collections import Counter
from pathlib import Path

import mediapipe as mp
from PIL import Image
from ultralytics import YOLO
from common import EXT, to_square
from roi_utils import crop_eyes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='data/webcam_test/eyes')
    parser.add_argument('--weights', default='runs/eyes_yolo26n-cls/weights/best.pt')
    parser.add_argument('--output', default='reports/eyes_roi_predictions.csv')
    parser.add_argument('--save-crops', action='store_true', help='Save eye crops for visual review')
    args = parser.parse_args()

    data = Path(args.data)
    weights = Path(args.weights)
    if not data.is_dir() or not weights.is_file():
        raise SystemExit(f'Input not found: data={data}, weights={weights}')

    model = YOLO(str(weights))
    names = model.names
    if set(names.values()) != {'Closed', 'Open'}:
        raise SystemExit(f'Unexpected model classes: {names}')
    closed_id = next(i for i, name in names.items() if name == 'Closed')
    classes = ('Closed', 'Open')
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    crop_dir = output.parent / 'eye_roi_examples'
    if args.save_crops:
        crop_dir.mkdir(parents=True, exist_ok=True)

    confusion = Counter()
    skipped = []
    rows = []

    # Static mode: each photo is independent, so face detection is performed per photo.
    with mp.solutions.face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1,
                                         refine_landmarks=True, min_detection_confidence=0.5) as mesh:
        for label in classes:
            folder = data / label
            if not folder.is_dir():
                raise SystemExit(f'Class folder missing: {folder}')
            files = sorted(p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in EXT)
            print(f'{label}: {len(files)} files')
            for p in files:
                try:
                    with Image.open(p) as im:
                        eye_crops = crop_eyes(im, mesh)
                    if not eye_crops:
                        skipped.append((str(p), 'face/eye not found'))
                        rows.append([str(p), label, 'NO_FACE', '', '', 0, ''])
                        continue
                    # Average probabilities from both eyes, NOT majority vote.
                    probs = []
                    for idx, eye in enumerate(eye_crops):
                        image = to_square(eye, 224)
                        pred = model.predict(image, imgsz=224, verbose=False)[0]
                        probs.append(float(pred.probs.data[closed_id]))
                        if args.save_crops:
                            image.save(crop_dir / f'{label}_{p.stem}_eye{idx+1}.jpg')
                    p_closed = sum(probs) / len(probs)
                    predicted = 'Closed' if p_closed >= 0.5 else 'Open'
                    confusion[(label, predicted)] += 1
                    rows.append([str(p), label, predicted, f'{p_closed:.6f}',
                                 f'{1-p_closed:.6f}', len(probs), ';'.join(f'{v:.4f}' for v in probs)])
                except (OSError, ValueError) as err:
                    skipped.append((str(p), str(err)))
                    rows.append([str(p), label, 'ERROR', '', '', 0, str(err)])

    with output.open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.writer(f)
        writer.writerow(['image', 'actual', 'predicted', 'p_closed', 'p_open', 'eyes_found', 'per_eye_p_closed'])
        writer.writerows(rows)

    n = sum(confusion.values())
    correct = sum(confusion[(c, c)] for c in classes)
    total_files = len(rows)
    print('\n=== Eye ROI evaluation ===')
    print(f'Whole dataset accuracy (detection failures counted as failures): {correct}/{total_files} = {correct / total_files * 100:.2f}%' if total_files else 'No images')
    print(f'Classification accuracy on detected faces: {correct}/{n} = {correct / n * 100:.2f}%' if n else 'No faces detected')
    print(f'Could not evaluate: {len(skipped)} images')
    print('Actual\\Pred      Closed    Open      Recall')
    for c in classes:
        tp = confusion[(c, c)]
        total = sum(confusion[(c, x)] for x in classes)
        recall = f'{tp / total * 100:.1f}%' if total else 'N/A'
        print(f'{c:<17}{confusion[(c,"Closed")]:<10}{confusion[(c,"Open")]:<10}{recall}')
    if skipped:
        print('Skipped examples:', skipped[:5])
    print('Results:', output.resolve())
    if args.save_crops:
        print('Review cropped eyes:', crop_dir.resolve())


if __name__ == '__main__':
    main()

"""MediaPipe FaceMesh -> left/right eye ROI for image and webcam inference.
Install: pip install mediapipe==0.10.21
"""
from PIL import Image, ImageOps
import numpy as np

# Eye outline landmark indices (includes eyelids and corners).
LEFT_EYE = (33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246)
RIGHT_EYE = (362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398)


def crop_eyes(image: Image.Image, face_mesh, horizontal_pad=0.38, vertical_pad=0.95):
    """Return eye PIL crops, or [] if no face/valid eye found.

    Same function can be called on a webcam frame converted to RGB PIL.
    """
    rgb = np.asarray(ImageOps.exif_transpose(image).convert("RGB"))
    h, w = rgb.shape[:2]
    result = face_mesh.process(rgb)
    if not result.multi_face_landmarks:
        return []
    lm = result.multi_face_landmarks[0].landmark
    crops = []
    for indices in (LEFT_EYE, RIGHT_EYE):
        pts = [(lm[i].x * w, lm[i].y * h) for i in indices]
        xs, ys = zip(*pts)
        x_min, x_max = min(xs), max(xs)
        y_min, y_max = min(ys), max(ys)
        ew = x_max - x_min
        eh = y_max - y_min
        if ew < 3 or eh < 1:
            continue
        # Use eye width to determine *vertical* crop too: closed eyelids yield tiny eh.
        cx, cy = (x_min + x_max) / 2, (y_min + y_max) / 2
        half_w = ew * (0.5 + horizontal_pad)
        half_h = max(eh / 2 + ew * vertical_pad * 0.35, ew * 0.38)
        box = (max(0, int(cx-half_w)), max(0, int(cy-half_h)),
               min(w, int(cx+half_w)), min(h, int(cy+half_h)))
        if box[2] - box[0] >= 8 and box[3] - box[1] >= 8:
            crops.append(Image.fromarray(rgb[box[1]:box[3], box[0]:box[2]].copy()))
    return crops

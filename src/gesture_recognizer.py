"""MediaPipe 손 인식: 계산대 위 손 감지 + 제스처로 결제 확인.

스캔앤고에서의 역할
  1) 트레이 위에 손이 있으면 아직 상품을 올리는 중 -> 인식을 미룬다 (가려진 상품 오인식 방지)
  2) 화면을 만지지 않고 제스처로 확인: Thumb_Up = 결제 진행, Open_Palm = 다시 스캔

  python src/gesture_recognizer.py [이미지 ...]
결과: docs/images/gesture.jpg
"""
import argparse
import urllib.request

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from PIL import Image, ImageDraw, ImageOps

from common import MODELS, OUT, SAMPLES, font, open_rgb

MODEL_URL = "https://storage.googleapis.com/mediapipe-models/gesture_recognizer/gesture_recognizer/float16/1/gesture_recognizer.task"
ACTIONS = {"Thumb_Up": "결제 진행", "Open_Palm": "다시 스캔"}
HAND_LINKS = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (5, 9), (9, 10), (10, 11), (11, 12),
              (9, 13), (13, 14), (14, 15), (15, 16), (13, 17), (17, 18), (18, 19), (19, 20), (0, 17)]

_recognizer = None


def recognizer():
    global _recognizer
    if _recognizer is None:
        path = MODELS / "gesture_recognizer.task"
        if not path.exists():
            urllib.request.urlretrieve(MODEL_URL, path)
        opts = vision.GestureRecognizerOptions(
            base_options=python.BaseOptions(model_asset_path=str(path)), num_hands=2)
        _recognizer = vision.GestureRecognizer.create_from_options(opts)
    return _recognizer


def recognize(img, pad=0.25):
    """PIL 이미지(RGB) -> (손 개수, [(제스처, 점수, 21점 랜드마크)])

    손이 화면을 꽉 채우면 손바닥 검출기가 놓치므로(실측) 테두리 여백을 둔다.
    계산대 천장 카메라에서는 손이 원래 작게 찍히므로 영향이 없다.
    """
    import numpy as np
    p = int(max(img.size) * pad)
    padded = ImageOps.expand(img, border=p, fill=(200, 200, 200)) if p else img
    res = recognizer().recognize(mp.Image(image_format=mp.ImageFormat.SRGB, data=np.array(padded)))
    W, H = padded.size
    hands = []
    for g, lm in zip(res.gestures, res.hand_landmarks):
        # 여백을 뺀 원본 좌표(0~1)로 되돌린다
        pts = [((q.x * W - p) / img.width, (q.y * H - p) / img.height) for q in lm]
        hands.append((g[0].category_name, g[0].score, pts))
    return len(res.hand_landmarks), hands


def draw(img, hands):
    img = img.copy()
    d = ImageDraw.Draw(img)
    W, H = img.size
    for name, score, lm in hands:
        pts = [(x * W, y * H) for x, y in lm]
        for a, b in HAND_LINKS:
            d.line([pts[a], pts[b]], fill="#06B6D4", width=max(3, W // 200))
        for x, y in pts:
            r = max(4, W // 150)
            d.ellipse([x - r, y - r, x + r, y + r], fill="#1D4ED8")
        label = f"{name} {score:.2f} → {ACTIONS.get(name, '손 감지: 인식 대기')}"
        f = font(max(18, W // 22))
        d.rectangle([10, 10, 30 + d.textbbox((0, 0), label, font=f)[2], 30 + f.size], fill="#0F172A")
        d.text((20, 18), label, fill="white", font=f)
    return img


def main():
    p = argparse.ArgumentParser()
    p.add_argument("images", nargs="*", default=[str(SAMPLES / "hand_thumb_up.jpg"), str(SAMPLES / "hand_open_palm.jpg")])
    args = p.parse_args()

    panels = []
    for path in args.images:
        img = open_rgb(path)
        n, hands = recognize(img)
        for name, score, _ in hands:
            print(f"{path}: 손 {n}개, {name} {score:.2f} -> {ACTIONS.get(name, '손 감지: 인식 대기')}")
        if not hands:
            print(f"{path}: 손 없음")
        panels.append(draw(img, hands))

    h = 480
    panels = [p.resize((int(p.width * h / p.height), h)) for p in panels]
    sheet = Image.new("RGB", (sum(p.width for p in panels) + 20 * (len(panels) - 1), h), "white")
    x = 0
    for p in panels:
        sheet.paste(p, (x, 0))
        x += p.width + 20
    sheet.save(OUT / "gesture.jpg", quality=88)


if __name__ == "__main__":
    main()

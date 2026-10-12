"""포장지 글자 OCR + 카탈로그 키워드 대조.

스캔앤고에서 OCR은 '교차 확인'용이다. SigLIP 2 유사도만으로 애매한 상품
(예: 바나나우유 vs 딸기우유처럼 용기가 같은 상품)을 포장지 글자로 가려낸다.

엔진 선택
  rapid (기본) : RapidOCR + PP-OCRv5 한국어 모델. 디자인된 한글 로고를 더 잘 읽었다.
  easy         : EasyOCR (보고서 1.5절). 같은 사진에서 '바나나맛우유' 로고를 읽지 못했다.

  python src/ocr_label.py [--engine rapid|easy] [이미지 ...]
결과: docs/images/ocr_label.jpg
"""
import argparse
import re

import numpy as np
from PIL import Image

from common import OUT, SAMPLES, draw_boxes, load_catalog, open_rgb

_engines = {}


def _rapid():
    from rapidocr import LangRec, ModelType, OCRVersion, RapidOCR
    eng = RapidOCR(params={"Rec.lang_type": LangRec.KOREAN, "Rec.ocr_version": OCRVersion.PPOCRV5,
                           "Rec.model_type": ModelType.MOBILE, "Global.text_score": 0.3})

    def run(arr):
        r = eng(arr)
        if r.txts is None:
            return []
        return [(b.tolist(), t, float(s)) for b, t, s in zip(r.boxes, r.txts, r.scores)]
    return run


def _easy():
    import easyocr
    reader = easyocr.Reader(["ko", "en"], gpu=False, verbose=False)
    return lambda arr: [(b, t, float(c)) for b, t, c in reader.readtext(arr)]


def read_text(img, engine="rapid", min_conf=0.30, rotations=(0, 90)):
    """PIL 이미지 -> [(text, conf, box 또는 None)]. 작은 사진은 키우고, 세로 글자를 위해 90도 회전본도 읽는다."""
    if engine not in _engines:
        _engines[engine] = {"rapid": _rapid, "easy": _easy}[engine]()
    s = 900 / max(img.size)
    big = img.resize((int(img.width * s), int(img.height * s)), Image.LANCZOS)
    out = []
    for rot in rotations:
        for b, t, c in _engines[engine](np.array(big.rotate(rot, expand=True))):
            if c >= min_conf:
                out.append((t, c, [(x / s, y / s) for x, y in b] if rot == 0 else None))
    return out


def normalize(t):
    return t.lower().replace(" ", "")


def check(texts, catalog, sku=None):
    """읽은 글자로 판정.
    반환: ('match', sku) 키워드 일치 / ('conflict', None) 한글 상품명이 읽혔는데 후보 키워드 없음 / ('none', None)
    """
    joined = normalize(" ".join(t[0] for t in texts))
    cands = [sku] if sku else list(catalog)
    for s in cands:
        if any(normalize(k) in joined for k in catalog[s]["keywords"]):
            return "match", s
    if re.search(r"[가-힣]{2,}", joined):
        return "conflict", None
    return "none", None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("images", nargs="*", default=[str(SAMPLES / "items" / f"{n}.jpg") for n in
                                                  ("banana_milk", "strawberry_milk", "choco_pie_cn")])
    p.add_argument("--engine", default="rapid", choices=["rapid", "easy"])
    args = p.parse_args()
    catalog = load_catalog()

    panels = []
    for path in args.images:
        img = open_rgb(path)
        texts = read_text(img, args.engine)
        verdict, sku = check(texts, catalog)
        print(f"{path}: {[t for t, _, _ in texts]} -> {verdict} {sku or ''}")
        dets = [(min(x for x, _ in b), min(y for _, y in b), max(x for x, _ in b), max(y for _, y in b),
                 f"{t} {c:.2f}", "#DC2626") for t, c, b in texts if b]
        panels.append(draw_boxes(img, dets, size=max(12, img.height // 22)))

    h = 420
    panels = [p.resize((int(p.width * h / p.height), h)) for p in panels]
    sheet = Image.new("RGB", (sum(p.width for p in panels) + 20 * (len(panels) - 1), h), "white")
    x = 0
    for p in panels:
        sheet.paste(p, (x, 0))
        x += p.width + 20
    sheet.save(OUT / "ocr_label.jpg", quality=88)


if __name__ == "__main__":
    main()

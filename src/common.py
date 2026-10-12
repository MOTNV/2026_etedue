"""스캔앤고 실습 코드 공통 설정: 경로, 상품 카탈로그, 한글 박스 그리기."""
import csv
import os
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
GALLERY = DATA / "gallery"          # 상품별 등록 사진: data/gallery/<SKU>/*.jpg
SAMPLES = DATA / "samples"          # 계산대 테스트 이미지
INDEX = DATA / "gallery_index.npz"  # register_product.py가 만드는 임베딩 갤러리
OUT = ROOT / "docs" / "images"      # 홈페이지에 쓰는 결과 이미지
MODELS = ROOT / "models"            # 내려받은 가중치 (git 제외)

for d in (OUT, MODELS):
    d.mkdir(parents=True, exist_ok=True)
CWD = Path.cwd()  # 사용자가 넘긴 상대 경로는 실행 위치 기준으로 푼다
os.chdir(MODELS)  # ultralytics가 가중치를 현재 폴더에 받으므로 models/로 모은다

_FONTS = ["C:/Windows/Fonts/malgun.ttf", "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
          "/System/Library/Fonts/AppleSDGothicNeo.ttc"]


def font(size=20):
    for f in _FONTS:
        if Path(f).exists():
            return ImageFont.truetype(f, size)
    return ImageFont.load_default()


def load_catalog():
    """data/catalog.csv -> {sku: {name, price, keywords}}"""
    with open(DATA / "catalog.csv", encoding="utf-8") as f:
        return {r["sku"]: {"name": r["name"], "price": int(r["price"]),
                           "keywords": [k for k in r["keywords"].split("|") if k]}
                for r in csv.DictReader(f)}


def draw_boxes(img, dets, size=22):
    """dets: [(x1, y1, x2, y2, label, color)] -> 한글 라벨이 들어간 박스 이미지"""
    img = img.convert("RGB").copy()
    d = ImageDraw.Draw(img)
    f = font(size)
    for x1, y1, x2, y2, label, color in dets:
        d.rectangle([x1, y1, x2, y2], outline=color, width=4)
        tw = d.textbbox((0, 0), label, font=f)[2]
        ty = max(y1 - size - 10, 0)
        d.rectangle([x1, ty, x1 + tw + 12, ty + size + 8], fill=color)
        d.text((x1 + 6, ty + 3), label, fill="white", font=f)
    return img


def resolve(path):
    path = Path(path)
    return path if path.is_absolute() else CWD / path


def open_rgb(path):
    return Image.open(resolve(path)).convert("RGB")

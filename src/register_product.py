"""상품 등록 = 갤러리 임베딩 만들기. 재학습 없이 신상품을 추가하는 핵심 단계.

data/gallery/<SKU>/*.jpg 사진을 SigLIP 2로 임베딩해 data/gallery_index.npz에 저장한다.
매주 신상품 50종은 사진 몇 장 + catalog.csv 한 줄만 추가하고 이 스크립트를 다시 돌리면 된다.
결과 파일은 수백 KB라서 밤사이 3,000개 매장 배포 부담이 모델 전체(250MB)보다 훨씬 작다.

  python src/register_product.py                               # 갤러리 전체 다시 만들기
  python src/register_product.py --add SKU "이름" 가격 키워드1 키워드2 --images a.jpg b.jpg
"""
import argparse
import csv
import shutil

import numpy as np

from common import DATA, GALLERY, INDEX, load_catalog, open_rgb
from siglip_embed import DEFAULT_CKPT, SigLIP


def add_product(sku, name, price, keywords, images):
    (GALLERY / sku).mkdir(parents=True, exist_ok=True)
    for img in images:
        shutil.copy(img, GALLERY / sku)
    if sku not in load_catalog():
        with open(DATA / "catalog.csv", "a", encoding="utf-8", newline="") as f:
            csv.writer(f).writerow([sku, name, price, "|".join(keywords)])
    print(f"등록: {sku} ({name}, {price}원) 사진 {len(images)}장")


def build_index(ckpt=DEFAULT_CKPT):
    catalog = load_catalog()
    enc = SigLIP(ckpt)
    skus, embs = [], []
    for sku in catalog:
        files = sorted(p for p in (GALLERY / sku).glob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
        if not files:
            print(f"  경고: {sku} 사진 없음")
            continue
        e = enc.images([open_rgb(f) for f in files])
        skus += [sku] * len(files)
        embs.append(e)
        print(f"  {sku:<16} 사진 {len(files)}장")
    embs = np.concatenate(embs).astype(np.float16)
    np.savez(INDEX, skus=np.array(skus), embs=embs, ckpt=np.array(ckpt))
    print(f"갤러리 저장: {INDEX.name} ({len(set(skus))}종, {INDEX.stat().st_size / 1024:.0f} KB)")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--add", nargs="+", metavar="SKU NAME PRICE [KEYWORDS...]")
    p.add_argument("--images", nargs="+", default=[])
    p.add_argument("--ckpt", default=DEFAULT_CKPT)
    args = p.parse_args()
    if args.add:
        sku, name, price, *kw = args.add
        add_product(sku, name, int(price), kw, args.images)
    build_index(args.ckpt)


if __name__ == "__main__":
    main()

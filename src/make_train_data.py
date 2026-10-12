"""YOLO26 '상품' 1클래스 검출기용 학습 데이터 만들기 (자동 라벨링 + 합성).

보고서 원칙: 제로샷(YOLOE)으로 검증 -> 부족하면 YOLO26 커스텀 학습으로 승격.
라벨링을 사람이 하지 않고, YOLOE 분할 마스크로 상품만 오려 낸 뒤(자동 라벨링)
여러 배경 위에 무작위로 붙여 박스 라벨이 정확한 학습 이미지를 만든다.

  1단계 python src/make_train_data.py cutouts   -> data/train/cutouts/*.png (오려 낸 상품) + 확인용 시트
  2단계 python src/make_train_data.py synth     -> data/train/{images,labels}/{train,val}, data/train/product.yaml

학습·평가 분리 원칙
  - 홈페이지 테스트 사진(계산대 합성, 바나나우유 banana_a, 사이다 cider_b, 중국판 초코파이, 삼각김밥)은 학습에 쓰지 않는다.
  - 삼각김밥은 비슷한 종류도 학습에 넣지 않아 '처음 보는 상품'에 대한 일반화를 본다.
  - 'te_' 로 시작하는 사진은 평가 전용이다 (eval_detectors.py).
"""
import json
import random
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter

from commons import RAW, download, write_credits
from common import DATA, ROOT, open_rgb

TRAIN = DATA / "train"
CUT = TRAIN / "cutouts"

# 상품(양성): 저장이름 -> Commons 제목. gallery_* 는 prepare_samples.py가 받은 등록 사진 원본.
POSITIVE = {
    "tr_cider_zero": "Chilsung cider zero.jpg", "tr_coke_life": "Bottle of Coca-Cola Life.png",
    "tr_grape_ade": "Lotte Squeeze Grappe Ade.jpg", "tr_milkis_1": "Korean drink-Milkis-01.jpg",
    "tr_milkis_bottles": "Milkis bottles.jpg", "tr_milkis_can": "Milkis can.jpg",
    "tr_pocari_jp": "POCARI SWEAT 500ml PET in Japan.jpg", "tr_pocari_pet": "POCARI SWEET, 500ml Plastic bottle type.jpg",
    "tr_yakult_pack": "A pack of flavoured Yakult sold in Singapore.jpg", "tr_yakult_drink": "Yakult drink.jpg",
    "tr_yakult_lite": "Yakult lite.jpg", "tr_yakult_flavours": "Yakult variant flavours.jpg",
    "tr_pringles": "Pringles-165g-to-134g.jpg", "tr_xylitol": "Lotte xylitol gum lime mint family bottle.jpg",
    "tr_cup_tofu": "Fried Tofu Cup Ramyeon.jpg",
}
GALLERY_RAW = ["banana_b", "cider_a", "chocopie_kr", "cupramen"]
# 상품 아님(음성): 조리한 음식·과자 내용물. 라벨 없이 배경·방해물로만 쓴다.
NEGATIVE = {
    "tr_shin_1": "Nongshim Shin Ramyun 20210529 001.jpg", "tr_shin_2": "Nongshim Shin Ramyun 20210529 002.jpg",
    "tr_shin_black": "Nongshim Shin Ramyun black 20201123 002.jpg", "tr_shin_fried_1": "Nongshim Shin Ramyun fried noodles 20210730 001.jpg",
    "tr_shin_fried_2": "Nongshim Shin Ramyun fried noodles 20210730 002.jpg", "tr_shin_light": "Nongshim Shin Ramyun Light 200210726 002.jpg",
    "tr_jjondeugi": "Jjondeugi 01.jpg", "tr_kkoedori": "Kkoedori.jpg", "tr_ggul_1": "꿀꽈배기 봄한정 패키지 1.jpg",
    "tr_ggul_3": "꿀꽈배기 봄한정 패키지 3.jpg", "tr_pringles_paprika": "Pringles Paprika reflection.JPG",
    "tr_bananakick_1": "Banana Kick 20201201 001.jpg", "tr_bananakick_3": "Banana Kick 20201201 003.jpg",
}
# 평가 전용 (eval_detectors.py에서 정답 박스와 함께 사용)
TEST = {
    "te_yakult_china": "Yakult china.jpg", "te_pocari_cn": "POCARI SWEAT 500ml PET in mainland China 20100317.jpg",
    "te_grape_ade_2": "Lotte Squeeze Grappe Ade 2.jpg", "te_shin_3": "Nongshim Shin Ramyun 20210529 003.jpg",
    "te_ggul_4": "꿀꽈배기 봄한정 패키지 4.jpg", "te_jjondeugi_2": "Jjondeugi 02.jpg",
}
PROMPTS = ["bottle", "can", "snack package", "cup noodle", "plastic container", "drink"]
# 시트로 확인한 뒤 빼기로 한 조각 (손이 같이 잘렸거나 상품이 아닌 부분)
EXCLUDE = set(json.loads((TRAIN / "exclude.json").read_text())) if (TRAIN / "exclude.json").exists() else set()


def cutouts():
    from ultralytics import YOLOE
    meta = download({**POSITIVE, **NEGATIVE, **TEST})
    CUT.mkdir(parents=True, exist_ok=True)
    model = YOLOE("yoloe-26s-seg.pt")
    model.set_classes(PROMPTS)
    names = []
    for key in list(POSITIVE) + GALLERY_RAW:
        img = open_rgb(RAW / f"{key}.jpg")
        r = model.predict(np.array(img)[:, :, ::-1], conf=0.25, agnostic_nms=True, verbose=False)[0]
        polys, confs = (r.masks.xy, r.boxes.conf.tolist()) if r.masks is not None else ([], [])
        if not polys:  # 기준을 넘는 검출이 없으면 기준을 낮춰 가장 높은 1개만 쓴다
            r = model.predict(np.array(img)[:, :, ::-1], conf=0.05, agnostic_nms=True, verbose=False)[0]
            if r.masks is None:
                print(f"  {key}: 검출 없음")
                continue
            polys, confs = r.masks.xy[:1], r.boxes.conf.tolist()[:1]
            print(f"  {key}: 낮은 기준으로 1개 사용 ({confs[0]:.2f})")
        for i, (poly, conf) in enumerate(zip(polys, confs)):
            if len(poly) < 3:
                continue
            mask = Image.new("L", img.size, 0)
            ImageDraw.Draw(mask).polygon([tuple(p) for p in poly], fill=255)
            rgba = img.copy()
            rgba.putalpha(mask)
            box = mask.getbbox()
            if not box or (box[2] - box[0]) * (box[3] - box[1]) < 0.02 * img.width * img.height:
                continue
            name = f"{key}_{i}"
            rgba.crop(box).save(CUT / f"{name}.png")
            names.append((name, conf))
    print(f"오려 낸 상품 {len(names)}개 -> {CUT}")

    # 확인용 시트: 마스크가 잘못된 조각은 exclude.json에 이름을 적어 뺀다
    S = 200
    sheet = Image.new("RGB", (8 * S, ((len(names) + 7) // 8) * (S + 18)), (235, 235, 235))
    d = ImageDraw.Draw(sheet)
    for k, (name, conf) in enumerate(names):
        c = Image.open(CUT / f"{name}.png")
        c.thumbnail((S - 8, S - 8))
        x, y = (k % 8) * S, (k // 8) * (S + 18)
        sheet.paste(c, (x + 4, y + 16), c)
        d.text((x + 4, y + 2), f"{name} {conf:.2f}", fill="black")
    sheet.save(TRAIN / "cutouts_sheet.jpg", quality=85)
    write_credits(meta, ROOT / "docs" / "CREDITS.md")


def background(rnd, size, negs):
    W, H = size
    if negs and rnd.random() < 0.35:  # 실제 사진 질감 (음식·접시·책상)
        n = rnd.choice(negs)
        s = max(W / n.width, H / n.height) * rnd.uniform(1.0, 1.6)
        n = n.resize((int(n.width * s) + 1, int(n.height * s) + 1))
        x, y = rnd.randint(0, n.width - W), rnd.randint(0, n.height - H)
        return n.crop((x, y, x + W, y + H))
    c1 = np.array([rnd.randint(40, 235) for _ in range(3)], dtype=np.float32)
    c2 = np.clip(c1 + rnd.uniform(-50, 50), 0, 255)
    t = np.linspace(0, 1, H)[:, None, None]
    arr = c1 * (1 - t) + c2 * t + np.random.default_rng(rnd.randint(0, 10**6)).normal(0, rnd.uniform(2, 12), (H, W, 3))
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


def overlap_ok(box, boxes, limit=0.2):
    for b in boxes:
        iw = max(0, min(box[2], b[2]) - max(box[0], b[0]))
        ih = max(0, min(box[3], b[3]) - max(box[1], b[1]))
        a = min((box[2] - box[0]) * (box[3] - box[1]), (b[2] - b[0]) * (b[3] - b[1]))
        if iw * ih > limit * a:
            return False
    return True


def synth(n_train=600, n_val=100, size=(640, 640), seed=0):
    rnd = random.Random(seed)
    np.random.seed(seed)
    cuts = [Image.open(p).convert("RGBA") for p in sorted(CUT.glob("*.png")) if p.stem not in EXCLUDE]
    negs = [open_rgb(RAW / f"{k}.jpg") for k in NEGATIVE]
    print(f"상품 조각 {len(cuts)}개, 음성 사진 {len(negs)}장")
    W, H = size
    for split, n in (("train", n_train), ("val", n_val)):
        (TRAIN / "images" / split).mkdir(parents=True, exist_ok=True)
        (TRAIN / "labels" / split).mkdir(parents=True, exist_ok=True)
        for i in range(n):
            img = background(rnd, size, negs).convert("RGB")
            boxes = []
            # 방해물: 음성 사진의 사각형 조각 (사각형 = 상품 이라는 지름길 학습 방지)
            for _ in range(rnd.randint(0, 2)):
                ng = rnd.choice(negs)
                w = int(W * rnd.uniform(0.2, 0.45)); h = int(w * rnd.uniform(0.6, 1.4))
                x0, y0 = rnd.randint(0, ng.width - min(w, ng.width)), rnd.randint(0, ng.height - min(h, ng.height))
                patch = ng.crop((x0, y0, x0 + w, y0 + h))
                img.paste(patch, (rnd.randint(0, W - w), rnd.randint(0, H - h)))
            # 상품: 2~6개
            for _ in range(rnd.randint(2, 6)):
                c = rnd.choice(cuts)
                hgt = H * rnd.uniform(0.18, 0.5)
                c = c.resize((max(8, int(c.width * hgt / c.height)), int(hgt)))
                if c.width > W * 0.6:
                    c = c.resize((int(W * 0.6), max(8, int(c.height * W * 0.6 / c.width))))
                c = c.rotate(rnd.uniform(-35, 35), expand=True, resample=Image.BICUBIC)
                rgb = ImageEnhance.Brightness(c.convert("RGB")).enhance(rnd.uniform(0.7, 1.3))
                c = Image.merge("RGBA", (*rgb.split(), c.split()[3]))
                for _ in range(20):
                    x, y = rnd.randint(0, max(0, W - c.width)), rnd.randint(0, max(0, H - c.height))
                    bb = c.split()[3].getbbox()
                    box = (x + bb[0], y + bb[1], x + bb[2], y + bb[3])
                    if overlap_ok(box, boxes):
                        img.paste(c, (x, y), c)
                        boxes.append(box)
                        break
            if rnd.random() < 0.3:
                img = img.filter(ImageFilter.GaussianBlur(rnd.uniform(0.3, 1.5)))
            stem = f"{split}_{i:04d}"
            img.save(TRAIN / "images" / split / f"{stem}.jpg", quality=90)
            with open(TRAIN / "labels" / split / f"{stem}.txt", "w") as f:
                for x1, y1, x2, y2 in boxes:
                    f.write(f"0 {(x1 + x2) / 2 / W:.6f} {(y1 + y2) / 2 / H:.6f} {(x2 - x1) / W:.6f} {(y2 - y1) / H:.6f}\n")
    (TRAIN / "product.yaml").write_text(
        f"path: {TRAIN.as_posix()}\ntrain: images/train\nval: images/val\nnames:\n  0: product\n", encoding="utf-8")
    print(f"합성 완료: train {n_train}, val {n_val} -> {TRAIN}")


if __name__ == "__main__":
    {"cutouts": cutouts, "synth": synth}[sys.argv[1] if len(sys.argv) > 1 else "cutouts"]()

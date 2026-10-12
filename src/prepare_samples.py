"""공개 사진(Wikimedia Commons)으로 실습 데이터를 만든다.

  data/gallery/<SKU>/   등록용 상품 사진 (본사가 신상품 등록할 때 찍는 사진 역할)
  data/samples/items/   계산대에 올라갈 상품 사진 (등록 사진과 '다른' 사진)
  data/samples/tray_demo.jpg  위 상품들을 한 장에 합성한 계산대 이미지
  data/samples/hand_*.jpg     제스처 인식용 손 사진
  docs/CREDITS.md       사진 출처·라이선스

실제 매장 사진이 생기면 이 단계는 건너뛰고 같은 폴더에 직접 넣으면 된다.
"""
import json
import random
import re
import time
import urllib.parse
import urllib.request

from PIL import Image, ImageFilter

from common import DATA, GALLERY, ROOT, SAMPLES

RAW = DATA / "raw"
UA = {"User-Agent": "scanandgo-edu/0.1 (student project)"}

# 파일명 -> Commons 원본 제목
SOURCES = {
    "banana_a": "Binggrae Banana Flavored Milk 03.jpg",
    "banana_b": "Binggrae Banana Flavored Milk 04 (15723530512).jpg",
    "cider_a": "Korea Chilsung Cider 01 (15891088219).jpg",
    "cider_b": "Korea Chilsung Cider 02 (15454873444).jpg",
    "chocopie_kr": "Orion-Choco-Pie-1.jpg",
    "chocopie_cn": "ORION CHOCO PIE CHINA VERSION (3).jpg",
    "cupramen": "Korea Cup noodle.jpg",
    "kimbap": "Samgak kimbap.jpg",
    "thumb_up": "Thumbs up 01.jpg",
    "open_palm": "Open Palm of the Left Hand, Fingers.jpg",
}

# 등록 사진 (SKU, 원본, 자르기 영역 @ 가로 960px 기준)
GALLERY_CROPS = [
    ("BANANA_MILK", "banana_b", (465, 55, 820, 550)),
    ("CHILSUNG_CIDER", "cider_a", (235, 55, 385, 525)),
    ("CHOCO_PIE", "chocopie_kr", (60, 70, 960, 680)),
    ("CUP_RAMEN", "cupramen", None),
]

# 계산대 위 상품 (이름, 원본, 자르기 영역, 정답 SKU) - 등록 사진과 다른 사진을 쓴다
TRAY_ITEMS = [
    ("banana_milk", "banana_a", (25, 150, 255, 465), "BANANA_MILK"),
    ("strawberry_milk", "banana_a", (262, 150, 482, 465), None),   # 미등록: 바나나우유와 비슷한 용기
    ("chilsung_cider", "cider_b", (175, 265, 268, 530), "CHILSUNG_CIDER"),
    ("choco_pie_cn", "chocopie_cn", (20, 0, 930, 645), "CHOCO_PIE"),  # 중국판 포장
    ("triangle_kimbap", "kimbap", (80, 180, 960, 950), None),      # 미등록 신상품
]


def fetch(url):
    for i in range(5):
        try:
            return urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=60).read()
        except Exception:
            time.sleep(3 * (i + 1))
    raise RuntimeError(f"다운로드 실패: {url}")


def download():
    RAW.mkdir(parents=True, exist_ok=True)
    meta_file = RAW / "credits.json"
    meta = json.loads(meta_file.read_text(encoding="utf-8")) if meta_file.exists() else {}
    for key, title in SOURCES.items():
        if (RAW / f"{key}.jpg").exists() and key in meta:
            continue
        q = urllib.parse.quote("File:" + title)
        api = ("https://commons.wikimedia.org/w/api.php?action=query&prop=imageinfo&format=json"
               f"&iiprop=url|extmetadata&iiurlwidth=960&titles={q}")
        info = list(json.loads(fetch(api))["query"]["pages"].values())[0]["imageinfo"][0]
        m = info["extmetadata"]
        (RAW / f"{key}.jpg").write_bytes(fetch(info["thumburl"]))
        meta[key] = {
            "title": title,
            "author": re.sub("<[^>]+>", "", m.get("Artist", {}).get("value", "?")).strip(),
            "license": m.get("LicenseShortName", {}).get("value", "?"),
            "url": info["descriptionurl"],
        }
        print("다운로드:", title)
        time.sleep(2)
    meta_file.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return meta


def crop(key, box):
    im = Image.open(RAW / f"{key}.jpg").convert("RGB")
    return im.crop(box) if box else im


def make_tray(items, size=(1600, 1000), seed=7):
    """상품 사진들을 계산대 트레이 위에 올린 것처럼 한 장으로 합성한다."""
    rnd = random.Random(seed)
    tray = Image.new("RGB", size, (206, 210, 216))
    noise = Image.effect_noise(size, 18).convert("RGB")
    tray = Image.blend(tray, noise, 0.08)
    slots = [(60, 80), (380, 80), (700, 60), (900, 520), (100, 560)]
    heights = [380, 380, 420, 330, 380]
    for (x, y), h, (name, img) in zip(slots, heights, items):
        img = img.resize((int(img.width * h / img.height), h))
        if img.width > 620:
            img = img.resize((620, int(img.height * 620 / img.width)))
        shadow = Image.new("RGBA", (img.width + 30, img.height + 30), (0, 0, 0, 0))
        shadow.paste((0, 0, 0, 90), (15, 15, img.width + 15, img.height + 15))
        shadow = shadow.filter(ImageFilter.GaussianBlur(10))
        tray.paste(shadow, (x - 5, y + 5), shadow)
        tray.paste(img.rotate(rnd.uniform(-3, 3), expand=False, fillcolor=(206, 210, 216)), (x, y))
    return tray


def main():
    meta = download()

    for sku, key, box in GALLERY_CROPS:
        (GALLERY / sku).mkdir(parents=True, exist_ok=True)
        crop(key, box).save(GALLERY / sku / f"{key}.jpg", quality=92)

    items_dir = SAMPLES / "items"
    items_dir.mkdir(parents=True, exist_ok=True)
    items, truth = [], {}
    for name, key, box, sku in TRAY_ITEMS:
        img = crop(key, box)
        img.save(items_dir / f"{name}.jpg", quality=92)
        items.append((name, img))
        truth[name] = sku
    (items_dir / "truth.json").write_text(json.dumps(truth, indent=2), encoding="utf-8")
    make_tray(items).save(SAMPLES / "tray_demo.jpg", quality=92)

    for key in ("thumb_up", "open_palm"):
        Image.open(RAW / f"{key}.jpg").convert("RGB").save(SAMPLES / f"hand_{key}.jpg", quality=92)

    lines = ["# 사진 출처", "", "실습용 사진은 모두 Wikimedia Commons의 자유 라이선스 이미지를 잘라 쓰거나 합성했습니다.",
             "`data/samples/tray_demo.jpg`와 결과 이미지들은 아래 사진을 합성·가공한 2차 저작물이며 원 라이선스를 따릅니다.", "",
             "| 파일 | 원본 | 저작자 | 라이선스 |", "|---|---|---|---|"]
    for key, m in meta.items():
        lines.append(f"| `{key}.jpg` | [{m['title']}]({m['url']}) | {m['author']} | {m['license']} |")
    (ROOT / "docs" / "CREDITS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("완료: data/gallery, data/samples, docs/CREDITS.md")


if __name__ == "__main__":
    main()

"""스캔앤고 전체 파이프라인 (A안): 손 확인 -> 상품 위치 -> 상품 판별 -> 글자 교차 확인 -> 영수증.

  1. MediaPipe   트레이 위에 손이 있으면 아직 올리는 중 -> 인식을 미룬다
  2. YOLOE       텍스트 프롬프트로 '상품처럼 생긴 것'의 위치만 찾는다 (재학습 0)
  3. SigLIP 2    잘라낸 상품을 등록 갤러리와 비교해 어떤 상품인지 판별한다
  4. OCR         유사도가 애매하면 포장지 글자로 교차 확인한다 (예: 바나나우유 vs 딸기우유)
  5. 미등록·불일치 상품은 바코드 스캔으로 넘긴다

  python src/scan_and_go.py [--source data/samples/tray_demo.jpg] [--detector yoloe|product]
결과: docs/images/scan_result.jpg/.json (YOLOE), scan_result_product.jpg/.json (직접 학습한 YOLO26n)
"""
import argparse
import json
import time

import numpy as np

from common import INDEX, MODELS, OUT, SAMPLES, draw_boxes, load_catalog, open_rgb
from detect_text_yoloe import PRODUCT_PROMPTS, load_yoloe
from siglip_embed import SigLIP, load_index

# 판정 기준. 샘플 5개로 정한 값이라 PoC에서 실제 매장 사진으로 다시 맞춰야 한다.
SURE = 0.95        # 이 이상 + 차이가 크면 OCR 없이 확정
SURE_MARGIN = 0.15
MIN_SIM = 0.75     # 이 미만이면 미등록 상품으로 본다
MIN_MARGIN = 0.05  # 1, 2위 상품 유사도 차이가 이보다 작으면 헷갈리는 것으로 본다
DET_KEEP = {"yoloe": 0.15,    # YOLOE 상품 후보 채택 기준 (놓치는 것보다 후보가 많은 편이 안전)
            "product": 0.25}  # 직접 학습한 YOLO26n-product (train_yolo26.py)

COLORS = {"ok": "#1D4ED8", "barcode": "#D97706"}


class Timer:
    def __init__(self):
        self.ms = {}

    def __call__(self, name):
        timer = self

        class _T:
            def __enter__(self):
                self.t = time.perf_counter()

            def __exit__(self, *a):
                timer.ms[name] = timer.ms.get(name, 0) + (time.perf_counter() - self.t) * 1000
        return _T()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source", default=str(SAMPLES / "tray_demo.jpg"))
    p.add_argument("--no-hands", action="store_true", help="손 확인 단계 생략")
    p.add_argument("--detector", default="yoloe", choices=["yoloe", "product"],
                   help="상품 위치 검출기: yoloe(제로샷) 또는 product(직접 학습한 YOLO26n)")
    args = p.parse_args()

    catalog = load_catalog()
    skus, gal, ckpt = load_index(INDEX)
    gal = gal.astype(np.float32)
    cols = list(dict.fromkeys(skus))

    # 모델 로드 (매장 AI 박스에서는 부팅 시 한 번)
    if args.detector == "yoloe":
        det = load_yoloe(PRODUCT_PROMPTS)
        det_kw = {"agnostic_nms": True}
    else:
        from ultralytics import YOLO
        det = YOLO(str(MODELS / "yolo26n-product.pt"))
        det_kw = {}
    enc = SigLIP(ckpt)
    from ocr_label import check, read_text
    if not args.no_hands:
        from gesture_recognizer import recognize

    img = open_rgb(args.source)
    bgr = np.array(img)[:, :, ::-1]  # ultralytics는 numpy 입력을 BGR로 받는다
    # 워밍업 (첫 호출은 느리므로 측정에서 뺀다)
    det.predict(bgr, verbose=False)
    enc.images([img])
    read_text(img.crop((0, 0, 64, 64)))
    T = Timer()

    if not args.no_hands:
        with T("1_hands"):
            n_hands, _ = recognize(img)
        if n_hands:
            print(f"손 {n_hands}개 감지 -> 상품을 다 올릴 때까지 대기")
            return

    with T(f"2_{args.detector}"):
        r = det.predict(bgr, conf=DET_KEEP[args.detector], verbose=False, **det_kw)[0]
    boxes = r.boxes.xyxy.tolist()

    crops = [img.crop(tuple(map(int, b))) for b in boxes]
    with T("3_siglip"):
        q = enc.images(crops) if crops else np.zeros((0, gal.shape[1]))
        sim_all = q @ gal.T
        sim = np.stack([sim_all[:, [i for i, s in enumerate(skus) if s == c]].max(1) for c in cols], 1) \
            if crops else np.zeros((0, len(cols)))

    lines, dets = [], []
    for box, crop, row in zip(boxes, crops, sim):
        order = row.argsort()[::-1]
        sku, s1 = cols[order[0]], float(row[order[0]])
        margin = s1 - float(row[order[1]]) if len(order) > 1 else 1.0
        ocr_texts = None

        if s1 >= SURE and margin >= SURE_MARGIN:
            status, how = "ok", "이미지"
        elif s1 < MIN_SIM or margin < MIN_MARGIN:
            status, how = "barcode", "미등록 상품"
        else:
            # 정방향 먼저 읽고, 판단이 안 서면 세로 글자용 90도 회전본을 추가로 읽는다
            texts = []
            for rot in (0, 90):
                with T("4_ocr"):
                    texts += read_text(crop, rotations=(rot,))
                verdict, _ = check(texts, catalog, sku)
                if verdict != "none":
                    break
            ocr_texts = [t for t, _, _ in texts]
            if verdict == "match":
                status, how = "ok", "이미지+글자"
            elif verdict == "conflict":
                status, how = "barcode", "글자 불일치"
            else:
                status, how = "ok", "이미지 (글자 확인 불가)"

        name = catalog[sku]["name"] if status == "ok" else None
        lines.append({"box": [round(v) for v in box], "sku": sku if status == "ok" else None, "name": name,
                      "price": catalog[sku]["price"] if status == "ok" else None, "status": status, "how": how,
                      "top1": sku, "sim": round(s1, 3), "margin": round(margin, 3), "ocr": ocr_texts})
        label = f"{name} {s1:.2f}" if status == "ok" else f"바코드 필요 ({how})"
        dets.append((*box, label, COLORS[status]))

    total_ms = sum(T.ms.values())
    total = sum(l["price"] for l in lines if l["status"] == "ok")
    print("\n===== 영수증 =====")
    for l in sorted(lines, key=lambda l: l["status"]):
        if l["status"] == "ok":
            print(f"  {l['name']:<16} {l['price']:>6,}원   [{l['how']}, 유사도 {l['sim']:.2f}]")
        else:
            print(f"  {'(바코드 스캔 필요)':<16} {'-':>7}    [{l['how']}, 최고 유사도 {l['top1']} {l['sim']:.2f}, OCR {l['ocr']}]")
    print(f"  합계 {total:,}원  / 인식 {sum(l['status'] == 'ok' for l in lines)}개, 바코드 {sum(l['status'] != 'ok' for l in lines)}개")
    print("단계별 시간(ms, CPU): " + ", ".join(f"{k} {v:.0f}" for k, v in sorted(T.ms.items())) + f" = {total_ms:.0f}")

    suffix = "" if args.detector == "yoloe" else "_product"
    draw_boxes(img, dets, size=26).save(OUT / f"scan_result{suffix}.jpg", quality=88)
    (OUT / f"scan_result{suffix}.json").write_text(json.dumps({"detector": args.detector,
        "source": str(args.source).replace("\\", "/").split("/")[-1], "lines": lines, "total": total,
        "timing_ms": {k: round(v) for k, v in sorted(T.ms.items())}, "total_ms": round(total_ms),
        "device": "CPU (개발 PC)", "siglip": ckpt,
    }, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()

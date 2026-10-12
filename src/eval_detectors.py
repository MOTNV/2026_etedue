"""상품 '위치' 검출기 3종 비교: YOLO26n(COCO) vs YOLOE(제로샷) vs YOLO26n-product(직접 학습).

평가 사진은 모두 학습에 쓰지 않은 사진이다 (make_train_data.py 참고).
  - 합성 계산대 1장, 실제 상품 사진 7장(삼각김밥은 학습에 없던 종류), 상품이 없는 음식 사진 3장
정답 박스는 사람이 표시했다 (data/eval_set.json).

  python src/eval_detectors.py
결과: docs/images/detector_compare.png, docs/images/eval_examples.jpg, docs/images/eval_result.json
"""
import json
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from ultralytics import YOLO

from common import DATA, MODELS, OUT, draw_boxes, font, open_rgb
from detect_text_yoloe import PRODUCT_PROMPTS, load_yoloe

IOU = 0.5


def iou(a, b):
    iw = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    ih = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = iw * ih
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter + 1e-9)


def match(preds, gts):
    """점수 높은 순으로 정답과 1:1 매칭 -> 각 예측의 TP 여부"""
    used, tps = set(), []
    for box, _ in sorted(preds, key=lambda p: -p[1]):
        best, bi = 0, None
        for i, g in enumerate(gts):
            if i not in used and iou(box, g) > best:
                best, bi = iou(box, g), i
        ok = best >= IOU
        if ok:
            used.add(bi)
        tps.append(ok)
    return tps


def ap50(all_preds, n_gt):
    """VOC 방식 AP@0.5: 모든 사진의 예측을 점수순으로 모아 정밀도-재현율 곡선 아래 넓이"""
    items = sorted(all_preds, key=lambda x: -x[0])
    tp = np.cumsum([t for _, t in items])
    fp = np.cumsum([not t for _, t in items])
    rec = tp / max(n_gt, 1)
    prec = tp / np.maximum(tp + fp, 1)
    mrec = np.concatenate([[0], rec, [1]])
    mpre = np.concatenate([[1], prec, [0]])
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))


def detectors():
    coco = YOLO("yolo26n.pt")
    yoloe = load_yoloe(PRODUCT_PROMPTS)
    prod = YOLO(str(MODELS / "yolo26n-product.pt"))

    def run(model, bgr, conf, **kw):
        r = model.predict(bgr, conf=conf, verbose=False, **kw)[0]
        return [(b, s) for b, s in zip(r.boxes.xyxy.tolist(), r.boxes.conf.tolist())]

    # (이름, 함수, 운영 기준값) - 운영 기준값은 각 모델을 실제로 쓸 때의 conf
    return [
        ("YOLO26n (COCO)", lambda b, c: run(coco, b, c), 0.25),
        ("YOLOE (제로샷)", lambda b, c: run(yoloe, b, c, agnostic_nms=True), 0.15),
        ("YOLO26n-product (학습)", lambda b, c: run(prod, b, c), 0.25),
    ]


def main():
    evalset = json.loads((DATA / "eval_set.json").read_text(encoding="utf-8"))
    imgs = []
    for e in evalset:
        img = open_rgb(DATA / e["image"])
        gts = e["boxes"] if e["boxes"] != "tray" else \
            [b[1:] for b in json.loads((DATA / "samples" / "tray_demo_boxes.json").read_text())]
        imgs.append((e["name"], img, np.array(img)[:, :, ::-1].copy(), gts))
    n_gt = sum(len(g) for *_, g in imgs)

    results, examples = {}, {}
    for name, fn, keep in detectors():
        fn(imgs[0][2], 0.25)  # 워밍업
        all_preds, per_img, ms = [], [], []
        tp_k = fp_k = 0
        for iname, img, bgr, gts in imgs:
            t0 = time.perf_counter()
            preds = fn(bgr, 0.01)
            ms.append((time.perf_counter() - t0) * 1000)
            all_preds += [(s, t) for (b, s), t in zip(sorted(preds, key=lambda p: -p[1]), match(preds, gts))]
            kept = [p for p in preds if p[1] >= keep]
            tps = match(kept, gts)
            tp_k += sum(tps)
            fp_k += len(tps) - sum(tps)
            per_img.append({"image": iname, "gt": len(gts), "tp": sum(tps), "fp": len(tps) - sum(tps)})
            examples[(name, iname)] = kept
        res = {"conf": keep, "precision": tp_k / max(tp_k + fp_k, 1), "recall": tp_k / n_gt,
               "ap50": ap50(all_preds, n_gt), "ms": float(np.mean(ms)), "per_image": per_img}
        results[name] = res
        print(f"{name:<24} AP50 {res['ap50']:.3f}  P {res['precision']:.2f}  R {res['recall']:.2f}  "
              f"(conf {keep}, 정답 {n_gt}개 중 {tp_k}개, 오검출 {fp_k}개)  {res['ms']:.0f} ms/장")
        for r in per_img:
            print(f"    {r['image']:<16} 정답 {r['gt']}  맞춤 {r['tp']}  오검출 {r['fp']}")

    (OUT / "eval_result.json").write_text(json.dumps({"n_images": len(imgs), "n_gt": n_gt, "iou": IOU,
                                                      "results": results}, ensure_ascii=False, indent=2),
                                          encoding="utf-8")

    # 막대 그래프
    plt.rcParams["font.family"] = font().getname()[0]
    names = list(results)
    fig, ax = plt.subplots(figsize=(8, 3.6), dpi=150)
    x = np.arange(3)
    colors = ["#94A3B8", "#06B6D4", "#1D4ED8"]
    for i, n in enumerate(names):
        vals = [results[n]["ap50"], results[n]["precision"], results[n]["recall"]]
        bars = ax.bar(x + (i - 1) * 0.26, vals, 0.26, label=n, color=colors[i])
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.015, f"{v:.2f}", ha="center", fontsize=8)
    ax.set_xticks(x, ["AP@0.5", "정밀도 (운영 기준값)", "재현율 (운영 기준값)"])
    ax.set_ylim(0, 1.12)
    ax.legend(fontsize=8, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.13))
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT / "detector_compare.png")

    # 예시 그림: 사진 4장 x 검출기 3종
    pick = ["tray_demo", "kimbap", "te_grape_ade_2", "te_shin_3"]
    cell = 300
    sheet = Image.new("RGB", (cell * 3 + 20, cell * len(pick) + 40), "white")
    from PIL import ImageDraw
    d = ImageDraw.Draw(sheet)
    for j, n in enumerate(names):
        d.text((j * (cell + 10) + 6, 8), n, fill="#0F172A", font=font(18))
    for i, iname in enumerate(pick):
        img = next(im for nm, im, _, _ in imgs if nm == iname)
        for j, n in enumerate(names):
            dets = [(*b, f"{s:.2f}", "#1D4ED8") for b, s in examples[(n, iname)]]
            v = draw_boxes(img, dets, size=max(18, img.width // 30))
            v.thumbnail((cell - 6, cell - 6))
            sheet.paste(v, (j * (cell + 10) + 3, 40 + i * cell + 3))
    sheet.save(OUT / "eval_examples.jpg", quality=88)


if __name__ == "__main__":
    main()

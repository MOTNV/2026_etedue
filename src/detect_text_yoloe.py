"""YOLOE 텍스트 프롬프트로 '상품처럼 생긴 것'을 학습 없이 검출한다.

스캔앤고에서 YOLOE는 상품이 '무엇인지'가 아니라 '어디에 있는지'만 찾는다.
상품 종류 판별은 SigLIP 2 갤러리 매칭(scan_and_go.py)이 맡는다.

  python src/detect_text_yoloe.py [--prompts bottle "snack package" ...]
결과: docs/images/tray_yoloe.jpg, docs/images/conf_chart.png
"""
import argparse
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from ultralytics import YOLOE

from common import OUT, SAMPLES, draw_boxes, font, open_rgb, resolve

PRODUCT_PROMPTS = ["bottle", "snack package", "seaweed wrapped rice", "cup noodle", "can"]


def load_yoloe(prompts, weights="yoloe-26s-seg.pt"):
    model = YOLOE(weights)
    model.set_classes(prompts)  # '말'이 곧 클래스 (재학습 0)
    return model


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source", default=str(SAMPLES / "tray_demo.jpg"))
    p.add_argument("--prompts", nargs="+", default=PRODUCT_PROMPTS)
    p.add_argument("--conf", type=float, default=0.05, help="차트에 낮은 점수도 보이도록 낮게 둔다")
    p.add_argument("--keep", type=float, default=0.15, help="이 값 이상만 상품 후보로 채택 (놓치는 것보다 후보가 많은 편이 안전)")
    args = p.parse_args()
    args.source = str(resolve(args.source))

    model = load_yoloe(args.prompts)
    model.predict(args.source, verbose=False)
    t0 = time.perf_counter()
    r = model.predict(args.source, conf=args.conf, agnostic_nms=True, verbose=False)[0]
    ms = (time.perf_counter() - t0) * 1000

    rows, dets = [], []
    for (x1, y1, x2, y2), c, s in zip(r.boxes.xyxy.tolist(), r.boxes.cls.tolist(), r.boxes.conf.tolist()):
        name = model.names[int(c)]
        keep = s >= args.keep
        rows.append((name, s, keep))
        print(f"  {name:<14} {s:.2f} {'keep' if keep else 'drop'}")
        if keep:
            dets.append((x1, y1, x2, y2, f"{name} {s:.2f}", "#0E7490"))
    print(f"YOLOE: {len(dets)}개 채택 / {len(rows)}개 후보, {ms:.0f} ms (CPU)")
    draw_boxes(open_rgb(args.source), dets).save(OUT / "tray_yoloe.jpg", quality=88)

    rows.sort(key=lambda x: -x[1])
    plt.rcParams["font.family"] = font().getname()[0] if font().getname() else "sans-serif"
    fig, ax = plt.subplots(figsize=(7, 3.2), dpi=150)
    ax.barh([f"{i + 1}. {n}" for i, (n, _, _) in enumerate(rows)][::-1], [s for _, s, _ in rows][::-1],
            color=["#1D4ED8" if k else "#CBD5E1" for _, _, k in rows][::-1])
    ax.axvline(args.keep, color="#D97706", ls="--", lw=1)
    ax.text(args.keep + 0.01, -0.6, f"threshold {args.keep}", color="#D97706", fontsize=8)
    ax.set_xlim(0, 1)
    ax.set_xlabel("confidence")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT / "conf_chart.png")


if __name__ == "__main__":
    main()

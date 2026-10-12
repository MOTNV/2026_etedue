"""[기준선] YOLO26 (COCO 80클래스) 으로 계산대 이미지를 검출한다.

COCO에는 '바나나우유', '초코파이' 같은 상품 클래스가 없다.
bottle·cup 정도의 일반 클래스로만 잡히거나 아예 놓친다는 것을 확인하는 용도.

  python src/detect_yolo26.py [--source data/samples/tray_demo.jpg]
"""
import argparse
import time

from ultralytics import YOLO

from common import OUT, SAMPLES, draw_boxes, open_rgb, resolve


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source", default=str(SAMPLES / "tray_demo.jpg"))
    p.add_argument("--conf", type=float, default=0.25)
    args = p.parse_args()
    args.source = str(resolve(args.source))

    model = YOLO("yolo26n.pt")
    model.predict(args.source, verbose=False)  # 워밍업
    t0 = time.perf_counter()
    r = model.predict(args.source, conf=args.conf, verbose=False)[0]
    ms = (time.perf_counter() - t0) * 1000

    dets = []
    for (x1, y1, x2, y2), c, s in zip(r.boxes.xyxy.tolist(), r.boxes.cls.tolist(), r.boxes.conf.tolist()):
        name = r.names[int(c)]
        print(f"  {name:<12} {s:.2f}")
        dets.append((x1, y1, x2, y2, f"{name} {s:.2f}", "#64748B"))
    print(f"YOLO26n: {len(dets)}개 검출, {ms:.0f} ms (CPU)")
    draw_boxes(open_rgb(args.source), dets).save(OUT / "tray_yolo26.jpg", quality=88)


if __name__ == "__main__":
    main()

"""YOLO26n '상품' 1클래스 검출기 학습 (COCO 사전학습 가중치에서 전이학습).

YOLOE를 대신해 '상품이 어디 있는지'만 찾는 가벼운 검출기다.
상품 종류 판별은 그대로 SigLIP 2 갤러리가 하므로, 신상품이 나와도 이 모델은 다시 학습할 필요가 없다.

  python src/make_train_data.py cutouts && python src/make_train_data.py synth
  python src/train_yolo26.py [--epochs 30]
결과: models/yolo26n-product.pt, docs/images/train_results.png
"""
import argparse
import shutil

from ultralytics import YOLO

from common import DATA, MODELS, OUT


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--imgsz", type=int, default=512, help="CPU 학습 시간을 줄이려고 512 (GPU면 640 권장)")
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--device", default="cpu", help="GPU가 있으면 0")
    args = p.parse_args()

    model = YOLO("yolo26n.pt")
    model.train(data=str(DATA / "train" / "product.yaml"), epochs=args.epochs, imgsz=args.imgsz,
                batch=args.batch, device=args.device, workers=2, patience=15, seed=0,
                project=str(MODELS / "train"), name="product", exist_ok=True, plots=True)

    run = MODELS / "train" / "product"
    shutil.copy(run / "weights" / "best.pt", MODELS / "yolo26n-product.pt")
    shutil.copy(run / "results.png", OUT / "train_results.png")
    print("저장:", MODELS / "yolo26n-product.pt")


if __name__ == "__main__":
    main()

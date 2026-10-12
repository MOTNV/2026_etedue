"""SigLIP 2 유사도 행렬: 계산대 위 상품 사진 x 등록 상품 갤러리.

각 행(계산대 상품)에서 가장 높은 칸이 인식 결과다. 미등록 상품은 모든 칸이 낮게 나와야 한다.

  python src/siglip_similarity.py
결과: docs/images/siglip_similarity.png
"""
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from common import INDEX, OUT, SAMPLES, font, open_rgb
from siglip_embed import SigLIP, load_index


def main():
    skus, gal, ckpt = load_index(INDEX)
    items = SAMPLES / "items"
    truth = json.loads((items / "truth.json").read_text(encoding="utf-8"))
    names = list(truth)

    enc = SigLIP(ckpt)
    q = enc.images([open_rgb(items / f"{n}.jpg") for n in names])
    sim_all = q @ gal.astype(np.float32).T
    cols = list(dict.fromkeys(skus))  # SKU별 최대값 (사진이 여러 장일 수 있음)
    sim = np.stack([sim_all[:, [i for i, s in enumerate(skus) if s == c]].max(1) for c in cols], 1)

    for n, row in zip(names, sim):
        print(f"  {n:<16} " + "  ".join(f"{c}:{v:.2f}" for c, v in zip(cols, row)))

    plt.rcParams["font.family"] = font().getname()[0]
    fig, ax = plt.subplots(figsize=(8, 4), dpi=150)
    im = ax.imshow(sim, cmap="Blues", vmin=0, vmax=1, aspect="auto")
    short = {"BANANA_MILK": "바나나우유", "CHILSUNG_CIDER": "칠성사이다", "CHOCO_PIE": "초코파이", "CUP_RAMEN": "컵라면"}
    ax.set_xticks(range(len(cols)), [short.get(c, c) for c in cols], fontsize=9)
    ylabels = {"banana_milk": "바나나우유", "strawberry_milk": "딸기우유 (미등록)", "chilsung_cider": "칠성사이다",
               "choco_pie_cn": "초코파이 (중국판 포장)", "triangle_kimbap": "삼각김밥 (미등록)"}
    ax.set_yticks(range(len(names)), [ylabels.get(n, n) for n in names])
    for i in range(len(names)):
        for j in range(len(cols)):
            ax.text(j, i, f"{sim[i, j]:.2f}", ha="center", va="center",
                    color="white" if sim[i, j] > 0.6 else "#0F172A", fontsize=9,
                    fontweight="bold" if j == sim[i].argmax() else "normal")
    ax.set_xlabel("등록 상품 갤러리")
    ax.set_title("SigLIP 2 코사인 유사도 (계산대 상품 x 등록 사진)", fontsize=10)
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    fig.savefig(OUT / "siglip_similarity.png")


if __name__ == "__main__":
    main()

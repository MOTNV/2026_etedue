"""SigLIP 2 이미지 임베딩 공용 함수.

기본 체크포인트는 CPU에서도 돌아가는 base 모델이다.
보고서의 so400m(1152차원)은 더 정확하지만 무거우므로 AI 박스(GPU)에서 --ckpt로 바꿔 쓴다.
"""
import numpy as np
import torch
from transformers import AutoModel, AutoProcessor

DEFAULT_CKPT = "google/siglip2-base-patch16-224"


class SigLIP:
    def __init__(self, ckpt=DEFAULT_CKPT):
        self.ckpt = ckpt
        self.model = AutoModel.from_pretrained(ckpt).eval()
        self.proc = AutoProcessor.from_pretrained(ckpt)

    @torch.no_grad()
    def images(self, imgs):
        """PIL 이미지 리스트 -> L2 정규화된 임베딩 (N, D)"""
        feats = self.model.get_image_features(**self.proc(images=imgs, return_tensors="pt"))
        if not torch.is_tensor(feats):  # transformers 버전에 따라 출력 객체로 올 수 있음
            feats = feats.pooler_output
        return torch.nn.functional.normalize(feats, dim=-1).numpy()

    @torch.no_grad()
    def texts(self, texts):
        inputs = self.proc(text=texts, padding="max_length", max_length=64, return_tensors="pt")
        feats = self.model.get_text_features(**inputs)
        if not torch.is_tensor(feats):
            feats = feats.pooler_output
        return torch.nn.functional.normalize(feats, dim=-1).numpy()


def load_index(path):
    z = np.load(path, allow_pickle=False)
    return list(z["skus"]), z["embs"], str(z["ckpt"])

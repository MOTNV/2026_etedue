# 스캔앤고 SCAN&GO — 나만의 피지컬 AI 제품

바코드를 찍지 않고 계산대에 올려놓기만 하면 카메라가 상품을 알아보고 금액을 계산해 주는 편의점 무인계산대.

- **홈페이지:** `index.html` (GitHub Pages)
- **제품 요구사항:** [`PRD/PRD.md`](PRD/PRD.md)
- **모델 검토 보고서:** [`docs/REPORT_zero-shot-models.md`](docs/REPORT_zero-shot-models.md) — 결론 **A안** (YOLOE + SigLIP 2 + OCR + MediaPipe 적용)

## 인식 파이프라인

```
카메라 → MediaPipe(손 없음 확인) → YOLOE(상품 위치, 재학습 0)
      → SigLIP 2(등록 사진 갤러리와 유사도) → [애매하면] OCR(포장지 글자 확인)
      → 영수증 / 미등록·불일치는 바코드
```

신상품은 모델 재학습 없이 **사진 등록**만으로 추가합니다 (`src/register_product.py`).

## 실행

```bash
python -m venv .venv
.venv\Scripts\activate            # macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt

python src/prepare_samples.py     # 공개 사진 내려받기 → data/gallery, data/samples
python src/register_product.py    # 상품 갤러리 임베딩 만들기
python src/scan_and_go.py         # 전체 파이프라인 → docs/images/scan_result.jpg

# 단계별 실습
python src/detect_yolo26.py       # 기준선 (COCO 80클래스)
python src/detect_text_yoloe.py   # YOLOE 텍스트 프롬프트 + 신뢰도 차트
python src/siglip_similarity.py   # 유사도 행렬
python src/ocr_label.py           # 포장지 OCR (--engine easy 로 EasyOCR 비교)
python src/gesture_recognizer.py  # 손 감지·제스처
```

### 모델 학습 (YOLO26 '상품' 검출기)

```bash
python src/make_train_data.py cutouts   # YOLOE 마스크로 상품 자동 라벨링 -> data/train/cutouts_sheet.jpg 확인
python src/make_train_data.py synth     # 합성 학습 데이터 700장
python src/train_yolo26.py              # YOLO26n 1클래스 전이학습 (CPU 약 1시간, GPU면 --device 0)
python src/eval_detectors.py            # 학습에 안 쓴 사진 11장으로 YOLO26(COCO)·YOLOE·학습 모델 비교
python src/scan_and_go.py --detector product   # 학습한 검출기로 전체 파이프라인
```

결과: 실제 사진 AP@0.5 기준 YOLOE 0.997 > 학습 모델 0.86 > COCO 0.83. 지금은 YOLOE를 유지한다 (보고서 3.5절).

가중치(YOLO26, YOLOE, SigLIP 2, OCR, MediaPipe)는 첫 실행 때 자동으로 내려받아 `models/`와 Hugging Face 캐시에 저장됩니다.

### 신상품 등록

```bash
python src/register_product.py --add KIMBAP_TUNA "삼각김밥 참치마요" 1500 참치 김밥 --images 사진1.jpg 사진2.jpg
```

## 폴더 구성

```
├── index.html              제품 소개 홈페이지
├── PRD/PRD.md              제품 요구사항 문서
├── report/                 과제 보고서
├── docs/                   모델 검토 보고서, 결과 이미지, 사진 출처
├── data/                   상품 카탈로그·등록 사진·테스트 이미지
├── src/                    모델 실습 코드
└── requirements.txt
```

실습 사진은 Wikimedia Commons의 자유 라이선스 사진을 잘라 쓰고 합성했습니다. 출처는 [`docs/CREDITS.md`](docs/CREDITS.md)에 있습니다.

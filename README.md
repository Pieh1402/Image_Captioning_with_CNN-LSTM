# Image Captioning bằng CNN-LSTM trên Flickr8k

Dự án đồ án tốt nghiệp xây dựng hệ thống mô tả hình ảnh tự động bằng kiến trúc CNN-LSTM:

- CNN encoder: EfficientNetB0 pretrained ImageNet, `include_top=False`, `pooling='avg'`.
- Decoder: Embedding + LSTM + Bahdanau Attention.
- Dataset: Flickr8k, file `archive/captions.txt` và thư mục `archive/Images/`.
- Đánh giá: BLEU-1..4, ROUGE-L, METEOR, CIDEr.
- Demo: Flask web app upload ảnh và sinh caption.

## Cấu trúc

```text
project/
├── archive/
│   ├── Images/
│   └── captions.txt
├── artifacts/
├── backend/
├── frontend/
├── evaluate_all.py
├── requirements.txt
└── README.md
```

## Cài đặt

```bash
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
python -c "import nltk; nltk.download('wordnet'); nltk.download('omw-1.4')"
```

## Chuẩn bị dữ liệu

Đặt Flickr8k theo cấu trúc:

```text
archive/
├── Images/
│   ├── 1000268201_693b08cb0e.jpg
│   └── ...
└── captions.txt
```

`captions.txt` cần có định dạng CSV:

```csv
image,caption
1000268201_693b08cb0e.jpg,A child in a pink dress is climbing up a set of stairs.
```

## Huấn luyện

```bash
python -m backend.train
```

Tùy chọn:

```bash
python -m backend.train --epochs 30 --batch-size 64 --learning-rate 0.0001
python -m backend.train --rebuild-features
```

Sau khi train, thư mục `artifacts/` sẽ có model, tokenizer, meta, features cache và history.

## Đánh giá

```bash
python -m backend.evaluate --decode-mode beam --beam-width 3
python evaluate_all.py
```

`evaluate_all.py` sinh thêm loss curve, biểu đồ metric, file dự đoán và báo cáo Markdown.

## Chạy web demo

```bash
python -m frontend.app
```

Mở `http://127.0.0.1:5000`.

## Lưu ý học thuật

Project chia dữ liệu theo Image ID với tỉ lệ 70/15/15 để tránh data leakage. Tokenizer được fit trên train captions. Evaluate dùng cùng seed và cùng hàm split với train.

Do EfficientNetB0 dùng `pooling='avg'`, feature ảnh là vector toàn cục `(1280,)`; vì vậy không thể tạo spatial attention heatmap thật trên vùng ảnh. Nếu cần heatmap không gian, hãy đổi encoder sang output feature map `7x7x1280` và attention theo spatial patches.


<h2 align="center">
    <a href="https://dainam.edu.vn/vi/khoa-cong-nghe-thong-tin">
    🎓 Faculty of Information Technology (DaiNam University)
    </a>
</h2>
<br>
<h2 align="center">
    IMAGE CAPTIONING BẰNG CNN-LSTM TRÊN FLICKR8K
</h2>
<br>
<div align="center">
    <p align="center">
        <img src="https://raw.githubusercontent.com/FIT-DNU/Cryptography-and-Cyber-Security/main/fitdnu_logo.png" alt="FIT DNU Logo" width="180"/>
        <img src="https://raw.githubusercontent.com/FIT-DNU/Cryptography-and-Cyber-Security/main/dnu_logo.png" alt="DaiNam University Logo" width="200"/>
    </p>


[![Faculty of Information Technology](https://img.shields.io/badge/Faculty%20of%20Information%20Technology-blue?style=for-the-badge)](https://dainam.edu.vn/vi/khoa-cong-nghe-thong-tin)
[![DaiNam University](https://img.shields.io/badge/DaiNam%20University-orange?style=for-the-badge)](https://dainam.edu.vn)

</div>

# Image Captioning bằng CNN-LSTM trên Flickr8k

Dự án đồ án tốt nghiệp xây dựng hệ thống mô tả hình ảnh tự động bằng kiến trúc CNN-LSTM:

- CNN encoder: EfficientNetB0 pretrained ImageNet, `include_top=False`, `pooling='avg'`.
- Decoder: Embedding + LSTM + Bahdanau Attention.
- Dataset: Flickr8k, file `archive/captions.txt` và thư mục `archive/Images/`.
- Đánh giá: BLEU-1..4, ROUGE-L, METEOR, CIDEr.
- Demo: Flask web app upload ảnh và sinh caption.

<br>

<details>
<summary>
    <h3>
        📁 Cấu trúc dự án
    </h3>
</summary>

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

</details>

<details>
<summary>
    <h3>
        ⚙️ Cài đặt
    </h3>
</summary>

```bash
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
python -c "import nltk; nltk.download('wordnet'); nltk.download('omw-1.4')"
```

</details>

<details>
<summary>
    <h3>
        🗂️ Chuẩn bị dữ liệu
    </h3>
</summary>

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

</details>

<details>
<summary>
    <h3>
        🏋️ Huấn luyện
    </h3>
</summary>

```bash
python -m backend.train
```

Tùy chọn:

```bash
python -m backend.train --epochs 30 --batch-size 64 --learning-rate 0.0001
python -m backend.train --rebuild-features
```

Sau khi train, thư mục `artifacts/` sẽ có model, tokenizer, meta, features cache và history.

</details>

<details>
<summary>
    <h3>
        📊 Đánh giá
    </h3>
</summary>

```bash
python -m backend.evaluate --decode-mode beam --beam-width 3
python evaluate_all.py
```

`evaluate_all.py` sinh thêm loss curve, biểu đồ metric, file dự đoán và báo cáo Markdown.

</details>

<details>
<summary>
    <h3>
        🌐 Chạy web demo
    </h3>
</summary>

```bash
python -m frontend.app
```

Mở `http://127.0.0.1:5000`.

</details>

<details>
<summary>
    <h3>
        📝 Lưu ý học thuật
    </h3>
</summary>

Project chia dữ liệu theo Image ID với tỉ lệ 70/15/15 để tránh data leakage. Tokenizer được fit trên train captions. Evaluate dùng cùng seed và cùng hàm split với train.

Do EfficientNetB0 dùng `pooling='avg'`, feature ảnh là vector toàn cục `(1280,)`; vì vậy không thể tạo spatial attention heatmap thật trên vùng ảnh. Nếu cần heatmap không gian, hãy đổi encoder sang output feature map `7x7x1280` và attention theo spatial patches.

</details>

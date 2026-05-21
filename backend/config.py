"""Central configuration for the image captioning project."""

from __future__ import annotations

import os


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Tự động phát hiện môi trường Kaggle/Colab
if os.path.exists("/kaggle/input/flickr8k"):
    # Kaggle Dataset
    ARCHIVE_DIR = "/kaggle/input/flickr8k"
    IMAGES_DIR = os.path.join(ARCHIVE_DIR, "Images")
    CAPTION_FILE = os.path.join(ARCHIVE_DIR, "captions.txt")
elif os.path.exists("/content/flickr8k"):
    # Colab local (nếu unzip vào đây)
    ARCHIVE_DIR = "/content/flickr8k"
    IMAGES_DIR = os.path.join(ARCHIVE_DIR, "Images")
    CAPTION_FILE = os.path.join(ARCHIVE_DIR, "captions.txt")
else:
    # Local development
    ARCHIVE_DIR = os.path.join(BASE_DIR, "archive")
    IMAGES_DIR = os.path.join(ARCHIVE_DIR, "Images")
    CAPTION_FILE = os.path.join(ARCHIVE_DIR, "captions.txt")

ARTIFACTS_DIR = os.path.join(BASE_DIR, "artifacts")
UPLOAD_DIR = os.path.join(ARTIFACTS_DIR, "uploads")

MODEL_PATH = os.path.join(ARTIFACTS_DIR, "caption_model.keras")
TOKENIZER_PATH = os.path.join(ARTIFACTS_DIR, "tokenizer.pkl")
META_PATH = os.path.join(ARTIFACTS_DIR, "meta.json")
FEATURES_PATH = os.path.join(ARTIFACTS_DIR, "features_effnetb0.pkl")
HISTORY_PATH = os.path.join(ARTIFACTS_DIR, "training_history.json")
CSV_LOG_PATH = os.path.join(ARTIFACTS_DIR, "training_log.csv")

EVALUATION_PATH = os.path.join(ARTIFACTS_DIR, "evaluation_results.json")
PREDICTIONS_PATH = os.path.join(ARTIFACTS_DIR, "eval_predictions.csv")
LOSS_CURVE_PATH = os.path.join(ARTIFACTS_DIR, "loss_curve.png")
RESULT_SAMPLES_PATH = os.path.join(ARTIFACTS_DIR, "result_samples.png")
ATTENTION_MAPS_PATH = os.path.join(ARTIFACTS_DIR, "attention_maps.png")
FINAL_REPORT_PATH = os.path.join(ARTIFACTS_DIR, "final_report.md")

IMG_SIZE = 224
FEAT_DIM = 1280
EMBEDDING_DIM = 256
LSTM_UNITS = 512
ATTENTION_UNITS = 512
DROPOUT_RATE = 0.5

BATCH_SIZE = 64
EPOCHS = 30
LEARNING_RATE = 1e-3
SEED = 42

VAL_SPLIT = 0.15
TEST_SPLIT = 0.15


for directory in (ARCHIVE_DIR, IMAGES_DIR, ARTIFACTS_DIR, UPLOAD_DIR):
    os.makedirs(directory, exist_ok=True)


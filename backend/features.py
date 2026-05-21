"""EfficientNetB0 feature extraction and caching utilities."""

from __future__ import annotations

import os
import pickle

import numpy as np
from PIL import Image, UnidentifiedImageError
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tqdm import tqdm


def build_feature_extractor(img_size: int = 224):
    """Build a frozen EfficientNetB0 feature extractor returning 1280-d vectors."""
    model = EfficientNetB0(
        weights="imagenet",
        include_top=False,
        input_shape=(img_size, img_size, 3),
    )
    model.trainable = False
    return model


def preprocess_image(img_path: str, img_size: int = 224) -> np.ndarray:
    """Load and preprocess a single image for EfficientNetB0."""
    if not os.path.exists(img_path):
        raise FileNotFoundError(f"Không tìm thấy ảnh: {img_path}")
    try:
        img = load_img(img_path, target_size=(img_size, img_size))
        arr = img_to_array(img)
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise ValueError(f"Không đọc được ảnh {img_path}: {exc}") from exc
    return preprocess_input(arr.astype(np.float32))


def extract_all_features(
    model,
    image_list: list[str],
    images_dir: str,
    img_size: int = 224,
) -> dict[str, np.ndarray]:
    """Extract features for all unique images, skipping corrupted files with warnings."""
    features: dict[str, np.ndarray] = {}
    for image_name in tqdm(sorted(set(image_list)), desc="Extracting features"):
        img_path = os.path.join(images_dir, image_name)
        try:
            img = preprocess_image(img_path, img_size)
            batch = np.expand_dims(img, axis=0)
            feature_map = model.predict(batch, verbose=0)[0]
            feature = feature_map.reshape(-1, feature_map.shape[-1]).astype(np.float32)
            features[image_name] = feature
        except (FileNotFoundError, ValueError, OSError) as exc:
            print(f"[WARN] Bỏ qua ảnh lỗi {image_name}: {exc}")
    if not features:
        raise ValueError("Không trích xuất được feature nào. Kiểm tra archive/Images.")
    return features


def save_features(features: dict[str, np.ndarray], path: str) -> None:
    """Save feature dictionary using pickle."""
    with open(path, "wb") as f:
        pickle.dump(features, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_features(path: str) -> dict[str, np.ndarray]:
    """Load cached feature dictionary."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Không tìm thấy cache features: {path}")
    with open(path, "rb") as f:
        return pickle.load(f)


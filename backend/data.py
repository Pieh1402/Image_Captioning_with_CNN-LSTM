"""Data loading, preprocessing, tokenization and batch generation."""

from __future__ import annotations

import json
import os
import pickle
import random
import re
from typing import Any

import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import Sequence, to_categorical


def load_captions(path: str) -> pd.DataFrame:
    """Load Flickr8k captions from a CSV file with image and caption columns."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Không tìm thấy file captions: {path}")

    df = pd.read_csv(path)
    if "image" not in df.columns or "caption" not in df.columns:
        raise ValueError("File captions.txt phải có 2 cột: 'image' và 'caption'.")
    return df[["image", "caption"]].dropna().reset_index(drop=True)


def preprocess_text(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize captions and add start/end sequence tokens."""
    out = df.copy()
    out["caption"] = out["caption"].astype(str).str.strip().str.lower()
    out["caption"] = out["caption"].apply(lambda x: re.sub(r"[^a-z0-9\s\.,'!?-]", "", x))
    out["caption"] = out["caption"].apply(lambda x: re.sub(r"\s+", " ", x).strip())
    out["caption"] = out["caption"].apply(lambda x: f"startseq {x} endseq")
    return out


def build_tokenizer(captions: list[str]) -> tuple[Tokenizer, int, int]:
    """Fit a Keras tokenizer and return tokenizer, vocab size and max length."""
    tokenizer = Tokenizer(oov_token=None)
    tokenizer.fit_on_texts(captions)
    vocab_size = len(tokenizer.word_index) + 1
    max_length = max(len(caption.split()) for caption in captions)
    return tokenizer, vocab_size, max_length


def split_by_image(
    df: pd.DataFrame,
    val_split: float = 0.15,
    test_split: float = 0.15,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split rows by unique image ID to avoid image leakage across sets."""
    if val_split < 0 or test_split < 0 or val_split + test_split >= 1:
        raise ValueError("val_split và test_split phải không âm và tổng nhỏ hơn 1.")

    images = sorted(df["image"].unique().tolist())
    rng = random.Random(seed)
    rng.shuffle(images)

    n_total = len(images)
    n_test = int(round(n_total * test_split))
    n_val = int(round(n_total * val_split))
    n_train = n_total - n_val - n_test

    train_images = set(images[:n_train])
    val_images = set(images[n_train : n_train + n_val])
    test_images = set(images[n_train + n_val :])

    train_df = df[df["image"].isin(train_images)].reset_index(drop=True)
    val_df = df[df["image"].isin(val_images)].reset_index(drop=True)
    test_df = df[df["image"].isin(test_images)].reset_index(drop=True)
    return train_df, val_df, test_df


class CaptionDataGenerator(Sequence):
    """Generate teacher-forcing batches from cached image features and captions."""

    def __init__(
        self,
        df: pd.DataFrame,
        tokenizer: Tokenizer,
        features: dict[str, np.ndarray],
        vocab_size: int,
        max_length: int,
        batch_size: int = 64,
        shuffle: bool = True,
    ) -> None:
        self.df = df[["image", "caption"]].copy()
        self.tokenizer = tokenizer
        self.features = features
        self.vocab_size = int(vocab_size)
        self.max_length = int(max_length)
        self.batch_size = int(batch_size)
        self.shuffle = bool(shuffle)
        self.samples = self._build_samples()
        self.indexes = np.arange(len(self.samples))
        self.on_epoch_end()

    def _build_samples(self) -> list[tuple[str, list[int], int]]:
        samples: list[tuple[str, list[int], int]] = []
        for _, row in self.df.iterrows():
            image = str(row["image"])
            if image not in self.features:
                continue
            seq = self.tokenizer.texts_to_sequences([row["caption"]])[0]
            for i in range(1, len(seq)):
                samples.append((image, seq[:i], int(seq[i])))
        if not samples:
            raise ValueError("Không tạo được sample train nào. Kiểm tra captions và features.")
        return samples

    def __len__(self) -> int:
        return int(np.ceil(len(self.samples) / float(self.batch_size)))

    def on_epoch_end(self) -> None:
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __getitem__(self, index: int) -> tuple[tuple[np.ndarray, np.ndarray], np.ndarray]:
        batch_ids = self.indexes[index * self.batch_size : (index + 1) * self.batch_size]
        image_features: list[np.ndarray] = []
        partial_sequences: list[np.ndarray] = []
        labels: list[np.ndarray] = []

        for sample_id in batch_ids:
            image, in_seq, out_token = self.samples[int(sample_id)]
            image_features.append(np.asarray(self.features[image], dtype=np.float32))
            partial_sequences.append(
                pad_sequences([in_seq], maxlen=self.max_length, padding="post", truncating="post")[0]
            )
            labels.append(to_categorical(out_token, num_classes=self.vocab_size))

        return (
            np.asarray(image_features, dtype=np.float32),
            np.asarray(partial_sequences, dtype=np.int32),
        ), np.asarray(labels, dtype=np.float32)


def save_tokenizer(tokenizer: Tokenizer, path: str) -> None:
    """Save tokenizer to pickle."""
    with open(path, "wb") as f:
        pickle.dump(tokenizer, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_tokenizer(path: str) -> Tokenizer:
    """Load tokenizer from pickle."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Không tìm thấy tokenizer: {path}")
    with open(path, "rb") as f:
        return pickle.load(f)


def save_meta(meta: dict[str, Any], path: str) -> None:
    """Save training metadata to JSON."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)


def load_meta(path: str) -> dict[str, Any]:
    """Load training metadata from JSON."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Không tìm thấy meta.json: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


"""Caption generation utilities for trained models."""

from __future__ import annotations

import math
import os
from functools import lru_cache
from typing import Any

import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences

from backend import config
from backend.data import load_meta, load_tokenizer
from backend.features import build_feature_extractor, preprocess_image
from backend.model import BahdanauAttention


_model = None
_tokenizer = None
_meta = None
_feature_extractor = None


def load_assets() -> tuple[Any, Any, dict[str, Any], Any]:
    """Lazy-load model, tokenizer, metadata and feature extractor."""
    global _model, _tokenizer, _meta, _feature_extractor
    if _tokenizer is None:
        _tokenizer = load_tokenizer(config.TOKENIZER_PATH)
    if _meta is None:
        _meta = load_meta(config.META_PATH)
    if _model is None:
        if not os.path.exists(config.MODEL_PATH):
            raise FileNotFoundError(f"Chưa có model đã train: {config.MODEL_PATH}")
        _model = tf.keras.models.load_model(
            config.MODEL_PATH,
            custom_objects={"BahdanauAttention": BahdanauAttention},
            compile=False,
        )
    if _feature_extractor is None:
        _feature_extractor = build_feature_extractor(int(_meta.get("img_size", config.IMG_SIZE)))
    return _model, _tokenizer, _meta, _feature_extractor


def preprocess_for_inference(img_path: str, feature_extractor, img_size: int = 224) -> np.ndarray:
    """Preprocess an image and extract its EfficientNetB0 feature vector."""
    img = preprocess_image(img_path, img_size)
    feature_map = feature_extractor.predict(np.expand_dims(img, axis=0), verbose=0)[0]
    feature = feature_map.reshape(-1, feature_map.shape[-1])
    return feature.astype(np.float32)


def _idx_to_word(tokenizer, index: int) -> str | None:
    return tokenizer.index_word.get(int(index))


def clean_caption(raw: str) -> str:
    """Remove special tokens and format a caption for display."""
    caption = raw.replace("startseq", "").replace("endseq", "").strip()
    caption = " ".join(caption.split())
    if not caption:
        return ""
    caption = caption[0].upper() + caption[1:]
    if caption[-1] not in ".!?":
        caption += "."
    return caption


def greedy_predict(model, tokenizer, feature: np.ndarray, max_length: int) -> str:
    """Generate one caption using greedy decoding."""
    start = tokenizer.word_index.get("startseq")
    end = tokenizer.word_index.get("endseq")
    seq = [start]

    for _ in range(max_length):
        padded = pad_sequences([seq], maxlen=max_length, padding="post")
        yhat = model.predict([feature[np.newaxis, ...], padded], verbose=0)[0]
        next_id = int(np.argmax(yhat))
        seq.append(next_id)
        if next_id == end:
            break

    words = [_idx_to_word(tokenizer, idx) for idx in seq]
    return clean_caption(" ".join(word for word in words if word))


def beam_search_predict(
    model,
    tokenizer,
    feature: np.ndarray,
    max_length: int,
    beam_width: int = 3,
) -> str:
    """Generate one caption using beam search with length normalization."""
    start = tokenizer.word_index.get("startseq")
    end = tokenizer.word_index.get("endseq")
    beams: list[tuple[list[int], float, bool]] = [([start], 0.0, False)]

    for _ in range(max_length):
        candidates: list[tuple[list[int], float, bool]] = []
        for seq, score, finished in beams:
            if finished:
                candidates.append((seq, score, True))
                continue
            padded = pad_sequences([seq], maxlen=max_length, padding="post")
            probs = model.predict([feature[np.newaxis, ...], padded], verbose=0)[0]
            top_ids = np.argsort(probs)[-beam_width:][::-1]
            for token_id in top_ids:
                prob = float(probs[token_id])
                if prob <= 0:
                    continue
                next_seq = seq + [int(token_id)]
                done = int(token_id) == end or len(next_seq) >= max_length
                length_norm = max(len(next_seq), 1) ** 0.7
                next_score = (score + math.log(prob + 1e-12)) / length_norm
                candidates.append((next_seq, next_score, done))
        beams = sorted(candidates, key=lambda item: item[1], reverse=True)[:beam_width]
        if beams and all(done for _, _, done in beams):
            break

    best_seq = beams[0][0]
    words = [_idx_to_word(tokenizer, idx) for idx in best_seq]
    return clean_caption(" ".join(word for word in words if word))


def get_attention_weights(model, feature: np.ndarray, sequence: list[int], max_length: int) -> np.ndarray | None:
    """Return temporal attention weights over decoder steps when the model exposes them."""
    try:
        att_layer = model.get_layer("bahdanau_attention")
        lstm_out = model.get_layer("caption_lstm").output
        image_embedding = model.get_layer("image_embedding").output
        _, weights = att_layer(lstm_out, image_embedding)
        att_model = tf.keras.Model(inputs=model.inputs, outputs=weights)
        padded = pad_sequences([sequence], maxlen=max_length, padding="post")
        return att_model.predict([feature[np.newaxis, ...], padded], verbose=0)[0]
    except (ValueError, AttributeError, TypeError):
        return None


def predict_from_path(img_path: str, beam_width: int = 3) -> tuple[str, np.ndarray | None]:
    """Public frontend API: generate a caption and optional temporal attention weights."""
    model, tokenizer, meta, extractor = load_assets()
    max_length = int(meta["max_length"])
    feature = preprocess_for_inference(img_path, extractor, int(meta.get("img_size", config.IMG_SIZE)))
    if beam_width <= 1:
        caption = greedy_predict(model, tokenizer, feature, max_length)
    else:
        caption = beam_search_predict(model, tokenizer, feature, max_length, beam_width)
    return caption, None


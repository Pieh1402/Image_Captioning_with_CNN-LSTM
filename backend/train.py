"""Training pipeline for EfficientNetB0 + LSTM image captioning."""

from __future__ import annotations

import argparse
import json
import os
import random

import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import CSVLogger, EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

from backend import config
from backend.data import (
    CaptionDataGenerator,
    build_tokenizer,
    load_captions,
    preprocess_text,
    save_meta,
    save_tokenizer,
    split_by_image,
)
from backend.features import build_feature_extractor, extract_all_features, load_features, save_features
from backend.model import build_model


def parse_args() -> argparse.Namespace:
    """Parse command-line options."""
    parser = argparse.ArgumentParser(description="Train CNN-LSTM image captioning on Flickr8k.")
    parser.add_argument("--epochs", type=int, default=config.EPOCHS)
    parser.add_argument("--batch-size", type=int, default=config.BATCH_SIZE)
    parser.add_argument("--learning-rate", type=float, default=config.LEARNING_RATE)
    parser.add_argument("--rebuild-features", action="store_true")
    return parser.parse_args()


def main() -> None:
    """Run the full training pipeline."""
    args = parse_args()
    random.seed(config.SEED)
    np.random.seed(config.SEED)
    tf.random.set_seed(config.SEED)

    print("[1/7] Loading captions")
    df = preprocess_text(load_captions(config.CAPTION_FILE))
    train_df, val_df, test_df = split_by_image(
        df,
        val_split=config.VAL_SPLIT,
        test_split=config.TEST_SPLIT,
        seed=config.SEED,
    )

    print("[2/7] Building tokenizer from train captions")
    tokenizer, vocab_size, max_length = build_tokenizer(train_df["caption"].tolist())
    save_tokenizer(tokenizer, config.TOKENIZER_PATH)

    meta = {
        "vocab_size": vocab_size,
        "max_length": max_length,
        "feat_dim": config.FEAT_DIM,
        "embedding_dim": config.EMBEDDING_DIM,
        "lstm_units": config.LSTM_UNITS,
        "attention_units": config.ATTENTION_UNITS,
        "img_size": config.IMG_SIZE,
        "seed": config.SEED,
        "val_split": config.VAL_SPLIT,
        "test_split": config.TEST_SPLIT,
        "train_images": int(train_df["image"].nunique()),
        "val_images": int(val_df["image"].nunique()),
        "test_images": int(test_df["image"].nunique()),
    }
    save_meta(meta, config.META_PATH)

    print("[3/7] Loading or extracting EfficientNetB0 features")
    if os.path.exists(config.FEATURES_PATH) and not args.rebuild_features:
        features = load_features(config.FEATURES_PATH)
    else:
        extractor = build_feature_extractor(config.IMG_SIZE)
        features = extract_all_features(
            extractor,
            df["image"].unique().tolist(),
            config.IMAGES_DIR,
            config.IMG_SIZE,
        )
        save_features(features, config.FEATURES_PATH)

    print("[4/7] Building model")
    model = build_model(
        vocab_size=vocab_size,
        max_length=max_length,
        feat_dim=config.FEAT_DIM,
        embedding_dim=config.EMBEDDING_DIM,
        lstm_units=config.LSTM_UNITS,
        attention_units=config.ATTENTION_UNITS,
        dropout_rate=config.DROPOUT_RATE,
        learning_rate=args.learning_rate,
    )
    model.summary()

    print("[5/7] Creating data generators")
    train_gen = CaptionDataGenerator(
        train_df,
        tokenizer,
        features,
        vocab_size,
        max_length,
        batch_size=args.batch_size,
        shuffle=True,
    )
    val_gen = CaptionDataGenerator(
        val_df,
        tokenizer,
        features,
        vocab_size,
        max_length,
        batch_size=args.batch_size,
        shuffle=False,
    )

    callbacks = [
        ModelCheckpoint(config.MODEL_PATH, monitor="val_loss", save_best_only=True, verbose=1),
        EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor="val_loss", patience=3, factor=0.5, min_lr=1e-7, verbose=1),
        CSVLogger(config.CSV_LOG_PATH),
    ]

    print("[6/7] Training")
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=args.epochs,
        callbacks=callbacks,
        verbose=1,
    )

    model.save(config.MODEL_PATH)
    with open(config.HISTORY_PATH, "w", encoding="utf-8") as f:
        json.dump(history.history, f, ensure_ascii=False, indent=2)

    best_val_loss = min(history.history.get("val_loss", [float("nan")]))
    best_epoch = int(np.argmin(history.history.get("val_loss", [0])) + 1)

    print("[7/7] Done")
    print(f"Best val_loss: {best_val_loss:.4f} at epoch {best_epoch}")
    print(f"Model: {config.MODEL_PATH}")
    print(f"Tokenizer: {config.TOKENIZER_PATH}")
    print(f"Meta: {config.META_PATH}")
    print(f"Features: {config.FEATURES_PATH}")


if __name__ == "__main__":
    main()


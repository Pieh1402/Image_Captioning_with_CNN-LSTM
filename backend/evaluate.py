"""Evaluate the trained image captioning model on the held-out test split."""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from tqdm import tqdm

from backend import config
from backend.data import load_captions, load_meta, load_tokenizer, preprocess_text, split_by_image
from backend.features import build_feature_extractor, extract_all_features, load_features, save_features
from backend.inference import beam_search_predict, clean_caption, greedy_predict
from backend.model import BahdanauAttention


def _load_model():
    import tensorflow as tf

    if not os.path.exists(config.MODEL_PATH):
        raise FileNotFoundError(f"Chưa có model: {config.MODEL_PATH}")
    return tf.keras.models.load_model(
        config.MODEL_PATH,
        custom_objects={"BahdanauAttention": BahdanauAttention},
        compile=False,
    )


def _tokens(text: str) -> list[str]:
    return clean_caption(text).lower().rstrip(".!?").split()


def _compute_bleu(refs: list[list[list[str]]], hyps: list[list[str]]) -> dict[str, float]:
    from nltk.translate.bleu_score import SmoothingFunction, corpus_bleu

    smooth = SmoothingFunction().method4
    weights = {
        "BLEU-1": (1.0, 0, 0, 0),
        "BLEU-2": (0.5, 0.5, 0, 0),
        "BLEU-3": (1 / 3, 1 / 3, 1 / 3, 0),
        "BLEU-4": (0.25, 0.25, 0.25, 0.25),
    }
    return {
        name: round(float(corpus_bleu(refs, hyps, weights=w, smoothing_function=smooth)), 4)
        for name, w in weights.items()
    }


def _compute_rouge(refs: list[list[list[str]]], hyps: list[list[str]]) -> float:
    from rouge_score import rouge_scorer

    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)
    scores = []
    for ref_set, hyp in zip(refs, hyps):
        hyp_text = " ".join(hyp)
        best = max(scorer.score(" ".join(ref), hyp_text)["rougeL"].fmeasure for ref in ref_set)
        scores.append(best)
    return round(float(np.mean(scores)), 4)


def _compute_meteor(refs: list[list[list[str]]], hyps: list[list[str]]) -> float:
    from nltk.translate.meteor_score import meteor_score

    return round(float(np.mean([meteor_score(ref_set, hyp) for ref_set, hyp in zip(refs, hyps)])), 4)


def _compute_cider(refs: list[list[list[str]]], hyps: list[list[str]]) -> float:
    try:
        from pycocoevalcap.cider.cider import Cider

        if len(refs) != len(hyps):
            raise ValueError(f"Số references ({len(refs)}) khác số hypotheses ({len(hyps)})")
        if len(hyps) < 2:
            print("[WARN] CIDEr cần ít nhất vài ảnh để tính document frequency ổn định.")

        gts: dict[int, list[str]] = {}
        res: dict[int, list[str]] = {}
        for idx, (ref_set, hyp) in enumerate(zip(refs, hyps)):
            ref_captions = [" ".join(ref).strip() for ref in ref_set]
            hyp_caption = " ".join(hyp).strip()
            if not ref_captions or any(not caption for caption in ref_captions):
                raise ValueError(f"Reference rỗng tại key={idx}")
            if not hyp_caption:
                raise ValueError(f"Hypothesis rỗng tại key={idx}")
            # Direct pycocoevalcap.cider.Cider expects list[str] values.
            # The COCO JSON API uses {"caption": "..."} dicts, but that format
            # raises AttributeError when passed directly to Cider().compute_score().
            gts[idx] = ref_captions
            res[idx] = [hyp_caption]
        if set(gts) != set(res):
            raise ValueError(
                "Key references/hypotheses không khớp: "
                f"chỉ có trong references={set(gts) - set(res)}, "
                f"chỉ có trong hypotheses={set(res) - set(gts)}"
            )
        score, _ = Cider().compute_score(gts, res)
        return round(float(score), 4)
    except Exception as exc:
        print(f"[WARN] Không tính được CIDEr bằng pycocoevalcap: {exc}")
        return 0.0


def run_evaluation(
    decode_mode: str = "beam",
    beam_width: int = 3,
    n_samples: int = 0,
    save_dir: str | None = None,
) -> dict[str, Any]:
    """Run caption generation and compute BLEU, ROUGE-L, METEOR and CIDEr."""
    save_path = Path(save_dir or config.ARTIFACTS_DIR)
    save_path.mkdir(parents=True, exist_ok=True)

    tokenizer = load_tokenizer(config.TOKENIZER_PATH)
    meta = load_meta(config.META_PATH)
    model = _load_model()

    df = preprocess_text(load_captions(config.CAPTION_FILE))
    _, _, test_df = split_by_image(df, config.VAL_SPLIT, config.TEST_SPLIT, config.SEED)
    test_images = test_df["image"].unique().tolist()

    if os.path.exists(config.FEATURES_PATH):
        features = load_features(config.FEATURES_PATH)
    else:
        print(f"[WARN] Không tìm thấy {config.FEATURES_PATH}. Đang tự động trích xuất đặc trưng cho tập Test...")
        extractor = build_feature_extractor(int(meta["img_size"]))
        features = extract_all_features(
            extractor,
            test_images,
            config.IMAGES_DIR,
            int(meta["img_size"]),
        )
    if n_samples > 0:
        test_images = test_images[:n_samples]

    references: dict[str, list[str]] = defaultdict(list)
    selected = set(test_images)
    for _, row in test_df.iterrows():
        if row["image"] in selected:
            references[str(row["image"])].append(str(row["caption"]))

    predictions: dict[str, str] = {}
    max_length = int(meta["max_length"])
    for image_id in tqdm(test_images, desc=f"Evaluating {decode_mode}"):
        feature = features.get(image_id)
        if feature is None:
            continue
        if decode_mode == "greedy" or beam_width <= 1:
            pred = greedy_predict(model, tokenizer, np.asarray(feature), max_length)
        else:
            pred = beam_search_predict(model, tokenizer, np.asarray(feature), max_length, beam_width)
        predictions[image_id] = pred

    image_ids = [img for img in test_images if img in predictions]
    refs_tok = [[_tokens(ref) for ref in references[img]] for img in image_ids]
    hyps_tok = [_tokens(predictions[img]) for img in image_ids]

    results: dict[str, Any] = {}
    results.update(_compute_bleu(refs_tok, hyps_tok))
    results["ROUGE-L"] = _compute_rouge(refs_tok, hyps_tok)
    results["METEOR"] = _compute_meteor(refs_tok, hyps_tok)
    results["CIDEr"] = _compute_cider(refs_tok, hyps_tok)
    results["decode_mode"] = decode_mode
    results["beam_width"] = beam_width
    results["n_test_images"] = len(image_ids)

    with open(save_path / "evaluation_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    rows = [
        {
            "image": img,
            "prediction": predictions[img],
            "references": " | ".join(clean_caption(ref) for ref in references[img]),
        }
        for img in image_ids
    ]
    pd.DataFrame(rows).to_csv(save_path / "eval_predictions.csv", index=False, encoding="utf-8-sig")

    print("\nEvaluation results")
    for key, value in results.items():
        print(f"{key:>14}: {value}")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--decode-mode", choices=["greedy", "beam"], default="beam")
    parser.add_argument("--beam-width", type=int, default=3)
    parser.add_argument("--n-samples", type=int, default=0)
    args = parser.parse_args()
    run_evaluation(args.decode_mode, args.beam_width, args.n_samples)

"""One-command evaluation, plotting and report generation."""

from __future__ import annotations

import json
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from backend import config
from backend.evaluate import run_evaluation


def plot_loss_curve() -> None:
    """Plot train and validation loss from training_history.json."""
    if not os.path.exists(config.HISTORY_PATH):
        print("[WARN] Không tìm thấy training_history.json, bỏ qua loss curve.")
        return
    with open(config.HISTORY_PATH, "r", encoding="utf-8") as f:
        history = json.load(f)
    loss = history.get("loss", [])
    val_loss = history.get("val_loss", [])
    if not loss:
        return
    plt.figure(figsize=(8, 5))
    plt.plot(loss, label="Train loss")
    if val_loss:
        plt.plot(val_loss, label="Validation loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss Curve")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(config.LOSS_CURVE_PATH, dpi=150)
    plt.close()


def plot_metric_bars(results: dict, path: str) -> None:
    """Plot a compact metric bar chart."""
    keys = ["BLEU-1", "BLEU-2", "BLEU-3", "BLEU-4", "ROUGE-L", "METEOR", "CIDEr"]
    values = [float(results.get(key, 0)) for key in keys]
    plt.figure(figsize=(9, 5))
    plt.bar(keys, values, color="#2563eb")
    plt.ylim(0, max(1.0, max(values) + 0.05))
    plt.title("Image Captioning Metrics")
    plt.ylabel("Score")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def make_sample_grid() -> None:
    """Create a 9-row visual prediction table when eval_predictions.csv exists."""
    pred_path = Path(config.PREDICTIONS_PATH)
    if not pred_path.exists():
        print("[WARN] Không tìm thấy eval_predictions.csv, bỏ qua sample grid.")
        return
    df = pd.read_csv(pred_path).head(9)
    if df.empty:
        return
    fig, axes = plt.subplots(len(df), 1, figsize=(10, max(4, len(df) * 1.2)))
    if len(df) == 1:
        axes = [axes]
    for ax, (_, row) in zip(axes, df.iterrows()):
        ax.axis("off")
        ref = str(row.get("references", "")).split("|")[0].strip()
        text = f"Image: {row['image']}\nPrediction: {row['prediction']}\nReference: {ref}"
        ax.text(0.01, 0.5, text, va="center", ha="left", fontsize=9)
    plt.tight_layout()
    plt.savefig(config.RESULT_SAMPLES_PATH, dpi=150)
    plt.close()


def make_attention_note() -> None:
    """Create a note image explaining the attention limitation with avg-pooled CNN features."""
    plt.figure(figsize=(9, 3))
    plt.axis("off")
    plt.text(
        0.02,
        0.65,
        "Attention visualization note",
        fontsize=14,
        weight="bold",
    )
    plt.text(
        0.02,
        0.35,
        "This project uses EfficientNetB0 with pooling='avg', producing a 1280-d global vector.\n"
        "Therefore, true spatial heatmaps over image regions are not available.\n"
        "For spatial attention maps, switch the feature extractor to a 7x7 feature map.",
        fontsize=10,
    )
    plt.tight_layout()
    plt.savefig(config.ATTENTION_MAPS_PATH, dpi=150)
    plt.close()


def write_report(results: dict) -> None:
    """Write a short Markdown report under artifacts/."""
    lines = [
        "# Báo cáo kết quả Image Captioning trên Flickr8k",
        "",
        "## Mô hình",
        "",
        "- Encoder: EfficientNetB0 pretrained ImageNet, `include_top=False`, `pooling='avg'`.",
        "- Decoder: Embedding + LSTM + Bahdanau Attention.",
        "- Suy luận: Greedy hoặc Beam Search.",
        "- Chia dữ liệu: 70% train, 15% validation, 15% test theo Image ID.",
        "",
        "## Kết quả",
        "",
        "| Metric | Score |",
        "|---|---:|",
    ]
    for key in ["BLEU-1", "BLEU-2", "BLEU-3", "BLEU-4", "ROUGE-L", "METEOR", "CIDEr"]:
        lines.append(f"| {key} | {float(results.get(key, 0)):.4f} |")
    lines.extend(
        [
            "",
            "## Nhận xét",
            "",
            "Mô hình đáp ứng pipeline chuẩn của bài toán image captioning: tiền xử lý văn bản, "
            "trích xuất đặc trưng CNN, huấn luyện decoder LSTM bằng teacher forcing, sinh caption "
            "và đánh giá bằng các độ đo phổ biến.",
            "",
            "Giới hạn chính: Flickr8k nhỏ, caption có thể chung chung; EfficientNetB0 dùng global average "
            "pooling nên không tạo được spatial attention heatmap thật.",
        ]
    )
    with open(config.FINAL_REPORT_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main() -> None:
    """Run evaluation, plots and report generation."""
    greedy = run_evaluation("greedy", beam_width=1, save_dir=config.ARTIFACTS_DIR)
    beam3 = run_evaluation("beam", beam_width=3, save_dir=config.ARTIFACTS_DIR)
    compare_dir = Path(config.ARTIFACTS_DIR) / "beam5_eval"
    beam5 = run_evaluation("beam", beam_width=5, save_dir=str(compare_dir), n_samples=200)

    comparison = {"greedy": greedy, "beam3": beam3, "beam5_200_samples": beam5}
    with open(Path(config.ARTIFACTS_DIR) / "metric_comparison.json", "w", encoding="utf-8") as f:
        json.dump(comparison, f, ensure_ascii=False, indent=2)

    plot_loss_curve()
    plot_metric_bars(beam3, str(Path(config.ARTIFACTS_DIR) / "metrics_bar.png"))
    make_sample_grid()
    make_attention_note()
    write_report(beam3)
    print(f"Done. Report: {config.FINAL_REPORT_PATH}")


if __name__ == "__main__":
    main()


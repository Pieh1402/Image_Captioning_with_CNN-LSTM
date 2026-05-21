"""Flask web demo for image captioning inference."""

from __future__ import annotations

import os
import sys
import time
import uuid
from pathlib import Path

from flask import Flask, jsonify, render_template, request, send_from_directory, url_for
from werkzeug.exceptions import RequestEntityTooLarge
from werkzeug.utils import secure_filename

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from backend import config


ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
MAX_CONTENT_LENGTH = 10 * 1024 * 1024


def create_app() -> Flask:
    """Create and configure the Flask application."""
    app = Flask(__name__, template_folder="templates", static_folder="static")
    app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH
    app.config["UPLOADS_DIR"] = config.UPLOAD_DIR

    def allowed(filename: str) -> bool:
        return Path(filename.lower()).suffix in ALLOWED_EXTENSIONS

    def model_ready() -> bool:
        required = (config.MODEL_PATH, config.TOKENIZER_PATH, config.META_PATH)
        return all(os.path.exists(path) for path in required)

    @app.errorhandler(RequestEntityTooLarge)
    def file_too_large(_exc):
        return jsonify({"error": "Ảnh quá lớn. Vui lòng chọn ảnh dưới 10MB"}), 413

    @app.get("/")
    def index():
        return render_template("index.html")

    @app.get("/health")
    def health():
        return jsonify({"status": "ok", "model_loaded": model_ready()})

    @app.post("/caption")
    def caption():
        file = request.files.get("file") or request.files.get("image")
        beam_width_raw = request.form.get("beam_width", "3")

        try:
            beam_width = max(1, min(5, int(beam_width_raw)))
        except ValueError:
            beam_width = 3

        if file is None or not file.filename:
            return jsonify({"error": "Bạn chưa chọn ảnh"}), 400

        original_name = secure_filename(file.filename)
        if not allowed(original_name):
            return jsonify({"error": "Chỉ hỗ trợ ảnh JPG, PNG, WEBP"}), 400

        if not model_ready():
            return jsonify({"error": "Model chưa sẵn sàng. Vui lòng train model trước"}), 503

        suffix = Path(original_name).suffix.lower()
        filename = f"{uuid.uuid4().hex}{suffix}"
        save_path = os.path.join(app.config["UPLOADS_DIR"], filename)
        file.save(save_path)

        started = time.perf_counter()
        try:
            from backend.data import load_meta
            from backend.inference import predict_from_path

            pred, attention = predict_from_path(save_path, beam_width=beam_width)
            meta = load_meta(config.META_PATH)
        except (FileNotFoundError, ValueError, OSError, RuntimeError) as exc:
            return jsonify({"error": f"Không tạo được caption: {exc}"}), 500

        process_time = round(time.perf_counter() - started, 2)
        mode = "greedy" if beam_width <= 1 else "beam"
        attention_payload = attention.tolist() if attention is not None else None

        return jsonify(
            {
                "caption": pred,
                "attention_weights": attention_payload,
                "process_time": process_time,
                "mode": mode,
                "beam_width": beam_width,
                "image_url": url_for("uploaded_file", filename=filename),
                "filename": original_name,
                "model": {
                    "architecture": "EfficientNetB0 + LSTM",
                    "vocab_size": int(meta.get("vocab_size", 0)),
                    "max_length": int(meta.get("max_length", 0)),
                    "beam_width": beam_width,
                },
            }
        )

    @app.get("/uploads/<path:filename>")
    def uploaded_file(filename: str):
        return send_from_directory(app.config["UPLOADS_DIR"], filename)

    return app


if __name__ == "__main__":
    create_app().run(host="127.0.0.1", port=5000, debug=True)

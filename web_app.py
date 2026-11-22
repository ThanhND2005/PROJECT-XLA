"""Ứng dụng web cục bộ cho hai quy trình GLCM."""

import argparse
import csv
import io
import json
import math
import os
import shutil
import subprocess
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from tempfile import gettempdir
from threading import Lock
from urllib.parse import urlsplit
from uuid import uuid4

import numpy as np
from PIL import Image, UnidentifiedImageError

from evaluate_glcm_models import FEATURES, ROOT, SOURCES, fit_lda, read_rows
from glcm import analyze, quantize

CLASSES = {
    "blanket": "Chăn",
    "floor": "Sàn",
    "grass": "Cỏ",
    "rice": "Gạo",
    "sand": "Cát",
}
FEATURE_NAMES = (*FEATURES[:3], "energy", *FEATURES[3:])
MAX_BYTES = 12 * 1024 * 1024
MAX_PIXELS = 4_000_000
FORMATS = ("PNG", "JPEG", "WEBP", "BMP", "TIFF")
MACRO = ROOT / "GLCM_Texture_Batch_L16_d1_a0.ijm"
FIJI_LOCK = Lock()
STATIC_FILES = {
    "/": ("index.html", "text/html; charset=utf-8"),
    "/app.css": ("app.css", "text/css; charset=utf-8"),
    "/app.js": ("app.js", "text/javascript; charset=utf-8"),
}


class AnalysisError(ValueError):
    def __init__(self, message, status=422):
        super().__init__(message)
        self.status = status


def sample_path(key):
    label = f"{key}1-without-rotations"
    return ROOT / "dataset_xla" / label / f"{key}1-d-p001.png"


def load_models():
    report = json.loads((ROOT / "glcm_classification_results.json").read_text(encoding="utf-8"))
    labels = [f"{key}1-without-rotations" for key in CLASSES]
    if report["classes"] != labels:
        raise ValueError("Nhãn trong báo cáo đánh giá không khớp với ứng dụng.")
    models = {}
    for source, path in SOURCES.items():
        rows = read_rows(path)
        train = [rows[name] for name in sorted(rows) if rows[name]["group"] in report["train_groups"]]
        x = np.array([row["x"] for row in train])
        y = np.array([labels.index(row["label"]) for row in train])
        mean, scale = x.mean(axis=0), x.std(axis=0)
        scale[scale == 0] = 1
        ridge = report["sources"][source]["LDA"]["chosen_parameter"]
        weights, intercept = fit_lda((x - mean) / scale, y, ridge, len(labels))
        models[source.lower()] = (mean, scale, weights, intercept)
    return models


def decode_image(data):
    try:
        with Image.open(io.BytesIO(data)) as image:
            if image.format not in FORMATS:
                raise AnalysisError("Hãy chọn ảnh PNG, JPG, WebP, BMP hoặc TIFF.")
            if image.width < 2 or image.width * image.height > MAX_PIXELS:
                raise AnalysisError("Ảnh cần rộng từ 2 pixel và tối đa 4 triệu pixel.")
            if getattr(image, "n_frames", 1) != 1:
                raise AnalysisError("Hãy chọn ảnh tĩnh chỉ có một khung hình.")
            return image.convert("L")
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError, SyntaxError) as error:
        raise AnalysisError("Không đọc được ảnh. Hãy kiểm tra định dạng và thử lại.") from error


def extract_python(gray):
    pixels = list(gray.tobytes())
    rows = [pixels[i:i + gray.width] for i in range(0, len(pixels), gray.width)]
    result = analyze(quantize(rows, 16), levels=16, distances=(1,), angles=(0,))["results"][0]
    return result["features"], result["probabilities"]


def find_fiji():
    configured = os.environ.get("GLCM_FIJI_EXE")
    if configured:
        path = Path(configured).expanduser()
        return path.resolve() if path.is_file() else None
    names = ("fiji.exe", "ImageJ-win64.exe", "fiji", "ImageJ-linux64")
    for directory in (ROOT / "Fiji.app", ROOT / "Fiji", ROOT):
        for name in names:
            candidate = directory / name
            if candidate.is_file():
                return candidate
    for name in names:
        executable = shutil.which(name)
        if executable:
            return Path(executable).resolve()
    return None


@contextmanager
def fiji_files(gray):
    workspace = Path(gettempdir()) / f"glcm-fiji-{uuid4().hex}"
    workspace.mkdir(mode=0o777 if os.name == "nt" else 0o700)
    source = workspace / "input"
    image = source / "image.png"
    output = workspace / "features.csv"
    try:
        source.mkdir()
        gray.save(image)
        yield source, output
    finally:
        output.unlink(missing_ok=True)
        image.unlink(missing_ok=True)
        if source.exists():
            source.rmdir()
        workspace.rmdir()


def extract_fiji(gray, executable):
    if executable is None:
        raise AnalysisError("Chưa tìm thấy Fiji. Hãy cài Fiji cùng GLCM Texture 0.4 và đặt GLCM_FIJI_EXE.", 503)
    try:
        with FIJI_LOCK, fiji_files(gray) as (source, output):
            argument = f"{source}|{output}"
            if os.environ.get("GLCM_FIJI_MODE") == "batch":
                command = [str(executable), "-port0", "-batch", str(MACRO), argument]
            else:
                command = [str(executable), "--headless", "--console", "-macro", str(MACRO), argument]
            process = subprocess.run(command, cwd=executable.parent, capture_output=True, timeout=90)
            if process.returncode or not output.is_file():
                log = (process.stderr or process.stdout).decode(errors="replace").strip()
                raise ValueError(log or f"Fiji không tạo CSV (mã thoát {process.returncode}).")
            with output.open(encoding="utf-8-sig", newline="") as stream:
                rows = list(csv.DictReader(stream))
            if len(rows) != 1:
                raise ValueError("Fiji không trả đúng một dòng đặc trưng.")
            row = rows[0]
            if row["status"] == "undefined_correlation":
                raise AnalysisError("Plugin Fiji không xác định được Correlation cho ảnh này.")
            features = {name: float(row[name]) for name in FEATURE_NAMES}
            if row["status"] != "ok" or not all(map(math.isfinite, features.values())):
                raise ValueError("Plugin trả đặc trưng không hợp lệ.")
            return features, None
    except AnalysisError:
        raise
    except (OSError, ValueError, subprocess.TimeoutExpired, KeyError) as error:
        print(f"Lỗi Fiji: {error}", flush=True)
        raise AnalysisError("Fiji không trả được đặc trưng. Hãy kiểm tra plugin GLCM Texture 0.4 và log máy chủ.", 502) from error


def run_analysis(data, method, predict, models, fiji):
    gray = decode_image(data)
    features, matrix = extract_python(gray) if method == "python" else extract_fiji(gray, fiji)
    result = {
        "method": method,
        "width": gray.width,
        "height": gray.height,
        "features": features,
        "matrix": matrix if not predict else None,
    }
    if predict:
        mean, scale, weights, intercept = models[method]
        vector = np.array([features[name] for name in FEATURES])
        scores = weights @ ((vector - mean) / scale) + intercept
        likelihoods = np.exp(scores - scores.max())
        probabilities = likelihoods / likelihoods.sum()
        ranking = sorted(zip(CLASSES, scores, probabilities), key=lambda item: item[1], reverse=True)
        result["ranking"] = [
            {"id": key, "label": CLASSES[key], "score": float(score), "probability": float(probability)}
            for key, score, probability in ranking
        ]
    return result


class WebHandler(BaseHTTPRequestHandler):
    def reply(self, status, body, content_type):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def json(self, status, value):
        self.reply(status, json.dumps(value, ensure_ascii=False, allow_nan=False).encode("utf-8"),
                   "application/json; charset=utf-8")

    def do_GET(self):
        path = urlsplit(self.path).path
        if path in STATIC_FILES:
            name, content_type = STATIC_FILES[path]
            return self.reply(200, (ROOT / "web" / name).read_bytes(), content_type)
        if path == "/api/status":
            return self.json(200, {"fijiConfigured": self.server.fiji is not None})
        key = path.removeprefix("/samples/").removesuffix(".png")
        if path == f"/samples/{key}.png" and key in CLASSES and sample_path(key).is_file():
            return self.reply(200, sample_path(key).read_bytes(), "image/png")
        self.json(404, {"error": "Không tìm thấy tài nguyên."})

    def do_POST(self):
        route = urlsplit(self.path).path.strip("/").split("/")
        if len(route) != 3 or route[0] != "api" or route[1] not in ("predict", "extract") or route[2] not in ("python", "fiji"):
            return self.json(404, {"error": "Không tìm thấy yêu cầu phân tích này."})
        try:
            size = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            return self.json(400, {"error": "Kích thước ảnh không hợp lệ."})
        if size <= 0 or size > MAX_BYTES:
            return self.json(413, {"error": "Hãy chọn ảnh có dung lượng từ 1 byte đến 12 MB."})
        try:
            result = run_analysis(self.rfile.read(size), route[2], route[1] == "predict",
                                  self.server.models, self.server.fiji)
        except AnalysisError as error:
            return self.json(error.status, {"error": str(error)})
        self.json(200, result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    models = load_models()
    with ThreadingHTTPServer((args.host, args.port), WebHandler) as server:
        server.models = models
        server.fiji = find_fiji()
        print(f"Texture Lab: http://{args.host}:{server.server_port}", flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()

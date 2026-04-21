#!/usr/bin/env python3
"""Benchmark de InsightFace con comparativa por det-size y lote de imagenes."""

from __future__ import annotations

import argparse
import statistics
import time
import urllib.request
from pathlib import Path

import cv2
import numpy as np
from insightface.app import FaceAnalysis
from insightface.utils import face_align


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return float("nan")
    sorted_vals = sorted(values)
    idx = int(round((pct / 100.0) * (len(sorted_vals) - 1)))
    return sorted_vals[idx]


def ensure_image(path: Path) -> Path:
    if path.exists():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    url = "https://raw.githubusercontent.com/opencv/opencv/master/samples/data/lena.jpg"
    urllib.request.urlretrieve(url, path.as_posix())
    return path


def load_images(base_image: Path, batch_images: int) -> list[np.ndarray]:
    image_path = ensure_image(base_image)
    img = cv2.imread(image_path.as_posix())
    if img is None:
        raise SystemExit(f"No se pudo leer imagen: {image_path}")
    return [img.copy() for _ in range(batch_images)]


def summarize(values: list[float]) -> tuple[float, float, float]:
    return statistics.mean(values), percentile(values, 50), percentile(values, 95)


def benchmark_once(
    app: FaceAnalysis,
    images: list[np.ndarray],
    max_num: int,
) -> tuple[float, float, float, int]:
    det_model = app.models.get("detection")
    rec_model = app.models.get("recognition")
    if det_model is None or rec_model is None:
        raise RuntimeError("No se encontraron modelos detection/recognition en FaceAnalysis")

    t0 = time.perf_counter()
    det_ms = 0.0
    emb_ms = 0.0
    total_faces = 0

    for img in images:
        td0 = time.perf_counter()
        bboxes, kpss = det_model.detect(img, max_num=max_num)
        td1 = time.perf_counter()
        det_ms += (td1 - td0) * 1000.0

        te0 = time.perf_counter()
        if bboxes is not None and len(bboxes) > 0:
            n_faces = min(len(bboxes), max_num)
            total_faces += n_faces
            for i in range(n_faces):
                kps = kpss[i]
                aligned = face_align.norm_crop(img, landmark=kps)
                _ = rec_model.get_feat(aligned)
        te1 = time.perf_counter()
        emb_ms += (te1 - te0) * 1000.0

    total_ms = (time.perf_counter() - t0) * 1000.0
    return det_ms, emb_ms, total_ms, total_faces


def run_for_det_size(
    det_size: int,
    images: list[np.ndarray],
    warmup: int,
    runs: int,
    max_num: int,
) -> dict[str, float]:
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    app = FaceAnalysis(name="buffalo_l", providers=providers)
    app.prepare(ctx_id=0, det_size=(det_size, det_size))

    for _ in range(warmup):
        benchmark_once(app, images, max_num=max_num)

    det_times: list[float] = []
    emb_times: list[float] = []
    total_times: list[float] = []
    faces: list[int] = []

    for _ in range(runs):
        d, e, t, n = benchmark_once(app, images, max_num=max_num)
        det_times.append(d)
        emb_times.append(e)
        total_times.append(t)
        faces.append(n)

    d_mean, d_p50, d_p95 = summarize(det_times)
    e_mean, e_p50, e_p95 = summarize(emb_times)
    t_mean, t_p50, t_p95 = summarize(total_times)

    batch_size = len(images)
    img_per_sec = batch_size / (t_mean / 1000.0)

    return {
        "det_size": float(det_size),
        "batch_size": float(batch_size),
        "det_mean_ms": d_mean,
        "det_p95_ms": d_p95,
        "emb_mean_ms": e_mean,
        "emb_p95_ms": e_p95,
        "total_mean_ms": t_mean,
        "total_p50_ms": t_p50,
        "total_p95_ms": t_p95,
        "img_per_sec": img_per_sec,
        "avg_faces_per_batch": float(statistics.mean(faces) if faces else 0.0),
        "avg_faces_per_image": float((statistics.mean(faces) / batch_size) if faces else 0.0),
    }


def print_table(results: list[dict[str, float]]) -> None:
    header = (
        "| det_size | batch | det_mean_ms | emb_mean_ms | total_mean_ms | total_p95_ms | "
        "img/s | faces/img |"
    )
    sep = "|---:|---:|---:|---:|---:|---:|---:|---:|"
    print("\n=== Comparative Table ===")
    print(header)
    print(sep)
    for r in results:
        print(
            "| {det_size:.0f} | {batch_size:.0f} | {det_mean_ms:.2f} | {emb_mean_ms:.2f} | "
            "{total_mean_ms:.2f} | {total_p95_ms:.2f} | {img_per_sec:.2f} | {avg_faces_per_image:.2f} |".format(
                **r
            )
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="InsightFace benchmark: det-size sweep + batch mode")
    parser.add_argument("--image", type=Path, default=Path("benchmarks/assets/lena.jpg"))
    parser.add_argument("--det-sizes", type=int, nargs="+", default=[320, 640, 1024])
    parser.add_argument("--batch-images", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=8)
    parser.add_argument("--runs", type=int, default=30)
    parser.add_argument("--max-num", type=int, default=1)
    args = parser.parse_args()

    images = load_images(args.image, args.batch_images)

    print("=== Environment ===")
    print(f"Image source: {args.image}")
    print(f"Resolution: {images[0].shape[1]}x{images[0].shape[0]}")
    print(f"Batch images per run: {len(images)}")
    print(f"Det sizes: {args.det_sizes}")

    results: list[dict[str, float]] = []
    for det_size in args.det_sizes:
        print(f"\nRunning det_size={det_size} ...")
        stats = run_for_det_size(
            det_size=det_size,
            images=images,
            warmup=args.warmup,
            runs=args.runs,
            max_num=args.max_num,
        )
        results.append(stats)

    print_table(results)

    best = max(results, key=lambda x: x["img_per_sec"])
    print("\n=== Best Throughput ===")
    print(
        "det_size={det_size:.0f} | img/s={img_per_sec:.2f} | total_mean_ms={total_mean_ms:.2f} | "
        "total_p95_ms={total_p95_ms:.2f}".format(**best)
    )


if __name__ == "__main__":
    main()

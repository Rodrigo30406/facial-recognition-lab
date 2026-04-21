#!/usr/bin/env python3
"""Simple GPU benchmark for face-recognition workloads.

Measures:
- Torch GEMM throughput (FP32/FP16)
- ONNXRuntime latency on a tiny embedding-like model
"""

from __future__ import annotations

import argparse
import statistics
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return float("nan")
    sorted_vals = sorted(values)
    idx = int(round((pct / 100.0) * (len(sorted_vals) - 1)))
    return sorted_vals[idx]


def benchmark_torch_matmul(
    size: int,
    dtype: torch.dtype,
    warmup: int,
    runs: int,
) -> dict[str, float]:
    device = torch.device("cuda")
    a = torch.randn((size, size), device=device, dtype=dtype)
    b = torch.randn((size, size), device=device, dtype=dtype)

    for _ in range(warmup):
        _ = a @ b
    torch.cuda.synchronize()

    timings_ms: list[float] = []
    for _ in range(runs):
        start = time.perf_counter()
        _ = a @ b
        torch.cuda.synchronize()
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        timings_ms.append(elapsed_ms)

    # Approximate FLOPs for NxN GEMM: 2 * N^3
    flops = 2.0 * (size**3)
    avg_sec = statistics.mean(timings_ms) / 1000.0
    tflops = (flops / avg_sec) / 1e12

    return {
        "mean_ms": statistics.mean(timings_ms),
        "p50_ms": percentile(timings_ms, 50),
        "p95_ms": percentile(timings_ms, 95),
        "tflops": tflops,
    }


def build_embedding_onnx_model(path: Path, in_dim: int = 512, out_dim: int = 512) -> None:
    import onnx
    from onnx import TensorProto, helper

    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [None, in_dim])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [None, out_dim])

    w_data = np.random.randn(in_dim, out_dim).astype(np.float32)
    b_data = np.random.randn(out_dim).astype(np.float32)
    w = helper.make_tensor("W", TensorProto.FLOAT, [in_dim, out_dim], w_data.flatten().tolist())
    b = helper.make_tensor("B", TensorProto.FLOAT, [out_dim], b_data.tolist())

    matmul = helper.make_node("MatMul", ["x", "W"], ["m"])
    add = helper.make_node("Add", ["m", "B"], ["a"])
    relu = helper.make_node("Relu", ["a"], ["y"])

    graph = helper.make_graph([matmul, add, relu], "embedding_graph", [x], [y], [w, b])
    model = helper.make_model(graph, producer_name="gpu_benchmark")
    # ORT 1.22 supports up to IR v10; pin for compatibility.
    model.ir_version = 10
    model.opset_import[0].version = 13
    onnx.save(model, path.as_posix())


def benchmark_onnxruntime(
    batch_size: int,
    in_dim: int,
    warmup: int,
    runs: int,
    providers: list[str],
    model_path: Path,
) -> dict[str, float | str]:
    build_embedding_onnx_model(model_path, in_dim=in_dim, out_dim=512)
    session = ort.InferenceSession(model_path.as_posix(), providers=providers)

    x = np.random.randn(batch_size, in_dim).astype(np.float32)

    for _ in range(warmup):
        _ = session.run(None, {"x": x})

    timings_ms: list[float] = []
    for _ in range(runs):
        start = time.perf_counter()
        _ = session.run(None, {"x": x})
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        timings_ms.append(elapsed_ms)

    throughput = batch_size / (statistics.mean(timings_ms) / 1000.0)

    return {
        "provider": session.get_providers()[0],
        "mean_ms": statistics.mean(timings_ms),
        "p50_ms": percentile(timings_ms, 50),
        "p95_ms": percentile(timings_ms, 95),
        "samples_per_sec": throughput,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="GPU benchmark for facial recognition stack")
    parser.add_argument("--size", type=int, default=4096, help="Matrix size for torch GEMM")
    parser.add_argument("--batch", type=int, default=128, help="Batch size for ONNX benchmark")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--runs", type=int, default=100)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is not available in torch. Activate the correct environment.")

    print("=== Environment ===")
    print(f"Torch version: {torch.__version__}")
    print(f"CUDA runtime: {torch.version.cuda}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Torch arch list: {torch.cuda.get_arch_list()}")
    print(f"ONNX providers: {ort.get_available_providers()}")

    print("\n=== Torch GEMM ===")
    fp32 = benchmark_torch_matmul(args.size, torch.float32, args.warmup, args.runs)
    print(
        "FP32 | mean={mean_ms:.2f}ms p50={p50_ms:.2f}ms p95={p95_ms:.2f}ms ~{tflops:.2f} TFLOPS".format(
            **fp32
        )
    )

    fp16 = benchmark_torch_matmul(args.size, torch.float16, args.warmup, args.runs)
    print(
        "FP16 | mean={mean_ms:.2f}ms p50={p50_ms:.2f}ms p95={p95_ms:.2f}ms ~{tflops:.2f} TFLOPS".format(
            **fp16
        )
    )

    print("\n=== ONNXRuntime Embedding-like Inference ===")
    model_path = Path("benchmarks") / "_tmp_embedding_bench.onnx"
    available = ort.get_available_providers()
    preferred = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    providers = [p for p in preferred if p in available]
    onnx_stats = benchmark_onnxruntime(
        batch_size=args.batch,
        in_dim=512,
        warmup=args.warmup,
        runs=args.runs,
        providers=providers,
        model_path=model_path,
    )
    print(
        "{provider} | mean={mean_ms:.2f}ms p50={p50_ms:.2f}ms p95={p95_ms:.2f}ms ~{samples_per_sec:.0f} samples/s".format(
            **onnx_stats
        )
    )

    if model_path.exists():
        model_path.unlink()


if __name__ == "__main__":
    main()

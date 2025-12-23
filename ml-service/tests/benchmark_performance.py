"""
Performance Benchmark for Food Classifier

Measures:
- Cold start (model loading) time
- Warm inference latency per method
- Memory usage
- Throughput (images/second)
- Latency distribution (min, max, p50, p95, p99)

Run: python tests/benchmark_performance.py
"""

import gc
import os
import sys
import time
import statistics
import tracemalloc
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import httpx
from PIL import Image
import torch

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))


@dataclass
class LatencyStats:
    """Statistics for latency measurements."""

    method: str
    n_runs: int
    mean_ms: float
    std_ms: float
    min_ms: float
    max_ms: float
    p50_ms: float
    p95_ms: float
    p99_ms: float
    throughput_per_sec: float


@dataclass
class MemoryStats:
    """Memory usage statistics."""

    method: str
    peak_mb: float
    current_mb: float


def percentile(data: List[float], p: float) -> float:
    """Calculate percentile of data."""
    sorted_data = sorted(data)
    k = (len(sorted_data) - 1) * p / 100
    f = int(k)
    c = f + 1 if f + 1 < len(sorted_data) else f
    return sorted_data[f] + (k - f) * (sorted_data[c] - sorted_data[f])


def fetch_test_image(url: str) -> Image.Image:
    """Fetch a test image from URL."""
    response = httpx.get(url, follow_redirects=True, timeout=30.0)
    response.raise_for_status()

    # Save temporarily and load
    tmp_path = Path("/tmp/perf_test_image.jpg")
    with open(tmp_path, "wb") as f:
        f.write(response.content)

    return Image.open(tmp_path).convert("RGB")


def measure_cold_start() -> Dict[str, float]:
    """Measure cold start time (importing and loading model)."""
    # Force garbage collection and clear any cached modules
    gc.collect()

    # Remove cached modules
    modules_to_remove = [k for k in sys.modules.keys() if "clip" in k.lower()]
    for mod in modules_to_remove:
        del sys.modules[mod]

    gc.collect()

    # Measure import time
    start = time.perf_counter()
    from app.ml_models.clip_food_classifier import CLIPFoodClassifier

    import_time = time.perf_counter() - start

    # Measure instantiation time
    start = time.perf_counter()
    classifier = CLIPFoodClassifier()
    instantiation_time = time.perf_counter() - start

    # Measure model loading time (first inference triggers lazy load)
    test_image = Image.new("RGB", (224, 224), color="white")
    start = time.perf_counter()
    classifier.classify(test_image, top_k=1)
    first_inference_time = time.perf_counter() - start

    return {
        "import_time_ms": import_time * 1000,
        "instantiation_time_ms": instantiation_time * 1000,
        "first_inference_time_ms": first_inference_time * 1000,
        "total_cold_start_ms": (import_time + instantiation_time + first_inference_time)
        * 1000,
    }


def benchmark_method(
    method_fn: Callable[[Image.Image], Any],
    image: Image.Image,
    n_warmup: int = 3,
    n_runs: int = 20,
    method_name: str = "unknown",
) -> LatencyStats:
    """Benchmark a classification method."""

    # Warmup runs (not measured)
    for _ in range(n_warmup):
        method_fn(image)

    # Measured runs
    latencies = []
    for _ in range(n_runs):
        start = time.perf_counter()
        method_fn(image)
        latency = (time.perf_counter() - start) * 1000  # Convert to ms
        latencies.append(latency)

    return LatencyStats(
        method=method_name,
        n_runs=n_runs,
        mean_ms=statistics.mean(latencies),
        std_ms=statistics.stdev(latencies) if len(latencies) > 1 else 0,
        min_ms=min(latencies),
        max_ms=max(latencies),
        p50_ms=percentile(latencies, 50),
        p95_ms=percentile(latencies, 95),
        p99_ms=percentile(latencies, 99),
        throughput_per_sec=1000 / statistics.mean(latencies),
    )


def measure_memory(
    method_fn: Callable[[Image.Image], Any],
    image: Image.Image,
    method_name: str = "unknown",
) -> MemoryStats:
    """Measure memory usage of a method."""
    gc.collect()

    tracemalloc.start()

    # Run the method
    method_fn(image)

    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    return MemoryStats(
        method=method_name,
        peak_mb=peak / 1024 / 1024,
        current_mb=current / 1024 / 1024,
    )


def benchmark_batch_throughput(
    classifier, images: List[Image.Image], method_name: str = "basic"
) -> Dict[str, float]:
    """Measure throughput for batch processing."""

    n_images = len(images)

    # Sequential processing
    start = time.perf_counter()
    for img in images:
        if method_name == "basic":
            classifier.classify(img, top_k=5)
        elif method_name == "with_tta":
            classifier.classify_with_tta(img, top_k=5)
        elif method_name == "hierarchical":
            classifier.classify_hierarchical(img, top_k=5)
        elif method_name == "enhanced":
            classifier.classify_enhanced(img, top_k=5)

    total_time = time.perf_counter() - start

    return {
        "method": method_name,
        "n_images": n_images,
        "total_time_sec": total_time,
        "throughput_per_sec": n_images / total_time,
        "avg_latency_ms": (total_time / n_images) * 1000,
    }


def get_gpu_memory_usage() -> Optional[Dict[str, float]]:
    """Get GPU memory usage if available."""
    if torch.cuda.is_available():
        return {
            "allocated_mb": torch.cuda.memory_allocated() / 1024 / 1024,
            "reserved_mb": torch.cuda.memory_reserved() / 1024 / 1024,
            "max_allocated_mb": torch.cuda.max_memory_allocated() / 1024 / 1024,
        }
    elif torch.backends.mps.is_available():
        # MPS doesn't have direct memory query, but we can get device info
        return {
            "device": "mps",
            "note": "MPS memory stats not directly available",
        }
    return None


def run_performance_benchmark():
    """Run complete performance benchmark."""
    print("=" * 80)
    print("FOOD CLASSIFIER PERFORMANCE BENCHMARK")
    print("=" * 80)

    # Get device info
    if torch.cuda.is_available():
        device = f"CUDA ({torch.cuda.get_device_name(0)})"
    elif torch.backends.mps.is_available():
        device = "MPS (Apple Silicon)"
    else:
        device = "CPU"
    print(f"\nDevice: {device}")

    # 1. Cold Start Benchmark
    print("\n" + "-" * 80)
    print("1. COLD START BENCHMARK")
    print("-" * 80)

    cold_start = measure_cold_start()
    print(f"  Import time:           {cold_start['import_time_ms']:>8.1f} ms")
    print(f"  Instantiation time:    {cold_start['instantiation_time_ms']:>8.1f} ms")
    print(f"  First inference time:  {cold_start['first_inference_time_ms']:>8.1f} ms")
    print(f"  Total cold start:      {cold_start['total_cold_start_ms']:>8.1f} ms")

    # 2. Load classifier and test image for warm benchmarks
    print("\n" + "-" * 80)
    print("2. LOADING TEST RESOURCES")
    print("-" * 80)

    from app.ml_models.clip_food_classifier import get_clip_classifier

    classifier = get_clip_classifier()

    # Fetch a test image
    test_url = "https://images.unsplash.com/photo-1568901346375-23c9450c58cd?w=640"
    print(f"  Fetching test image: {test_url}")
    test_image = fetch_test_image(test_url)
    print(f"  Image size: {test_image.size}")

    # Create multiple test images for batch benchmark
    test_images = [test_image] * 10  # 10 copies for batch test

    # 3. Warm Latency Benchmark
    print("\n" + "-" * 80)
    print("3. WARM LATENCY BENCHMARK (20 runs each, 3 warmup)")
    print("-" * 80)

    methods = {
        "basic": lambda img: classifier.classify(img, top_k=5),
        "with_tta": lambda img: classifier.classify_with_tta(img, top_k=5),
        "hierarchical": lambda img: classifier.classify_hierarchical(img, top_k=5),
        "enhanced": lambda img: classifier.classify_enhanced(img, top_k=5),
    }

    latency_results = []
    for name, method_fn in methods.items():
        print(f"\n  Benchmarking: {name}")
        stats = benchmark_method(
            method_fn, test_image, n_warmup=3, n_runs=20, method_name=name
        )
        latency_results.append(stats)
        print(f"    Mean: {stats.mean_ms:>7.1f} ms (std: {stats.std_ms:.1f})")
        print(f"    Min:  {stats.min_ms:>7.1f} ms | Max: {stats.max_ms:>7.1f} ms")
        print(
            f"    P50:  {stats.p50_ms:>7.1f} ms | P95: {stats.p95_ms:>7.1f} ms | P99: {stats.p99_ms:>7.1f} ms"
        )
        print(f"    Throughput: {stats.throughput_per_sec:.2f} img/sec")

    # 4. Latency Summary Table
    print("\n" + "-" * 80)
    print("4. LATENCY SUMMARY")
    print("-" * 80)
    print(
        f"{'Method':<15} {'Mean':>8} {'Std':>8} {'P50':>8} {'P95':>8} {'P99':>8} {'Throughput':>12}"
    )
    print("-" * 80)
    for stats in latency_results:
        print(
            f"{stats.method:<15} {stats.mean_ms:>7.1f}ms {stats.std_ms:>7.1f}ms {stats.p50_ms:>7.1f}ms {stats.p95_ms:>7.1f}ms {stats.p99_ms:>7.1f}ms {stats.throughput_per_sec:>10.2f}/s"
        )

    # 5. Memory Benchmark
    print("\n" + "-" * 80)
    print("5. MEMORY USAGE")
    print("-" * 80)

    memory_results = []
    for name, method_fn in methods.items():
        mem_stats = measure_memory(method_fn, test_image, method_name=name)
        memory_results.append(mem_stats)
        print(
            f"  {name:<15}: Peak: {mem_stats.peak_mb:>6.1f} MB, Current: {mem_stats.current_mb:>6.1f} MB"
        )

    # GPU memory if available
    gpu_mem = get_gpu_memory_usage()
    if gpu_mem:
        print(f"\n  GPU Memory:")
        for k, v in gpu_mem.items():
            if isinstance(v, float):
                print(f"    {k}: {v:.1f} MB")
            else:
                print(f"    {k}: {v}")

    # 6. Batch Throughput Benchmark
    print("\n" + "-" * 80)
    print("6. BATCH THROUGHPUT (10 images)")
    print("-" * 80)

    batch_results = []
    for name in ["basic", "with_tta", "hierarchical", "enhanced"]:
        result = benchmark_batch_throughput(classifier, test_images, method_name=name)
        batch_results.append(result)
        print(
            f"  {name:<15}: {result['throughput_per_sec']:>6.2f} img/sec (avg: {result['avg_latency_ms']:.1f}ms)"
        )

    # 7. Image Size Impact
    print("\n" + "-" * 80)
    print("7. IMAGE SIZE IMPACT (basic method)")
    print("-" * 80)

    sizes = [(224, 224), (320, 320), (480, 480), (640, 640), (1024, 1024)]
    size_results = []

    for size in sizes:
        resized = test_image.resize(size)
        stats = benchmark_method(
            lambda img: classifier.classify(img, top_k=5),
            resized,
            n_warmup=2,
            n_runs=10,
            method_name=f"{size[0]}x{size[1]}",
        )
        size_results.append(stats)
        print(
            f"  {size[0]:>4}x{size[1]:<4}: {stats.mean_ms:>7.1f} ms (throughput: {stats.throughput_per_sec:.2f}/s)"
        )

    # 8. Summary & Recommendations
    print("\n" + "=" * 80)
    print("SUMMARY & RECOMMENDATIONS")
    print("=" * 80)

    fastest = min(latency_results, key=lambda x: x.mean_ms)
    slowest = max(latency_results, key=lambda x: x.mean_ms)

    print(f"\n  Fastest method:  {fastest.method} ({fastest.mean_ms:.1f} ms)")
    print(f"  Slowest method:  {slowest.method} ({slowest.mean_ms:.1f} ms)")
    print(f"  Speedup ratio:   {slowest.mean_ms / fastest.mean_ms:.1f}x")

    print("\n  Recommendations:")
    print("  - For real-time: Use 'basic' or 'hierarchical' method")
    print("  - For accuracy:  Use 'enhanced' or 'with_tta' method")
    print("  - For balance:   Use 'with_tta' (good accuracy with reasonable latency)")

    # Check if any method is too slow for real-time
    for stats in latency_results:
        if stats.mean_ms > 500:
            print(
                f"  - WARNING: '{stats.method}' exceeds 500ms, may not be suitable for real-time"
            )

    return {
        "cold_start": cold_start,
        "latency": latency_results,
        "memory": memory_results,
        "batch_throughput": batch_results,
        "size_impact": size_results,
    }


if __name__ == "__main__":
    results = run_performance_benchmark()

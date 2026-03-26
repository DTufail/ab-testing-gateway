"""
Latency benchmark for all three endpoint variants.
Runs N requests per variant and reports p50/p95/p99/mean.
Saves results to benchmarks/phase1_baseline.json.
"""
import argparse
import sys
import os
import json
import time
import statistics
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

import boto3

ENDPOINT_NAME_FILE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".endpoint_name"
)
BENCHMARKS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "benchmarks"
)
OUTPUT_FILE = os.path.join(BENCHMARKS_DIR, "phase1_baseline.json")

TEST_QUERY = "I lost my card and need a new one"


def parse_args():
    parser = argparse.ArgumentParser(description="Measure latency across all endpoint variants")
    parser.add_argument(
        "--endpoint-name",
        default=None,
        help="Endpoint name (default: read from .endpoint_name)",
    )
    parser.add_argument(
        "--n",
        type=int,
        default=50,
        help="Number of requests per variant (default: 50)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output JSON filename under benchmarks/ (default: phase1_baseline.json)",
    )
    return parser.parse_args()


def get_endpoint_name(args):
    if args.endpoint_name:
        return args.endpoint_name
    if os.path.exists(ENDPOINT_NAME_FILE):
        with open(ENDPOINT_NAME_FILE) as f:
            name = f.read().strip()
        if name:
            return name
    print("ERROR: No endpoint name provided and .endpoint_name file not found.")
    print("Pass --endpoint-name or run deploy_endpoint.py first.")
    sys.exit(1)


def percentile(data, p):
    """Compute the p-th percentile of a sorted list."""
    sorted_data = sorted(data)
    k = (len(sorted_data) - 1) * p / 100
    f = int(k)
    c = f + 1
    if c >= len(sorted_data):
        return sorted_data[f]
    return sorted_data[f] + (k - f) * (sorted_data[c] - sorted_data[f])


def benchmark_variant(runtime_client, endpoint_name, variant_name, n):
    """Send n requests to a variant and return latency stats in milliseconds."""
    print(f"  Benchmarking {variant_name} ({n} requests)...")
    payload = json.dumps({"inputs": TEST_QUERY})
    latencies = []

    for i in range(n):
        t0 = time.perf_counter()
        runtime_client.invoke_endpoint(
            EndpointName=endpoint_name,
            TargetVariant=variant_name,
            ContentType="application/json",
            Body=payload,
        )
        t1 = time.perf_counter()
        latencies.append((t1 - t0) * 1000)  # ms

        if (i + 1) % 10 == 0:
            print(f"    {i + 1}/{n} requests completed")

    return {
        "p50": round(percentile(latencies, 50), 2),
        "p95": round(percentile(latencies, 95), 2),
        "p99": round(percentile(latencies, 99), 2),
        "mean": round(statistics.mean(latencies), 2),
    }


def main():
    args = parse_args()
    endpoint_name = get_endpoint_name(args)

    runtime_client = boto3.client("sagemaker-runtime", region_name=config.AWS_REGION)

    print(f"=== Latency Benchmark ===")
    print(f"Endpoint: {endpoint_name}")
    print(f"Requests per variant: {args.n}")
    print(f"Query: \"{TEST_QUERY}\"\n")

    variant_stats = {}
    for variant_name in config.ALL_VARIANTS:
        stats = benchmark_variant(runtime_client, endpoint_name, variant_name, args.n)
        variant_stats[variant_name] = stats

    # Print comparison table
    print("\n" + "=" * 72)
    print(f"{'Variant':<30} {'p50 (ms)':>10} {'p95 (ms)':>10} {'p99 (ms)':>10} {'mean (ms)':>10}")
    print("-" * 72)
    for variant_name in config.ALL_VARIANTS:
        s = variant_stats[variant_name]
        print(f"{variant_name:<30} {s['p50']:>10.1f} {s['p95']:>10.1f} {s['p99']:>10.1f} {s['mean']:>10.1f}")
    print("=" * 72)

    # Save results
    os.makedirs(BENCHMARKS_DIR, exist_ok=True)
    out_file = os.path.join(BENCHMARKS_DIR, args.output) if args.output else OUTPUT_FILE
    output = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "endpoint_name": endpoint_name,
        "n_requests_per_variant": args.n,
        "variants": variant_stats,
    }
    with open(out_file, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nResults saved to: {out_file}")


if __name__ == "__main__":
    main()

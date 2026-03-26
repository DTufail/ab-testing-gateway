"""
04_endpoint_latency_detail.py
Re-run endpoint latency with warm-up and more samples.
Run from repo root: python audit/04_endpoint_latency_detail.py
"""
import sys
import os
import json
import time
import statistics
from datetime import datetime, timezone

print("=== 04_endpoint_latency_detail.py ===")

WARM_UP  = 10
TIMED    = 100
TEST_QUERY = "What is my card limit?"
OUTPUT_FILE = "./benchmarks/phase1_audit_latency.json"

# Variant names (matches deploy_endpoint.py)
VARIANTS = [
    "VariantA-BERT-FP32",
    "VariantB-BERT-INT8",
    "VariantC-DistilBERT",
]


# --- Resolve endpoint name ---
def load_endpoint_name():
    # Try spec-specified path first, then actual repo path, then baseline JSON
    candidates = [
        "./benchmarks/endpoint_name.txt",
        "./.endpoint_name",
        ".endpoint_name",
    ]
    for path in candidates:
        if os.path.exists(path):
            name = open(path).read().strip()
            if name:
                print(f"Endpoint name from {path}: {name}")
                return name

    # Fall back to phase1_baseline.json
    baseline = "./benchmarks/phase1_baseline.json"
    if os.path.exists(baseline):
        with open(baseline) as f:
            data = json.load(f)
        name = data.get("endpoint_name", "").strip()
        if name:
            print(f"Endpoint name from {baseline}: {name}")
            return name

    print("ERROR: Could not find endpoint name.")
    print("  Searched: benchmarks/endpoint_name.txt, .endpoint_name")
    print("  Also checked benchmarks/phase1_baseline.json")
    print("  Pass --endpoint-name <name> or create benchmarks/endpoint_name.txt")
    return None


# --- Check for --endpoint-name flag ---
endpoint_name = None
if "--endpoint-name" in sys.argv:
    idx = sys.argv.index("--endpoint-name")
    if idx + 1 < len(sys.argv):
        endpoint_name = sys.argv[idx + 1]
        print(f"Endpoint name from CLI: {endpoint_name}")

if not endpoint_name:
    endpoint_name = load_endpoint_name()

if not endpoint_name:
    sys.exit(1)

# --- AWS / boto3 ---
try:
    import boto3
except ImportError:
    print("ERROR: boto3 is not installed. Run: pip install boto3")
    sys.exit(1)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    import config as cfg
    region = cfg.AWS_REGION
except Exception:
    region = os.environ.get("AWS_DEFAULT_REGION", "us-east-1")

runtime_client = boto3.client("sagemaker-runtime", region_name=region)
payload = json.dumps({"inputs": TEST_QUERY})

print(f"\nEndpoint: {endpoint_name}")
print(f"Region:   {region}")
print(f"Query:    \"{TEST_QUERY}\"")
print(f"Warm-up:  {WARM_UP} requests per variant (discarded)")
print(f"Timed:    {TIMED} requests per variant\n")


def percentile(data, p):
    sorted_data = sorted(data)
    k = (len(sorted_data) - 1) * p / 100
    f = int(k)
    c = f + 1
    if c >= len(sorted_data):
        return sorted_data[f]
    return sorted_data[f] + (k - f) * (sorted_data[c] - sorted_data[f])


def benchmark_variant(variant_name, warm_up, timed):
    print(f"  [{variant_name}] warming up ({warm_up} requests)...")
    for _ in range(warm_up):
        runtime_client.invoke_endpoint(
            EndpointName=endpoint_name,
            TargetVariant=variant_name,
            ContentType="application/json",
            Body=payload,
        )

    print(f"  [{variant_name}] timing ({timed} requests)...")
    times = []
    for i in range(timed):
        t0 = time.perf_counter()
        runtime_client.invoke_endpoint(
            EndpointName=endpoint_name,
            TargetVariant=variant_name,
            ContentType="application/json",
            Body=payload,
        )
        times.append((time.perf_counter() - t0) * 1000)
        if (i + 1) % 25 == 0:
            print(f"    {i + 1}/{timed} done")

    n = len(times)
    return {
        "p50_ms":  round(statistics.median(times), 2),
        "p95_ms":  round(sorted(times)[int(0.95 * n)], 2),
        "p99_ms":  round(sorted(times)[int(0.99 * n)], 2),
        "mean_ms": round(statistics.mean(times), 2),
    }


# --- Run benchmarks ---
variant_results = {}
for v in VARIANTS:
    try:
        stats = benchmark_variant(v, WARM_UP, TIMED)
        variant_results[v] = stats
        print(f"  → p50={stats['p50_ms']}ms  p95={stats['p95_ms']}ms  p99={stats['p99_ms']}ms  mean={stats['mean_ms']}ms\n")
    except Exception as e:
        print(f"  ERROR benchmarking {v}: {e}")
        variant_results[v] = None

# --- Print comparison table ---
def fmt(ms):
    return f"{ms:.1f}ms"

print(f"\n=== Endpoint Latency ({WARM_UP} warm-up + {TIMED} timed, batch=1) ===")
header = f"{'Variant':<28} {'p50':>8} {'p95':>8} {'p99':>8} {'mean':>8} {'speedup_vs_A':>14}"
print(header)
print("-" * len(header))

fp32_p50 = None
for label, vname in [
    ("BERT FP32",  "VariantA-BERT-FP32"),
    ("BERT INT8",  "VariantB-BERT-INT8"),
    ("DistilBERT", "VariantC-DistilBERT"),
]:
    s = variant_results.get(vname)
    if s is None:
        print(f"{label:<28}  (failed)")
        continue
    if label == "BERT FP32":
        fp32_p50 = s["p50_ms"]
    speedup = f"{fp32_p50 / s['p50_ms']:.2f}x" if fp32_p50 and s["p50_ms"] > 0 else "N/A"
    print(f"{label:<28} {fmt(s['p50_ms']):>8} {fmt(s['p95_ms']):>8} {fmt(s['p99_ms']):>8} {fmt(s['mean_ms']):>8} {speedup:>14}")

# --- Network overhead estimate ---
print("\nNetwork overhead estimate (endpoint p50 - local p50):")
print("  Run 02_local_latency_compare.py first to get local p50 numbers.")
print("  Formula: endpoint_p50 - local_p50 = SageMaker serialization + routing overhead")

for label, vname in [
    ("BERT FP32",  "VariantA-BERT-FP32"),
    ("BERT INT8",  "VariantB-BERT-INT8"),
    ("DistilBERT", "VariantC-DistilBERT"),
]:
    s = variant_results.get(vname)
    if s:
        print(f"  {label}: endpoint p50 = {s['p50_ms']}ms  (subtract local p50 from script 02)")

# --- Save results ---
os.makedirs("./benchmarks", exist_ok=True)
output = {
    "timestamp": datetime.now(timezone.utc).isoformat(),
    "endpoint": endpoint_name,
    "warm_up_requests": WARM_UP,
    "timed_requests": TIMED,
    "variants": variant_results,
}
with open(OUTPUT_FILE, "w") as f:
    json.dump(output, f, indent=2)
print(f"\nResults saved to: {OUTPUT_FILE}")

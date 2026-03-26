"""
Pull per-variant ModelLatency and OverheadLatency from the AWS/SageMaker
CloudWatch namespace and save to benchmarks/phase1_sagemaker_native_latency.json.

These are SageMaker's own measurements: pure container inference time and
SageMaker routing overhead, isolated from API Gateway and Lambda noise.

Run once after 30+ minutes of endpoint traffic to get stable numbers.
These feed into the Phase 5 benchmark narrative.

Usage:
    python scripts/pull_sagemaker_native_latency.py

Output:
    benchmarks/phase1_sagemaker_native_latency.json
"""
import boto3
import json
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# Read endpoint name — try both known locations
for candidate in (REPO_ROOT / ".endpoint_name",
                  REPO_ROOT / "benchmarks" / "endpoint_name.txt"):
    if candidate.exists():
        endpoint_name = candidate.read_text().strip()
        break
else:
    sys.exit(
        "ERROR: endpoint name not found. "
        "Expected .endpoint_name or benchmarks/endpoint_name.txt. "
        "Run Phase 1 first."
    )

cw = boto3.client("cloudwatch")
now   = datetime.now(timezone.utc)
start = now - timedelta(hours=1)

VARIANTS = [
    "VariantA-BERT-FP32",
    "VariantB-BERT-INT8",
    "VariantC-DistilBERT",
]

print(f"Pulling SageMaker native latency metrics...")
print(f"Endpoint : {endpoint_name}")
print(f"Window   : last 1 hour ({start.strftime('%H:%M')} → {now.strftime('%H:%M')} UTC)\n")

results = {}
for variant in VARIANTS:
    for metric_name in ["ModelLatency", "OverheadLatency"]:
        resp = cw.get_metric_statistics(
            Namespace="AWS/SageMaker",
            MetricName=metric_name,
            Dimensions=[
                {"Name": "EndpointName", "Value": endpoint_name},
                {"Name": "VariantName",  "Value": variant},
            ],
            StartTime=start,
            EndTime=now,
            Period=3600,
            Statistics=["Average", "Minimum", "Maximum"],
            ExtendedStatistics=["p50", "p95", "p99"],
        )
        points = resp.get("Datapoints", [])
        if not points:
            print(f"  WARNING: No data for {variant}/{metric_name} in last 1 hour.")
            print(f"           Make sure the endpoint has received traffic.")
            results.setdefault(variant, {})[metric_name] = None
            continue

        p = points[0]
        ext = p.get("ExtendedStatistics", {})
        results.setdefault(variant, {})[metric_name] = {
            "avg_us": round(p.get("Average", 0), 1),
            "min_us": round(p.get("Minimum", 0), 1),
            "max_us": round(p.get("Maximum", 0), 1),
            "p50_us": round(ext.get("p50", 0), 1),
            "p95_us": round(ext.get("p95", 0), 1),
            "p99_us": round(ext.get("p99", 0), 1),
            # Convenience ms values for comparison with Phase 1 boto3 benchmarks
            "p50_ms": round(ext.get("p50", 0) / 1000, 1),
            "p95_ms": round(ext.get("p95", 0) / 1000, 1),
        }

# ── Print table ──────────────────────────────────────────────────────────────
print(f"\n=== SageMaker Native Latency (last 1 hour) ===")
print(f"Endpoint: {endpoint_name}\n")
header = f"{'Variant':<25} {'Metric':<18} {'p50 (ms)':<12} {'p95 (ms)':<12} {'avg (ms)'}"
print(header)
print("-" * len(header))
for variant, metrics in results.items():
    for metric_name, vals in metrics.items():
        if vals:
            print(
                f"{variant:<25} {metric_name:<18} "
                f"{vals['p50_ms']:<12} {vals['p95_ms']:<12} "
                f"{vals['avg_us'] / 1000:.1f}"
            )
        else:
            print(f"{variant:<25} {metric_name:<18} {'NO DATA'}")

# ── Save JSON ─────────────────────────────────────────────────────────────────
output_path = REPO_ROOT / "benchmarks" / "phase1_sagemaker_native_latency.json"
with open(output_path, "w") as f:
    json.dump(
        {
            "endpoint_name":  endpoint_name,
            "pulled_at":      now.isoformat(),
            "lookback_hours": 1,
            "note": (
                "Units: _us = microseconds, _ms = milliseconds. "
                "ModelLatency = pure container inference. "
                "OverheadLatency = SageMaker routing overhead."
            ),
            "variants": results,
        },
        f,
        indent=2,
    )
print(f"\nSaved to {output_path}")

"""
Seed the DynamoDB routing config table with initial values derived from
benchmarks/phase1_baseline.json.

Extracts p50 endpoint latency per variant for the latency_cache field.
DistilBERT gets 999.9 if its p50 is missing from the baseline file.

Usage:
    python scripts/seed_dynamodb.py
"""
import json
import sys
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path

import boto3

REPO_ROOT  = Path(__file__).resolve().parent.parent
BENCHMARKS = REPO_ROOT / "benchmarks"

sys.path.insert(0, str(REPO_ROOT))
import config as cfg

TABLE_NAME = "ab-gateway-routing-config"
DISTILBERT_LATENCY_PLACEHOLDER = 999.9

VARIANT_A = cfg.VARIANT_A   # "VariantA-BERT-FP32"
VARIANT_B = cfg.VARIANT_B   # "VariantB-BERT-INT8"
VARIANT_C = cfg.VARIANT_C   # "VariantC-DistilBERT"


def load_p50_latencies() -> dict:
    """Read p50 endpoint latencies from phase1_baseline.json."""
    baseline_path = BENCHMARKS / "phase1_baseline.json"
    if not baseline_path.exists():
        print(f"WARNING: {baseline_path} not found — using placeholder values for all variants.")
        return {}

    with open(baseline_path) as f:
        data = json.load(f)

    variants_data = data.get("variants", {})
    latencies = {}
    for variant in [VARIANT_A, VARIANT_B, VARIANT_C]:
        p50 = variants_data.get(variant, {}).get("p50")
        if p50 is not None:
            latencies[variant] = float(p50)
        else:
            print(f"  WARNING: p50 missing for {variant} — will use placeholder.")

    return latencies


def seed_table() -> None:
    latencies = load_p50_latencies()

    latency_cache = {
        VARIANT_A: latencies.get(VARIANT_A, DISTILBERT_LATENCY_PLACEHOLDER),
        VARIANT_B: latencies.get(VARIANT_B, DISTILBERT_LATENCY_PLACEHOLDER),
        # DistilBERT always gets the placeholder per spec (local p50 not captured)
        VARIANT_C: DISTILBERT_LATENCY_PLACEHOLDER,
    }

    dynamodb = boto3.resource("dynamodb", region_name=cfg.AWS_REGION)
    table    = dynamodb.Table(TABLE_NAME)

    # DynamoDB resource requires Decimal instead of float
    item = {
        "config_id": "active",
        "strategy":  "weighted_random",
        "weights": {
            VARIANT_A: Decimal("0.6"),
            VARIANT_B: Decimal("0.2"),
            VARIANT_C: Decimal("0.2"),
        },
        "latency_cache": {
            k: Decimal(str(v))
            for k, v in latency_cache.items()
        },
        "shadow_target": VARIANT_B,
        "updated_at":    datetime.now(timezone.utc).isoformat(),
        "version":       1,
    }

    table.put_item(Item=item)

    # Read back and print to confirm
    response = table.get_item(Key={"config_id": "active"})
    print("Seeded config item:")
    print(json.dumps(
        {k: (float(v) if hasattr(v, '__float__') and not isinstance(v, str) else
             {ik: float(iv) for ik, iv in v.items()} if isinstance(v, dict) else v)
         for k, v in response["Item"].items()},
        indent=2,
    ))


if __name__ == "__main__":
    print(f"Seeding DynamoDB table: {TABLE_NAME}")
    seed_table()
    print("\nDone.")

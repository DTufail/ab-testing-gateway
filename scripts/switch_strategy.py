"""
CLI to switch the active routing strategy in DynamoDB without redeploying.

Usage:
    python scripts/switch_strategy.py --strategy weighted_random
    python scripts/switch_strategy.py --strategy least_latency
    python scripts/switch_strategy.py --strategy shadow
    python scripts/switch_strategy.py --strategy header_pinned
    python scripts/switch_strategy.py --strategy weighted_random --weights "A=0.8,B=0.1,C=0.1"

Weight shorthand: A = VariantA-BERT-FP32, B = VariantB-BERT-INT8, C = VariantC-DistilBERT
"""
import argparse
import json
import sys
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path

import boto3

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
import config as cfg

TABLE_NAME = "ab-gateway-routing-config"

VALID_STRATEGIES = {"weighted_random", "least_latency", "shadow", "header_pinned"}
WEIGHT_ALIASES = {
    "A": cfg.VARIANT_A,
    "B": cfg.VARIANT_B,
    "C": cfg.VARIANT_C,
}


def parse_weights(weights_str: str) -> dict:
    """Parse 'A=0.8,B=0.1,C=0.1' into variant-keyed dict."""
    result = {}
    for part in weights_str.split(","):
        key, value = part.strip().split("=")
        key = key.strip().upper()
        variant = WEIGHT_ALIASES.get(key, key)
        result[variant] = Decimal(value.strip())
    return result


def normalize_weights(weights: dict) -> dict:
    """Ensure weights sum to 1.0 (normalise Decimal values)."""
    total = sum(weights.values())
    if total == 0:
        raise ValueError("Weights sum to zero.")
    return {k: Decimal(str(round(float(v) / float(total), 6))) for k, v in weights.items()}


def get_current_config(table) -> dict:
    response = table.get_item(Key={"config_id": "active"})
    return response.get("Item", {})


def print_diff(before: dict, after: dict) -> None:
    print("\nChanges:")
    all_keys = set(before) | set(after)
    changed = False
    for k in sorted(all_keys):
        bv = before.get(k)
        av = after.get(k)
        if bv != av:
            changed = True
            print(f"  {k}:")
            print(f"    before: {bv}")
            print(f"    after:  {av}")
    if not changed:
        print("  (no changes)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Switch A/B gateway routing strategy")
    parser.add_argument(
        "--strategy", required=True, choices=sorted(VALID_STRATEGIES),
        help="Routing strategy to activate",
    )
    parser.add_argument(
        "--weights", default=None,
        help="Weight override: 'A=0.8,B=0.1,C=0.1' (only used with weighted_random)",
    )
    args = parser.parse_args()

    if args.strategy == "header_pinned":
        print(
            "NOTE: 'header_pinned' is request-driven — it activates automatically when "
            "the X-Target-Variant header is present and overrides the table strategy. "
            "You can still set 'header_pinned' in the table as the default fallback, "
            "which will behave like weighted_random for requests without the header."
        )

    dynamodb = boto3.resource("dynamodb", region_name=cfg.AWS_REGION)
    table    = dynamodb.Table(TABLE_NAME)

    before = get_current_config(table)
    if not before:
        sys.exit(f"ERROR: No config item found in {TABLE_NAME}. Run seed_dynamodb.py first.")

    updates = {
        "strategy":   args.strategy,
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "version":    int(before.get("version", 0)) + 1,
    }

    if args.weights:
        new_weights = normalize_weights(parse_weights(args.weights))
        # Merge with existing weights for any missing variants
        merged = dict(before.get("weights", {}))
        merged.update(new_weights)
        updates["weights"] = merged

    # Build UpdateExpression
    expr_parts = []
    expr_names = {}
    expr_values = {}
    for i, (key, value) in enumerate(updates.items()):
        placeholder = f"#k{i}"
        val_placeholder = f":v{i}"
        expr_parts.append(f"{placeholder} = {val_placeholder}")
        expr_names[placeholder] = key
        expr_values[val_placeholder] = value

    table.update_item(
        Key={"config_id": "active"},
        UpdateExpression="SET " + ", ".join(expr_parts),
        ExpressionAttributeNames=expr_names,
        ExpressionAttributeValues=expr_values,
    )

    after = get_current_config(table)
    print_diff(before, after)
    print(f"\nStrategy switched to: {args.strategy}")
    print("NOTE: Lambda picks up the new config within 30 seconds (TTL cache).")


if __name__ == "__main__":
    main()

"""
Shift canary variant to CANARY_WEIGHT% of traffic.
Distributes remaining traffic proportionally across other variants.
Stores pre-canary weights for rollback.
"""
import boto3
import json
import logging
import os

logger = logging.getLogger()
logger.setLevel(logging.INFO)

sm            = boto3.client("sagemaker")
ENDPOINT_NAME = os.environ["ENDPOINT_NAME"]
CANARY_WEIGHT = float(os.environ["CANARY_WEIGHT"])   # 0.10


def lambda_handler(event, context):
    variant_name = event["variant_name"]

    # Read current weights from the live endpoint (endpoint is source of truth)
    desc     = sm.describe_endpoint(EndpointName=ENDPOINT_NAME)
    variants = desc["ProductionVariants"]

    # SageMaker weights are relative integers — normalise to fractions
    total_weight = sum(v.get("CurrentWeight", 0) for v in variants) or 1
    current_fractions = {
        v["VariantName"]: v.get("CurrentWeight", 0) / total_weight
        for v in variants
    }

    if variant_name not in current_fractions:
        raise ValueError(
            f"Variant {variant_name!r} not found in endpoint. "
            f"Available: {list(current_fractions.keys())}"
        )

    # Distribute remaining weight proportionally across non-canary variants
    other       = {v: w for v, w in current_fractions.items() if v != variant_name}
    other_total = sum(other.values()) or 1.0
    remaining   = 1.0 - CANARY_WEIGHT

    new_fractions = {variant_name: CANARY_WEIGHT}
    for v, w in other.items():
        new_fractions[v] = round((w / other_total) * remaining, 4)

    # Renormalise to exactly 1.0
    total         = sum(new_fractions.values())
    new_fractions = {v: round(w / total, 4) for v, w in new_fractions.items()}

    # SageMaker accepts float weights; multiply by 100 for readability
    desired = [
        {"VariantName": v, "DesiredWeight": round(w * 100, 2)}
        for v, w in new_fractions.items()
    ]
    sm.update_endpoint_weights_and_capacities(
        EndpointName=ENDPOINT_NAME,
        DesiredWeightsAndCapacities=desired,
    )

    logger.info(json.dumps({
        "action":        "canary_weight_shifted",
        "variant":       variant_name,
        "canary_weight": CANARY_WEIGHT,
        "new_weights":   new_fractions,
    }))

    return {
        **event,
        "pre_canary_weights": current_fractions,
        "canary_weights_set": new_fractions,
    }

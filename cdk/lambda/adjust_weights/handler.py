"""
Adjust endpoint weights for promote or rollback.

Promote: set canary variant to 100%, all others to 0%.
Rollback: restore pre_canary_weights from execution context.

Action is determined by event["action"]: "promote" | "rollback"
"""
import boto3
import json
import logging
import os

logger = logging.getLogger()
logger.setLevel(logging.INFO)

sm            = boto3.client("sagemaker")
ENDPOINT_NAME = os.environ["ENDPOINT_NAME"]


def lambda_handler(event, context):
    action       = event["action"]       # "promote" or "rollback"
    variant_name = event["variant_name"]

    if action == "promote":
        desc     = sm.describe_endpoint(EndpointName=ENDPOINT_NAME)
        variants = [v["VariantName"] for v in desc["ProductionVariants"]]
        weights  = {v: (100.0 if v == variant_name else 0.0) for v in variants}

    elif action == "rollback":
        pre_weights = event.get("pre_canary_weights", {})
        if not pre_weights:
            raise ValueError("pre_canary_weights missing from event — cannot rollback")
        # Convert fractions back to the 0-100 scale SageMaker uses
        weights = {v: round(w * 100, 2) for v, w in pre_weights.items()}

    else:
        raise ValueError(f"Unknown action: {action!r}. Must be 'promote' or 'rollback'.")

    desired = [
        {"VariantName": v, "DesiredWeight": w}
        for v, w in weights.items()
    ]
    sm.update_endpoint_weights_and_capacities(
        EndpointName=ENDPOINT_NAME,
        DesiredWeightsAndCapacities=desired,
    )

    logger.info(json.dumps({
        "action":        action,
        "variant":       variant_name,
        "final_weights": weights,
    }))

    return {
        **event,
        "final_action":  action,
        "final_weights": weights,
    }

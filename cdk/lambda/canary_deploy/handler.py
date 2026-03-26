"""
Canary deployment Lambda — Phase 4a STUB.

This handler is a placeholder that logs the EventBridge event and returns 200.
The real canary deployment logic (validation polling, weight shifting, baking
period, promote/rollback) is implemented in Phase 4b using Step Functions.

Triggered by: EventBridge rule `ab-gateway-model-approved`
Event source:  aws.sagemaker
Detail type:   SageMaker Model Package State Change
"""
import json
import logging
import os

logger = logging.getLogger()
logger.setLevel(logging.INFO)


def lambda_handler(event, context):
    logger.info(json.dumps({
        "source":   "canary-deploy-stub",
        "phase":    "4a",
        "message":  "Phase 4a stub — canary logic not yet implemented",
        "event":    event,
        "env": {
            "ENDPOINT_NAME":           os.environ.get("ENDPOINT_NAME", "NOT_SET"),
            "VALIDATION_PROJECT_NAME": os.environ.get("VALIDATION_PROJECT_NAME", "NOT_SET"),
            "CANARY_WEIGHT":           os.environ.get("CANARY_WEIGHT", "NOT_SET"),
        },
    }))
    return {
        "statusCode": 200,
        "body": json.dumps({
            "phase":   "4a",
            "status":  "STUB_OK",
            "message": "Phase 4a infrastructure deployed. Canary logic in Phase 4b.",
        }),
    }

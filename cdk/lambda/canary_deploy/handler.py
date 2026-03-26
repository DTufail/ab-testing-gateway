"""
Canary deployment Lambda — Phase 4b.

Receives the EventBridge event from SageMaker Model Registry approval,
starts the Step Functions state machine, and returns immediately.
All deployment logic is in the state machine.

Triggered by: EventBridge rule `ab-gateway-model-approved`
Event source:  aws.sagemaker
Detail type:   SageMaker Model Package State Change
"""
import boto3
import json
import logging
import os
from datetime import datetime, timezone

logger = logging.getLogger()
logger.setLevel(logging.INFO)

sfn = boto3.client("stepfunctions")
STATE_MACHINE_ARN = os.environ["STATE_MACHINE_ARN"]


def lambda_handler(event, context):
    now = datetime.now(timezone.utc).isoformat()
    detail = event.get("detail", {})

    model_package_arn = detail.get("ModelPackageArn", "")
    model_group_name  = detail.get("ModelPackageGroupName", "")

    if not model_package_arn or not model_group_name:
        logger.error(f"Missing required fields in event: {json.dumps(event)}")
        return {"statusCode": 400, "body": "Missing ModelPackageArn or ModelPackageGroupName"}

    # Map group name → variant name by stripping suffix
    variant_name = model_group_name.replace("-ModelGroup", "")

    execution_input = json.dumps({
        "model_package_arn":      model_package_arn,
        "model_group_name":       model_group_name,
        "variant_name":           variant_name,
        "triggered_at":           now,
        "validation_build_id":    None,
        "validation_poll_count":  0,
    })

    # Execution name must be unique and <= 80 chars, alphanumeric + hyphens only
    exec_name = f"canary-{variant_name[:20]}-{int(datetime.now(timezone.utc).timestamp())}"

    resp = sfn.start_execution(
        stateMachineArn=STATE_MACHINE_ARN,
        name=exec_name,
        input=execution_input,
    )

    logger.info(json.dumps({
        "source":        "canary-deploy",
        "action":        "state_machine_started",
        "execution_arn": resp["executionArn"],
        "variant":       variant_name,
        "model_package": model_package_arn,
    }))

    return {
        "statusCode": 200,
        "body": json.dumps({
            "execution_arn": resp["executionArn"],
            "variant":       variant_name,
        }),
    }

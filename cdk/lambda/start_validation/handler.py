"""
Start CodeBuild validation job.
Returns the build ID so the state machine can poll it.
"""
import boto3
import logging
import os

logger = logging.getLogger()
logger.setLevel(logging.INFO)

cb = boto3.client("codebuild")
VALIDATION_PROJECT = os.environ["VALIDATION_PROJECT_NAME"]


def lambda_handler(event, context):
    model_package_arn = event["model_package_arn"]
    variant_name      = event["variant_name"]

    resp = cb.start_build(
        projectName=VALIDATION_PROJECT,
        environmentVariablesOverride=[
            {"name": "MODEL_PACKAGE_ARN", "value": model_package_arn, "type": "PLAINTEXT"},
            {"name": "VARIANT_NAME",      "value": variant_name,      "type": "PLAINTEXT"},
        ],
    )
    build_id = resp["build"]["id"]
    logger.info(f"CodeBuild started: {build_id} for variant={variant_name}")

    return {
        **event,
        "validation_build_id":   build_id,
        "validation_poll_count": 0,
    }

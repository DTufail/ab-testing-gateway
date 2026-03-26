"""
Poll CodeBuild job status.
Returns validation_status: SUCCEEDED | IN_PROGRESS | FAILED
State machine loops back to WaitForValidation if IN_PROGRESS.
"""
import boto3
import logging

logger = logging.getLogger()
logger.setLevel(logging.INFO)

cb = boto3.client("codebuild")
MAX_POLLS = 20   # 20 × 3-min wait = 60 minutes max before giving up


def lambda_handler(event, context):
    build_id   = event["validation_build_id"]
    poll_count = event.get("validation_poll_count", 0) + 1

    resp   = cb.batch_get_builds(ids=[build_id])
    build  = resp["builds"][0]
    status = build["buildStatus"]  # SUCCEEDED | FAILED | FAULT | TIMED_OUT | IN_PROGRESS | STOPPED

    logger.info(f"CodeBuild {build_id} status={status} poll={poll_count}")

    # Normalise: map all terminal failures to FAILED
    if status in ("FAULT", "TIMED_OUT", "STOPPED"):
        status = "FAILED"

    # If still running but poll limit exceeded, treat as failed
    if status == "IN_PROGRESS" and poll_count >= MAX_POLLS:
        status = "FAILED"
        logger.error(f"Validation polling limit ({MAX_POLLS}) exceeded — treating as FAILED")

    return {
        **event,
        "validation_status":     status,      # SUCCEEDED | IN_PROGRESS | FAILED
        "validation_poll_count": poll_count,
    }

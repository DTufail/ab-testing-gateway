"""
validate_model.py — Golden dataset validation for AB Gateway canary deployment.

Calls the live SageMaker endpoint with TargetVariant pinned to the variant
being validated. Compares predicted_id (int) against label (int) from the
golden test set. Exits 0 if accuracy >= ACCURACY_THRESHOLD, else exits 1.

Environment variables (all set by CodeBuild project):
    ENDPOINT_NAME           Live SageMaker endpoint name
    VARIANT_NAME            Variant to validate (e.g. VariantB-BERT-INT8)
    ARTIFACT_BUCKET         S3 bucket for golden test set
    GOLDEN_S3_KEY           S3 key of golden_test_set.jsonl
    ACCURACY_THRESHOLD      Float, default 0.80
"""
import boto3
import concurrent.futures
import json
import logging
import os
import sys
import time
from datetime import datetime, timezone

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

# ── Config from environment ───────────────────────────────────────────────────
ENDPOINT_NAME      = os.environ["ENDPOINT_NAME"]
VARIANT_NAME       = os.environ["VARIANT_NAME"]
ARTIFACT_BUCKET    = os.environ["ARTIFACT_BUCKET"]
GOLDEN_S3_KEY      = os.environ.get("GOLDEN_S3_KEY", "validation/golden_test_set.jsonl")
ACCURACY_THRESHOLD = float(os.environ.get("ACCURACY_THRESHOLD", "0.80"))
MAX_WORKERS        = 10
MAX_RETRIES        = 1
RETRY_BACKOFF_S    = 2.0

# ── AWS clients ───────────────────────────────────────────────────────────────
s3      = boto3.client("s3")
runtime = boto3.client("sagemaker-runtime")


def load_golden_set() -> list:
    """Download golden_test_set.jsonl from S3 and parse it."""
    logger.info(f"Loading golden test set from s3://{ARTIFACT_BUCKET}/{GOLDEN_S3_KEY}")
    obj  = s3.get_object(Bucket=ARTIFACT_BUCKET, Key=GOLDEN_S3_KEY)
    body = obj["Body"].read().decode("utf-8")
    examples = [json.loads(line) for line in body.strip().splitlines() if line.strip()]
    logger.info(f"Loaded {len(examples)} examples")
    return examples


def invoke_single(example: dict) -> dict:
    """
    Call the endpoint for one example with retry.
    Returns {"correct": bool, "predicted_id": int|None, "label": int, "error": str|None}
    """
    label = example["label"]
    body  = json.dumps({"inputs": example["text"]})

    for attempt in range(MAX_RETRIES + 1):
        try:
            resp = runtime.invoke_endpoint(
                EndpointName=ENDPOINT_NAME,
                TargetVariant=VARIANT_NAME,
                ContentType="application/json",
                Accept="application/json",
                Body=body,
            )
            result       = json.loads(resp["Body"].read().decode("utf-8"))
            # HuggingFace DLC wraps predict_fn output in a list; unwrap it
            if isinstance(result, list):
                result = result[0]
            predicted_id = result.get("predicted_id")
            return {
                "correct":      predicted_id == label,
                "predicted_id": predicted_id,
                "label":        label,
                "error":        None,
            }
        except Exception as e:
            if attempt < MAX_RETRIES:
                logger.warning(f"Invoke failed (attempt {attempt+1}), retrying: {e}")
                time.sleep(RETRY_BACKOFF_S)
            else:
                logger.error(f"Invoke failed after {MAX_RETRIES+1} attempts: {e}")
                return {
                    "correct":      False,
                    "predicted_id": None,
                    "label":        label,
                    "error":        str(e),
                }


def main():
    start = datetime.now(timezone.utc)
    logger.info(
        f"Starting validation | endpoint={ENDPOINT_NAME} | variant={VARIANT_NAME} | "
        f"threshold={ACCURACY_THRESHOLD}"
    )

    examples = load_golden_set()
    if not examples:
        logger.error("Golden test set is empty — cannot validate")
        sys.exit(1)

    # Parallel inference
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {pool.submit(invoke_single, ex): ex for ex in examples}
        for i, future in enumerate(concurrent.futures.as_completed(futures)):
            result = future.result()
            results.append(result)
            if (i + 1) % 50 == 0:
                partial = sum(1 for r in results if r["correct"]) / len(results)
                logger.info(f"Progress: {i+1}/{len(examples)} | partial_accuracy={partial:.4f}")

    # Compute metrics
    total    = len(results)
    correct  = sum(1 for r in results if r["correct"])
    errors   = sum(1 for r in results if r["error"] is not None)
    accuracy = correct / total if total > 0 else 0.0
    elapsed  = (datetime.now(timezone.utc) - start).total_seconds()

    logger.info(
        f"Validation complete | accuracy={accuracy:.4f} | correct={correct}/{total} | "
        f"invoke_errors={errors} | elapsed={elapsed:.1f}s"
    )

    # Abort if >50% of calls failed — something is seriously wrong
    if errors / total > 0.50:
        logger.error(
            f"ABORT: {errors}/{total} invoke calls failed (>{50}%). "
            "Check endpoint health and IAM permissions."
        )
        sys.exit(1)

    passed = accuracy >= ACCURACY_THRESHOLD

    validation_result = {
        "variant":            VARIANT_NAME,
        "endpoint_name":      ENDPOINT_NAME,
        "total_examples":     total,
        "correct":            correct,
        "invoke_errors":      errors,
        "accuracy":           accuracy,
        "accuracy_threshold": ACCURACY_THRESHOLD,
        "passed":             passed,
        "elapsed_seconds":    elapsed,
        "timestamp":          start.isoformat(),
    }

    with open("validation_result.json", "w") as f:
        json.dump(validation_result, f, indent=2)
    logger.info(f"validation_result.json written: {validation_result}")

    if not passed:
        logger.error(
            f"GATE FAILED: accuracy={accuracy:.4f} < threshold={ACCURACY_THRESHOLD}"
        )
        sys.exit(1)

    logger.info(f"GATE PASSED: accuracy={accuracy:.4f} >= threshold={ACCURACY_THRESHOLD}")
    sys.exit(0)


if __name__ == "__main__":
    main()

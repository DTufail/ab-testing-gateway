"""
Traffic Controller Lambda handler.

Triggered every 15 minutes by EventBridge. Reads CloudWatch metrics for all
3 variants, scores them, applies dampening and floor, then writes updated
weights back to DynamoDB. The Router Lambda picks up the change within 30
seconds via its TTL cache, closing the feedback loop.

Why idempotency guard at 10 minutes (not 15):
    EventBridge guarantees at-least-once delivery. Two invocations 14 minutes
    apart are separate cycles and should both run. Two invocations 3 minutes
    apart are likely a retry. 10-minute guard catches retries while allowing
    genuinely close cycles.

Why update_item (not put_item):
    put_item would overwrite the entire DynamoDB item, losing the `strategy`
    and `shadow_target` fields that the Router Lambda depends on. update_item
    with specific expression attributes is the safe contract.
"""
import boto3
import json
import os
import logging
from datetime import datetime, timezone
from decimal import Decimal

from metrics_reader import collect_all_metrics
from scorer import compute_scores, apply_dampening_and_floor, validate_weights

logger = logging.getLogger()
logger.setLevel(logging.INFO)


def _json_default(obj):
    """
    Custom JSON encoder for types that boto3/DynamoDB return but json.dumps
    cannot handle natively.

    - Decimal: boto3's DynamoDB resource always returns numbers as Decimal.
      Return int for whole numbers (Decimal("1") → 1), float otherwise
      (Decimal("1.5") → 1.5). This preserves type fidelity in logs.
    - datetime: belt-and-suspenders in case any timestamp object leaks in
      from CloudWatch or DynamoDB responses.
    - Anything else: re-raise TypeError so genuine bugs are still surfaced,
      not silently swallowed.
    """
    if isinstance(obj, Decimal):
        return int(obj) if obj % 1 == 0 else float(obj)
    if isinstance(obj, datetime):
        return obj.isoformat()
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")

TABLE_NAME    = os.environ["ROUTING_CONFIG_TABLE"]
ENDPOINT_NAME = os.environ["SAGEMAKER_ENDPOINT_NAME"]

_dynamodb = boto3.resource("dynamodb")
_table    = _dynamodb.Table(TABLE_NAME)


def lambda_handler(event, context):
    """
    Traffic Controller entry point.

    Steps:
    1. Idempotency check: skip if last run was < 10 minutes ago
    2. Read current routing config from DynamoDB
    3. Collect metrics from CloudWatch (ABGateway + AWS/SageMaker)
    4. Compute new weights with scoring + dampening + floor
    5. Validate weights
    6. Write updated config back to DynamoDB (update_item, never put_item)
    7. Emit structured decision log (queryable via Logs Insights)
    """
    now     = datetime.now(timezone.utc)
    now_iso = now.isoformat()

    # ── Step 1: Idempotency guard ──────────────────────────────────────────
    config_resp = _table.get_item(Key={"config_id": "active"})
    config = config_resp.get("Item", {})

    last_run_str = config.get("last_controller_run")
    if last_run_str:
        try:
            last_run = datetime.fromisoformat(last_run_str.replace("Z", "+00:00"))
            minutes_since = (now - last_run).total_seconds() / 60
            if minutes_since < 10:
                logger.info(json.dumps({
                    "source": "traffic-controller",
                    "event": "DUPLICATE_INVOCATION_SKIPPED",
                    "minutes_since_last_run": round(minutes_since, 2),
                    "timestamp": now_iso,
                }, default=_json_default))
                return {"statusCode": 200, "body": "DUPLICATE_INVOCATION_SKIPPED"}
        except Exception as e:
            logger.warning(f"Could not parse last_controller_run: {e}")

    current_weights = {
        k: float(v)
        for k, v in config.get("weights", {}).items()
    }
    current_version = int(config.get("version", 1))

    # ── Step 2: Collect metrics from CloudWatch ────────────────────────────
    try:
        all_metrics = collect_all_metrics(ENDPOINT_NAME)
    except Exception as e:
        logger.error(f"Metrics collection failed: {e}")
        # Update last_controller_run to prevent rapid retries on error
        _table.update_item(
            Key={"config_id": "active"},
            UpdateExpression="SET last_controller_run = :t",
            ExpressionAttributeValues={":t": now_iso},
        )
        raise

    # ── Step 3: Score variants ─────────────────────────────────────────────
    raw_scores, skip_log, killed_log = compute_scores(all_metrics, current_weights)
    new_weights = apply_dampening_and_floor(raw_scores, current_weights)

    if not validate_weights(new_weights):
        logger.error(f"Weight validation failed: {new_weights}")
        raise ValueError(f"Invalid weights computed: {new_weights}")

    # ── Step 4: Build updated latency_cache ───────────────────────────────
    # Prefer SageMaker native ModelLatency (µs → ms) for cleaner signal.
    # Falls back to EMF p50 if native metrics unavailable.
    # If neither available, keep existing cached value — never zero it out.
    updated_latency_cache = dict(config.get("latency_cache", {}))
    for variant, m in all_metrics.items():
        native_p50_us = m.get("model_latency_p50_us")
        if native_p50_us is not None:
            updated_latency_cache[variant] = round(native_p50_us / 1000.0, 2)
        elif m.get("p50_ms") is not None:
            updated_latency_cache[variant] = round(m["p50_ms"], 2)

    # ── Step 5: Write to DynamoDB ──────────────────────────────────────────
    new_version = current_version + 1

    def to_decimal(d: dict) -> dict:
        return {k: Decimal(str(v)) for k, v in d.items()}

    _table.update_item(
        Key={"config_id": "active"},
        UpdateExpression="""
            SET weights             = :w,
                latency_cache       = :lc,
                updated_at          = :t,
                version             = :v,
                last_controller_run = :t
        """,
        ExpressionAttributeValues={
            ":w":  to_decimal(new_weights),
            ":lc": to_decimal(updated_latency_cache),
            ":t":  now_iso,
            ":v":  new_version,
        },
    )

    # ── Step 6: Structured decision log ───────────────────────────────────
    # Queryable in Logs Insights with: filter source = "traffic-controller"
    # Also the source for Dashboard Panel 7 (log widget).
    decision_log = {
        "source": "traffic-controller",
        "timestamp": now_iso,
        "config_version": new_version,
        "previous_weights": {k: round(v, 4) for k, v in current_weights.items()},
        "new_weights": {k: round(v, 4) for k, v in new_weights.items()},
        "weight_deltas": {
            k: round(new_weights[k] - current_weights.get(k, 0), 4)
            for k in new_weights
        },
        "updated_latency_cache": {k: round(float(v), 2) for k, v in updated_latency_cache.items()},
        "cycle_metrics": {
            variant: {
                "sample_count":         m.get("sample_count", 0),
                "error_rate":           round(m.get("error_rate", 0), 4),
                "avg_confidence":       round(m.get("avg_confidence", 0), 4),
                "p50_ms":               m.get("p50_ms"),
                "model_latency_p50_us": m.get("model_latency_p50_us"),
            }
            for variant, m in all_metrics.items()
        },
        "skipped_variants": skip_log,
        "killed_variants":  killed_log,
    }
    logger.info(json.dumps(decision_log, default=_json_default))

    return {
        "statusCode": 200,
        "body": json.dumps({
            "new_weights": new_weights,
            "version": new_version,
        }, default=_json_default),
    }

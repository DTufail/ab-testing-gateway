"""
Read CloudWatch metrics for the canary variant over the baking window.
Returns canary_healthy boolean for the state machine health choice.
"""
import boto3
import json
import logging
import os
from datetime import datetime, timezone, timedelta

logger = logging.getLogger()
logger.setLevel(logging.INFO)

cw                   = boto3.client("cloudwatch")
EMF_NAMESPACE        = os.environ["EMF_NAMESPACE"]
ERROR_RATE_THRESHOLD = float(os.environ["ERROR_RATE_THRESHOLD"])   # 0.05
P95_THRESHOLD_MS     = float(os.environ["P95_LATENCY_THRESHOLD"])  # 5000
BAKING_MINUTES       = int(os.environ["CANARY_WAIT_MINUTES"])      # 15


def lambda_handler(event, context):
    variant_name = event["variant_name"]
    end_time     = datetime.now(timezone.utc)
    start_time   = end_time - timedelta(minutes=BAKING_MINUTES + 2)  # +2 min EMF buffer
    period       = int((end_time - start_time).total_seconds())

    def _get_sum(metric_name):
        resp = cw.get_metric_statistics(
            Namespace=EMF_NAMESPACE,
            MetricName=metric_name,
            Dimensions=[
                {"Name": "Variant",   "Value": variant_name},
                {"Name": "Strategy",  "Value": "weighted_random"},
            ],
            StartTime=start_time,
            EndTime=end_time,
            Period=period,
            Statistics=["Sum"],
        )
        pts = resp.get("Datapoints", [])
        return pts[0]["Sum"] if pts else None

    def _get_p95(metric_name):
        resp = cw.get_metric_statistics(
            Namespace=EMF_NAMESPACE,
            MetricName=metric_name,
            Dimensions=[
                {"Name": "Variant",   "Value": variant_name},
                {"Name": "Strategy",  "Value": "weighted_random"},
            ],
            StartTime=start_time,
            EndTime=end_time,
            Period=period,
            ExtendedStatistics=["p95"],
        )
        pts = resp.get("Datapoints", [])
        if not pts:
            return None
        return pts[0].get("ExtendedStatistics", {}).get("p95")

    errors      = _get_sum("ErrorCount")
    invocations = _get_sum("InvocationCount")
    p95_ms      = _get_p95("RequestLatency")

    error_rate = (
        (errors / invocations)
        if (errors is not None and invocations and invocations > 0)
        else None
    )

    # If no data at all, assume healthy — can't penalise a variant with no traffic
    if error_rate is None and p95_ms is None:
        canary_healthy = True
        reason = "NO_DATA — assumed healthy (insufficient traffic during baking)"
    else:
        error_ok       = error_rate is None or error_rate <= ERROR_RATE_THRESHOLD
        latency_ok     = p95_ms     is None or p95_ms     <= P95_THRESHOLD_MS
        canary_healthy = error_ok and latency_ok
        reason = (
            f"error_rate={error_rate} (threshold={ERROR_RATE_THRESHOLD}), "
            f"p95_ms={p95_ms} (threshold={P95_THRESHOLD_MS})"
        )

    logger.info(json.dumps({
        "action":         "canary_health_check",
        "variant":        variant_name,
        "error_rate":     error_rate,
        "p95_ms":         p95_ms,
        "canary_healthy": canary_healthy,
        "reason":         reason,
    }))

    return {
        **event,
        "canary_healthy":    canary_healthy,
        "health_error_rate": error_rate,
        "health_p95_ms":     p95_ms,
        "health_reason":     reason,
    }

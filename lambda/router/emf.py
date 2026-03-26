import json
import time
from typing import Optional

NAMESPACE = "ABGateway"


def emit_request_metrics(
    variant: str,
    strategy: str,
    request_latency_ms: float,
    sagemaker_latency_ms: float,
    dynamodb_latency_ms: float,
    confidence: float,
    predicted_label: str,
    request_id: str,
    input_length: int,
    error: Optional[str] = None,
    is_shadow: bool = False,
    shadow_variant: Optional[str] = None,
    shadow_latency_ms: Optional[float] = None,
) -> None:
    """
    Emit one EMF log line per request.
    Automatically extracted into CloudWatch Metrics by CloudWatch Logs.
    Also queryable as structured log via CloudWatch Logs Insights.
    """
    emf_payload = {
        "_aws": {
            "Timestamp": int(time.time() * 1000),
            "CloudWatchMetrics": [
                {
                    "Namespace": NAMESPACE,
                    "Dimensions": [["Variant", "Strategy"]],
                    "Metrics": [
                        {"Name": "RequestLatency",  "Unit": "Milliseconds"},
                        {"Name": "InvocationCount", "Unit": "Count"},
                        {"Name": "ErrorCount",      "Unit": "Count"},
                    ],
                },
                {
                    "Namespace": NAMESPACE,
                    "Dimensions": [["Variant"]],
                    "Metrics": [
                        {"Name": "ConfidenceScore", "Unit": "None"},
                    ],
                },
                {
                    "Namespace": NAMESPACE,
                    "Dimensions": [[]],
                    "Metrics": [
                        {"Name": "DynamoDBReadLatency", "Unit": "Milliseconds"},
                    ],
                },
            ],
        },
        # Dimensions
        "Variant":  variant,
        "Strategy": strategy,
        # Metrics
        "RequestLatency":      request_latency_ms,
        "InvocationCount":     1,
        "ErrorCount":          1 if error else 0,
        "ConfidenceScore":     confidence,
        "DynamoDBReadLatency": dynamodb_latency_ms,
        # High-cardinality properties (queryable in Logs Insights, not turned into metrics)
        "request_id":           request_id,
        "variant_selected":     variant,
        "strategy":             strategy,
        "input_length":         input_length,
        "predicted_label":      predicted_label,
        "total_latency_ms":     request_latency_ms,
        "sagemaker_latency_ms": sagemaker_latency_ms,
        "dynamodb_latency_ms":  dynamodb_latency_ms,
        "is_shadow":            is_shadow,
        "shadow_variant":       shadow_variant,
        "error":                error,
    }

    # Add shadow metrics if applicable
    if is_shadow and shadow_latency_ms is not None:
        emf_payload["_aws"]["CloudWatchMetrics"].append({
            "Namespace": NAMESPACE,
            "Dimensions": [["Variant"]],
            "Metrics": [{"Name": "ShadowLatency", "Unit": "Milliseconds"}],
        })
        emf_payload["ShadowLatency"] = shadow_latency_ms

    # Print to stdout — CloudWatch Logs picks this up automatically in Lambda
    print(json.dumps(emf_payload))

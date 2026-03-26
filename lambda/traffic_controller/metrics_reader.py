"""
CloudWatch metric reads for the Traffic Controller.

Reads from two namespaces:
  - ABGateway       : custom EMF metrics published by the Router Lambda
  - AWS/SageMaker   : built-in SageMaker metrics (free, 1-min resolution)

Design notes
------------
Why 20-minute lookback with Period=1200:
    CloudWatch EMF has ~2-minute ingestion delay. A 15-minute window risks
    reading incomplete data from the most recent EventBridge cycle. 20 minutes
    guarantees the full last cycle is always fully ingested before the
    Controller reads it.

Why strategy = "weighted_random" only:
    header_pinned and shadow invocations don't reflect organic traffic
    distribution. Scoring on those would bias weights toward whatever the
    human operator pinned, not actual variant performance.

All reads handle empty Datapoints gracefully — the endpoint may have been
cold for the full lookback window.
"""
import boto3
from datetime import datetime, timedelta, timezone
from typing import Dict, Any, Optional

cw = boto3.client("cloudwatch")

LOOKBACK_MINUTES = 20       # 20-min window covers the full 15-min EventBridge cycle
                             # plus 2-min EMF ingestion delay buffer
PERIOD_SECONDS   = 1200     # One bucket covering the full lookback window

VARIANTS = [
    "VariantA-BERT-FP32",
    "VariantB-BERT-INT8",
    "VariantC-DistilBERT",
]


def get_emf_metrics(variant: str, end_time: datetime) -> Dict[str, Any]:
    """
    Read from ABGateway namespace (custom EMF metrics from Router Lambda).

    Returns:
        {p50_ms, mean_ms, error_rate, avg_confidence, sample_count, error_count}

    Only includes strategy=weighted_random to exclude header_pinned and shadow
    traffic from the scoring signal.
    """
    start_time = end_time - timedelta(minutes=LOOKBACK_MINUTES)

    def _get_stat(metric_name: str, stat: str, extended: bool = False) -> Optional[float]:
        kwargs = dict(
            Namespace="ABGateway",
            MetricName=metric_name,
            Dimensions=[
                {"Name": "Variant",   "Value": variant},
                {"Name": "Strategy",  "Value": "weighted_random"},
            ],
            StartTime=start_time,
            EndTime=end_time,
            Period=PERIOD_SECONDS,
        )
        if extended:
            kwargs["ExtendedStatistics"] = [stat]
        else:
            kwargs["Statistics"] = [stat]
        resp = cw.get_metric_statistics(**kwargs)
        points = resp.get("Datapoints", [])
        if not points:
            return None
        if extended:
            return points[0].get("ExtendedStatistics", {}).get(stat)
        return points[0].get(stat)

    sample_count   = _get_stat("InvocationCount", "Sum")     or 0
    error_count    = _get_stat("ErrorCount",       "Sum")     or 0
    avg_confidence = _get_stat("ConfidenceScore",  "Average") or 0.0
    p50_ms         = _get_stat("RequestLatency",   "p50", extended=True)
    mean_ms        = _get_stat("RequestLatency",   "Average") or 0.0

    return {
        "sample_count":   int(sample_count),
        "error_count":    int(error_count),
        "error_rate":     error_count / sample_count if sample_count > 0 else 0.0,
        "avg_confidence": avg_confidence,
        "p50_ms":         p50_ms,
        "mean_ms":        mean_ms,
    }


def get_sagemaker_model_latency(
    variant: str, endpoint_name: str, end_time: datetime
) -> Dict[str, Optional[float]]:
    """
    Read ModelLatency from AWS/SageMaker namespace.
    Built-in, free, 1-minute resolution.

    Returns:
        {model_latency_p50_us, model_latency_mean_us}

    Units are microseconds — conversion to ms happens in scorer.py.
    Returns None values when no datapoints are available (cold endpoint).
    """
    start_time = end_time - timedelta(minutes=LOOKBACK_MINUTES)
    resp = cw.get_metric_statistics(
        Namespace="AWS/SageMaker",
        MetricName="ModelLatency",
        Dimensions=[
            {"Name": "EndpointName", "Value": endpoint_name},
            {"Name": "VariantName",  "Value": variant},
        ],
        StartTime=start_time,
        EndTime=end_time,
        Period=PERIOD_SECONDS,
        Statistics=["Average"],
        ExtendedStatistics=["p50"],
    )
    points = resp.get("Datapoints", [])
    if not points:
        return {"model_latency_p50_us": None, "model_latency_mean_us": None}
    point = points[0]
    return {
        "model_latency_p50_us":  point.get("ExtendedStatistics", {}).get("p50"),
        "model_latency_mean_us": point.get("Average"),
    }


def collect_all_metrics(endpoint_name: str) -> Dict[str, Dict[str, Any]]:
    """
    Collect metrics for all 3 variants from both namespaces.

    Returns:
        {variant_name: {all fields merged from EMF + SageMaker}}
    """
    now = datetime.now(timezone.utc)
    result = {}
    for variant in VARIANTS:
        emf  = get_emf_metrics(variant, now)
        sagi = get_sagemaker_model_latency(variant, endpoint_name, now)
        result[variant] = {**emf, **sagi}
    return result

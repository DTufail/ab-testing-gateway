import json
import os
import time
import uuid
import threading
from typing import Any, Dict, Optional

import boto3
from botocore.exceptions import ClientError

import config_cache
import emf
import strategies


class ModelUnavailableError(Exception):
    """Raised when the SageMaker endpoint does not exist."""

# Module-level: initialized once at cold start, reused across warm invocations
ENDPOINT_NAME = os.environ["SAGEMAKER_ENDPOINT_NAME"]
_sagemaker_runtime = boto3.client("runtime.sagemaker")


def _invoke_sagemaker(variant: str, text: str) -> Dict[str, Any]:
    """Invoke the SageMaker endpoint for a given variant. Returns parsed response body."""
    try:
        response = _sagemaker_runtime.invoke_endpoint(
            EndpointName=ENDPOINT_NAME,
            ContentType="application/json",
            Body=json.dumps({"inputs": text}),
            TargetVariant=variant,
        )
        body = json.loads(response["Body"].read())
        # HuggingFace PyTorch DLC wraps predict_fn output in a list; unwrap it
        if isinstance(body, list):
            body = body[0]
        return body
    except ClientError as e:
        code = e.response["Error"]["Code"]
        msg  = e.response["Error"]["Message"]
        if code in ("ValidationError", "ValidationException") and "not found" in msg:
            raise ModelUnavailableError(msg)
        raise


def _shadow_invoke(variant: str, text: str, request_id: str) -> None:
    """
    Fire-and-forget shadow invocation. Runs in a background thread.
    Logs result to CloudWatch via EMF. Never raises — all exceptions are caught.
    """
    try:
        t0 = time.perf_counter()
        result = _invoke_sagemaker(variant, text)
        shadow_latency_ms = (time.perf_counter() - t0) * 1000

        emf.emit_request_metrics(
            variant=variant,
            strategy="shadow",
            request_latency_ms=shadow_latency_ms,
            sagemaker_latency_ms=shadow_latency_ms,
            dynamodb_latency_ms=0.0,
            confidence=result.get("confidence", 0.0),
            predicted_label=result.get("predicted_label", ""),
            request_id=request_id,
            input_length=len(text),
            is_shadow=True,
            shadow_variant=variant,
            shadow_latency_ms=shadow_latency_ms,
        )
    except Exception as exc:
        emf.emit_request_metrics(
            variant=variant,
            strategy="shadow",
            request_latency_ms=0.0,
            sagemaker_latency_ms=0.0,
            dynamodb_latency_ms=0.0,
            confidence=0.0,
            predicted_label="",
            request_id=request_id,
            input_length=len(text) if text else 0,
            error=str(exc),
            is_shadow=True,
            shadow_variant=variant,
        )


def lambda_handler(event: Dict[str, Any], context: Any) -> Dict[str, Any]:
    request_id = str(uuid.uuid4())
    total_start = time.perf_counter()

    # Warming ping — return immediately without touching DynamoDB or SageMaker
    if event.get("source") == "warming-ping":
        return {"statusCode": 200, "headers": {"Content-Type": "application/json"}, "body": '{"status":"warm"}'}

    try:
        # Parse body — API Gateway may send it as a string or pre-parsed dict
        raw_body = event.get("body", "{}")
        if isinstance(raw_body, str):
            body = json.loads(raw_body)
        else:
            body = raw_body or {}

        text = body.get("inputs", "")

        # Check for header pin before reading DynamoDB
        headers = event.get("headers") or {}
        # API Gateway lowercases header names
        header_variant = headers.get("x-target-variant") or headers.get("X-Target-Variant")
        pinned = strategies.route_header_pinned(header_variant)

        # Read routing config (with TTL cache)
        config, dynamodb_ms = config_cache.get_routing_config()

        # Select strategy and variant
        if pinned:
            selected_variant = pinned
            active_strategy = "header_pinned"
            shadow_variant: Optional[str] = None
        else:
            active_strategy = config.get("strategy", "weighted_random")
            weights = config.get("weights", {})
            shadow_variant = None

            if active_strategy == "weighted_random":
                selected_variant = strategies.route_weighted_random(weights)
            elif active_strategy == "least_latency":
                selected_variant = strategies.route_least_latency(config.get("latency_cache", {}))
            elif active_strategy == "shadow":
                selected_variant, shadow_variant = strategies.route_shadow(
                    weights, config.get("shadow_target", "VariantB-BERT-INT8")
                )
            else:
                # Unknown strategy — fall back to weighted_random
                selected_variant = strategies.route_weighted_random(weights)
                active_strategy = "weighted_random"

        # Invoke primary variant
        sm_start = time.perf_counter()
        result = _invoke_sagemaker(selected_variant, text)
        sagemaker_ms = (time.perf_counter() - sm_start) * 1000

        predicted_label = result.get("predicted_label", "")
        predicted_id    = result.get("predicted_id", -1)
        confidence      = result.get("confidence", 0.0)

        total_ms = (time.perf_counter() - total_start) * 1000

        # Fire-and-forget shadow invocation if shadow mode is active
        if shadow_variant:
            t = threading.Thread(
                target=_shadow_invoke,
                args=(shadow_variant, text, request_id),
                daemon=True,
            )
            t.start()

        # Emit primary EMF metrics
        emf.emit_request_metrics(
            variant=selected_variant,
            strategy=active_strategy,
            request_latency_ms=total_ms,
            sagemaker_latency_ms=sagemaker_ms,
            dynamodb_latency_ms=dynamodb_ms,
            confidence=confidence,
            predicted_label=predicted_label,
            request_id=request_id,
            input_length=len(text),
            is_shadow=False,
            shadow_variant=shadow_variant,
        )

        response_body = {
            "predicted_label": predicted_label,
            "predicted_id":    predicted_id,
            "confidence":      confidence,
            "variant":         selected_variant,
            "strategy":        active_strategy,
            "request_id":      request_id,
            "latency_ms":      total_ms,
        }

        return {
            "statusCode": 200,
            "headers": {"Content-Type": "application/json"},
            "body": json.dumps(response_body),
        }

    except ModelUnavailableError:
        emf.emit_request_metrics(
            variant="unknown",
            strategy="unknown",
            request_latency_ms=0.0,
            sagemaker_latency_ms=0.0,
            dynamodb_latency_ms=0.0,
            confidence=0.0,
            predicted_label="",
            request_id=request_id,
            input_length=0,
            error="model_unavailable",
        )
        return {
            "statusCode": 503,
            "headers": {"Content-Type": "application/json"},
            "body": json.dumps({
                "error": "model_unavailable",
                "message": "SageMaker endpoint not running",
                "request_id": request_id,
            }),
        }

    except Exception as exc:
        total_ms = (time.perf_counter() - total_start) * 1000

        emf.emit_request_metrics(
            variant="unknown",
            strategy="unknown",
            request_latency_ms=0.0,
            sagemaker_latency_ms=0.0,
            dynamodb_latency_ms=0.0,
            confidence=0.0,
            predicted_label="",
            request_id=request_id,
            input_length=0,
            error=str(exc),
        )

        return {
            "statusCode": 500,
            "headers": {"Content-Type": "application/json"},
            "body": json.dumps({"error": "inference failed", "request_id": request_id}),
        }

import time
import boto3
import os
from typing import Dict, Any, Tuple

TABLE_NAME = os.environ["ROUTING_CONFIG_TABLE"]

# Module-level: initialized once at cold start, reused across invocations
_dynamodb = boto3.resource("dynamodb")
_table    = _dynamodb.Table(TABLE_NAME)

_config_cache: Dict[str, Any] = {"item": None, "fetched_at": 0.0}
CACHE_TTL_SECONDS = 30


def get_routing_config() -> Tuple[Dict[str, Any], float]:
    """
    Returns routing config from DynamoDB with 30-second in-memory TTL.
    On cache hit: returns cached item, zero DynamoDB calls.
    On cache miss: reads DynamoDB, updates cache, returns item.
    Also returns how long the DynamoDB read took (0ms on cache hit).
    """
    now = time.monotonic()
    if _config_cache["item"] and (now - _config_cache["fetched_at"]) < CACHE_TTL_SECONDS:
        return _config_cache["item"], 0.0

    # Cache miss — read DynamoDB
    t0 = time.perf_counter()
    response = _table.get_item(Key={"config_id": "active"})
    dynamodb_ms = (time.perf_counter() - t0) * 1000

    item = response["Item"]

    # DynamoDB returns numeric values as Decimal. Cast to float here so the
    # rest of the codebase (strategies.py arithmetic) always gets native floats.
    if "weights" in item:
        item["weights"] = {k: float(v) for k, v in item["weights"].items()}
    if "latency_cache" in item:
        item["latency_cache"] = {k: float(v) for k, v in item["latency_cache"].items()}

    _config_cache["item"] = item
    _config_cache["fetched_at"] = now
    return item, dynamodb_ms

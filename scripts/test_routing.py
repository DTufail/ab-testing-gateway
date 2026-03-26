"""
Integration test suite for the Phase 2 routing layer.

Reads API URL from benchmarks/phase2_outputs.json.
Sends real HTTP requests to API Gateway.

Usage:
    python scripts/test_routing.py

Exit code 0 if all tests pass, 1 if any fail.
"""
import json
import re
import sys
import time
from decimal import Decimal
from pathlib import Path

import boto3
import requests

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
import config as cfg

OUTPUTS_PATH = REPO_ROOT / "benchmarks" / "phase2_outputs.json"
TABLE_NAME   = "ab-gateway-routing-config"

VARIANT_A = cfg.VARIANT_A  # "VariantA-BERT-FP32"
VARIANT_B = cfg.VARIANT_B  # "VariantB-BERT-INT8"
VARIANT_C = cfg.VARIANT_C  # "VariantC-DistilBERT"
ALL_VARIANTS = set(cfg.ALL_VARIANTS)

SAMPLE_TEXT = "Why was my card declined?"

# Results accumulator
_results: list[dict] = []


def load_outputs() -> dict:
    if not OUTPUTS_PATH.exists():
        sys.exit(f"ERROR: {OUTPUTS_PATH} not found. Run phase2_deploy.py first.")
    with open(OUTPUTS_PATH) as f:
        return json.load(f)


def get_table():
    dynamodb = boto3.resource("dynamodb", region_name=cfg.AWS_REGION)
    return dynamodb.Table(TABLE_NAME)


def set_weights(table, a: float, b: float, c: float) -> None:
    table.update_item(
        Key={"config_id": "active"},
        UpdateExpression="SET #w = :w, strategy = :s",
        ExpressionAttributeNames={"#w": "weights"},
        ExpressionAttributeValues={
            ":w": {
                VARIANT_A: Decimal(str(a)),
                VARIANT_B: Decimal(str(b)),
                VARIANT_C: Decimal(str(c)),
            },
            ":s": "weighted_random",
        },
    )
    # Wait for Lambda cache TTL to expire (30s + buffer)
    time.sleep(35)


def set_strategy(table, strategy: str) -> None:
    table.update_item(
        Key={"config_id": "active"},
        UpdateExpression="SET strategy = :s",
        ExpressionAttributeValues={":s": strategy},
    )
    time.sleep(35)


def post(api_url: str, text: str = SAMPLE_TEXT, headers: dict = None) -> requests.Response:
    h = {"Content-Type": "application/json"}
    if headers:
        h.update(headers)
    return requests.post(api_url, json={"inputs": text}, headers=h, timeout=30)


def record(name: str, passed: bool, detail: str = "") -> None:
    status = "PASS" if passed else "FAIL"
    _results.append({"name": name, "status": status, "detail": detail})
    print(f"  [{status}] {name}" + (f": {detail}" if detail else ""))


# ---------------------------------------------------------------------------
# Test 1 — weighted_random distribution
# ---------------------------------------------------------------------------
def test_weighted_random(api_url: str, table) -> None:
    print("\nTest 1: weighted_random distribution")

    # 100% to Variant A
    set_weights(table, 1.0, 0.0, 0.0)
    variants = [post(api_url).json().get("variant") for _ in range(50)]
    a_count = variants.count(VARIANT_A)
    record("100% weight → all land on Variant A", a_count == 50, f"{a_count}/50")

    # 100% to Variant B
    set_weights(table, 0.0, 1.0, 0.0)
    variants = [post(api_url).json().get("variant") for _ in range(50)]
    b_count = variants.count(VARIANT_B)
    record("100% weight → all land on Variant B", b_count == 50, f"{b_count}/50")

    # Reset to default
    set_weights(table, 0.6, 0.2, 0.2)
    print("  Weights reset to A=0.6, B=0.2, C=0.2")


# ---------------------------------------------------------------------------
# Test 2 — header_pinned
# ---------------------------------------------------------------------------
def test_header_pinned(api_url: str) -> None:
    print("\nTest 2: header_pinned")

    resps = [post(api_url, headers={"X-Target-Variant": VARIANT_A}).json() for _ in range(5)]
    a_hits = sum(1 for r in resps if r.get("variant") == VARIANT_A)
    record("X-Target-Variant: VariantA → all return VariantA", a_hits == 5, f"{a_hits}/5")

    resps = [post(api_url, headers={"X-Target-Variant": VARIANT_B}).json() for _ in range(5)]
    b_hits = sum(1 for r in resps if r.get("variant") == VARIANT_B)
    record("X-Target-Variant: VariantB → all return VariantB", b_hits == 5, f"{b_hits}/5")

    resp = post(api_url, headers={"X-Target-Variant": "InvalidVariantName"})
    record(
        "Invalid header value does not 500 (falls back to weighted_random)",
        resp.status_code == 200,
        f"status={resp.status_code}",
    )


# ---------------------------------------------------------------------------
# Test 3 — least_latency
# ---------------------------------------------------------------------------
def test_least_latency(api_url: str, table) -> None:
    print("\nTest 3: least_latency")

    # Seed cache so VariantB has the lowest latency
    table.update_item(
        Key={"config_id": "active"},
        UpdateExpression="SET latency_cache = :lc, strategy = :s",
        ExpressionAttributeValues={
            ":lc": {
                VARIANT_A: Decimal("1496.04"),
                VARIANT_B: Decimal("131.2"),
                VARIANT_C: Decimal("999.9"),
            },
            ":s": "least_latency",
        },
    )
    time.sleep(35)

    resps = [post(api_url).json() for _ in range(10)]
    b_hits = sum(1 for r in resps if r.get("variant") == VARIANT_B)
    record(
        "least_latency always routes to VariantB (lowest cached latency)",
        b_hits == 10,
        f"{b_hits}/10",
    )


# ---------------------------------------------------------------------------
# Test 4 — shadow mode
# ---------------------------------------------------------------------------
def test_shadow(api_url: str, table) -> None:
    print("\nTest 4: shadow mode")

    set_strategy(table, "shadow")
    resps = [post(api_url).json() for _ in range(5)]

    # Primary variant should be highest-weight (VariantA at 0.6)
    non_shadow_variants = {r.get("variant") for r in resps}
    all_valid = all(v in ALL_VARIANTS for v in non_shadow_variants)
    record("Shadow responses come from a valid primary variant", all_valid, str(non_shadow_variants))

    # Response body must NOT contain shadow predictions
    no_shadow_in_body = all("shadow" not in str(r.get("predicted_label", "")).lower() for r in resps)
    record("Response body does not expose shadow results", no_shadow_in_body)

    # Check CloudWatch Logs for shadow invocations
    # CloudWatch log ingestion typically takes 10-20s after Lambda writes them
    print("  Waiting 20s for CloudWatch log ingestion...")
    time.sleep(20)
    try:
        logs = boto3.client("logs", region_name=cfg.AWS_REGION)
        log_group = "/aws/lambda/ab-gateway-router"
        now_ms = int(time.time() * 1000)
        five_min_ago_ms = now_ms - 5 * 60 * 1000

        events = logs.filter_log_events(
            logGroupName=log_group,
            startTime=five_min_ago_ms,
            endTime=now_ms,
            filterPattern='{ $.is_shadow IS TRUE }',
            limit=10,
        )["events"]

        record(
            "CloudWatch Logs contain shadow invocation entries",
            len(events) > 0,
            f"{len(events)} shadow log events found",
        )
    except Exception as exc:
        record("CloudWatch Logs shadow check", False, str(exc))


# ---------------------------------------------------------------------------
# Test 5 — response schema
# ---------------------------------------------------------------------------
def test_response_schema(api_url: str, table) -> None:
    print("\nTest 5: response schema")

    set_strategy(table, "weighted_random")
    resps = [post(api_url).json() for _ in range(20)]

    uuid4_re = re.compile(
        r"^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$"
    )

    failures = []
    for i, r in enumerate(resps):
        if not isinstance(r.get("predicted_label"), str) or not r["predicted_label"]:
            failures.append(f"resp {i}: predicted_label missing or empty")
        if not isinstance(r.get("predicted_id"), int) or not (0 <= r["predicted_id"] <= 76):
            failures.append(f"resp {i}: predicted_id={r.get('predicted_id')} out of range 0-76")
        if not isinstance(r.get("confidence"), float) or not (0.0 <= r["confidence"] <= 1.0):
            failures.append(f"resp {i}: confidence={r.get('confidence')} out of range")
        if r.get("variant") not in ALL_VARIANTS:
            failures.append(f"resp {i}: invalid variant={r.get('variant')}")
        if not isinstance(r.get("strategy"), str):
            failures.append(f"resp {i}: strategy missing")
        if not uuid4_re.match(r.get("request_id", "")):
            failures.append(f"resp {i}: invalid request_id={r.get('request_id')}")
        if not isinstance(r.get("latency_ms"), (int, float)) or r["latency_ms"] <= 0:
            failures.append(f"resp {i}: latency_ms={r.get('latency_ms')} not positive")

    record("All 20 responses match expected schema", len(failures) == 0, "; ".join(failures[:3]))


# ---------------------------------------------------------------------------
# Test 6 — error handling
# ---------------------------------------------------------------------------
def test_error_handling(api_url: str) -> None:
    print("\nTest 6: error handling")

    # Malformed JSON body
    resp = requests.post(
        api_url,
        data="not-json-{{{",
        headers={"Content-Type": "application/json"},
        timeout=30,
    )
    record("Malformed JSON returns HTTP 500", resp.status_code == 500, f"status={resp.status_code}")

    # Empty inputs string — should be 200 (valid BERT input)
    resp = post(api_url, text="")
    record("Empty inputs string returns HTTP 200", resp.status_code == 200, f"status={resp.status_code}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    outputs = load_outputs()
    api_url = outputs["api_gateway_url"]
    print(f"API URL: {api_url}")

    table = get_table()

    try:
        test_weighted_random(api_url, table)
        test_header_pinned(api_url)
        test_least_latency(api_url, table)
        test_shadow(api_url, table)
        test_response_schema(api_url, table)
        test_error_handling(api_url)
    finally:
        # Always restore default config
        print("\nRestoring default config (weighted_random, A=0.6, B=0.2, C=0.2)...")
        set_weights(table, 0.6, 0.2, 0.2)

    # Summary
    passed = sum(1 for r in _results if r["status"] == "PASS")
    failed = sum(1 for r in _results if r["status"] == "FAIL")
    total  = len(_results)
    print(f"\n{'='*50}")
    print(f"Results: {passed}/{total} passed, {failed} failed")

    if failed > 0:
        print("\nFailed tests:")
        for r in _results:
            if r["status"] == "FAIL":
                print(f"  - {r['name']}: {r['detail']}")
        sys.exit(1)
    else:
        print("All tests passed.")
        sys.exit(0)


if __name__ == "__main__":
    main()

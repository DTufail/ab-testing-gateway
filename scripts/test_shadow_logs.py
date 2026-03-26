"""
Standalone test for shadow mode CloudWatch log check.
Sends 5 requests in shadow mode, waits for log ingestion, then verifies.
"""
import json
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
LOG_GROUP    = "/aws/lambda/ab-gateway-router"

VARIANT_A = cfg.VARIANT_A
VARIANT_B = cfg.VARIANT_B
VARIANT_C = cfg.VARIANT_C
ALL_VARIANTS = set(cfg.ALL_VARIANTS)


def main():
    with open(OUTPUTS_PATH) as f:
        api_url = json.load(f)["api_gateway_url"]
    print(f"API URL: {api_url}")

    dynamodb = boto3.resource("dynamodb", region_name=cfg.AWS_REGION)
    table    = dynamodb.Table(TABLE_NAME)

    # Set shadow strategy
    print("\nSetting strategy=shadow, waiting 35s for Lambda cache TTL...")
    table.update_item(
        Key={"config_id": "active"},
        UpdateExpression="SET strategy = :s",
        ExpressionAttributeValues={":s": "shadow"},
    )
    time.sleep(35)

    # Send 5 requests
    print("Sending 5 requests...")
    resps = []
    for i in range(5):
        r = requests.post(api_url, json={"inputs": "Why was my card declined?"}, timeout=30)
        body = r.json()
        resps.append(body)
        print(f"  [{i+1}] variant={body.get('variant')}  strategy={body.get('strategy')}")

    # Wait for CloudWatch log ingestion
    print("\nWaiting 20s for CloudWatch log ingestion...")
    time.sleep(20)

    # Query CloudWatch
    logs_client = boto3.client("logs", region_name=cfg.AWS_REGION)
    now_ms          = int(time.time() * 1000)
    five_min_ago_ms = now_ms - 5 * 60 * 1000

    events = logs_client.filter_log_events(
        logGroupName=LOG_GROUP,
        startTime=five_min_ago_ms,
        endTime=now_ms,
        filterPattern='{ $.is_shadow IS TRUE }',
        limit=10,
    )["events"]

    print(f"\nShadow log events found: {len(events)}")
    for e in events:
        print(f"  {e['message'][:120]}")

    # Restore default
    print("\nRestoring strategy=weighted_random...")
    table.update_item(
        Key={"config_id": "active"},
        UpdateExpression="SET strategy = :s, #w = :w",
        ExpressionAttributeNames={"#w": "weights"},
        ExpressionAttributeValues={
            ":s": "weighted_random",
            ":w": {
                VARIANT_A: Decimal("0.6"),
                VARIANT_B: Decimal("0.2"),
                VARIANT_C: Decimal("0.2"),
            },
        },
    )

    if len(events) > 0:
        print("\n[PASS] CloudWatch Logs contain shadow invocation entries")
        sys.exit(0)
    else:
        print("\n[FAIL] No shadow log events found")
        sys.exit(1)


if __name__ == "__main__":
    main()

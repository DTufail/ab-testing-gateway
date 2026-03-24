"""
Validate all three endpoint variants return correct, well-formed predictions.
Exits with code 1 if any variant fails.
"""
import argparse
import sys
import os
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

import boto3

ENDPOINT_NAME_FILE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".endpoint_name"
)

TEST_QUERIES = [
    "I lost my card and need a new one",
    "What is the exchange rate for USD to EUR?",
    "Why was my transaction declined?",
    "How do I activate my new card?",
    "I want to cancel my subscription",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Validate all endpoint variants")
    parser.add_argument(
        "--endpoint-name",
        default=None,
        help="Endpoint name (default: read from .endpoint_name)",
    )
    return parser.parse_args()


def get_endpoint_name(args):
    if args.endpoint_name:
        return args.endpoint_name
    if os.path.exists(ENDPOINT_NAME_FILE):
        with open(ENDPOINT_NAME_FILE) as f:
            name = f.read().strip()
        if name:
            return name
    print("ERROR: No endpoint name provided and .endpoint_name file not found.")
    print("Pass --endpoint-name or run deploy_endpoint.py first.")
    sys.exit(1)


def normalize_response(raw):
    """
    Normalize to {"predicted_label": str, "confidence": float}.

    Handles two formats:
      - Custom inference.py:   {"predicted_label": "...", "confidence": 0.xx, ...}
      - HF default pipeline:   [{"label": "...", "score": 0.xx}]
    """
    if isinstance(raw, dict) and "predicted_label" in raw:
        return raw
    if isinstance(raw, list) and raw and "label" in raw[0]:
        return {
            "predicted_label": raw[0]["label"],
            "confidence": float(raw[0]["score"]),
        }
    raise ValueError(f"Unrecognised response format: {raw}")


def validate_variant(runtime_client, endpoint_name, variant_name):
    """Send 5 test queries to a variant and validate responses."""
    print(f"\n  Testing {variant_name}...")
    failures = []

    for i, query in enumerate(TEST_QUERIES):
        payload = json.dumps({"inputs": query})
        try:
            response = runtime_client.invoke_endpoint(
                EndpointName=endpoint_name,
                TargetVariant=variant_name,
                ContentType="application/json",
                Body=payload,
            )
            raw    = json.loads(response["Body"].read())
            result = normalize_response(raw)

            # Assertions
            assert isinstance(result["confidence"], float), "confidence must be a float"
            assert 0.0 <= result["confidence"] <= 1.0, (
                f"confidence {result['confidence']} is out of range [0.0, 1.0]"
            )

            label = result["predicted_label"]
            conf  = result["confidence"]
            print(f"    [{i+1}] \"{query[:45]}...\"")
            print(f"         → {label} (confidence: {conf:.4f})")

        except AssertionError as e:
            failures.append(f"Query {i+1}: {e}")
            print(f"    [{i+1}] ASSERTION FAILED: {e}")
        except Exception as e:
            failures.append(f"Query {i+1}: {e}")
            print(f"    [{i+1}] ERROR: {e}")

    return failures


def main():
    args = parse_args()
    endpoint_name = get_endpoint_name(args)

    runtime_client = boto3.client("sagemaker-runtime", region_name=config.AWS_REGION)

    print(f"=== Validating endpoint: {endpoint_name} ===")
    print(f"Testing {len(TEST_QUERIES)} queries per variant...\n")

    results = {}
    for variant_name in config.ALL_VARIANTS:
        failures = validate_variant(runtime_client, endpoint_name, variant_name)
        results[variant_name] = failures

    # Summary
    print("\n" + "=" * 60)
    print("VALIDATION SUMMARY")
    print("=" * 60)
    any_failed = False
    for variant_name in config.ALL_VARIANTS:
        failures = results[variant_name]
        if failures:
            status = "FAIL"
            any_failed = True
        else:
            status = "PASS"
        print(f"  {variant_name:<30} {status}")
        for failure in failures:
            print(f"    - {failure}")
    print("=" * 60)

    if any_failed:
        print("\nSome variants FAILED. See details above.")
        sys.exit(1)
    else:
        print("\nAll variants PASSED.")


if __name__ == "__main__":
    main()

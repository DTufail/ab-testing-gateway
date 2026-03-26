"""
Traffic generator for Phase 3 — populates CloudWatch metrics so the
Traffic Controller and pull_sagemaker_native_latency.py have real data.

Sends real HTTP requests through API Gateway → Router Lambda → SageMaker.
Each request lands on a variant, emitting EMF metrics (ABGateway namespace)
and incrementing SageMaker's built-in ModelLatency counters.

Usage:
    # Quick warm-up: 50 requests, no delay (< 1 min)
    python3 scripts/generate_traffic.py --requests 50

    # Sustained load: 200 requests, 1 req/sec (~3 min)
    python3 scripts/generate_traffic.py --requests 200 --delay 1

    # Enough to satisfy Traffic Controller MIN_SAMPLES=30 per variant:
    # At 20% weight, B and C each need ~150 total requests to get 30 samples.
    python3 scripts/generate_traffic.py --requests 200 --delay 0.5

    # After running, wait 2 min for CloudWatch EMF ingestion, then:
    #   python3 scripts/pull_sagemaker_native_latency.py
    #   python3 scripts/test_traffic_controller.py
"""
import argparse
import json
import sys
import time
import random
from pathlib import Path

try:
    import requests
except ImportError:
    sys.exit("ERROR: requests not installed. Run: pip install requests")

REPO_ROOT = Path(__file__).resolve().parent.parent

# Load API URL from phase2 outputs
OUTPUTS_PATH = REPO_ROOT / "benchmarks" / "phase2_outputs.json"
if not OUTPUTS_PATH.exists():
    sys.exit(f"ERROR: {OUTPUTS_PATH} not found. Run infra/phase2_deploy.py first.")
with open(OUTPUTS_PATH) as f:
    API_URL = json.load(f)["api_gateway_url"]

# Diverse banking intents for realistic inputs
SAMPLE_TEXTS = [
    "Why was my card declined?",
    "I need to transfer money to another account",
    "What is my current balance?",
    "How do I activate my new credit card?",
    "Can I increase my credit limit?",
    "I was charged twice for the same transaction",
    "How do I set up direct deposit?",
    "I lost my debit card, what should I do?",
    "What are the fees for international transfers?",
    "How do I dispute a charge on my statement?",
    "Can I get a cash advance on my credit card?",
    "How do I change my PIN?",
    "My account has been locked, help please",
    "What is the interest rate on my savings account?",
    "I want to open a new checking account",
    "How do I link my PayPal to my bank account?",
    "What documents do I need for a loan application?",
    "My mortgage payment was not processed",
    "How do I set up automatic bill payments?",
    "Can I get a replacement card sent to a different address?",
]


def send_request(text: str) -> dict:
    resp = requests.post(
        API_URL,
        json={"inputs": text},
        headers={"Content-Type": "application/json"},
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()


def main():
    parser = argparse.ArgumentParser(description="Generate traffic to the A/B gateway")
    parser.add_argument("--requests", type=int, default=150,
                        help="Total number of requests to send (default: 150)")
    parser.add_argument("--delay", type=float, default=0.2,
                        help="Seconds between requests (default: 0.2)")
    args = parser.parse_args()

    n         = args.requests
    delay     = args.delay
    eta_sec   = n * delay
    eta_str   = f"{eta_sec:.0f}s" if eta_sec < 60 else f"{eta_sec/60:.1f}min"

    print(f"Traffic Generator")
    print(f"  API URL  : {API_URL}")
    print(f"  Requests : {n}")
    print(f"  Delay    : {delay}s between requests")
    print(f"  ETA      : ~{eta_str}")
    print(f"  Note     : After finishing, wait 2 min for CloudWatch EMF ingestion\n")

    counts   = {}
    errors   = 0
    start    = time.time()

    for i in range(1, n + 1):
        text = random.choice(SAMPLE_TEXTS)
        try:
            result  = send_request(text)
            variant = result.get("variant", "unknown")
            counts[variant] = counts.get(variant, 0) + 1

            if i % 10 == 0 or i == n:
                elapsed  = time.time() - start
                rps      = i / elapsed
                dist_str = "  ".join(f"{v.split('-')[1]}:{c}" for v, c in sorted(counts.items()))
                print(f"  [{i:>4}/{n}]  {rps:.1f} req/s  |  {dist_str}  |  errors={errors}")

        except Exception as e:
            errors += 1
            print(f"  [{i:>4}/{n}]  ERROR: {e}")

        if delay > 0 and i < n:
            time.sleep(delay)

    elapsed = time.time() - start
    print(f"\nDone. {n} requests in {elapsed:.1f}s ({n/elapsed:.1f} req/s)")
    print(f"Variant distribution: {counts}")
    print(f"Errors: {errors}/{n}")

    if errors == n:
        print("\nERROR: All requests failed. Check your API Gateway URL and endpoint status.")
        sys.exit(1)

    print("\nNext steps:")
    print("  1. Wait ~2 minutes for CloudWatch EMF ingestion")
    print("  2. python3 scripts/pull_sagemaker_native_latency.py")
    print("  3. python3 scripts/test_traffic_controller.py")


if __name__ == "__main__":
    main()

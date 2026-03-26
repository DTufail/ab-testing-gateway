"""
Traffic Controller scoring logic.

Pure functions only — no boto3, no I/O. Fully unit-testable without AWS
credentials. No imports from metrics_reader.py or handler.py.

Scoring formula
---------------
    score = confidence × (1 / normalized_latency) × (1 − error_rate)

Normalized latency = variant_p50 / baseline_p50.
  - Running faster than baseline → normalized_latency < 1 → higher score.
  - Running at baseline → normalized_latency = 1.0 → neutral score.

Design decisions
----------------
MIN_SAMPLES = 30
    At 20% traffic (Variants B and C), each minor variant gets ~6 requests per
    15-minute window. That's statistically meaningless — a single outlier flips
    the score. 30 samples requires sustained traffic to make autonomous decisions.

KILL_SWITCH_ERROR = 0.10
    Zero out a variant above 10% error rate. Immediate protective action.

DAMPENING_FACTOR = 0.30
    30% new + 70% old per cycle. Without dampening, a single noisy window can
    shift 20% of traffic to a lucky variant. With 0.30, weights converge in
    5-7 cycles (~75-105 min) at sustained traffic — stable, evidence-based.

WEIGHT_FLOOR = 0.05
    Never drop a variant below 5%. If a variant reaches zero traffic, its
    CloudWatch metrics go stale and recovery becomes undetectable. The floor
    keeps the measurement signal alive permanently.
"""
from typing import Dict, Optional, Tuple

VARIANTS = [
    "VariantA-BERT-FP32",
    "VariantB-BERT-INT8",
    "VariantC-DistilBERT",
]

MIN_SAMPLES        = 30     # Skip reweighting below this sample count
KILL_SWITCH_ERROR  = 0.10   # Zero out a variant above 10% error rate
DAMPENING_FACTOR   = 0.30   # 30% new + 70% old prevents oscillation
WEIGHT_FLOOR       = 0.05   # Never starve a variant below 5%

# Phase 1 audit local p50 values (ms) — used as baseline for latency normalization.
# VariantC placeholder (999.9) is conservative; replaced once real traffic flows.
BASELINE_LATENCY_MS = {
    "VariantA-BERT-FP32": 405.0,
    "VariantB-BERT-INT8": 131.2,
    "VariantC-DistilBERT": 999.9,
}


def compute_scores(
    metrics: Dict[str, Dict],
    current_weights: Dict[str, float],
) -> Tuple[Dict[str, Optional[float]], Dict[str, int], Dict[str, float]]:
    """
    Compute raw (pre-dampening) target weights from metrics.

    Returns:
        scores      : {variant: float | None}
                        None  = insufficient samples, carry forward current weight
                        0.0   = kill-switch triggered (error rate too high)
                        >0.0  = raw score for normalization
        skip_log    : {variant: sample_count} for variants skipped due to MIN_SAMPLES
        killed_log  : {variant: error_rate} for variants zeroed by kill-switch

    Latency source priority:
        1. SageMaker ModelLatency p50 (µs → ms conversion): cleaner signal,
           not polluted by Lambda cold starts or DynamoDB cache-miss spikes.
        2. EMF RequestLatency p50 (ms, end-to-end): fallback when native
           metrics are unavailable (cold endpoint, no recent traffic).
    """
    scores     = {}
    skip_log   = {}
    killed_log = {}

    for variant in VARIANTS:
        m = metrics.get(variant, {})
        sample_count = m.get("sample_count", 0)

        if sample_count < MIN_SAMPLES:
            skip_log[variant] = sample_count
            scores[variant] = None  # carry forward current weight
            continue

        error_rate = m.get("error_rate", 0.0)

        if error_rate > KILL_SWITCH_ERROR:
            killed_log[variant] = error_rate
            scores[variant] = 0.0
            continue

        # Latency: prefer SageMaker native (µs → ms), fall back to EMF
        native_p50_us = m.get("model_latency_p50_us")
        if native_p50_us is not None:
            latency_ms = native_p50_us / 1000.0
        else:
            latency_ms = m.get("p50_ms") or BASELINE_LATENCY_MS[variant]

        baseline     = BASELINE_LATENCY_MS[variant]
        norm_latency = latency_ms / baseline  # 1.0 = at baseline, <1.0 = faster

        # Guard against zero/negative latency (defensive; shouldn't happen)
        if norm_latency <= 0:
            norm_latency = 1.0

        confidence = m.get("avg_confidence", 0.5)

        scores[variant] = confidence * (1.0 / norm_latency) * (1.0 - error_rate)

    return scores, skip_log, killed_log


def apply_dampening_and_floor(
    raw_scores: Dict[str, Optional[float]],
    current_weights: Dict[str, float],
) -> Dict[str, float]:
    """
    Convert raw scores to new weights with dampening and floor.

    Steps:
    1. For variants with None (insufficient data), carry forward current weight.
    2. For variants with 0.0 (kill switch), set to 0.0.
    3. Normalize non-zero scores to sum to 1.0.
    4. Apply dampening: new = old × (1 − DAMPENING_FACTOR) + computed × DAMPENING_FACTOR
    5. Apply floor: clamp minimum to WEIGHT_FLOOR for all variants.
    6. Renormalize after floor to guarantee weights sum to 1.0.
    """
    # Step 1+2: resolve None → carry forward, 0.0 → zero
    computed = {}
    for variant in VARIANTS:
        score = raw_scores.get(variant)
        if score is None:
            computed[variant] = current_weights.get(variant, 1.0 / len(VARIANTS))
        elif score == 0.0:
            computed[variant] = 0.0
        else:
            computed[variant] = score

    # Step 3: normalize (ignore zeros so they don't dilute the distribution)
    total = sum(v for v in computed.values() if v > 0)
    if total == 0:
        # All variants killed or missing — return current weights unchanged
        return current_weights.copy()
    normalized = {
        v: (computed[v] / total) if computed[v] > 0 else 0.0
        for v in VARIANTS
    }

    # Step 4: dampening
    dampened = {
        v: current_weights.get(v, 1.0 / len(VARIANTS)) * (1 - DAMPENING_FACTOR)
           + normalized[v] * DAMPENING_FACTOR
        for v in VARIANTS
    }

    # Step 5: floor
    floored = {v: max(dampened[v], WEIGHT_FLOOR) for v in VARIANTS}

    # Step 6: renormalize
    total_after_floor = sum(floored.values())
    final = {v: round(floored[v] / total_after_floor, 4) for v in VARIANTS}

    return final


def validate_weights(weights: Dict[str, float]) -> bool:
    """
    Sanity check: weights must sum to ~1.0 and all be >= WEIGHT_FLOOR.
    """
    total = sum(weights.values())
    if not (0.99 <= total <= 1.01):
        return False
    if any(w < WEIGHT_FLOOR - 0.001 for w in weights.values()):
        return False
    return True

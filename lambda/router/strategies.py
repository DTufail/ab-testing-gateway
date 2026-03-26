import random
from typing import Dict, Any, Optional, Tuple

VALID_VARIANTS = {
    "VariantA-BERT-FP32",
    "VariantB-BERT-INT8",
    "VariantC-DistilBERT",
}


def route_weighted_random(weights: Dict[str, float]) -> str:
    """
    Weighted random selection using cumulative threshold.
    weights: {"VariantA-BERT-FP32": 0.6, "VariantB-BERT-INT8": 0.2, ...}
    Normalizes weights before comparing to handle floating-point drift.
    """
    # Sort by name for deterministic ordering
    sorted_variants = sorted(weights.keys())
    total = sum(weights[v] for v in sorted_variants)
    if total == 0:
        return sorted_variants[0]

    r = random.random()
    cumulative = 0.0
    for variant in sorted_variants:
        cumulative += weights[variant] / total
        if r < cumulative:
            return variant

    # Fallback: floating-point edge case where r == 1.0
    return sorted_variants[-1]


def route_header_pinned(header_value: Optional[str]) -> Optional[str]:
    """
    Returns variant name if header_value is a valid variant, else None.
    Caller falls back to weighted_random on None.
    """
    if header_value and header_value.strip() in VALID_VARIANTS:
        return header_value.strip()
    return None


def route_least_latency(latency_cache: Dict[str, float]) -> str:
    """
    Returns variant with minimum cached p50 latency.
    Ignores zero or missing values. Falls back to first variant alphabetically
    if all values are zero or cache is empty.
    """
    valid = {k: v for k, v in latency_cache.items() if v and v > 0}
    if not valid:
        return sorted(latency_cache.keys())[0] if latency_cache else "VariantA-BERT-FP32"
    return min(valid, key=lambda k: valid[k])


def route_shadow(weights: Dict[str, float], shadow_target: str) -> Tuple[str, str]:
    """
    Returns (primary_variant, shadow_variant).
    Primary is the highest-weight variant.
    Shadow is shadow_target from config.
    If shadow_target == primary, use second-highest weight variant instead.
    """
    sorted_by_weight = sorted(weights.keys(), key=lambda k: weights[k], reverse=True)
    primary = sorted_by_weight[0]

    if shadow_target != primary:
        shadow = shadow_target
    else:
        # shadow_target is the primary — use second-highest weight variant
        shadow = sorted_by_weight[1] if len(sorted_by_weight) > 1 else sorted_by_weight[0]

    return primary, shadow

from typing import Dict


def clip_and_normalize_weights(
    w1: float,
    w2: float,
    w3: float,
    base_w1: float,
    base_w2: float,
    base_w3: float,
) -> Dict[str, float]:
    """Clip each weight to +/- 0.15 of its baseline, then normalize to sum to 1.0."""

    def clip_weight(proposed_val, base_val):
        return max(base_val - 0.15, min(base_val + 0.15, proposed_val))

    c_w1 = clip_weight(w1, base_w1)
    c_w2 = clip_weight(w2, base_w2)
    c_w3 = clip_weight(w3, base_w3)

    total = max(c_w1, 0) + max(c_w2, 0) + max(c_w3, 0)
    return {
        "w1": round(c_w1 / total, 3) if total > 0 else 0.333,
        "w2": round(c_w2 / total, 3) if total > 0 else 0.333,
        "w3": round(c_w3 / total, 3) if total > 0 else 0.334,
    }


# used by tests
_clip_and_normalize_weights = clip_and_normalize_weights

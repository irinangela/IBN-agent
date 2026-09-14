# Thesis multi-objective setup configuration
#
# This config is designed to drive:
#   1) topology generation (generator.py)
#   2) workload generation (generator.py)
#   3) objective weighting for cost/security/latency (main.py + heuristics)
#
# Near-neighbor instance family
# Continuous ranges [a, b] are collapsed to a tight band around the original
# midpoint (INSTANCE_BAND_REL). Discrete draws (security tier, mTLS, unikernel,
# cluster membership) stay stochastic, so two seeds are similar but not clones.
# WARM_START_SEEDS are the historical near-matches. SEED is tonight's live batch.
#
# One dataset covers every retrieval condition. 
# The CSV holds all of WARM_START_SEEDS plus SEED
# The pruning only changes which of them the retriever is allowed to read:
#   WARM_START_MODE = "oracle"       -> retrieve SEED (idealized, upper bound)
#   WARM_START_MODE = "near_neighbor" + WARM_START_K = 0 / 1 / 2 / 5 / None
#   0 is cold start (empty retrieval). None is the full WARM_START_SEEDS pool.
# Keep SEED fixed across conditions so retrieval size is the only variable.

def _band(lo, hi, rel=None):
    """Collapse [lo, hi] to a tight band around the midpoint."""
    if rel is None:
        rel = INSTANCE_BAND_REL
    mid = 0.5 * (float(lo) + float(hi))
    half = max(abs(mid) * float(rel), 1e-9)
    a = max(float(lo), round(mid - half, 4))
    b = min(float(hi), round(mid + half, 4))
    if a > b:
        a, b = b, a
    return (a, b)


def _fixed_int(lo, hi):
    """Pin an inclusive integer range to its midpoint."""
    mid = int(round(0.5 * (int(lo) + int(hi))))
    return (mid, mid)


def _simplex_weight_grid(step=0.2):
    """Sparse W1+W2+W3=1 grid. step=0.2 yields 21 points."""
    n = int(round(1.0 / step))
    out = []
    for i in range(n + 1):
        for j in range(n + 1 - i):
            k = n - i - j
            out.append((round(i * step, 10), round(j * step, 10), round(k * step, 10)))
    return tuple(out)


def retrieval_seeds():
    """Seeds for the warm-start / feasibility pre-check.

    WARM_START_K trims the pool to its first K entries so the retrieval-size
    pruning needs no separate dataset. None means "use the whole pool".
    0 is cold start: no historical seeds, so feasibility has no evidence and
    the proposer gets an empty warm-start table.
    """
    if WARM_START_MODE == "oracle":
        return (int(SEED),)
    seeds = tuple(int(s) for s in WARM_START_SEEDS)
    if WARM_START_K is not None:
        k = int(WARM_START_K)
        if k <= 0:
            return ()
        seeds = seeds[:k]
    return seeds


# --- Simulation Parameters ---
INSTANCE_BAND_REL = 0.04

# Live instance the agent actually places. Keep this fixed across pruning.
SEED = 210
# Historical near-neighbor pool stored in valid_runs.csv.
WARM_START_SEEDS = tuple(range(200, 210))
# "near_neighbor": retrieve WARM_START_SEEDS only (thesis default).
# "oracle": retrieve SEED (idealized warm-start).
WARM_START_MODE = "near_neighbor"
# None = whole pool. Set 5 / 2 / 1 for pruning, 0 for cold start.
# k=0 does not HITL and does not invent a historical clip center -> search
# starts at the simplex midpoint and Recalibrator uses live attempts only.
WARM_START_K = None

# Seeds written by scripts/generate_dataset.py (pool + live, so "oracle" is a
# switch rather than another sweep).
DATASET_SEEDS = WARM_START_SEEDS + (SEED,)

# --- Topology Constants ---
# Matches the "small" topology used in the repo's root `config.py`
NUM_EDGE = 40
NUM_FOG = 10
NUM_CLOUD = 5
NUM_NODES = NUM_EDGE + NUM_FOG + NUM_CLOUD

# --- Cluster/Domain Constants ---
NUM_CLUSTERS = 2
INTRA_CLUSTER_LATENCY_FACTOR = 0.8  # latency discount for intra-cluster comms

# Machines per Node (range inclusive) — originally (1,2), (2,4), (5,5)
MACHINES_EDGE = _fixed_int(1, 2)
MACHINES_FOG = _fixed_int(2, 4)
MACHINES_CLOUD = _fixed_int(5, 5)

# --- Machine Performance Coefficients ---
# Exec time = typical_runtime / speed_coeff
SPEED_COEFF_EDGE = _band(0.5, 0.8)
SPEED_COEFF_FOG = _band(1.0, 1.2)
SPEED_COEFF_CLOUD = _band(1.4, 2.0)

# Machine Specifications (CPU range and COST range) by tier
SPECS_EDGE_CPU = _band(1.0, 4.0)
SPECS_EDGE_COST = _band(3.0, 5.0)

SPECS_FOG_CPU = _band(2.0, 12.0)
SPECS_FOG_COST = _band(2.0, 3.0)

SPECS_CLOUD_CPU = _band(8.0, 16.0)
SPECS_CLOUD_COST = _band(1.0, 2.0)

# Residual cost noise after the speed-correlated base cost (was ±10%).
COST_DISTURBANCE = _band(0.9, 1.1)

# --- Latency Constants (ms) ---
INTRA_MACHINE_LATENCY = 2.0
INTRA_NODE_LATENCY = 5.0

# Network latencies (ms) by tier pair
DELAY_EDGE_EDGE = _band(20.0, 40.0)
DELAY_EDGE_FOG = _band(60.0, 150.0)
DELAY_EDGE_CLOUD = _band(350.0, 500.0)
DELAY_FOG_FOG = _band(60.0, 150.0)
DELAY_FOG_CLOUD = _band(220.0, 350.0)
DELAY_CLOUD_CLOUD = _band(250.0, 400.0)

# User (data source) to node latencies
DELAY_USER_EDGE = _band(20.0, 40.0)
DELAY_USER_FOG = _band(100.0, 150.0)
DELAY_USER_CLOUD = _band(400.0, 500.0)

# --- Application Constants ---
NUM_APPS = 50
MIN_SERVICES = 5
MAX_SERVICES = 5

# --- Return Path (optional) ---
RETURN_PATH_PROBABILITY = 1.0

# --- Service Generation Parameters ---
# 1) Runtime (ms)
PROB_HEAVY_SERVICE = 0.0
RUNTIME_LIGHT = _band(10.0, 100.0)
RUNTIME_HEAVY = _band(100.0, 200.0)

# 2) CPU Demands for services
DEMAND_CPU_LIGHT = _band(0.1, 1.0)
DEMAND_CPU_HEAVY = _band(1.0, 2.0)

# 3) Security Requirements
MIN_SEC = 0
MAX_SEC = 3
PROB_UNIKERNEL_IMAGE = 0.20

# 4) Data size (MB)
DATA_SIZE = _band(0.5, 5.0)

# 5) mTLS
MTLS_PROBABILITY = 0.25
MTLS_CPU_OVERHEAD = _band(0.1, 0.4)  # CPU overhead for each mTLS link
MTLS_LATENCY_FIXED = 5.0
MTLS_LATENCY_PER_MB = 10.0

# --- Security Tier Resource Overheads (CPU multipliers) ---
# Tiers: 0, 1, 2, 3
import numpy as np

OVERHEAD_CPU_BASE = np.array([1.0, 1.74, 1.36, 0.67])
# Relative jitter around OVERHEAD_CPU_BASE for tiers 1-3 (was ±20%).
SEC_OVERHEAD_JITTER = INSTANCE_BAND_REL

# Cost Multipliers per Security Tier (applied to base cost)
COST_MULT_EDGE = np.array([1.0, 1.1, 1.25, 1.5])
COST_MULT_FOG = np.array([1.0, 1.1, 1.75, 2.5])
COST_MULT_CLOUD = np.array([1.0, 1.3, 2.25, 3.5])

# Node Capability Distributions:
# A service with min_security s can only be placed on a machine whose max tier
# is >= s, so high-tier hosts caused security-heavy placements to
# crowd a few machines and exhaust their CPU (the invalid runs).
# Every machine now supports at least tier 2, with tier-3 capability increasing
# edge -> fog -> cloud. This keeps a graded isolation story (edge is
# the least capable, cloud the most) while removing the tier-support bottleneck.
PROB_EDGE = [0.0, 0.0, 0.55, 0.45]
PROB_FOG = [0.0, 0.0, 0.40, 0.60]
PROB_CLOUD = [0.0, 0.0, 0.25, 0.75]

# --- Normalization Bounds (Estimated) ---
MAX_NORM_COST = 5000.0
MIN_NORM_COST = 10.0

MAX_NORM_LATENCY = 2000.0
MIN_NORM_LATENCY = 5.0

MAX_NORM_SEC = 40.0
MIN_NORM_SEC = 1.0

# --- Security normalization weights ---
# When True, per-service security is weighted by typical_runtime.
SEC_RUNTIME_WEIGHTED = True

# --- Current weights (updated by main.py during sweeps) ---
# Score = W1*NormCost + W3*NormLatency + W2*(1 - NormSecurity)
W1 = 0.1
W2 = 0.8
W3 = 0.1

# --- Objective weight sweep ---
# Small set for a manual `python simulation/main.py` smoke run.
WEIGHT_SETS = [
    (1.0, 0.0, 0.0),   # Cost only
    (0.0, 1.0, 0.0),   # Security only
    (0.0, 0.0, 1.0),   # Latency only
    (0.35, 0.20, 0.45),  # Balanced_1 (example_1)
    (0.40, 0.25, 0.35),  # Balanced_2 (example_2)
]

# Weight grid for the dataset sweep (66 points at step 0.1).
# Both algorithms must share this grid: giving one algorithm more points 
# would make it look artificially strong in the feasibility pre-check and 
# that may suppress the auto-switch to app_rollout.
# Density experiments subsample this grid at read time so no re-generation needed.
WARM_START_WEIGHT_SETS = _simplex_weight_grid(0.1)

# --- Algorithm toggles ---
RUN_BEST_FIT = True
RUN_APP_ROLLOUT = True

# Heuristic implementation choices (used by the repo's rollout/greedy)
# RolloutAllocatorV2 uses resolve_greedy_allocator(topo) which checks USE_FAST_GREEDY.
USE_FAST_GREEDY = True

# RolloutAllocatorV2 uses ProcessPoolExecutor(max_workers=MAX_WORKERS).
# Keep it modest for thesis runs.
MAX_WORKERS = 6

# Optional progress print flag (repo behavior)
ROLLOUT_PROGRESS = False

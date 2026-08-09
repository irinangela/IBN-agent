# Thesis multi-objective setup configuration
#
# This config is designed to drive:
#   1) topology generation (generator.py)
#   2) workload generation (generator.py)
#   3) objective weighting for cost/security/latency (main.py + heuristics)

# --- Simulation Parameters ---
SEED = 100

# --- Topology Constants ---
# Matches the "small" topology used in the repo's root `config.py`
NUM_EDGE = 40
NUM_FOG = 10
NUM_CLOUD = 5
NUM_NODES = NUM_EDGE + NUM_FOG + NUM_CLOUD

# --- Cluster/Domain Constants ---
NUM_CLUSTERS = 2
INTRA_CLUSTER_LATENCY_FACTOR = 0.8  # latency discount for intra-cluster comms

# Machines per Node (range inclusive)
MACHINES_EDGE = (1, 2)
MACHINES_FOG = (2, 4)
MACHINES_CLOUD = (5, 5)

# --- Machine Performance Coefficients ---
# Exec time = typical_runtime / speed_coeff
SPEED_COEFF_EDGE = (0.5, 0.8)
SPEED_COEFF_FOG = (1.0, 1.2)
SPEED_COEFF_CLOUD = (1.4, 2.0)

# Machine Specifications (CPU range and COST range) by tier
SPECS_EDGE_CPU = (1.0, 4.0)
SPECS_EDGE_COST = (3.0, 5.0)

SPECS_FOG_CPU = (2.0, 12.0)
SPECS_FOG_COST = (2.0, 3.0)

SPECS_CLOUD_CPU = (8.0, 16.0)
SPECS_CLOUD_COST = (1.0, 2.0)

# --- Latency Constants (ms) ---
INTRA_MACHINE_LATENCY = 2.0
INTRA_NODE_LATENCY = 5.0

# Network latencies (ms) by tier pair
DELAY_EDGE_EDGE = (20.0, 40.0)
DELAY_EDGE_FOG = (60.0, 150.0)
DELAY_EDGE_CLOUD = (350.0, 500.0)
DELAY_FOG_FOG = (60.0, 150.0)
DELAY_FOG_CLOUD = (220.0, 350.0)
DELAY_CLOUD_CLOUD = (250.0, 400.0)

# User (data source) to node latencies
DELAY_USER_EDGE = (20.0, 40.0)
DELAY_USER_FOG = (100.0, 150.0)
DELAY_USER_CLOUD = (400.0, 500.0)

# --- Application Constants ---
NUM_APPS = 50
MIN_SERVICES = 2
MAX_SERVICES = 10

# --- Return Path (optional) ---
RETURN_PATH_PROBABILITY = 1.0

# --- Service Generation Parameters ---
# 1) Runtime (ms)
PROB_HEAVY_SERVICE = 0.0
RUNTIME_LIGHT = (10.0, 100.0)
RUNTIME_HEAVY = (100.0, 200.0)

# 2) CPU Demands for services
DEMAND_CPU_LIGHT = (0.1, 1.0)
DEMAND_CPU_HEAVY = (1.0, 2.0)

# 3) Security Requirements
MIN_SEC = 0
MAX_SEC = 3
PROB_UNIKERNEL_IMAGE = 0.20

# 4) Data size (MB)
DATA_SIZE = (0.5, 5.0)

# 5) mTLS
MTLS_PROBABILITY = 0.25
MTLS_CPU_OVERHEAD = (0.1, 0.4)  # CPU overhead for each mTLS link
MTLS_LATENCY_FIXED = 5.0
MTLS_LATENCY_PER_MB = 10.0

# --- Security Tier Resource Overheads (CPU multipliers) ---
# Tiers: 0, 1, 2, 3
import numpy as np

OVERHEAD_CPU_BASE = np.array([1.0, 1.74, 1.36, 0.67])

# Cost Multipliers per Security Tier (applied to base cost)
COST_MULT_EDGE = np.array([1.0, 1.1, 1.25, 1.5])
COST_MULT_FOG = np.array([1.0, 1.1, 1.75, 2.5])
COST_MULT_CLOUD = np.array([1.0, 1.3, 2.25, 3.5])

# Node Capability Distributions (security tier probabilities 0-3)
PROB_EDGE = [0.50, 0.2, 0.2, 0.1]
PROB_FOG = [0.30, 0.3, 0.2, 0.2]
PROB_CLOUD = [0.20, 0.25, 0.25, 0.30]

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
# Keep these weights aligned with the manuscript's cost/security/latency objectives.
WEIGHT_SETS = [
    (1.0, 0.0, 0.0),   # Cost only
    (0.0, 1.0, 0.0),   # Security only
    (0.0, 0.0, 1.0),   # Latency only
    (0.35, 0.20, 0.45),  # Balanced_1 (example_1)
    (0.40, 0.25, 0.35),  # Balanced_2 (example_2)
]

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
ROLLOUT_PROGRESS = True


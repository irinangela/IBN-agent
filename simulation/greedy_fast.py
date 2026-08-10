"""
Vectorized best-fit Greedy allocator — same decisions as GreedyAllocator in heuristics.py.

Optimizations (not objective-specific):
  - Precomputed per-machine node / category / cluster arrays
  - Precomputed trusted-region masks per source cluster
  - Precomputed cost-multiplier lookup table
  - Hoisted config weights and latency constants
  - NumPy batch scoring over all feasible machines per (tier, mTLS) branch
"""
from typing import Tuple

import numpy as np
from simulation import config as cfg
from simulation.structs import Application, Microservice, Topology
from simulation.heuristics import GreedyAllocator, compute_service_bounds


def use_fast_greedy() -> bool:
    return bool(getattr(cfg, "USE_FAST_GREEDY", False))


def resolve_greedy_allocator(topo: Topology) -> GreedyAllocator:
    """Return GreedyAllocatorFast when USE_FAST_GREEDY is set in config."""
    if use_fast_greedy():
        return GreedyAllocatorFast(topo)
    return GreedyAllocator(topo)


def _normalize_vec(values: np.ndarray, min_val: float, max_val: float) -> np.ndarray:
    if max_val <= min_val:
        return np.zeros_like(values, dtype=np.float64)
    return np.clip((values - min_val) / (max_val - min_val), 0.0, 1.0)


class GreedyAllocatorFast(GreedyAllocator):
    """Drop-in replacement for GreedyAllocator with a faster placement search."""

    def __init__(self, topo: Topology):
        super().__init__(topo)
        t = topo
        machines = t.machines
        self._machines = machines
        self._cpu_idx = t.CPU_IDX
        self._cost_idx = t.COST_IDX
        self._sec_idx = t.SEC_IDX
        self._speed_idx = t.SPEED_IDX

        self._machine_nodes = machines[:, t.NODE_IDX].astype(np.int32)
        self._machine_cats = t.node_categories[self._machine_nodes].astype(np.int32)
        self._machine_clusters = t.node_clusters[self._machine_nodes].astype(np.int32)

        self._cost_mult = np.zeros((4, 4), dtype=np.float64)
        self._cost_mult[1] = np.asarray(cfg.COST_MULT_EDGE, dtype=np.float64)
        self._cost_mult[2] = np.asarray(cfg.COST_MULT_FOG, dtype=np.float64)
        self._cost_mult[3] = np.asarray(cfg.COST_MULT_CLOUD, dtype=np.float64)

        if t.trust_matrix is not None:
            self._trusted_by_cluster = t.trust_matrix[:, self._machine_clusters].astype(bool)
        else:
            self._trusted_by_cluster = None

        self._W1 = float(cfg.W1)
        self._W2 = float(cfg.W2)
        self._W3 = float(cfg.W3)
        self._w_tier = float(getattr(cfg, "SEC_TIER_WEIGHT", 1.0))
        self._w_mtls = float(getattr(cfg, "SEC_MTLS_WEIGHT", 1.0))
        self._sec_rt_weighted = bool(getattr(cfg, "SEC_RUNTIME_WEIGHTED", True))
        self._intra_machine = float(cfg.INTRA_MACHINE_LATENCY)
        self._intra_node = float(cfg.INTRA_NODE_LATENCY)

    def _find_best_placement(
        self,
        app: Application,
        ms: Microservice,
        ms_idx: int,
        prev_mach_idx: int,
    ) -> Tuple[int, int, bool]:
        best_score = float("inf")
        best_cand = (-1, -1, False)

        is_last = ms_idx == len(app.microservices) - 1
        sb = compute_service_bounds(ms, ms_idx, is_last=is_last, return_to_ue=app.return_to_ue)

        base_required_mtls = ms_idx > 0 and app.mtls_requirements[ms_idx - 1]

        machines = self._machines
        cpu_all = machines[:, self._cpu_idx]
        m_sec = machines[:, self._sec_idx]
        sec_mask = m_sec >= ms.min_security
        if not np.any(sec_mask):
            return best_cand

        idx = np.flatnonzero(sec_mask)
        n = len(idx)

        base_rate = machines[idx, self._cost_idx]
        coeff = machines[idx, self._speed_idx]
        exec_time = ms.typical_runtime / coeff
        cats = self._machine_cats[idx]
        tier_cap = np.minimum(int(ms.max_security), m_sec[idx].astype(np.int32))

        if self._trusted_by_cluster is not None:
            trusted = self._trusted_by_cluster[app.source_cluster, idx]
        else:
            trusted = np.ones(n, dtype=bool)

        valid_min = np.where(trusted, int(ms.min_security), max(int(ms.min_security), 1))

        node_ids = self._machine_nodes[idx]
        if prev_mach_idx == -1:
            comm = self.topo.node_to_user_latency[node_ids, app.gen_point]
        else:
            prev_node = int(self._machine_nodes[prev_mach_idx])
            same_node = node_ids == prev_node
            same_machine = idx == prev_mach_idx
            comm = np.where(
                same_node,
                np.where(same_machine, self._intra_machine, self._intra_node),
                self.topo.network_latency[prev_node, node_ids],
            )

        lat_base = exec_time + comm
        if app.return_to_ue and is_last:
            lat_base = lat_base + self.topo.node_to_user_latency[node_ids, app.gen_point]

        sec_rt = ms.typical_runtime if self._sec_rt_weighted else 1.0
        mtls_lat_pen = 0.0
        if ms_idx > 0:
            prev_data = app.microservices[ms_idx - 1].data_size
            mtls_lat_pen = cfg.MTLS_LATENCY_FIXED + (cfg.MTLS_LATENCY_PER_MB * prev_data)

        egress_cpu_fixed = 0.0
        if ms_idx < len(app.microservices) - 1 and app.mtls_requirements[ms_idx]:
            egress_cpu_fixed = app.mtls_overheads[ms_idx]

        prev_mtls_ok = True
        if ms_idx > 0 and not base_required_mtls and prev_mach_idx != -1:
            prev_mtls_ok = cpu_all[prev_mach_idx] >= app.mtls_overheads[ms_idx - 1]

        overheads = np.asarray(ms.security_overheads, dtype=np.float64)
        cpu_cand = cpu_all[idx]

        if ms_idx == 0:
            mtls_options = (False,)
        elif base_required_mtls:
            mtls_options = (True,)
        else:
            mtls_options = (True, False)

        sb_min_cost, sb_max_cost = sb["min_cost"], sb["max_cost"]
        sb_min_lat, sb_max_lat = sb["min_lat"], sb["max_lat"]
        sb_min_sec, sb_max_sec = sb["min_sec"], sb["max_sec"]

        for use_mtls in mtls_options:
            if use_mtls and ms_idx > 0 and not base_required_mtls and not prev_mtls_ok:
                continue

            mtls_ingress = app.mtls_overheads[ms_idx - 1] if (use_mtls and ms_idx > 0) else 0.0
            mtls_cpu_local = mtls_ingress + egress_cpu_fixed

            lat_add = mtls_lat_pen if (use_mtls and ms_idx > 0) else 0.0
            curr_lat = lat_base + lat_add

            for tier in range(4):
                tier_ok = (tier >= valid_min) & (tier <= tier_cap)
                if not np.any(tier_ok):
                    continue

                req_cpu = ms.cpu_demand * overheads[tier] + mtls_cpu_local
                feasible = tier_ok & (cpu_cand >= req_cpu)

                if ms_idx > 0 and not use_mtls:
                    feasible &= trusted

                if not np.any(feasible):
                    continue

                cost_mult_val = self._cost_mult[cats, tier]
                cost_val = base_rate * cost_mult_val * exec_time

                sec_val = (float(tier) * self._w_tier + (self._w_mtls if use_mtls else 0.0)) * sec_rt

                n_cost = _normalize_vec(cost_val, sb_min_cost, sb_max_cost)
                n_lat = _normalize_vec(curr_lat, sb_min_lat, sb_max_lat)
                n_sec = _normalize_vec(np.full(n, sec_val, dtype=np.float64), sb_min_sec, sb_max_sec)

                scores = self._W1 * n_cost + self._W3 * n_lat + self._W2 * (1.0 - n_sec)
                scores = np.where(feasible, scores, np.inf)

                local_best = int(np.argmin(scores))
                local_score = float(scores[local_best])
                if local_score < best_score:
                    best_score = local_score
                    best_cand = (int(idx[local_best]), tier, use_mtls)

        return best_cand

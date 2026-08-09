import numpy as np
import copy
import time
import math
import random
import concurrent.futures
from typing import List, Tuple, Optional
from simulation import config as cfg
from simulation.structs import Topology, Application, Microservice

# --- Helper: Normalization ---
def normalize(value, min_val, max_val, metric_type='default'):
    mode = getattr(cfg, 'NORMALIZATION_MODE', 'linear')
    if metric_type == 'sec':
        mode = 'linear'
        
    if mode == 'log':
        if max_val <= min_val: return 0.0
        clamped_val = max(min_val, min(max_val, value))
        return math.log(clamped_val - min_val + 1.0) / math.log(max_val - min_val + 1.0)
    elif mode == 'soft_cap':
        if max_val <= min_val: return 0.0
        cap_mult = getattr(cfg, 'SOFT_CAP_MULTIPLIER', 0.5)
        soft_max = min_val + cap_mult * (max_val - min_val)
        return min(1.0, max(0.0, (value - min_val) / (soft_max - min_val)))
    else:
        if max_val <= min_val: return 0.0
        return min(1.0, max(0.0, (value - min_val) / (max_val - min_val)))

# --- Helper: Per-Service Bounds (Used for Summation) ---
def compute_service_bounds(ms, ms_idx=0, is_last=False, return_to_ue=False) -> dict:
    """
    Compute static per-service normalization bounds.
    ms_idx: position in chain (0 = first service, >0 = intermediate)
    is_last / return_to_ue: when both True, fold the return-to-user latency into
        this service's bounds (matches the global adjustment in
        compute_app_global_bounds). Used by Greedy to evaluate the final service
        consistently; left False for the global-bounds summation to avoid
        double-counting the return path.
    """
    # Constants derived from config.py
    MIN_SPEED_COEFF = cfg.SPEED_COEFF_EDGE[0] # 0.6
    MAX_SPEED_COEFF = cfg.SPEED_COEFF_CLOUD[1] # 2.5
    
    # Cost Rates (Base + Overhead)
    MAX_COST_RATE = cfg.SPECS_EDGE_COST[1] * cfg.COST_MULT_EDGE[3]
    MIN_COST_RATE = cfg.SPECS_CLOUD_COST[0] * cfg.COST_MULT_CLOUD[0]
    
    # Latency: differentiate first service (UE-to-node) vs intermediate (node-to-node)
    if ms_idx == 0:
        # First service: comm is from UE to placement node (can be up to cloud)
        MAX_NET_DELAY = cfg.DELAY_USER_CLOUD[0]  # 400.0
    else:
        # Intermediate services: benefit from collocation (edge-to-edge max)
        MAX_NET_DELAY = cfg.DELAY_EDGE_EDGE[1]   # 40.0
    mtls_delay = cfg.MTLS_LATENCY_FIXED + (cfg.MTLS_LATENCY_PER_MB * ms.data_size)
    
    # --- COST ---
    max_cost = (ms.typical_runtime / MIN_SPEED_COEFF) * MAX_COST_RATE
    min_cost = (ms.typical_runtime / MAX_SPEED_COEFF) * MIN_COST_RATE
    
    # --- LATENCY ---
    max_exec = ms.typical_runtime / MIN_SPEED_COEFF
    max_lat = max_exec + MAX_NET_DELAY + mtls_delay
    
    min_exec = ms.typical_runtime / MAX_SPEED_COEFF
    min_lat = min_exec
    
    # Return-path adjustment for the final service (mirrors compute_app_global_bounds)
    if is_last and return_to_ue:
        max_lat += cfg.DELAY_USER_CLOUD[0]  # 400.0 (last service on cloud)
        min_lat += cfg.DELAY_USER_EDGE[0]   # 20.0  (last service on nearby edge)
    
    # --- SECURITY (Weighted by Runtime) ---
    # Min: min_security tier, no mTLS
    # Max: tier 3 (theoretical max) + mTLS bonus
    w_tier = getattr(cfg, 'SEC_TIER_WEIGHT', 1.0)
    w_mtls = getattr(cfg, 'SEC_MTLS_WEIGHT', 1.0)
    sec_rt = ms.typical_runtime if getattr(cfg, 'SEC_RUNTIME_WEIGHTED', True) else 1.0
    min_sec = (float(ms.min_security) * w_tier) * sec_rt
    max_sec = ((float(ms.max_security) * w_tier) + (1.0 * w_mtls)) * sec_rt
    
    return {
        'min_cost': min_cost, 'max_cost': max_cost,
        'min_lat': min_lat, 'max_lat': max_lat,
        'min_sec': min_sec, 'max_sec': max_sec
    }

def compute_app_global_bounds(app: Application) -> dict:
    """
    Computes the Global Min/Max Bounds for the entire application chain.
    """
    total_min_cost = 0.0
    total_max_cost = 0.0
    total_min_lat = 0.0
    total_max_lat = 0.0
    total_min_sec = 0.0
    total_max_sec = 0.0
    
    for i, ms in enumerate(app.microservices):
        b = compute_service_bounds(ms, ms_idx=i)
        total_min_cost += b['min_cost']
        total_max_cost += b['max_cost']
        total_min_lat += b['min_lat']
        total_max_lat += b['max_lat']
        total_min_sec += b['min_sec']
        total_max_sec += b['max_sec']
        
    # Return Path Adjustment for Latency (last service back to UE, can be up to cloud)
    if app.return_to_ue:
        total_max_lat += cfg.DELAY_USER_CLOUD[0]  # 400.0 (last service on cloud)
        total_min_lat += cfg.DELAY_USER_EDGE[0]   # 20.0 (last service on nearby edge)
        
    return {
        'total_min_cost': total_min_cost,
        'total_max_cost': total_max_cost,
        'range_cost': total_max_cost - total_min_cost,
        
        'total_min_lat': total_min_lat,
        'total_max_lat': total_max_lat,
        'range_lat': total_max_lat - total_min_lat,
        
        'total_min_sec': total_min_sec,
        'total_max_sec': total_max_sec,
        'range_sec': total_max_sec - total_min_sec
    }

# --- Helper: Score Calculation ---
def get_app_normalized_metrics(topo: Topology, app: Application) -> Optional[Tuple[float, float, float]]:
    """
    Calculates the components of the Objective Function using GLOBAL NORMALIZATION.
    Returns (GlobalNormCost, GlobalNormSecurity, GlobalNormLatency)
    """
    if not hasattr(app, 'global_bounds'):
         app.global_bounds = compute_app_global_bounds(app)
         
    bounds = app.global_bounds
    
    total_raw_cost = 0.0
    total_raw_sec = 0.0 # Weighted
    total_raw_lat = 0.0
    
    prev_node = app.gen_point
    prev_mach_idx = -1
    
    for i, ms in enumerate(app.microservices):
        if ms.assigned_machine_id == -1: 
            return None # Incomplete
            
        m_idx = ms.assigned_machine_id
        tier = ms.assigned_tier
        node = int(topo.machines[m_idx, topo.NODE_IDX])
        cat = topo.node_categories[node]
        
        # --- 1. Cost ---
        base_rate = topo.machines[m_idx, topo.COST_IDX]
        cost_mult = 1.0
        if cat == 1: cost_mult = cfg.COST_MULT_EDGE[tier]
        elif cat == 2: cost_mult = cfg.COST_MULT_FOG[tier]
        elif cat == 3: cost_mult = cfg.COST_MULT_CLOUD[tier]
        
        total_rate = base_rate * cost_mult
        coeff = topo.machines[m_idx, topo.SPEED_IDX]
        exec_time = ms.typical_runtime / coeff
        
        total_raw_cost += (total_rate * exec_time)
        
        # --- 2. Latency (Exec + Ingress) ---
        comm_lat = 0.0
        if prev_mach_idx == -1:
            comm_lat = topo.node_to_user_latency[node, prev_node]
        else:
            if m_idx == prev_mach_idx: comm_lat = cfg.INTRA_MACHINE_LATENCY
            elif node == prev_node: comm_lat = cfg.INTRA_NODE_LATENCY
            else: comm_lat = topo.network_latency[prev_node, node]
            
            # mTLS Penalty (Ingress)
            if i > 0 and i-1 < len(app.mtls_decisions) and app.mtls_decisions[i-1]:
                prev_ms = app.microservices[i-1]
                penalty = cfg.MTLS_LATENCY_FIXED + (cfg.MTLS_LATENCY_PER_MB * prev_ms.data_size)
                comm_lat += penalty
        
        lat_val = exec_time + comm_lat
        
        # Add Return Path to Last Service Latency?
        if app.return_to_ue and i == len(app.microservices) - 1:
             ret_lat = topo.node_to_user_latency[node, app.gen_point]
             lat_val += ret_lat
             
        total_raw_lat += lat_val
        
        # --- 3. Security (Weighted by Runtime) ---
        w_tier = getattr(cfg, 'SEC_TIER_WEIGHT', 1.0)
        w_mtls = getattr(cfg, 'SEC_MTLS_WEIGHT', 1.0)
        sec_base = float(tier) * w_tier
        if i < len(app.microservices) - 1:
            if i < len(app.mtls_decisions) and app.mtls_decisions[i]:
                sec_base += 1.0 * w_mtls
        
        sec_rt = ms.typical_runtime if getattr(cfg, 'SEC_RUNTIME_WEIGHTED', True) else 1.0
        weighted_sec = sec_base * sec_rt
        total_raw_sec += weighted_sec
        
        prev_node = node
        prev_mach_idx = m_idx

    # --- Global Normalization ---
    n_cost = normalize(total_raw_cost, bounds['total_min_cost'], bounds['total_max_cost'], 'cost')
    n_lat = normalize(total_raw_lat, bounds['total_min_lat'], bounds['total_max_lat'], 'lat')
    n_sec = normalize(total_raw_sec, bounds['total_min_sec'], bounds['total_max_sec'], 'sec')
    
    return (n_cost, n_sec, n_lat)

def calculate_app_score_static(topo: Topology, app: Application) -> float:
    """
    Calculates the Total Normalized Objective Score for an application
    using GLOBAL APPLICATION NORMALIZATION.
    """
    metrics = get_app_normalized_metrics(topo, app)
    if metrics is None:
        return float('inf')
        
    n_cost, n_sec, n_lat = metrics
    
    total_score = (cfg.W1 * n_cost) + (cfg.W3 * n_lat) + (cfg.W2 * (1 - n_sec))
    return total_score

# --- Helper: Get Raw Metrics (Cost, Security, Latency) ---
def get_app_raw_metrics(topo: Topology, app: Application) -> Tuple[float, float, float]:
    """
    Returns (TotalRawCost, TotalRawSecScore, TotalRawLatency) for an application.
    """
    total_cost = 0.0
    total_sec_score = 0.0 
    total_latency = 0.0
    
    # 1. Cost & Security
    for i, ms in enumerate(app.microservices):
        if ms.assigned_machine_id == -1: return (0,0,0)
            
        m_idx = ms.assigned_machine_id
        tier = ms.assigned_tier
        
        # Cost Rate
        base_rate = topo.machines[m_idx, topo.COST_IDX]
        node = int(topo.machines[m_idx, topo.NODE_IDX])
        cat = topo.node_categories[node]
        
        cost_mult = 1.0
        if cat == 1: cost_mult = cfg.COST_MULT_EDGE[tier]
        elif cat == 2: cost_mult = cfg.COST_MULT_FOG[tier]
        elif cat == 3: cost_mult = cfg.COST_MULT_CLOUD[tier]
        
        total_rate = base_rate * cost_mult
        
        # Execution Time
        runtime = ms.typical_runtime
        coeff = topo.machines[m_idx, topo.SPEED_IDX]
        exec_time = runtime / coeff
        
        total_cost += (total_rate * exec_time)
        
        # Security (Unweighted for Reporting)
        w_tier = getattr(cfg, 'SEC_TIER_WEIGHT', 1.0)
        w_mtls = getattr(cfg, 'SEC_MTLS_WEIGHT', 1.0)
        total_sec_score += float(tier) * w_tier
        if i < len(app.microservices) - 1:
            if i < len(app.mtls_decisions) and app.mtls_decisions[i]:
                total_sec_score += 1.0 * w_mtls

    # 2. Latency
    prev_node = app.gen_point
    prev_mach_idx = -1
    
    for i, ms in enumerate(app.microservices):
        curr_mach_idx = ms.assigned_machine_id
        curr_node = int(topo.machines[curr_mach_idx, topo.NODE_IDX])
        
        runtime = ms.typical_runtime
        coeff = topo.machines[curr_mach_idx, topo.SPEED_IDX]
        exec_time = runtime / coeff
        total_latency += exec_time
        
        comm_lat = 0.0
        if prev_mach_idx == -1:
            comm_lat = topo.node_to_user_latency[curr_node, prev_node]
        else:
            if curr_mach_idx == prev_mach_idx: comm_lat = cfg.INTRA_MACHINE_LATENCY
            elif curr_node == prev_node: comm_lat = cfg.INTRA_NODE_LATENCY
            else: comm_lat = topo.network_latency[prev_node, curr_node]
            
            # mTLS Penalty
            is_mtls = False
            if (i-1) < len(app.mtls_decisions) and app.mtls_decisions[i-1]:
                is_mtls = True
            
            if is_mtls:
                prev_ms = app.microservices[i-1]
                penalty = cfg.MTLS_LATENCY_FIXED + (cfg.MTLS_LATENCY_PER_MB * prev_ms.data_size)
                comm_lat += penalty
        
        total_latency += comm_lat
        prev_node = curr_node
        prev_mach_idx = curr_mach_idx
        
    # Return Path
    if app.return_to_ue:
        if app.microservices:
            last_ms = app.microservices[-1]
            if last_ms.assigned_machine_id != -1:
                last_node = int(topo.machines[last_ms.assigned_machine_id, topo.NODE_IDX])
                ret_lat = topo.node_to_user_latency[last_node, app.gen_point]
                total_latency += ret_lat
                
    return (total_cost, total_sec_score, total_latency)


def compute_norm_mtls_latency(assignments) -> float:
    """
    Normalized mTLS latency contribution across apps.
    With linear normalization, each link penalty adds penalty/range_lat to norm latency.
    """
    import random
    from simulation import generator
    import numpy as np

    if not assignments:
        return 0.0

    np.random.seed(cfg.SEED)
    random.seed(cfg.SEED)
    apps = generator.generate_workload(len(assignments))
    for app in apps:
        app.global_bounds = compute_app_global_bounds(app)

    norm_mtls = 0.0
    for app, app_entry in zip(apps, assignments):
        bounds = app.global_bounds
        range_lat = bounds['total_max_lat'] - bounds['total_min_lat']
        if range_lat <= 0:
            continue
        for link in app_entry.get("links", []):
            if link.get("mtls_active", False):
                norm_mtls += float(link.get("mtls_latency", 0.0)) / range_lat
    return norm_mtls


def get_app_norm_mtls_latency(app: Application) -> float:
    """Normalized latency contribution from mTLS penalties for one app."""
    bounds = app.global_bounds if hasattr(app, 'global_bounds') else compute_app_global_bounds(app)
    range_lat = bounds['total_max_lat'] - bounds['total_min_lat']
    if range_lat <= 0:
        return 0.0
    mtls = 0.0
    for i in range(len(app.microservices) - 1):
        if i < len(app.mtls_decisions) and app.mtls_decisions[i]:
            prev_ms = app.microservices[i]
            mtls += cfg.MTLS_LATENCY_FIXED + (cfg.MTLS_LATENCY_PER_MB * prev_ms.data_size)
    return mtls / range_lat


class GreedyAllocator:
    def __init__(self, topo: Topology):
        self.topo = topo

    def solve(self, apps: List[Application]):
        for app in apps:
            # Pre-calculate Global Bounds
            app.global_bounds = compute_app_global_bounds(app)
            
            # Init decisions list
            app.mtls_decisions = [False] * (len(app.microservices) - 1)
            self.allocate_application(app)

    def allocate_application(self, app: Application) -> bool:
        # Sequential allocation for chain
        prev_mach_idx = -1
       
        for i, ms in enumerate(app.microservices):
            # If already assigned (e.g., by Rollout prefix), skip but update prev_mach_idx
            if ms.assigned_machine_id != -1:
                prev_mach_idx = ms.assigned_machine_id
                continue
                
            best_m_idx, best_tier, best_mtls = self._find_best_placement(app, ms, i, prev_mach_idx)
            
            if best_m_idx != -1:
                self._apply_assignment(ms, best_m_idx, best_tier, best_mtls, app, i)
                prev_mach_idx = best_m_idx
            else:
                self._rollback(app, i)
                return False
        return True

    def _find_best_placement(self, app: Application, ms: Microservice, ms_idx: int, prev_mach_idx: int) -> Tuple[int, int, bool]:
        best_score = float('inf')
        best_cand = (-1, -1, False)
        
        is_last = (ms_idx == len(app.microservices) - 1)
        sb = compute_service_bounds(ms, ms_idx, is_last=is_last, return_to_ue=app.return_to_ue)
        
        # Check mTLS requirement (Constraint)
        base_required_mtls = False
        if ms_idx > 0 and app.mtls_requirements[ms_idx-1]:
            base_required_mtls = True # Hard Constraint
            
        # Filter Machines
        m_sec = self.topo.machines[:, self.topo.SEC_IDX]
        sec_mask = (m_sec >= ms.min_security)
        candidates = np.where(sec_mask)[0]
            
        if len(candidates) == 0:
            return best_cand
        
        # Iterate candidates
        for m_idx in candidates:
            tier_cap = int(self.topo.machines[m_idx, self.topo.SEC_IDX])
            valid_max = min(ms.max_security, tier_cap)
            valid_min = ms.min_security
            
            # Machine Metrics
            base_rate = self.topo.machines[m_idx, self.topo.COST_IDX]
            node_id = int(self.topo.machines[m_idx, self.topo.NODE_IDX])
            cat = self.topo.node_categories[node_id]
            coeff = self.topo.machines[m_idx, self.topo.SPEED_IDX]
            exec_time = ms.typical_runtime / coeff
            
            # --- Trust Constraint ---
            target_cluster = int(self.topo.node_clusters[node_id])
            is_trusted = True
            if self.topo.trust_matrix is not None:
                is_trusted = bool(self.topo.trust_matrix[app.source_cluster, target_cluster])
            
            if not is_trusted:
                # Untrusted: tier 0 forbidden (must be >= 1)
                valid_min = max(valid_min, 1)
                if valid_min > valid_max:
                    continue  # Machine can't satisfy trust tier constraint
            
            # Determine mTLS requirement (base or trust-forced)
            required_mtls = base_required_mtls
            if not is_trusted and ms_idx > 0:
                required_mtls = True  # Force mTLS for untrusted placement
            
            # Latency Base
            lat_base = exec_time
            comm_base = 0.0
            
            if prev_mach_idx == -1:
                gen_p = app.gen_point
                comm_base = self.topo.node_to_user_latency[node_id, gen_p]
            else:
                prev_node = int(self.topo.machines[prev_mach_idx, self.topo.NODE_IDX])
                if node_id == prev_node:
                    if m_idx == prev_mach_idx: comm_base = cfg.INTRA_MACHINE_LATENCY
                    else: comm_base = cfg.INTRA_NODE_LATENCY
                else:
                    comm_base = self.topo.network_latency[prev_node, node_id]
            
            lat_base += comm_base
            
            # Return Path (Partial Estimate)
            if app.return_to_ue and ms_idx == len(app.microservices) - 1:
                 lat_base += self.topo.node_to_user_latency[node_id, app.gen_point]
            
            # --- Branching: mTLS Options ---
            mtls_options = [True] if required_mtls else [True, False]
            if ms_idx == 0: mtls_options = [False]
            
            for use_mtls in mtls_options:
                # Check Capacity on PREVIOUS machine if use_mtls is True (Optional Check)
                # Guard matches _apply_assignment: deduction happens whenever the link is not
                # base-required, which covers both optional-trusted AND trust-forced-untrusted.
                if use_mtls and ms_idx > 0:
                     if not base_required_mtls:
                         if prev_mach_idx != -1 and self.topo.machines[prev_mach_idx, self.topo.CPU_IDX] < app.mtls_overheads[ms_idx - 1]:
                             continue 
                
                # Check Local Machine Capacity
                mtls_cpu_local = 0.0
                if use_mtls: mtls_cpu_local += app.mtls_overheads[ms_idx - 1] # Ingress
                if ms_idx < len(app.microservices)-1 and app.mtls_requirements[ms_idx]:
                    mtls_cpu_local += app.mtls_overheads[ms_idx] # Egress (Required)
                    
                # Iterate Tiers
                for tier in range(valid_min, valid_max + 1):
                     req_cpu = ms.cpu_demand * ms.security_overheads[tier] + mtls_cpu_local
                     
                     if (self.topo.machines[m_idx, self.topo.CPU_IDX] < req_cpu):
                        continue
                        
                     # Calculate Local Objective Contribution (Weighted Raw Metrics)
                     
                     # 1. Cost
                     cost_mult_val = 1.0
                     if cat == 1: cost_mult_val = cfg.COST_MULT_EDGE[tier]
                     elif cat == 2: cost_mult_val = cfg.COST_MULT_FOG[tier]
                     elif cat == 3: cost_mult_val = cfg.COST_MULT_CLOUD[tier]
                     
                     cost_val = base_rate * cost_mult_val * exec_time
                     
                     # 2. Latency
                     curr_lat = lat_base
                     if use_mtls and ms_idx > 0:
                         prev_data = app.microservices[ms_idx-1].data_size
                         pen = cfg.MTLS_LATENCY_FIXED + (cfg.MTLS_LATENCY_PER_MB * prev_data)
                         curr_lat += pen
                     
                     # 3. Security (Weighted by Runtime)
                     w_tier = getattr(cfg, 'SEC_TIER_WEIGHT', 1.0)
                     w_mtls = getattr(cfg, 'SEC_MTLS_WEIGHT', 1.0)
                     sec_val = float(tier) * w_tier
                     if use_mtls: sec_val += 1.0 * w_mtls
                     sec_rt = ms.typical_runtime if getattr(cfg, 'SEC_RUNTIME_WEIGHTED', True) else 1.0
                     sec_val *= sec_rt # WEIGHTING (runtime-weighted unless disabled)
                     
                     # Range-based normalization using per-service local bounds
                     n_cost = normalize(cost_val, sb['min_cost'], sb['max_cost'], 'cost')
                     n_lat = normalize(curr_lat, sb['min_lat'], sb['max_lat'], 'lat')
                     n_sec = normalize(sec_val, sb['min_sec'], sb['max_sec'], 'sec')
                     
                     score = (cfg.W1 * n_cost) + (cfg.W3 * n_lat) + (cfg.W2 * (1 - n_sec))
                     
                     if score < best_score:
                         best_score = score
                         best_cand = (m_idx, tier, use_mtls)
                         
        return best_cand

    def _apply_assignment(self, ms, m_idx, tier, use_mtls, app, ms_idx):
        ms.assigned_machine_id = m_idx
        ms.assigned_tier = tier
        ms.assigned_node_id = int(self.topo.machines[m_idx, self.topo.NODE_IDX])
        ms.rflag = True
        
        # Record mTLS Decision
        if ms_idx > 0:
            app.mtls_decisions[ms_idx-1] = use_mtls
            
        # Deduct Resources
        mtls_cpu_ingress = app.mtls_overheads[ms_idx - 1] if use_mtls else 0.0
        mtls_cpu_egress_reserved = 0.0
        if ms_idx < len(app.microservices)-1 and app.mtls_requirements[ms_idx]:
             mtls_cpu_egress_reserved = app.mtls_overheads[ms_idx]
             
        cpu_d = ms.cpu_demand * ms.security_overheads[tier]
        cpu_d += mtls_cpu_ingress + mtls_cpu_egress_reserved
        
        self.topo.machines[m_idx, self.topo.CPU_IDX] -= cpu_d
        self.topo.machines[m_idx, self.topo.LOAD_IDX] += 1
        
        # Deduct from PREVIOUS machine (Egress) if use_mtls
        if use_mtls and ms_idx > 0:
            prev_ms = app.microservices[ms_idx-1]
            p_idx = prev_ms.assigned_machine_id
            if p_idx != -1:
                if not app.mtls_requirements[ms_idx-1]:
                     self.topo.machines[p_idx, self.topo.CPU_IDX] -= app.mtls_overheads[ms_idx - 1]

    def _rollback(self, app, failed_at):
        for i in range(failed_at):
            ms = app.microservices[i]
            m_idx = ms.assigned_machine_id
            tier = ms.assigned_tier
            
            if m_idx == -1: continue
            
            # 1. Self Resources
            mtls_ingress = 0.0
            if i > 0 and app.mtls_decisions[i-1]: mtls_ingress = app.mtls_overheads[i - 1]
            
            mtls_egress = 0.0
            if i < len(app.microservices)-1 and app.mtls_requirements[i]:
                mtls_egress = app.mtls_overheads[i] if i < len(app.microservices)-1 else 0.0
                
            cpu_d = ms.cpu_demand * ms.security_overheads[tier] + mtls_ingress + mtls_egress
            self.topo.machines[m_idx, self.topo.CPU_IDX] += cpu_d
            self.topo.machines[m_idx, self.topo.LOAD_IDX] -= 1
            
            # 2. Prev Machine (Optional Egress)
            if i > 0 and app.mtls_decisions[i-1]:
                if not app.mtls_requirements[i-1]:
                     p_idx = app.microservices[i-1].assigned_machine_id
                     if p_idx != -1:
                        self.topo.machines[p_idx, self.topo.CPU_IDX] += app.mtls_overheads[i - 1]
                        
            ms.assigned_machine_id = -1
            ms.assigned_node_id = -1
            ms.rflag = False

# --- Worker Function for Rollout ---
def evaluate_batch_v2(args):
    topo_master, app_master, ms_index, candidates, use_fast = args
    # Note: app_master MUST have global_bounds attached!
    
    res_backup = topo_master.machines.copy()
    
    # BACKUP APP MASTER STATE
    mtls_backup = list(app_master.mtls_decisions)
    ms_state_backup = []
    for m in app_master.microservices:
        ms_state_backup.append((m.assigned_machine_id, m.assigned_tier, getattr(m, 'assigned_node_id', -1), getattr(m, 'rflag', False)))
    
    best_local_decision = None
    best_local_score = float('inf')
    
    if use_fast:
        from greedy_fast import GreedyAllocatorFast
        allocator = GreedyAllocatorFast(topo_master)
    else:
        allocator = GreedyAllocator(topo_master)
    
    prev_mach_idx = -1
    if ms_index > 0:
        prev_mach_idx = app_master.microservices[ms_index-1].assigned_machine_id
    
    for m_idx, tier, use_mtls in candidates:
        # 1. Apply Move
        ms = app_master.microservices[ms_index]
        
        mtls_cpu_ingress = app_master.mtls_overheads[ms_index - 1] if use_mtls else 0.0
        mtls_cpu_egress_reserved = 0.0
        if ms_index < len(app_master.microservices) - 1 and app_master.mtls_requirements[ms_index]:
             mtls_cpu_egress_reserved = app_master.mtls_overheads[ms_index]

        cpu_d = ms.cpu_demand * ms.security_overheads[tier]
        cpu_d += mtls_cpu_ingress + mtls_cpu_egress_reserved
        
        topo_master.machines[m_idx, topo_master.CPU_IDX] -= cpu_d
        topo_master.machines[m_idx, topo_master.LOAD_IDX] += 1
        
        ms.assigned_machine_id = m_idx
        ms.assigned_tier = tier
        ms.assigned_node_id = int(topo_master.machines[m_idx, topo_master.NODE_IDX])
        
        if ms_index > 0:
            app_master.mtls_decisions[ms_index-1] = use_mtls
            
        if use_mtls and ms_index > 0:
             if not app_master.mtls_requirements[ms_index-1]:
                  if prev_mach_idx != -1:
                        topo_master.machines[prev_mach_idx, topo_master.CPU_IDX] -= app_master.mtls_overheads[ms_index - 1]
        
        # 2. Run Greedy Tail
        success = allocator.allocate_application(app_master) 
        
        if success:
            # Score now uses Global Normalization (app_master has bounds)
            score = calculate_app_score_static(topo_master, app_master)
            if score < best_local_score:
                best_local_score = score
                best_local_decision = (m_idx, tier, use_mtls)
        
        # 3. Revert
        topo_master.machines[:] = res_backup
        app_master.mtls_decisions = list(mtls_backup)
        for idx, m in enumerate(app_master.microservices):
            m.assigned_machine_id, m.assigned_tier, m.assigned_node_id, m.rflag = ms_state_backup[idx]
             
    if best_local_decision:
        return (best_local_decision[0], best_local_decision[1], best_local_decision[2], best_local_score)
    return None

class RolloutAllocatorV2:
    def __init__(self, topo: Topology, max_branching=None):
        self.topo = topo
        from greedy_fast import resolve_greedy_allocator
        self.greedy = resolve_greedy_allocator(topo)
        self.total_heuristic_calls = 0
        self.max_branching = max_branching
        self._use_fast_greedy = getattr(cfg, 'USE_FAST_GREEDY', False)
        
    def solve(self, apps: List[Application]):
         with concurrent.futures.ProcessPoolExecutor(max_workers=cfg.MAX_WORKERS) as executor:
            n_apps = len(apps)
            for app_idx, app in enumerate(apps):
                # Pre-calculate Global Bounds (Required for worker)
                app.global_bounds = compute_app_global_bounds(app)
                
                app.mtls_decisions = [False] * (len(app.microservices) - 1)
                self.rollout_app(app, executor)
                if getattr(cfg, 'ROLLOUT_PROGRESS', False):
                    print(f"  [App-Rollout] {app_idx + 1}/{n_apps} apps", flush=True)
                
    def rollout_app(self, app, executor):
        for i, ms in enumerate(app.microservices):
            candidates = self._get_candidates(app, ms, i)
            
            if not candidates:
                self.greedy._rollback(app, i)
                return False
                 
            self.total_heuristic_calls += len(candidates)
                 
            chunk_size = max(1, len(candidates) // cfg.MAX_WORKERS)
            chunks = [candidates[k:k+chunk_size] for k in range(0, len(candidates), chunk_size)]
            
            tasks = [(self.topo, app, i, chunk, self._use_fast_greedy) for chunk in chunks]
            
            results = executor.map(evaluate_batch_v2, tasks)
            
            best_res = None
            best_s = float('inf')
            
            for res in results:
                if res and res[3] < best_s:
                    best_s = res[3]
                    best_res = res
            
            if best_res:
                m, t, use_mtls, s = best_res
                self.greedy._apply_assignment(ms, m, t, use_mtls, app, i)
            else:
                self.greedy._rollback(app, i)
                return False
        return True

    def _get_candidates(self, app, ms, i):
        candidates = []
        
        base_required_mtls = False
        if i > 0 and app.mtls_requirements[i-1]: base_required_mtls = True
        
        m_sec = self.topo.machines[:, self.topo.SEC_IDX]
        
        sec_mask = (m_sec >= ms.min_security)
        candidate_indices = np.where(sec_mask)[0]
        
        for m_idx in candidate_indices:
            tier_cap = int(self.topo.machines[m_idx, self.topo.SEC_IDX])
            valid_max = min(ms.max_security, tier_cap)
            valid_min = ms.min_security
            
            # --- Trust Constraint ---
            node_id = int(self.topo.machines[m_idx, self.topo.NODE_IDX])
            target_cluster = int(self.topo.node_clusters[node_id])
            is_trusted = True
            if self.topo.trust_matrix is not None:
                is_trusted = bool(self.topo.trust_matrix[app.source_cluster, target_cluster])
            
            if not is_trusted:
                valid_min = max(valid_min, 1)  # Ban tier 0 for untrusted
                if valid_min > valid_max:
                    continue
            
            required_mtls = base_required_mtls
            if not is_trusted and i > 0:
                required_mtls = True  # Force mTLS for untrusted
            
            mtls_options = [True] if required_mtls else [True, False]
            if i == 0: mtls_options = [False]
            
            if cfg.W2 <= 0 and not required_mtls:
                mtls_options = [False]
            
            for use_mtls in mtls_options:
                if use_mtls and i > 0:
                     p_idx = app.microservices[i-1].assigned_machine_id
                     if not base_required_mtls:
                         if p_idx != -1 and self.topo.machines[p_idx, self.topo.CPU_IDX] < app.mtls_overheads[i - 1]:
                               continue
            
                for tier in range(valid_min, valid_max + 1):
                    mtls_ingress = app.mtls_overheads[i - 1] if use_mtls else 0.0
                    mtls_egress_reserved = 0.0
                    if i < len(app.microservices)-1 and app.mtls_requirements[i]:
                         mtls_egress_reserved = app.mtls_overheads[i]
                         
                    req_cpu = ms.cpu_demand * ms.security_overheads[tier] + mtls_ingress + mtls_egress_reserved
                    
                    if (self.topo.machines[m_idx, self.topo.CPU_IDX] >= req_cpu):
                        candidates.append((m_idx, tier, use_mtls))
                        
        if self.max_branching is not None and len(candidates) > self.max_branching:
            candidates = random.sample(candidates, self.max_branching)
                
        return candidates


# --- Simulated Annealing Allocator ---
class SimulatedAnnealingAllocator:
    """
    Simulated Annealing metaheuristic for microservice placement.
    Starts from the Greedy best-fit baseline and performs local
    single-service reassignment moves to improve the objective score.
    Operates per-application (consistent with Greedy/Rollout framework).
    """
    def __init__(self, topo: Topology):
        self.topo = topo
        self.greedy = GreedyAllocator(topo)

    def solve(self, apps: List[Application], time_budget: float = 60.0, max_iterations: int = 1000):
        """
        For each application:
        1. Run Greedy to get baseline placement.
        2. Run SA (max_iterations attempts per app) to improve the placement.
        """
        # Phase 1: Run Greedy baseline for all apps
        self.greedy.solve(apps)
        
        # Count successfully placed apps
        feasible_apps = []
        for app in apps:
            all_placed = all(ms.assigned_machine_id != -1 for ms in app.microservices)
            if all_placed and len(app.microservices) > 0:
                feasible_apps.append(app)
        
        if not feasible_apps:
            return
        
        # Phase 2: SA improvement per app
        for app in feasible_apps:
            self._sa_optimize_app(app, max_attempts=max_iterations)
    
    def _sa_optimize_app(self, app: Application, max_attempts: int = 1000):
        """
        Run Simulated Annealing on a single application's placement.
        """
        n_services = len(app.microservices)
        if n_services == 0:
            return
        
        # Calculate current score (baseline from Greedy)
        current_score = calculate_app_score_static(self.topo, app)
        if current_score == float('inf'):
            return  # Infeasible baseline, skip
        
        best_score = current_score
        
        # Save best solution state
        best_assignments = self._save_app_state(app)
        best_topo_cpu = self.topo.machines[:, self.topo.CPU_IDX].copy()
        best_topo_load = self.topo.machines[:, self.topo.LOAD_IDX].copy()
        
        initial_temp = getattr(cfg, 'SA_INITIAL_TEMP', 1.0)
        min_temp = getattr(cfg, 'SA_MIN_TEMP', 0.001)
        # Geometric cooling factor over max_attempts steps
        cooling_rate = (min_temp / initial_temp) ** (1.0 / max_attempts)
        
        T = initial_temp
        attempts = 0
        
        # Track consecutive failures to find any neighbor across all services to prevent infinite loop
        consecutive_no_neighbors = 0
        
        while attempts < max_attempts:
            # 1. Pick a random microservice to reposition (one service at a time)
            ms_idx = random.randint(0, n_services - 1)
            ms = app.microservices[ms_idx]
            
            # 2. Mask infeasible actions: get only feasible alternatives (excluding current assignment)
            neighbors = self._get_feasible_neighbors(app, ms, ms_idx)
            
            if not neighbors:
                consecutive_no_neighbors += 1
                if consecutive_no_neighbors > 200:
                    # If we tried 200 times and found zero neighbors for chosen services,
                    # check if the app is completely locked down (no moves possible) and break
                    all_locked = True
                    for s_idx in range(n_services):
                        if self._get_feasible_neighbors(app, app.microservices[s_idx], s_idx):
                            all_locked = False
                            break
                    if all_locked:
                        break
                    consecutive_no_neighbors = 0
                continue
            
            consecutive_no_neighbors = 0
            
            # 3. Pick a random feasible neighbor
            new_m_idx, new_tier, new_mtls = random.choice(neighbors)
            
            # 4. Save current old state to allow exact revert
            old_m_idx = ms.assigned_machine_id
            old_tier = ms.assigned_tier
            old_node_id = ms.assigned_node_id
            old_mtls = None
            if ms_idx > 0:
                old_mtls = app.mtls_decisions[ms_idx - 1]
            
            # 5. Tentatively unapply old service assignment (freeing CPU & loads)
            self._unapply_service(app, ms, ms_idx)
            
            # 6. Verify absolute feasibility of the new placement (strict capacity check)
            # a) CPU demand on new machine
            mtls_cpu_ingress = app.mtls_overheads[ms_idx - 1] if new_mtls else 0.0
            mtls_cpu_egress_reserved = 0.0
            if ms_idx < n_services - 1 and app.mtls_requirements[ms_idx]:
                mtls_cpu_egress_reserved = app.mtls_overheads[ms_idx]
            
            req_cpu = ms.cpu_demand * ms.security_overheads[new_tier] + mtls_cpu_ingress + mtls_cpu_egress_reserved
            
            # Add own optional egress if applicable
            if ms_idx < n_services - 1:
                egress_active_optional = app.mtls_decisions[ms_idx] and not app.mtls_requirements[ms_idx]
                if egress_active_optional:
                    req_cpu += app.mtls_overheads[ms_idx]
            
            # b) Previous machine's optional mTLS egress if applicable
            req_prev_cpu = 0.0
            if new_mtls and ms_idx > 0 and not app.mtls_requirements[ms_idx - 1]:
                req_prev_cpu = app.mtls_overheads[ms_idx - 1]
                
            fit = True
            if self.topo.machines[new_m_idx, self.topo.CPU_IDX] < req_cpu:
                fit = False
                
            if fit and req_prev_cpu > 0:
                prev_ms = app.microservices[ms_idx - 1]
                prev_m_idx = prev_ms.assigned_machine_id
                if prev_m_idx == new_m_idx:
                    # Previous service is on the same machine
                    if self.topo.machines[new_m_idx, self.topo.CPU_IDX] < (req_cpu + req_prev_cpu):
                        fit = False
                elif prev_m_idx != -1:
                    if self.topo.machines[prev_m_idx, self.topo.CPU_IDX] < req_prev_cpu:
                        fit = False
                        
            if not fit:
                # Infeasible under exact check — revert to previous assignment
                self._reapply_service(app, ms, ms_idx, old_m_idx, old_tier, old_node_id, old_mtls)
                continue
                
            # 7. Apply the new placement
            self._apply_service(app, ms, ms_idx, new_m_idx, new_tier, new_mtls)
            
            # 8. Compute new objective score
            new_score = calculate_app_score_static(self.topo, app)
            
            if new_score == float('inf'):
                # Infeasible total chain state — revert
                self._unapply_service(app, ms, ms_idx)
                self._reapply_service(app, ms, ms_idx, old_m_idx, old_tier, old_node_id, old_mtls)
                continue
                
            # This is a valid feasible repositioning attempt
            attempts += 1
            delta = new_score - current_score
            
            # 9. Accept or Reject with Metropolis criteria
            if delta < 0 or random.random() < math.exp(-delta / T):
                # Accept the move
                current_score = new_score
                if current_score < best_score:
                    best_score = current_score
                    best_assignments = self._save_app_state(app)
                    best_topo_cpu = self.topo.machines[:, self.topo.CPU_IDX].copy()
                    best_topo_load = self.topo.machines[:, self.topo.LOAD_IDX].copy()
            else:
                # Reject the move — revert
                self._unapply_service(app, ms, ms_idx)
                self._reapply_service(app, ms, ms_idx, old_m_idx, old_tier, old_node_id, old_mtls)
                
            # 10. Cool the temperature
            T *= cooling_rate
            
        # Restore best solution found during search
        self._restore_app_state(app, best_assignments)
        self.topo.machines[:, self.topo.CPU_IDX] = best_topo_cpu
        self.topo.machines[:, self.topo.LOAD_IDX] = best_topo_load
    
    def _get_feasible_neighbors(self, app: Application, ms: Microservice, ms_idx: int) -> List[Tuple[int, int, bool]]:
        """
        Get all feasible (machine, tier, mTLS) alternatives for a microservice,
        excluding the current assignment.
        """
        neighbors = []
        n_services = len(app.microservices)
        
        current_m_idx = ms.assigned_machine_id
        current_tier = ms.assigned_tier
        current_mtls = app.mtls_decisions[ms_idx - 1] if ms_idx > 0 else False
        
        # mTLS requirement on ingress link
        base_required_mtls = False
        if ms_idx > 0 and app.mtls_requirements[ms_idx - 1]:
            base_required_mtls = True
        
        # Filter machines by minimum security
        m_sec = self.topo.machines[:, self.topo.SEC_IDX]
        sec_mask = (m_sec >= ms.min_security)
        candidate_indices = np.where(sec_mask)[0]
        
        for m_idx in candidate_indices:
            tier_cap = int(self.topo.machines[m_idx, self.topo.SEC_IDX])
            valid_max = min(ms.max_security, tier_cap)
            valid_min = ms.min_security
            
            # Trust Constraint
            node_id = int(self.topo.machines[m_idx, self.topo.NODE_IDX])
            target_cluster = int(self.topo.node_clusters[node_id])
            is_trusted = True
            if self.topo.trust_matrix is not None:
                is_trusted = bool(self.topo.trust_matrix[app.source_cluster, target_cluster])
            
            if not is_trusted:
                valid_min = max(valid_min, 1)
                if valid_min > valid_max:
                    continue
            
            required_mtls = base_required_mtls
            if not is_trusted and ms_idx > 0:
                required_mtls = True
            
            mtls_options = [True] if required_mtls else [True, False]
            if ms_idx == 0:
                mtls_options = [False]
            
            if cfg.W2 <= 0 and not required_mtls:
                mtls_options = [False]
            
            for use_mtls in mtls_options:
                # Check previous machine capacity for mTLS egress
                if use_mtls and ms_idx > 0:
                    p_idx = app.microservices[ms_idx - 1].assigned_machine_id
                    if not base_required_mtls:
                        if p_idx != -1 and self.topo.machines[p_idx, self.topo.CPU_IDX] < app.mtls_overheads[ms_idx - 1]:
                            continue
                
                for tier in range(valid_min, valid_max + 1):
                    # Skip current assignment
                    if m_idx == current_m_idx and tier == current_tier and use_mtls == current_mtls:
                         continue
                    
                    # CPU capacity check (approximate — actual check done on apply)
                    mtls_ingress = app.mtls_overheads[ms_idx - 1] if use_mtls else 0.0
                    mtls_egress_reserved = 0.0
                    if ms_idx < n_services - 1 and app.mtls_requirements[ms_idx]:
                        mtls_egress_reserved = app.mtls_overheads[ms_idx]
                    
                    req_cpu = ms.cpu_demand * ms.security_overheads[tier] + mtls_ingress + mtls_egress_reserved
                    
                    # Available CPU: current capacity + resources that will be freed if moving away
                    avail_cpu = self.topo.machines[m_idx, self.topo.CPU_IDX]
                    if m_idx == current_m_idx:
                        old_ingress = app.mtls_overheads[ms_idx - 1] if current_mtls else 0.0
                        old_egress = 0.0
                        if ms_idx < n_services - 1 and app.mtls_requirements[ms_idx]:
                            old_egress = app.mtls_overheads[ms_idx]
                        old_cpu = ms.cpu_demand * ms.security_overheads[current_tier] + old_ingress + old_egress
                        avail_cpu += old_cpu
                    
                    if avail_cpu >= req_cpu:
                        neighbors.append((m_idx, tier, use_mtls))
        
        return neighbors
    
    def _compute_service_cpu(self, app: Application, ms: Microservice, ms_idx: int, tier: int, use_mtls: bool) -> float:
        """Compute total CPU demand for a service assignment."""
        n_services = len(app.microservices)
        mtls_cpu_ingress = app.mtls_overheads[ms_idx - 1] if use_mtls else 0.0
        mtls_cpu_egress_reserved = 0.0
        if ms_idx < n_services - 1 and app.mtls_requirements[ms_idx]:
            mtls_cpu_egress_reserved = app.mtls_overheads[ms_idx]
        return ms.cpu_demand * ms.security_overheads[tier] + mtls_cpu_ingress + mtls_cpu_egress_reserved
    
    def _unapply_service(self, app: Application, ms: Microservice, ms_idx: int):
        """Remove the current assignment of a service, restoring resources."""
        m_idx = ms.assigned_machine_id
        tier = ms.assigned_tier
        
        if m_idx == -1:
            return
        
        use_mtls = app.mtls_decisions[ms_idx - 1] if ms_idx > 0 else False
        cpu_used = self._compute_service_cpu(app, ms, ms_idx, tier, use_mtls)
        
        # Restore resources on the machine
        self.topo.machines[m_idx, self.topo.CPU_IDX] += cpu_used
        self.topo.machines[m_idx, self.topo.LOAD_IDX] -= 1
        
        # Restore previous machine's mTLS egress cost (if optional mTLS was used)
        if use_mtls and ms_idx > 0:
            if not app.mtls_requirements[ms_idx - 1]:
                prev_ms = app.microservices[ms_idx - 1]
                p_idx = prev_ms.assigned_machine_id
                if p_idx != -1:
                    self.topo.machines[p_idx, self.topo.CPU_IDX] += app.mtls_overheads[ms_idx - 1]
        
        # Restore own machine's mTLS optional egress cost (if optional mTLS is active on egress link)
        if ms_idx < len(app.microservices) - 1:
            egress_active_optional = app.mtls_decisions[ms_idx] and not app.mtls_requirements[ms_idx]
            if egress_active_optional:
                self.topo.machines[m_idx, self.topo.CPU_IDX] += app.mtls_overheads[ms_idx]
        
        # Clear assignment
        ms.assigned_machine_id = -1
        ms.assigned_node_id = -1
        ms.assigned_tier = 0
        ms.rflag = False
        if ms_idx > 0:
            app.mtls_decisions[ms_idx - 1] = False
    
    def _apply_service(self, app: Application, ms: Microservice, ms_idx: int,
                       m_idx: int, tier: int, use_mtls: bool):
        """Apply a new assignment for a service."""
        ms.assigned_machine_id = m_idx
        ms.assigned_tier = tier
        ms.assigned_node_id = int(self.topo.machines[m_idx, self.topo.NODE_IDX])
        ms.rflag = True
        
        if ms_idx > 0:
            app.mtls_decisions[ms_idx - 1] = use_mtls
        
        cpu_used = self._compute_service_cpu(app, ms, ms_idx, tier, use_mtls)
        self.topo.machines[m_idx, self.topo.CPU_IDX] -= cpu_used
        self.topo.machines[m_idx, self.topo.LOAD_IDX] += 1
        
        # Deduct from previous machine (egress) if optional mTLS
        if use_mtls and ms_idx > 0:
            if not app.mtls_requirements[ms_idx - 1]:
                prev_ms = app.microservices[ms_idx - 1]
                p_idx = prev_ms.assigned_machine_id
                if p_idx != -1:
                    self.topo.machines[p_idx, self.topo.CPU_IDX] -= app.mtls_overheads[ms_idx - 1]
                    
        # Deduct from own machine (egress) if optional mTLS is active on egress link
        if ms_idx < len(app.microservices) - 1:
            egress_active_optional = app.mtls_decisions[ms_idx] and not app.mtls_requirements[ms_idx]
            if egress_active_optional:
                self.topo.machines[m_idx, self.topo.CPU_IDX] -= app.mtls_overheads[ms_idx]
    
    def _reapply_service(self, app: Application, ms: Microservice, ms_idx: int,
                         m_idx: int, tier: int, node_id: int, old_mtls: Optional[bool]):
        """Re-apply a previously saved assignment (revert after unapply)."""
        use_mtls = old_mtls if old_mtls is not None else False
        self._apply_service(app, ms, ms_idx, m_idx, tier, use_mtls)
        ms.assigned_node_id = node_id
    
    def _save_app_state(self, app: Application) -> dict:
        """Save the full assignment state of an application."""
        state = {
            'assignments': [],
            'mtls_decisions': app.mtls_decisions.copy()
        }
        for ms in app.microservices:
            state['assignments'].append({
                'machine_id': ms.assigned_machine_id,
                'tier': ms.assigned_tier,
                'node_id': ms.assigned_node_id,
                'rflag': ms.rflag
            })
        return state
    
    def _restore_app_state(self, app: Application, state: dict):
        """Restore an application to a previously saved state."""
        app.mtls_decisions = state['mtls_decisions'].copy()
        for i, ms in enumerate(app.microservices):
            s = state['assignments'][i]
            ms.assigned_machine_id = s['machine_id']
            ms.assigned_tier = s['tier']
            ms.assigned_node_id = s['node_id']
            ms.rflag = s['rflag']

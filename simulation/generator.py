import numpy as np
from simulation import config as cfg
from simulation.structs import Topology, Application, Microservice
import random

def segment_nodes_into_clusters(num_nodes: int, num_clusters: int) -> np.ndarray:
    """
    Segments nodes into clusters ensuring every cluster has at least one node.
    """
    if num_clusters > num_nodes:
        raise ValueError("Cannot have more clusters than nodes")
        
    clusters = np.zeros(num_nodes, dtype=int)
    
    # 1. Ensure each cluster has at least one node
    # Allow shuffling of node indices to make it random which specific nodes get picked
    node_indices = np.arange(num_nodes)
    np.random.shuffle(node_indices)
    
    # Assign first k nodes to k clusters
    for i in range(num_clusters):
        clusters[node_indices[i]] = i
        
    # Assign remaining nodes randomly
    for i in range(num_clusters, num_nodes):
        clusters[node_indices[i]] = np.random.randint(0, num_clusters)
        
    return clusters

def generate_topology() -> Topology:
    # 1. Define Nodes based on Config Counts
    # 50 Edge (1), 20 Fog (2), 5 Cloud (3) -> Derived from cfg
    
    # We construct the array explicitly
    cat_edge = np.full(cfg.NUM_EDGE, 1)
    cat_fog = np.full(cfg.NUM_FOG, 2)
    cat_cloud = np.full(cfg.NUM_CLOUD, 3)
    categories = np.concatenate([cat_edge, cat_fog, cat_cloud])
    
    num_nodes = len(categories) # Should equal cfg.NUM_NODES
        
    # 1A. Segment Nodes into Clusters
    node_clusters = segment_nodes_into_clusters(num_nodes, cfg.NUM_CLUSTERS)
        
    # 2. Generate Network Latency Matrix (Symmetric)
    lat_matrix = np.zeros((num_nodes, num_nodes))
    
    # We iterate over unique pairs (i, j) where i < j
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            cat_i = categories[i]
            cat_j = categories[j]
            
            # Key is sorted tuple of categories to handle (Edge, Fog) == (Fog, Edge)
            pair = tuple(sorted((cat_i, cat_j)))
            
            latency = 0.0
            
            if pair == (1, 1): # Edge-Edge
                latency = round(np.random.uniform(*cfg.DELAY_EDGE_EDGE), 2)
            elif pair == (1, 2): # Edge-Fog
                latency = round(np.random.uniform(*cfg.DELAY_EDGE_FOG), 2)
            elif pair == (1, 3): # Edge-Cloud
                latency = round(np.random.uniform(*cfg.DELAY_EDGE_CLOUD), 2)
            elif pair == (2, 2): # Fog-Fog
                latency = round(np.random.uniform(*cfg.DELAY_FOG_FOG), 2)
            elif pair == (2, 3): # Fog-Cloud
                latency = round(np.random.uniform(*cfg.DELAY_FOG_CLOUD), 2)
            elif pair == (3, 3): # Cloud-Cloud
                 latency = round(np.random.uniform(*cfg.DELAY_CLOUD_CLOUD), 2)
            
            # --- CLUSTER LATENCY DISCOUNT ---
            # If nodes are in the same cluster, apply discount
            if node_clusters[i] == node_clusters[j]:
                latency *= cfg.INTRA_CLUSTER_LATENCY_FACTOR
                latency = round(latency, 2)
            
            lat_matrix[i, j] = latency
            lat_matrix[j, i] = latency

    # Diagonal is 0 (handled by intra_node property logic elsewhere)
    
    # 3. User Latency Matrix (Node to Cluster Gateway)
    # Dimensions: [TargetNode, ClusterID]
    # user_lat[n, c] = Latency from Node n to Cluster Gateway c
    
    # Logic:
    # If node n is in cluster c: Low Latency (Base * Discount)
    # If node n is NOT in cluster c: Base Latency (No Discount, No Penalty)
    
    user_lat = np.zeros((num_nodes, cfg.NUM_CLUSTERS))
    
    # Pre-calculate Base User Access Delays for each node
    # This represents "standard access" delay (Inter-cluster baseline)
    base_access_delays = np.zeros(num_nodes)
    for n in range(num_nodes):
        cat = categories[n]
        if cat == 1: # Edge
            base_access_delays[n] = round(np.random.uniform(*cfg.DELAY_USER_EDGE), 2)
        elif cat == 2: # Fog
            base_access_delays[n] = round(np.random.uniform(*cfg.DELAY_USER_FOG), 2)
        else: # Cloud
            base_access_delays[n] = round(np.random.uniform(*cfg.DELAY_USER_CLOUD), 2)
            
    for n in range(num_nodes):
        for c in range(cfg.NUM_CLUSTERS):
            base = base_access_delays[n]
            
            if node_clusters[n] == c:
                # Same Cluster: Apply Discount (Cooperative Domain)
                # "node will have generally less latency towards that application"
                # Logic: Base Access * Discount
                val = base * cfg.INTRA_CLUSTER_LATENCY_FACTOR
            else:
                # Different Cluster: Base (No Penalty)
                val = base
                
            user_lat[n, c] = round(val, 2)
    
 
    
    # 3. Generate Machines
    machines_list = []
    machine_rows = []
    
    for n_id in range(num_nodes):
        cat = categories[n_id]
        
        # Determine num machines
        if cat == 1: # Edge
            num_mach = np.random.randint(cfg.MACHINES_EDGE[0], cfg.MACHINES_EDGE[1] + 1)
            base_cpu = cfg.SPECS_EDGE_CPU
            # base_ram/sto removed
            base_cost = cfg.SPECS_EDGE_COST
            security_prob = cfg.PROB_EDGE
            
        elif cat == 2: # Fog
            num_mach = np.random.randint(cfg.MACHINES_FOG[0], cfg.MACHINES_FOG[1] + 1)
            base_cpu = cfg.SPECS_FOG_CPU
            # base_ram/sto removed
            base_cost = cfg.SPECS_FOG_COST
            security_prob = cfg.PROB_FOG
            
        else: # Cloud
            num_mach = np.random.randint(cfg.MACHINES_CLOUD[0], cfg.MACHINES_CLOUD[1] + 1)
            base_cpu = cfg.SPECS_CLOUD_CPU
            # base_ram/sto removed
            base_cost = cfg.SPECS_CLOUD_COST
            security_prob = cfg.PROB_CLOUD
        
        for _ in range(num_mach):
            # Speed (Stochastic)
            if cat == 1: sp_rng = cfg.SPEED_COEFF_EDGE
            elif cat == 2: sp_rng = cfg.SPEED_COEFF_FOG
            else: sp_rng = cfg.SPEED_COEFF_CLOUD
            
            # DEBUG
            # print(f"Cat: {cat}, sp_rng: {sp_rng}, type: {type(sp_rng)}")
            
            mach_speed = round(np.random.uniform(*sp_rng), 2)
            
            # Cost Generation (User Request: Correlate with Speed)
            # 1. Normalize Speed in Range [0, 1]
            s_min, s_max = sp_rng
            if s_max > s_min:
                norm_speed = (mach_speed - s_min) / (s_max - s_min)
            else:
                norm_speed = 0.5 # Default if range is single point
                
            # 2. Interpolate Base Cost
            c_min, c_max = base_cost
            base_c = c_min + (norm_speed * (c_max - c_min))
            
            # 3. Add Stochastic Disturbance (tight band; was ±10%)
            dist_factor = np.random.uniform(*cfg.COST_DISTURBANCE)
            cost = round(base_c * dist_factor, 2)
            
            # Sample Specs from Base Ranges (Restored)
            cpu = round(np.random.uniform(*base_cpu), 2)
            
            # Security (Max Supported - Restored)
            sec = np.random.choice([0, 1, 2, 3], p=security_prob)

            # Create Machine Row
            # [CPU, RAM, STO, COST, SEC, LOAD, NODE_ID, SPEED, INTRA_NODE, INTRA_MACH]
            # RAM/STO columns are 999999.0 used as placeholders (effectively infinite)
            machine_rows.append([cpu, 999999.0, 999999.0, cost, sec, 0, n_id, mach_speed, cfg.INTRA_NODE_LATENCY, cfg.INTRA_MACHINE_LATENCY])
            
    machines = np.array(machine_rows)

    return Topology(
        num_nodes=num_nodes,
        machines=machines,
        network_latency=lat_matrix,
        node_to_user_latency=user_lat,
        node_categories=categories,
        node_clusters=node_clusters
    )

def generate_workload(num_apps=cfg.NUM_APPS) -> list[Application]:
    apps = []
    ms_counter = 0
    
    for i in range(num_apps):
        num_ms = np.random.randint(cfg.MIN_SERVICES, cfg.MAX_SERVICES + 1)
        services = []
        
        for _ in range(num_ms):
            # constraints
            # User Request: Unikernel (Tier 3) requires specialized image.
            # Only a percentage of services (defined by PROB_UNIKERNEL_IMAGE) have this specialized image (s_max=3).
            # The rest are limited to s_max=2.
            # Services with s_max=3 CAN run on Tiers 0,1,2 as well (Portable).
            has_specialized_image = (np.random.random() < cfg.PROB_UNIKERNEL_IMAGE)
            limit_max = 3 if has_specialized_image else 2
            
            # Generate min_sec
            # Existing probs: [0: 0.5, 1: 0.2, 2: 0.2, 3: 0.1]
            min_sec = np.random.choice([0, 1, 2, 3], p=[0.50, 0.30, 0.10, 0.10])
            
            # Consistency Check: min_sec <= max_sec
            # If min_sec > limit_max, we must clamp it.
            if min_sec > limit_max:
                min_sec = limit_max
                
            # Generate max_sec (Must be <= limit_max and >= min_sec)
            # We allow it to be anything in [min_sec, limit_max]
            # But the user said "Put all s_max=2... besides 25%... that have s_max=3"
            # This implies the *capability* is 2 or 3. 
            # Does it mean the service *requires* exactly 3? No, "portable across Tiers 0 through 2".
            # So max_sec defining the "maximum supported tier" is exactly limit_max.
            max_sec = limit_max
            
            # Additional logic: If the original code allowed max_sec to be *lower* than the capability (e.g. strict subset),
            # we should preserve that? 
            # Original: max_sec = randint(min_sec, cfg.MAX_SEC)
            # User says: "Put ALL s_max=2 ... besides 25% ... that have s_max=3"
            # This sounds like a hard assignment of the upper bound.
            # So I will set max_sec = limit_max.
            
            # Double check: if min_sec was 3, and limit_max was 2 -> min_sec became 2. max_sec is 2. Range [2, 2]. Correct.
            
            # V2: Typical Runtime & Demands (Updated Bimodal)
            # User Request: First service (Ingress) should never be Heavy, unless it's the only service
            is_heavy = (np.random.random() < cfg.PROB_HEAVY_SERVICE)
            if _ == 0 and num_ms > 1: 
                is_heavy = False # Force Light for first service (if chain > 1)
                
            if is_heavy:
                # Heavy
                runtime = round(np.random.uniform(*cfg.RUNTIME_HEAVY), 2)
                cpu = round(np.random.uniform(*cfg.DEMAND_CPU_HEAVY), 2)
                # ram/sto ignored
            else:
                # Light
                runtime = round(np.random.uniform(*cfg.RUNTIME_LIGHT), 2)
                cpu = round(np.random.uniform(*cfg.DEMAND_CPU_LIGHT), 2)
                # ram/sto ignored
            
            # Data Size (MB)
            d_size = round(np.random.uniform(*cfg.DATA_SIZE), 2)
            
            # Stochastic Security Overheads
            # Tier 0 is Baseline (1.0)
            # Tiers 1, 2, 3: [Base ± SEC_OVERHEAD_JITTER]
            jitter = float(getattr(cfg, "SEC_OVERHEAD_JITTER", 0.2))
            sec_ov = [1.0] * 4
            for t in range(1, 4):
                 base = cfg.OVERHEAD_CPU_BASE[t]
                 low = base * (1.0 - jitter)
                 high = base * (1.0 + jitter)
                 val = np.random.uniform(low, high)
                 sec_ov[t] = round(val, 3)
            
            ms = Microservice(
                id=ms_counter,
                app_id=i,
                cpu_demand=cpu,
                ram_demand=0.0,
                storage_demand=0.0,
                min_security=min_sec,
                max_security=max_sec,
                typical_runtime=runtime,
                data_size=d_size,
                security_overheads=sec_ov
            )
            services.append(ms)
            ms_counter += 1
            
        # V2: Chain Logic & mTLS
        mtls_reqs = []
        mtls_ovs = []
        for _ in range(num_ms - 1):
            req = (np.random.random() < cfg.MTLS_PROBABILITY)
            mtls_reqs.append(req)
            if isinstance(cfg.MTLS_CPU_OVERHEAD, tuple):
                ov = round(np.random.uniform(*cfg.MTLS_CPU_OVERHEAD), 3)
            else:
                ov = cfg.MTLS_CPU_OVERHEAD
            mtls_ovs.append(ov)
            
        return_path = (np.random.random() < cfg.RETURN_PATH_PROBABILITY)

        # Cluster Selection (Uniform Random)
        source_cluster = np.random.randint(0, cfg.NUM_CLUSTERS)
        
        app = Application(
            id=i,
            microservices=services,
            mtls_requirements=mtls_reqs,
            mtls_decisions=mtls_reqs.copy(), # Initialize decisions with requirements (Required=True)
            mtls_overheads=mtls_ovs,
            gen_point=source_cluster, # Cluster ID
            source_cluster=source_cluster,
            return_to_ue=return_path
        )
        apps.append(app)
        
    print(f"Generated {len(apps)} apps with total {ms_counter} services (Avg {ms_counter/len(apps):.2f})")
    return apps

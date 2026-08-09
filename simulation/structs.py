from dataclasses import dataclass, field
from typing import List, Dict, Optional
import numpy as np

@dataclass
class Microservice:
    id: int
    app_id: int
    
    # Demands (Base)
    cpu_demand: float
    ram_demand: float
    storage_demand: float
    
    # Security
    min_security: int
    max_security: int 
    
    # Runtime
    typical_runtime: float # ms on standard CPU
    
    # Data Transfer
    data_size: float # MB (Data output to next service)
    
    # Stochastic Security Overheads (CPU multiplier per tier [0, 1, 2, 3])
    # Defaults to [1.0, 1.0, 1.0, 1.0] if not set, but generator will set it.
    security_overheads: List[float] = field(default_factory=lambda: [1.0, 1.0, 1.0, 1.0])
    
    # Assignment State
    assigned_machine_id: int = -1
    assigned_node_id: int = -1
    assigned_tier: int = 0
    assigned_obj_cost: float = 0.0
    rflag: bool = False

@dataclass
class Application:
    id: int
    microservices: List[Microservice]
    
    # Chain Logic
    # The chain is defined implicitly by the list order: MS[0] -> MS[1] -> ... -> MS[N]
    
    # Constraints
    # latency_demand: float  # vs Total End-to-End Latency
    gen_point: int # Cluster ID (0..NUM_CLUSTERS-1) where request originates
    source_cluster: int = -1 # Explicit cluster ID (same as gen_point)
    
    # mTLS Requirements
    # List of booleans. index i corresponds to link MS[i] -> MS[i+1]
    # Length is len(microservices) - 1
    mtls_requirements: List[bool] = field(default_factory=list)
    
    # Decisions (Track actual usage)
    # Length is len(microservices) - 1. True if mTLS enabled on link.
    mtls_decisions: List[bool] = field(default_factory=list)
    
    # Overheads
    # Length is len(microservices) - 1. Stores the CPU overhead for each mTLS link.
    mtls_overheads: List[float] = field(default_factory=list)
    
    return_to_ue: bool = False # If True, last service sends data back to UE

@dataclass
class Topology:
    num_nodes: int
    machines: np.ndarray 
    # Machine Matrix Columns:
    # 0: CPU, 1: RAM, 2: STORAGE, 3: COST, 4: SEC_LEVEL, 
    # 5: LOAD, 6: NODE_ID, 7: SPEED_COEFF, 8: INTRA_NODE_LAT, 9: INTRA_MACH_LAT
    
    node_categories: np.ndarray # 1: Near-Edge, 2: Far-Edge, 3: Cloud
    node_clusters: np.ndarray # Cluster ID (0..NUM_CLUSTERS-1) for each node
    
    node_to_user_latency: np.ndarray # (NumNodes x NumClusters) - Latency from Node to Cluster Gateway
    network_latency: np.ndarray # (NumNodes x NumNodes)
    
    trust_matrix: np.ndarray = None # (NUM_CLUSTERS x NUM_CLUSTERS) binary trust matrix

    # Column Constants
    CPU_IDX = 0
    RAM_IDX = 1
    STR_IDX = 2
    COST_IDX = 3
    SEC_IDX = 4
    LOAD_IDX = 5
    NODE_IDX = 6
    SPEED_IDX = 7
    INODE_LAT_IDX = 8 # Intra-Node
    IMACH_LAT_IDX = 9 # Intra-Machine

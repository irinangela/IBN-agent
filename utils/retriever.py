import pandas as pd
from typing import Dict, Any, Optional

df_non_zero_sec = pd.read_csv("../simulation/runs/valid_runs.csv").assign(zero_sec=False)
df_zero_sec = pd.read_csv("../simulation/runs/valid_runs_zero_sec.csv").assign(zero_sec=True)
df = pd.concat([df_zero_sec, df_non_zero_sec], ignore_index=True)

def get_warm_start_context(parsed_intent: Dict[str, Any], top_k: int = 3) -> str:
    """
    Filters the historical dataset based on hard constraints and primary objective,
    returning the best historical weight combinations for a 'Warm Start'.
    """
    # Filter for valid and successful runs using the rollout algorithm
    filtered_df = df[
        (df["valid"] == True) & 
        (df["failures"] == 0) & 
        (df["algorithm"] == "app_rollout")
    ].copy()

    metric_cols = {
        "cost": "norm_cost",
        "security": "norm_sec",
        "latency": "norm_lat"
    }

    for constraint in parsed_intent.get("hard_constraints", []):
        metric_name = constraint.get("metric", "").lower()
        operator = constraint.get("operator")
        threshold = constraint.get("threshold")

        if metric_name in metric_cols:
            col = metric_cols[metric_name]
            if operator in ["<=", "<"]:
                filtered_df = filtered_df[filtered_df[col] <= threshold]
            elif operator in [">=", ">"]:
                filtered_df = filtered_df[filtered_df[col] >= threshold]

    # Fallback: if filters are too strict and filtered_df is empty, reset the search to have a valid df and be able to negotiate trade-offs
    if filtered_df.empty:
        filtered_df = df[
            (df["valid"] == True) & 
            (df["failures"] == 0) & 
            (df["algorithm"] == "app_rollout")
        ].copy()

    primary = parsed_intent.get("primary_objective", "").lower()
    if primary in metric_cols:
        target_col = metric_cols[primary]
        filtered_df = filtered_df.sort_values(by=target_col, ascending=True)

    # Format top_k results into a clean string for the LLM prompt
    top_runs = filtered_df.head(top_k)
    
    if top_runs.empty:
        return "No matching historical runs found."

    context_str = "Historical Successful Runs:\n"
    for idx, row in top_runs.iterrows():
        w1_val = row.get('W1', row.get('w1', 0.0))
        w2_val = row.get('W2', row.get('w2', 0.0))
        w3_val = row.get('W3', row.get('w3', 0.0))
        
        context_str += (
            f"- Candidate Weights: w1(cost)={w1_val:.2f}, w2(sec)={w2_val:.2f}, w3(lat)={w3_val:.2f} "
            f"| Results -> Cost: {row['norm_cost']:.3f}, Security: {row['norm_sec']:.3f}, Latency: {row['norm_lat']:.3f}\n"
        )

    return context_str
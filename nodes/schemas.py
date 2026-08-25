from typing import Literal

from pydantic import BaseModel, Field

MetricType = Literal["latency", "cost", "security"]

UNIT_LABELS = {
    "latency": "ms (avg per app)",
    "cost": "cost units (avg per app)",
    "security": "security score (avg per app, higher is better)",
}

METRIC_RESULT_KEYS = {
    "latency": "avg_latency",
    "cost": "avg_cost",
    "security": "avg_security",
}

HIGHER_IS_BETTER = {"security"}


class HardConstraint(BaseModel):
    metric: MetricType = Field(
        description="The metric being constrained: 'latency', 'cost', or 'security'."
    )
    operator: str = Field(
        description="The comparison operator ('<=', '>=', '<', '>', '==')."
    )
    threshold: float = Field(
        description="The numeric threshold in operator units (ms / cost units / security score)."
    )


class IntentSchema(BaseModel):
    primary_objective: MetricType = Field(
        description="The main metric to minimize or maximize."
    )
    hard_constraints: list[HardConstraint] = Field(
        default_factory=list,
        description="Strict numeric limits in operator units.",
    )
    soft_preferences: list[str] = Field(
        default_factory=list,
        description="General preferences without strict numeric bounds.",
    )
    non_relaxable_constraints: list[MetricType] = Field(
        default_factory=list,
        description="List of metric names that must NEVER be violated or relaxed.",
    )
    relaxation_order: list[MetricType] = Field(
        default_factory=list,
        description="The order in which metric names can be sacrificed if a solution is infeasible.",
    )

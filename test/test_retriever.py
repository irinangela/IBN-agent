import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.retriever import get_warm_start_context


def test_warm_start_warns_when_no_historical_run_meets_sla():
    """Impossible latency SLA must surface the fallback, not silent top-k by objective."""
    context = get_warm_start_context({
        "primary_objective": "latency",
        "hard_constraints": [
            {"metric": "latency", "operator": "<=", "threshold": 1.0}
        ],
    })
    assert context.startswith("WARNING: 0 historical runs satisfied the hard constraints")
    assert "latency <= 1.0" in context
    assert "Best observed latency in the corpus:" in context
    assert "Historical Successful Runs" in context


def test_warm_start_omits_warning_when_runs_meet_sla():
    context = get_warm_start_context({
        "primary_objective": "latency",
        "hard_constraints": [
            {"metric": "latency", "operator": "<=", "threshold": 10000.0}
        ],
    })
    assert not context.startswith("WARNING:")
    assert "Historical Successful Runs" in context

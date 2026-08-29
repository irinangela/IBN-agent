import os
import sys
import copy

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.plots import categorize_run, save_run_figures, save_dataset_figures, save_batch_figures
from utils.plots.io import attempts_of, load_run, load_valid_runs
from utils.plots.run import plot_convergence, plot_weights, plot_requested_vs_delivered
from utils.plots.dataset import (
    plot_algorithm_comparison,
    plot_pareto,
    plot_rugged_front,
    plot_weight_simplex,
)
from utils.plots.aggregate import plot_category_summary, plot_relaxation_correctness


FIXTURES = os.path.join(os.path.dirname(__file__), "fixtures")
SAMPLE_JSON = os.path.join(FIXTURES, "sample_run.json")
SAMPLE_CSV = os.path.join(FIXTURES, "sample_runs.csv")


def test_attempts_of_reads_fixture():
    run = load_run(SAMPLE_JSON)
    attempts = attempts_of(run)
    assert [row["attempt"] for row in attempts] == [1, 2, 3]
    assert attempts[-1]["avg_latency"] == 920.0


def test_per_run_plotters_return_figures():
    run = load_run(SAMPLE_JSON)
    assert plot_convergence(run) is not None
    assert plot_weights(run) is not None
    assert plot_requested_vs_delivered(run) is not None


def test_save_run_figures_writes_png_and_pdf(tmp_path):
    dest = tmp_path / "29-08-2026-run-1.json"
    dest.write_text(open(SAMPLE_JSON, encoding="utf-8").read(), encoding="utf-8")
    written = save_run_figures(str(dest))
    stems = {os.path.splitext(os.path.basename(path))[0] for path in written}
    assert "29-08-2026-run-1-convergence" in stems
    assert "29-08-2026-run-1-weights" in stems
    assert "29-08-2026-run-1-requested-vs-delivered" in stems
    assert any(path.endswith(".png") for path in written)
    assert any(path.endswith(".pdf") for path in written)


def test_dataset_plotters_on_sparse_sheet():
    frame = load_valid_runs(SAMPLE_CSV, seed=100)
    assert plot_rugged_front(frame) is not None
    assert plot_pareto(frame, run=load_run(SAMPLE_JSON)) is not None
    assert plot_algorithm_comparison(frame) is not None
    assert plot_weight_simplex() is not None
    assert plot_weight_simplex(run=load_run(SAMPLE_JSON)) is not None


def test_save_dataset_figures(tmp_path):
    frame = load_valid_runs(SAMPLE_CSV, seed=100)
    written = save_dataset_figures(frame, str(tmp_path))
    names = {os.path.splitext(os.path.basename(path))[0] for path in written}
    assert "rugged-front" in names
    assert "pareto" in names
    assert "algorithm-comparison" in names
    assert "weight-simplex" in names


def test_categorize_and_batch_plots(tmp_path):
    feasible = load_run(SAMPLE_JSON)
    feasible["relaxations"] = []
    feasible["relaxation_events"] = []
    feasible["active_intent"] = feasible["initial_intent"]

    needs_relax = copy.deepcopy(feasible)
    needs_relax["success"] = False
    needs_relax["feasibility_report"]["needs_operator"] = True
    needs_relax["feasibility_report"]["per_algorithm"]["best_fit"]["jointly_feasible"] = False
    needs_relax["feasibility_report"]["per_algorithm"]["best_fit"]["infeasible_metrics"] = ["latency"]
    needs_relax["relaxations"] = [{"metric": "latency", "original": 900, "active": 945}]
    needs_relax["relaxation_events"] = needs_relax["relaxations"]

    assert categorize_run(feasible) == "feasible"
    assert categorize_run(needs_relax) == "infeasible-requiring-relaxation"

    written = save_batch_figures([feasible, needs_relax], str(tmp_path))
    names = {os.path.splitext(os.path.basename(path))[0] for path in written}
    assert "category-summary" in names
    assert "relaxation-correctness" in names
    assert plot_category_summary([feasible, needs_relax]) is not None
    assert plot_relaxation_correctness([needs_relax]) is not None

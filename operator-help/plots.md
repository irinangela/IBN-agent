# Plots guide for the operator

- Plotting is **read-only**: it never edits analytics JSON or the historical CSV.
- Each figure is saved as `PNG` and `PDF` for later use in the thesis chapters. 
- Shared colours, axis labels, and `savefig` live in `utils/plots/style.py` for consistency.
- Weight labels are fixed: **w1 = cost**, **w2 = security**, **w3 = latency**.
- If a plotter has nothing useful to draw (empty attempts, no hard constraints, empty CSV), it returns `None` and that file is skipped.

Two metric families appear on plots and must not be mixed:

- Operator units: `avg_cost`, `avg_latency`, `avg_security` -> Mean per successful app. These are what the verifier and HITL use. 
- Heuristic values: `norm_cost`, `norm_lat`, `norm_sec` -> Sums of per-app min-max scores over the workload. Used by the optimizer objective. 

---

## Three categories

1. **Per-run**:
    After every run a `results-analytics/*-run-*.json` is created. Using these analytics we plot figures that are shown in the UI and also saved in `results-analytics/` in `PNG` and `PDF` format. We can regenerate these using `python scripts/plot_run.py`.

2. **Dataset**:
    This is a category regarding the historical sweep, independent of a single agent session. The data source is `simulation/runs/valid_runs.csv` and the results can be generated using `python scripts/plot_dataset.py`. These graphs are not shown in the UI. They are saved in `results-analytics/dataset/` and are useful for later analysis and thesis usage. 

3. **Aggregate**:
    Comparison over a batch of agent runs. Data source is the `results-analytics/` folder with `*-run-*.json` files. These are not shown in the UI. We can generate them using `python scripts/plot_batch.py`. Useful for showing results in comparisons in the thesis.

Parse failures never write a JSON, so they never appear in per-run or aggregate figures.

---

## 1. Per-run 

Written automatically in `_finish_run` (`main.py`) next to the analytics JSON. Streamlit shows the PNGs, while PDFs sit beside them for the thesis.

Default output directory: `results-analytics/`.

**Filenames** (run = JSON basename, e.g. `30-08-2026-run-1`):

- `{run}-convergence.png/.pdf`
- `{run}-weights.png/.pdf`
- `{run}-requested-vs-delivered.png/.pdf`

**Data:** `attempts` on the JSON (weights + `avg_*` + `norm_*` + verifier feedback per iteration). 
The series of attempts is complete as analytics appends the last simulation even when SUCCESS / max-iterations skip the Recalibrator.

### Convergence trajectory

- One row per hard-constraint metric (or all three if the JSON has no hard constraints).
- Left column: `avg_*` vs iteration, markers on each attempt.
- Solid black horizontal line: **requested** threshold from `initial_intent`.
- Dashed black horizontal line: **relaxed** threshold from `active_intent`, only if it differs from requested.
- Dotted vertical line: a relaxation event, if `relaxation_events` includes an `iteration` (current analytics often leave `iteration` as `null`, so the vline may be absent even when thresholds changed).
- Right column (if `norm_*` are present): the same iterations in heuristic unitss.

### Weight trajectory

- `w1`, `w2`, `w3` vs iteration, y-axis `[0, 1]`.
- Shows whether search stayed local (`±0.15` clip) or walked toward a corner.

### Requested vs delivered

- Grouped bars per hard-constraint metric, in operator units.
- **requested** = original SLA, **delivered** = `final_metrics` (or last attempt), **relaxed** = active threshold (hatched bar, only if it changed).
- Title includes `success` or `did not meet SLA`.

### Regenerate one run

```bash
python scripts/plot_run.py --json results-analytics/30-08-2026-run-1.json
```

---

## 2. Dataset 

These describe the simulator, not one agent loop. They use `valid == True` and `failures == 0` only. `load_valid_runs` derives `avg_*` as `total_* / successful_apps` and, by default, keeps the live `SEED`.

Default output directory: `results-analytics/dataset/`.

### Rugged front (`rugged-front`)

- Simplex: **x = w1**, **y = w2**, **w3 = 1 − w1 − w2**
- Three panels coloured by `norm_lat`, `norm_cost`, `norm_sec`.
- **Sparse sheet** (fewer than 50 points, or few distinct w1 values): scatter.
- **Dense sheet** (more than 50 points): filled triangulation (`tricontourf`), triangles outside the simplex masked.
- Default algorithm panel: `best_fit`. Use `--algorithm app_rollout` for the other heuristic.

This is the figure that shows **non-monotonic** weight -> metric behaviour (why the Recalibrator must not treat “more w3” as “better latency”, non-intuitive nature).
### Weight simplex (`weight-simplex`) -> used in the thesis

- Same right-triangle simplex as rugged-front: x = w1, y = w2, w3 = 1 − w1 − w2. Corners are pure cost / security / latency.
- Dots: the 66-point dataset grid (`WARM_START_WEIGHT_SETS`, step 0.1).
- Shaded polygon: weights the proposer/recalibrator can reach in one step by clipping each coordinate to +-0.15 of the baseline and renormalizing. Blue dots are grid points inside that region.
- Black star: baseline. Default is `(0.35, 0.20, 0.45)`. With `--run`, the first attempt is the baseline and the orange path is later attempts.

### Approximate Pareto front (`pareto`)

- Historical points: `avg_cost` vs `avg_latency`, colour = `avg_security`.
- Optional `--run path/to/*-run-*.json` overlays that agent’s attempt path (orange line, star on the last point).

### Algorithm comparison (`algorithm-comparison`)

- Pairs `best_fit` and `app_rollout` rows that share **seed, W1, W2, W3**.
- Four identity plots: avg cost, avg latency, avg security, runtime (`duration_s`). Points off the dashed y = x line are where the two algorithms disagree at the same weights.

### Generate dataset figures

```bash
python scripts/plot_dataset.py
```

---

## 3. Aggregate 

Thesis figures over every `*-run-*.json` in a folder. Default input: `results-analytics/`. Default output: `results-analytics/aggregate/`.

Auto-switch, HITL, and in-loop relaxation are included if those runs finished search. Parse failures are excluded (because there is no JSON). 

### Intent categories (`categorize_run`)

Used by the category-summary bars:

1. **infeasible-requiring-relaxation** — any threshold change (`relaxations` / `relaxation_events`), or the operator chose HITL `relax`.
2. **feasible** — historical check said the chosen algorithm was `jointly feasible`, and either the run succeeded or it failed (still labelled feasible: history said the SLA was in range).
3. **ambiguous** — jointly infeasible, no individual metric beyond its best-known bound (`infeasible_metrics` empty), HITL was required, and search **failed** without relaxing. Example: two SLAs that never co-occur (Pareto conflict) and the operator did not loosen either.
4. Otherwise HITL-needed or unsuccessful search is also considered **infeasible-requiring-relaxation**.
5. Default is **feasible**.

### Category summary (`category-summary`)

Three bars per category (label `infeasible-requiring-relaxation` is shortened to `needs relaxation`):

- **Success rate** — fraction of runs with `success == true`.
- **Iterations to success** — mean `iterations_done` on successes; failures use `max_iterations` (or `iterations_done` if max is missing).
- **Token cost** — mean `token_usage.total_tokens`.

### Relaxation-decision correctness (`relaxation-correctness`)

Only runs that actually have relaxation events:

- **matched order[0]** — first relaxed metric equals the first entry of `initial_intent.relaxation_order`.
- **did not match** — a different metric was loosened first.
- **no order declared** — events exist but `relaxation_order` was empty.

Omitted entirely if no run in the folder relaxed.

### Effort vs outcome (`success-vs-tokens`)

Scatter of **total tokens** vs **iterations**, circles = success, crosses = failure.

### Generate aggregate figures

```bash
python scripts/plot_batch.py
```

---


## Notes

- **UI failure:** if figure generation throws, Streamlit still finishes the run and shows a warning (`figure_error`). The JSON is already saved.
- **Re-plot after changing style:** re-run the matching script so existing PNG/PDF are overwritten.
- **Sparse vs dense CSV:** today’s `valid_runs.csv` is a small weight set, so rugged-front is a scatter. A dense sweep would switch that panel to a filled contour automatically (`DENSE_SIMPLEX_MIN_POINTS = 50`).
- **Seed:** dataset plots default to the live simulation seed so they match warm-start and feasibility. (Pass another `--seed` only if you intend a different topology/workload slice)
- **Dependencies:** `matplotlib` is in `requirements.txt`.

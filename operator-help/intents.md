| | `best_fit` | `app_rollout` | live `best_fit` | live `app_rollout` |
|---|---|---|---|---|
| Best cost (lower) | 307 | **305** | 329 | 328 |
| Best security (higher) | 15.2 | **15.38** | 14.86 | 14.94 |
| Best latency (ms, lower) | 508 | **485** | 520 | **492** |

`app_rollout` dominates `best_fit` for almost all rows in the dataset, so feasibility **auto-switches** when `best_fit` cannot meet an SLA and `app_rollout` can. Combined latency floor is **484.9 ms**. HITL only fires if **neither** algorithm is jointly feasible (use **470 ms**). Relax then offers **two** algorithm-tagged thresholds: `best_fit` at about **533.5 ms** (508.1 × 1.05, fast) and `app_rollout` at about **509.1 ms** (484.9 × 1.05, ~45s per simulation). Picking the `app_rollout` offer also switches live search to that algorithm. The default UI intent (1000 ms) is still an easy success.

---

## Intents to try and what features each one will demonstrate:

### 1. Simple warm-start

```text
Optimize for low latency. Average latency must stay under 1000ms. Cost and security are secondary.
```

**What it shows:** First Parser, then Feasibility proceeds with `best_fit` (no operator pause). Proposer lists historical weights with no WARNING so one (the first and last) simulation takes place. Verifier returns **SUCCESS**.

(This is the **Latency** sidebar preset. 1000 ms is well above both historical floors, so history already contains matching runs. Use the Cost and Security presets for the other two easy-feasible demos.)

### 2. Unrecognised intent

```text
Keep average bananas at least 8. Prefer low apples. Watermelons can be sacrificed first if needed.
```

**What it shows:** Parser cannot map this intent into the structured JSON. It shows a message to the operator asking for a rewrite of the intent and a corresponding field to fill in the revised intent. After submitting the revised intent, the process continuous as it should. 

### 3. Security metric (higher is better)

```text
Keep average security at least 8. Prefer low latency. Cost can be sacrificed first if needed.
```

**What it shows:** Parser emits `security >= 8` (not `<=`). Feasibility stays on `best_fit` because that algorithm is jointly feasible. Warm-start sorts by highest neighborhood-averaged security.

### 4. Tight but feasible latency (no HITL, stays on `best_fit`)

```text
Average latency must stay under 550ms. Security and cost are soft preferences only.
```

**What it shows:** 550 ms is above the 508 ms `best_fit` floor, so Feasibility proceeds with `best_fit` and does not pause. Live seed 210 can do about 520 ms, so the verifier should still succeed. Contrast with case 5 (500 ms), which auto-switches, and case 6 (470 ms), which pauses.

### 5. Auto-switch to `app_rollout`

```text
Optimize for low latency. Average latency must stay under 500ms. Cost and security are secondary.
```

**What it shows:** 500 ms is below the `best_fit` floor (508 ms) but above the `app_rollout` floor (485 ms). Feasibility auto-switches to `app_rollout` and does **not** pause. Live rollout can do about 492 ms, so the verifier should succeed. Live `best_fit` cannot meet 500 ms.

### 6. HITL — relax to the offered threshold

```text
Optimize for low latency. Average latency must stay under 470ms. Cost and security are secondary.
```

**What it shows:** 470 ms is below both historical floors, so operator input is required. The run pauses with three options: `continue anyway`, `relax constraints to offered thresholds`, `enter a new intent`. Choose the second option. You should see two algorithm-tagged offers:

- `best_fit`: about **533.5 ms** (508.1 × 1.05), fast (~0.25s per simulation). Live `best_fit` can do about 520 ms, so this offer should succeed.
- `app_rollout`: about **509.1 ms** (484.9 × 1.05), slower (~45s per simulation). Live rollout can do about 492 ms, so this tighter offer should also succeed, and search switches to `app_rollout`.

You can still edit the number before confirming. Search keeps the algorithm you selected. You can also choose `Enter a new intent` at the same pause.

### 7. Joint Pareto infeasibility (two SLAs that never co-occur)

```text
Average latency must stay under 700ms and average cost must stay under 400. Prefer low latency. If needed, relax cost first, then latency.
```

**What it shows:** Each bound is feasible on its own (latency 700 and cost 400 exist in history), but no stored run has both. HITL still offers: `continue searching anyway`, `relax constraints to offered thresholds` and `enter a new intent`. Choose the second one. Pick one metric to loosen. Each metric then shows algorithm-tagged conditional offers (and the runtime note) so the joint intent becomes feasible on the allocator you select.

### 8. HITL — continue and in-loop graduated relaxation

```text
Optimize for low latency. Average latency should stay under 470ms. If that is not possible you may relax.
```

At the pause, choose `Continue searching anyway`.

**What it shows:** Search starts with the 470ms SLA. Warm-start should show `WARNING: 0 historical runs satisfied…`. After 3 distinct failed weight vectors, Recalibrator may loosen the threshold (code will not drop the constraint). The agent will not strictly relax after the 3 failed attempts. It may continue searching if it still wants to try more empirical steps. When all indicators show that the search is stuck you should see a `[IMPORTANT]: latency constraint relaxed from 470 to …` line and `relaxations` in the analytics JSON.

**Note:** Raise max iterations to 7 or 8 so the relaxation can take place before the loop exits. Live `best_fit` cannot beat about 520 ms, so 470 ms will not succeed without a relaxation (and the first 5% step to ~509 ms is still below live `best_fit`).

### 9. Non-relaxable SLA (Relax option hidden)

```text
We are time-sensitive. Average latency must never exceed 470ms. Cost and security are secondary.
```

**What it shows:** Because we used keywords like ``never``, the parser put `latency` in `non_relaxable_constraints`. To make sure we do not violate this, the UI does not show the option `Relax constraints…`. Only `Continue` and `New intent` are shown. 

If you choose `Continue` Recalibrator is blocked (`[SYSTEM]: No relaxable hard constraint remains. Weight search only.`). Loop ends on max iterations. 


### 10. Hard conflict, both constraints non-relaxable

```text
This involves self-driving vehicles so latency should never exceed 550ms. Security must never drop below 14. Keep cost as low as possible.
```

**What it shows:** Each bound is feasible on its own (550 ms and security 14 both exist in history), but they never co-occur. Both metrics are non-relaxable, so no `Relax` button. `Continue` dies at max iterations and `New intent` is the only way out.

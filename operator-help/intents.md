| | `best_fit` | `app_rollout` | live `best_fit` |
|---|---|---|---|
| Best cost (lower) | 307 | 529 | 329 |
| Best security (higher) | 15.2 | 14.9 | 14.9 |
| Best latency (ms, lower) | 508 | 566 | 520 |

`best_fit` dominates `app_rollout` on all three axes in the current dataset, so feasibility never auto-switches on this CSV. A latency SLA is infeasible only if it is **stricter than 508 ms** (use **500 ms** for HITL). The default UI intent (1000 ms) is still an easy success.

---

## Intents to try and what features each one will demonstrate:

### 1. Simple warm-start

```text
Optimize for low latency. Average latency must stay under 1000ms. Cost and security are secondary.
```

**What it shows:** First Parser, then Feasibility proceeds with `best_fit` (no operator pause). Proposer lists historical weights with no WARNING so one (the first and last) simulation takes place. Verifier returns **SUCCESS**.

(This is the default in the UI. 1000 ms is well above the 508 ms historical floor, so history already contains matching runs.)

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

### 4. Tight but feasible latency (no HITL)

```text
Average latency must stay under 550ms. Security and cost are soft preferences only.
```

**What it shows:** 550 ms is above the 508 ms historical floor, so Feasibility proceeds with `best_fit` and does not pause. Live seed 210 can do about 520 ms, so the verifier should still succeed. Use this as the contrast with case 5 (500 ms), which is just below the floor.

**Auto-switch to `app_rollout` cannot be shown on this dataset.** `best_fit` is better on cost, security, and latency, so any SLA `app_rollout` can meet, `best_fit` can also meet. The code path still exists, this CSV just never needs it.

### 5. HITL — relax to the offered threshold

```text
Optimize for low latency. Average latency must stay under 500ms. Cost and security are secondary.
```

**What it shows:** 500 ms is below the 508 ms historical floor, so no historical run meets the hard constraint and operator input is required. The run pauses with three options: `continue anyway`, `relax constraints to offered threshold`, `enter a new intent`. Choose the second option, which offers the best-known value loosened by a small margin (about 534ms = 508 × 1.05) and the ability to change it before confirming. Live `best_fit` can do about 520ms, so accepting the offer should succeed.

### 6. Joint Pareto infeasibility (two SLAs that never co-occur)

```text
Average latency must stay under 900ms and average cost must stay under 450. Prefer low latency. If needed, relax cost first, then latency.
```

**What it shows:** Each bound is feasible on its own (latency 900 and cost 450 exist in history), but no stored run has both. HITL still offers: `continue searching anyway`, `relax constraints to offered thresholds` and `enter a new intent`. Choose the second one which offers now two distinct conditional relaxations, one for each metric so that the joint intent becomes feasible. Pick one metric to loosen.


### 7. HITL — new intent

```text
Optimize for low latency. Average latency must stay under 500ms. Cost and security are secondary.
```

**What it shows:** The same intent as number 5, but this time choose `Enter a new intent`. A field to enter the revised intent is shown and after confirming the new input the run continues with the updated JSON.

### 8. HITL — continue and in-loop graduated relaxation

```text
Optimize for low latency. Average latency should stay under 500ms. If that is not possible you may relax.
```

At the pause, choose `Continue searching anyway`.

**What it shows:** Search starts with the 500ms SLA. Warm-start should show `WARNING: 0 historical runs satisfied…`. After 3 distinct failed weight vectors, Recalibrator may loosen the threshold (code will not drop the constraint). The agent will not strictly relax after the 3 failed attempts. It may continue searching if it still wants to try more empirical steps. When all indicators show that the search is stuck you should see a `[IMPORTANT]: latency constraint relaxed from 500 to …` line and `relaxations` in the analytics JSON.

**Note:** Raise max iterations to 7 or 8 so the relaxation can take place before the loop exits. Live latency cannot beat about 520ms, so 500 ms will not succeed without a relaxation.

### 9. Non-relaxable SLA (Relax option hidden)

```text
We are time-sensitive. Average latency must never exceed 500ms. Cost and security are secondary.
```

**What it shows:** Because we used keywords like ``never``, the parser put `latency` in `non_relaxable_constraints`. To make sure we do not violate this, the UI does not show the option `Relax constraints…`. Only `Continue` and `New intent` are shown. 

If you choose `Continue` Recalibrator is blocked (`[SYSTEM]: No relaxable hard constraint remains. Weight search only.`). Loop ends on max iterations. 


### 10. Hard conflict, both constraints non-relaxable

```text
This involves self-driving vehicles so latency should never exceed 550ms. Security must never drop below 14. Keep cost as low as possible.
```

**What it shows:** Each bound is feasible on its own (550 ms and security 14 both exist in history), but they never co-occur. Both metrics are non-relaxable, so no `Relax` button. `Continue` dies at max iterations and `New intent` is the only way out.

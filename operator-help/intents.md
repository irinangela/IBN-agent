## Intents to try and what features each one will demonstrate:

### 1. Simple warm-start

```text
Optimize for low latency. Average latency must stay under 1000ms. Cost and security are secondary.
```

**What it shows:** First Parser, then Feasibility proceeds with `best_fit` (no operator pause). Proposer lists historical weights with no WARNING so one (the first and last) simulation takes place. Verifier returns **SUCCESS**.

(This is the default in the UI, 1000ms is above 889 ms, so history already contains matching runs).

### 2. Unrecognised intent

```text
Keep average bananas at least 8. Prefer low apples. Watermelons can be sacrificed first if needed.
```

**What it shows:** Parser cannot map this intent into the structured JSON. It shows a message to the operator asking for a rewrite of the intent and a corresponding field to fill in the revised intent. After submitting the revised intent, the process continuous as it should. 

### 3. Security metric (higher is better)

```text
Keep average security at least 8. Prefer low latency. Cost can be sacrificed first if needed.
```

**What it shows:** Parser emits `security >= 8` (not `<=`). Feasibility stays on `best_fit` (`app_rollout` never reaches 8). Warm-start sorts by highest security.


### 4. Auto-switch to  `app_rollout`

```text
Average latency must stay under 860ms. Security and cost are soft preference only.
```

**What it shows:** Best_fit shows non jointly feasible, but app_rollout shows jointly feasible. Feasibility prints **Auto-switched live simulator to app_rollout**, which has historically feasible weight triplets. Search uses rollout 
warm-start. 
***IMPORTANT***: This run will fail. Check the email to see the problem. This use case shows only the auto-switch. 


### 5. HITL — relax to the offered threshold

```text
Optimize for low latency. Average latency must stay under 500ms. Cost and security are secondary.
```

**What it shows:** Feasibility check shows that no historical runs meet the hard contraint and thus operator input is required. The run pauses and waits for the operator to choose among three options: `continue anyway`, `relax constraints to offered threshold`, `enter a new intent`. Choose the second option which offers a best-known value loosened by a small margin (e.g. 3% or 5%) and the ability to change it before confirming and running the agent with the revised threshold.


### 6. Joint Pareto infeasibility (two SLAs that never co-occur)

```text
Average latency must stay under 900ms and average cost must stay under 450. Prefer low latency. If needed, relax cost first, then latency.
```

**What it shows:** Each bound is feasible on its own (latency 900 and cost 450 exist in history), but no stored run has both. HITL still offers: `continue searching anyway`, `relax constraints to offered thresholds` and `enter a new intent`. Choose the second one which offers now two distinct conditional relaxations, one for each metric so that the joint intent becomes feasible. Pick one metric to loosen.


### 7. HITL — new intent

```text
Optimize for low latency. Average latency must stay under 500ms. Cost and security are secondary.
```

**What it shows:** The same intent as number 5, but this time we choose the `Enter a new intent`. A field to enter the revised intent is shown and after confirming the new input the run continuous with the updated JSON.



### 8. HITL - continue and in-loop graduated relaxation

```text
Optimize for low latency. Average latency should stay under 500ms. If that is not possible you may relax.
```

At the pause, choose `Continue searching anyway`.

**What it shows:** Search starts with the 500ms SLA. Warm-start should show `WARNING: 0 historical runs satisfied…`. After 3 distinct failed weight vectors, and if the gap between current and desired values is over 70%, Recalibrator may loosen the threshold (code will not drop the constraint). The agent will not strictly relax after the 3 failed attempts. It may continue searching if considered necessary to try more empirical steps. When all indicators show that the search is stuck you should see a `[IMPORTANT]: latency constraint relaxed from 500 to …` line and `relaxations` in the analytics JSON.

**Note:** For this case, you should raise the max iterations to 7 or 8 to be sure the relaxation will take place before it exits.


### 9. Non-relaxable SLA (Relax option hidden)

```text
We are time-sensitive. Average latency must never exceed 800ms. Cost and security are secondary.
```

**What it shows:** Because we used keywords like ``never``, the parser put `latency` in `non_relaxable_constraints`. To make sure we do not violate this, the UI does not show the option `Relax constraints…`. Only `Continue` and `New intent` are shown. 

If you choose `Continue` Recalibrator is blocked (`[SYSTEM]: No relaxable hard constraint remains. Weight search only.`). Loop ends on max iterations. 


### 10. Hard conflict, both constraints non-relaxable

```text
This involves self-driving vehicles so latency should never exceed 850ms. Security must stay at least 8. Keep cost as low as possible.
```

**What it shows:** Rollout can do 850 ms but not security 8; `best_fit` can do security 8 but not 850 ms. Both offers are non-relaxable so no `Relax` button. `Continue` dies at max iterations and `New intent` is the only way out.

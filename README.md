# Agentic Intent-Based Network (IBN) Optimizer

This repository contains an LLM-powered Agentic Framework designed to solve multi-objective service placement problems on the edge-cloud continuum. It acts as an intelligent middleware that translates natural language operator intents into strict mathematical weights, orchestrates a black-box heuristic simulator, actively recalibrates those weights to navigate the approximate Pareto front, and finds optimal solutions or negotiates trade-offs when solutions are infeasible.

## Architecture Overview

The system is built using **LangGraph**, structuring the LLM operations in a ReAct (Reason + Act) loop method. This architecture prevents LLM hallucinations by separating semantic reasoning from deterministic mathematical verification.

### System Workflow (State Graph)

```mermaid
graph TD
    A([User Input in Natural Language]) --> B[Intent Parser]
    B --> C[Feasibility Pre-check]
    C --> D[Weight Proposer]
    D --> E[Simulator]
    E --> F[Verifier]
    F --> G{Constraints Satisfied?}
    G -- Yes (Target Achieved) --> H([END])
    G -- No (Thresholds Failed) --> I{Max Iterations?}
    I -- Yes --> H
    I -- No --> J[Recalibrator]
    I --> E
```

---


## File Structure

```text
 ┣━  prompts/              # System prompts for LLM tuning
 ┃ ┣━  parser_system_prompt.md
 ┃ ┣━  proposer_system_prompt.md
 ┃ ┗━  recalibration_system_prompt.md
 ┣━  simulation/           # Containes two algorithms, topology and workload generators and core simulation parameters and weights.
 ┃ ┣━  config.py           
 ┃ ┣━  generator.py        
 ┃ ┣━  greedy_fast.py        
 ┃ ┣━  heuristics.py        
 ┃ ┣━  main.py            
 ┃ ┣━  rollout_worker.py        
 ┃ ┣━  structs.py        
 ┃ ┗━  runs/               # Contains valid runs in csv format for warm-starting
 ┣━  test/                 # Test scripts for node verification (unit)
 ┃ ┣━  test_parser.py
 ┃ ┣━  test_proposer.py
 ┃ ┣━  test_recalibrator.py
 ┃ ┣━  test_feasibility.py
 ┃ ┣━  test_retriever.py
 ┃ ┣━  test_safety_guards.py
 ┃ ┗━  test_verifier.py
 ┣━  tools/
 ┃ ┗━  tools.py            # LangChain tool wrapping the simulator
 ┣━  utils/
 ┃ ┣━  retriever.py        # RAG functionality for querying historical runs
 ┃ ┗━  analytics.py        # Analyses runs and gathers information for later evaluation
 ┣━  state.py              # LangGraph AgentState definition
 ┣━  nodes/                # LangGraph Node implementations
 ┃ ┣━ __init__.py          # Public API re-exports
 ┃ ┣━ llm.py               # Shared ChatAnthropic client
 ┃ ┣━ parser.py            # Natural Language to structured intent
 ┃ ┣━ proposer.py          # Warm-start + micro-adjust weights
 ┃ ┣━ feasibility.py       # Deterministic historical pre-check + HITL apply
 ┃ ┣━ recalibrator.py      # Search + graduated threshold relaxation
 ┃ ┣━ schemas.py           # MetricType, IntentSchema, unit labels
 ┃ ┣━ verifier.py          # Deterministic SLA checks
 ┃ ┗━ weights.py           # Clip + normalization
 ┣━  app.py                # LangGraph orchestrator (Edges & Conditional Routing)
 ┗━  main.py               # Streamlit Frontend

```

---

## Core Components

### 1. State Management (`state.py`)

To solve the problem of "Agent Amnesia" during extended optimization loops, the state maintains a strict memory hierarchy:

* `original_parsed_intent`: Immutable: Stores the operator's exact initial goals once after intent_parser_node runs.
* `active_parsed_intent`: Mutable working memory. Hard-constraint **thresholds** may be loosened after evidence-based HITL or in-loop graduated relaxation.
* `recalibration_history`: An array tracking every failed weight combination and the exact simulation feedback, preventing the LLM from getting stuck in loops.

### 2. The Nodes (`nodes.py`)

* **Intent Parser (Inverse Translator):** Uses Pydantic structured output to translate human intents ("keep latency under 400ms") into a machine-readable JSON schema (with `primary_objective`, `hard_constraints`, `soft_preferences`, `non_relaxable_constraints` and `relaxation_order`). 
* **Feasibility:** Deterministic pre-check against historical runs. If the SLA is beyond best-known bounds, the operator is asked to continue, loosen thresholds, or enter a new intent. `app_rollout` may be auto-selected when it is jointly feasible and `best_fit` is not.
* **Weight Proposer (Warm-Start):** Addresses the "Rugged" Pareto Front. Because LLMs possess semantic bias (e.g., assuming higher latency weights always yield better latency), this node uses a RAG-oriented `retriever.py` to query historical successful runs. The LLM is structured to follow Pydantic baselines and makes micro-adjustments.
* **Verifier (Deterministic):** Written as pure Python logic (no LLM). It evaluates mean operator-unit metrics (`avg_latency`, `avg_cost`, `avg_security`) against the active constraints and catches capacity failures (`failures > 0`), ensuring zero arithmetic hallucination.
* **Recalibrator (Adaptive Search):** Analyzes negative feedback from the Verifier. It performs fine-tuned weight adjustments or, after at least three distinct failed weight vectors, may loosen a relaxable hard-constraint threshold and report the gap versus the original SLA.

### 3. The Tool Layer (`tools.py`)

Bridges the LangGraph agent and the underlying simulation.

* **Optimization:** Employs global caching for the network topology and application workload (`_CACHED_TOPO`, `_CACHED_APPS`). This ensures they are generated only once per session based on `config.SEED`, cutting out redundant computation during the agentic loop.
* **Safeguards:** Instead of allowing the LLM to write raw tool-call parameters, the orchestration extracts the proposed weights from the state and injects them into `config.py`.

### 4. Graph Orchestration (`app.py`)

Constructs two graphs: `parse_app` (Parser → Feasibility) and `search_app` (Proposer → Simulator → Verifier ⇄ Recalibrator). Streamlit pauses after Feasibility when historical evidence shows the SLA is infeasible. `route_verification` still enforces `max_iterations`.

---

## Interactive UI (`main.py`)

The project features a **Streamlit** frontend designed to visualize the internal reasoning (chain of thought) of the agent. Once we run an optimization with an intent as input, a JSON file is generated containing analytics about the process (fail/success, final weights, constraint relaxations, number of iterations, token usage, etc), which is saved in a `results-analytics` directory ready to be used for further evaluation. 

## Automated Testing

The repository includes a comprehensive pytest suite that is integrated into a GitHub Actions CI/CD pipeline to ensure metric verification, mathematical guarantees and LLM invoking are preserved on every commit.



### How to Run

Due to the `ProcessPoolExecutor` utilized in the simulation's `app_rollout`, the entry point is strictly protected with `multiprocessing.freeze_support()` to prevent recursive process spawning on Windows.

1. Clone the repo and install dependencies that might be needed.
2. Ensure you have your `ANTHROPIC_API_KEY` set in a `.env` file. (or similar key for API calls)
3. Run the Streamlit server:
```bash
streamlit run main.py
```
4. The UI allows you to input natural language intents, adjust iteration limits, and watch the real-time execution logs as the agent iterates through the Parser, Proposer, Simulator, Verifier, and Recalibrator nodes.

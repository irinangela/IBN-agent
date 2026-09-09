import os
import time
import streamlit as st
import multiprocessing

from app import parse_app, search_app
from nodes.feasibility import apply_operator_decision
from utils.analytics import save_run_analytics


def _initial_state(user_intent: str, max_iterations: int) -> dict:
    return {
        "user_intent": user_intent,
        "iteration_count": 0,
        "max_iterations": max_iterations,
        "recalibration_history": [],
        "total_input_tokens": 0,
        "total_output_tokens": 0,
        "use_rollout": False,
        "needs_operator": False,
        "operator_decision": {},
        "feasibility_report": {},
        "parse_failed": False,
        "parse_feedback": "",
    }


def _render_node(node_name: str, node_state: dict) -> None:
    with st.expander(f"Node Executed: {node_name}", expanded=True):
        if node_name == "Parser":
            if node_state.get("parse_failed"):
                st.warning(node_state.get("parse_feedback") or "")
            else:
                st.write("**Original Parsed Intent:**")
                st.json(node_state.get("original_parsed_intent"))

        elif node_name == "Feasibility":
            st.write("**Feasibility Pre-check:**")
            st.info(node_state.get("reasoning") or "")
            st.write("**Live algorithm:**")
            st.write(
                "app_rollout" if node_state.get("use_rollout") else "best_fit"
            )
            st.write("**Report:**")
            st.json(node_state.get("feasibility_report"))

        elif node_name == "Proposer":
            st.write("**Historical Warm-start Context:**")
            st.text(node_state.get("historical_context"))
            st.write("**LLM Reasoning:**")
            st.info(node_state.get("reasoning"))
            st.write("**Proposed Weights:**")
            st.json(node_state.get("current_weights"))

        elif node_name == "Simulator":
            st.write("**Simulator Metrics:**")
            st.json(node_state.get("simulation_results"))

        elif node_name == "Verifier":
            if node_state.get("constraints_satisfied"):
                st.success(node_state.get("verifier_feedback"))
            else:
                st.error(node_state.get("verifier_feedback"))

        elif node_name == "Recalibrator":
            st.write("**Recalibration Reasoning & Adjustments:**")
            st.info(node_state.get("reasoning"))
            st.write("**New Proposed Weights:**")
            st.json(node_state.get("current_weights"))
            st.write("**Updated Active Intent (Checking for relaxations):**")
            st.json(node_state.get("active_parsed_intent"))


def _replay_log() -> None:
    for node_name, node_state in st.session_state.get("execution_log") or []:
        _render_node(node_name, node_state)


def _stream_graph(graph, incoming_state: dict) -> dict:
    current_state = dict(incoming_state)
    for output in graph.stream(incoming_state):
        for node_name, node_state in output.items():
            current_state.update(node_state)
            st.session_state.execution_log.append((node_name, node_state))
            _render_node(node_name, node_state)
    return current_state


def _finish_run(final_state: dict) -> None:
    start_time = st.session_state.get("start_time") or time.time()
    filepath = save_run_analytics(final_state, time.time() - start_time)
    st.session_state.agent_state = final_state
    st.session_state.analytics_path = filepath
    st.session_state.figure_paths = []
    st.session_state.figure_error = ""
    try:
        from utils.plots import save_run_figures
        st.session_state.figure_paths = save_run_figures(filepath)
    except Exception as exc:
        st.session_state.figure_error = str(exc)
    st.session_state.phase = "done"


def _show_done_banner() -> None:
    final_state = st.session_state.get("agent_state") or {}
    filepath = st.session_state.get("analytics_path", "")
    is_success = final_state.get("constraints_satisfied", False)
    if is_success:
        st.success(
            f"Optimization Loop Complete! Target achieved. Analytics saved to `{filepath}`"
        )
    else:
        st.error(
            f"Optimization loop stopped due to max iterations "
            f"({final_state.get('iteration_count')}). "
            f"Target constraints were not met. Analytics saved to `{filepath}`"
        )
    tokens = (
        final_state.get("total_input_tokens", 0)
        + final_state.get("total_output_tokens", 0)
    )
    st.markdown(
        f"**Quick Stats:** used **{tokens} tokens**, over "
        f"**{final_state.get('iteration_count', 0)} iterations**."
    )
    if st.session_state.get("figure_error"):
        st.warning(f"Figures could not be generated: {st.session_state.figure_error}")
    pngs = [
        path for path in (st.session_state.get("figure_paths") or [])
        if path.endswith(".png") and os.path.isfile(path)
    ]
    if pngs:
        st.markdown("### Run figures")
        for path in pngs:
            st.caption(os.path.splitext(os.path.basename(path))[0])
            st.image(path)


def main():
    st.set_page_config(page_title="Intent-Based Optimization Agent", layout="wide")
    st.title("Agentic Intent-Based Network Optimizer")
    st.markdown(
        "Translates natural language into mathematical weights for edge-cloud placement."
    )

    if "phase" not in st.session_state:
        st.session_state.phase = "idle"
        st.session_state.execution_log = []
        st.session_state.agent_state = {}

    with st.sidebar:
        st.header("Agent Settings")
        max_iterations = st.slider(
            "Max Iterations", min_value=1, max_value=10, value=5
        )
        st.markdown("---")
        st.info(
            "The agent will autonomously navigate multiple weight combinations "
            "to satisfy your constraints, negotiate trade-offs and find optimal solutions."
        )

    user_intent = st.text_area(
        "Enter your network intent:",
        value="Optimize for low latency. Average latency must stay under 1000ms. Cost and security are secondary.",
        height=100,
    )

    phase = st.session_state.phase
    run_disabled = phase in ("parsing", "searching", "negotiating")
    if st.button("Run Agentic Optimization", type="primary", disabled=run_disabled):
        if not user_intent.strip():
            st.warning("Please enter an intent first.")
        else:
            st.session_state.phase = "parsing"
            st.session_state.pending_intent = user_intent
            st.session_state.max_iterations = max_iterations
            st.session_state.execution_log = []
            st.session_state.start_time = time.time()
            st.rerun()

    st.markdown("### Agent Execution Log")
    _replay_log()

    if phase == "negotiating":
        agent_state = st.session_state.agent_state or {}
        if agent_state.get("parse_failed"):
            revised_intent = st.text_area(
                "Revised intent",
                value=agent_state.get("user_intent", ""),
                key="parse_revised_intent",
            )
            if st.button("Submit revised intent"):
                if not (revised_intent or "").strip():
                    st.warning("Please enter a revised intent.")
                else:
                    st.session_state.phase = "parsing"
                    st.session_state.pending_intent = revised_intent.strip()
                    st.session_state.execution_log = []
                    st.rerun()
            return

        report = agent_state.get("feasibility_report") or {}
        st.warning(
            "No historical run on this seed meets all hard constraints. "
            "Choose how to proceed."
        )
        suggestions = report.get("suggested_relaxations") or []
        relaxable = [item for item in suggestions if item.get("relaxable")]

        options = ["Continue searching anyway"]
        if relaxable:
            options.append("Relax constraints to offered thresholds")
        options.append("Enter a new intent")

        choice = st.radio("How should the agent proceed?", options)
        thresholds = {}
        selected_algorithm = None
        revised_intent = None
        if choice.startswith("Relax"):
            st.caption(
                "Each offer is that algorithm's best-known value, while keeping "
                "the other constraints, plus a 5% safety margin. Search will run on "
                "the algorithm you pick. The app_rollout algorithm can usually meet "
                "a tighter threshold, but each simulation is much slower."
            )
            if len(relaxable) > 1:
                metric_labels = []
                by_metric_label = {}
                for item in relaxable:
                    label = (
                        f"{item['metric']}: requested {item.get('operator')} "
                        f"{item.get('requested')}"
                    )
                    metric_labels.append(label)
                    by_metric_label[label] = item
                selected_metric = st.radio(
                    "Which metric are you willing to relax?", metric_labels
                )
                item = by_metric_label[selected_metric]
            else:
                item = relaxable[0]

            algo_options = list(item.get("options") or [])
            if not algo_options and item.get("offered") is not None:
                algo_options = [{
                    "algorithm": "best_fit",
                    "offered": item["offered"],
                    "runtime_note": "",
                    "given": item.get("given") or [],
                }]
            option_labels = []
            by_option_label = {}
            for opt in algo_options:
                given = opt.get("given") or []
                keep = f" — keep {', '.join(given)}" if given else ""
                note = opt.get("runtime_note") or ""
                note_txt = f" — {note}" if note else ""
                label = (
                    f"{opt.get('algorithm')}: {item.get('operator')} "
                    f"{opt.get('offered')}{note_txt}{keep}"
                )
                option_labels.append(label)
                by_option_label[label] = opt
            if not option_labels:
                st.error("No algorithm-specific offers are available for this metric.")
            else:
                selected_option = st.radio(
                    "Which algorithm threshold do you want to use?",
                    option_labels,
                )
                chosen = by_option_label[selected_option]
                selected_algorithm = chosen.get("algorithm")
                thresholds[item["metric"]] = st.number_input(
                    (
                        f"New {item['metric']} threshold "
                        f"(requested {item.get('operator')} {item.get('requested')})"
                    ),
                    value=float(chosen.get("offered") or 0.0),
                    key=(
                        f"relax_{item['metric']}_"
                        f"{chosen.get('algorithm') or 'best_fit'}"
                    ),
                )
        elif choice.startswith("Enter"):
            revised_intent = st.text_area(
                "Revised intent",
                value=st.session_state.agent_state.get("user_intent", ""),
                key="revised_intent",
            )

        if st.button("Confirm"):
            if choice.startswith("Enter"):
                if not (revised_intent or "").strip():
                    st.warning("Please enter a revised intent.")
                else:
                    st.session_state.phase = "parsing"
                    st.session_state.pending_intent = revised_intent.strip()
                    st.session_state.execution_log = []
                    st.rerun()
            else:
                action = "relax" if choice.startswith("Relax") else "continue"
                decision = {"action": action}
                if action == "relax":
                    decision["thresholds"] = thresholds
                    if selected_algorithm:
                        decision["algorithm"] = selected_algorithm
                updated = apply_operator_decision(
                    st.session_state.agent_state, decision
                )
                st.session_state.agent_state = {
                    **st.session_state.agent_state,
                    **updated,
                }
                st.session_state.phase = "searching"
                st.rerun()

    if phase == "parsing":
        with st.spinner("Checking historical feasibility..."):
            incoming = _initial_state(
                st.session_state.pending_intent,
                st.session_state.get("max_iterations", max_iterations),
            )
            parsed_state = _stream_graph(parse_app, incoming)
            st.session_state.agent_state = parsed_state
            if parsed_state.get("needs_operator"):
                st.session_state.phase = "negotiating"
            else:
                st.session_state.phase = "searching"
            st.rerun()

    if phase == "searching":
        with st.spinner("Agent is reasoning and simulating..."):
            final_state = _stream_graph(search_app, st.session_state.agent_state)
            _finish_run(final_state)
            st.rerun()

    if phase == "done":
        _show_done_banner()


if __name__ == "__main__":
    import multiprocessing
    
    multiprocessing.freeze_support()
    main()

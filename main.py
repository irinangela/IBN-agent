import time
import streamlit as st
import multiprocessing
import json

from app import app as agent_app
from utils.analytics import save_run_analytics

def main():
    st.set_page_config(page_title="Intent-Based Optimization Agent", layout="wide")
    st.title("🌐 Agentic Intent-Based Network Optimizer")
    st.markdown("Translates natural language into mathematical weights for edge-cloud placement.")

    # Sidebar for settings
    with st.sidebar:
        st.header("Agent Settings")
        max_iterations = st.slider("Max Recalibration Iterations", min_value=1, max_value=10, value=5)
        st.markdown("---")
        st.info("The agent will autonomously navigate multiple weight combinations to satisfy your constraints, negotiate trade-offs and find optimal solutions.")

    # Input area
    user_intent = st.text_area(
        "Enter your network intent:",
        value="Minimize latency. I need cost to be strictly under 3.0. If you can't hit the cost, relax security.",
        height=100
    )

    if st.button("🚀 Run Agentic Optimization", type="primary"):
        if not user_intent.strip():
            st.warning("Please enter an intent first.")
            return

        initial_state = {
            "user_intent": user_intent,
            "iteration_count": 0,
            "max_iterations": max_iterations,
            "recalibration_history": [],
            "total_input_tokens": 0,
            "total_output_tokens": 0
        }

        current_state = initial_state.copy()

        st.markdown("### Agent Execution Log")
        log_container = st.container()

        start_time = time.time()

        with st.spinner("Agent is reasoning and simulating..."):
            for output in agent_app.stream(initial_state):
                
                for node_name, node_state in output.items():
                    current_state.update(node_state)
                    with log_container.expander(f"⚙️ Node Executed: {node_name}", expanded=True):
                        
                        if node_name == "Parser":
                            st.write("**Original Parsed Intent:**")
                            st.json(node_state.get("original_parsed_intent"))
                            
                        elif node_name == "Proposer":
                            st.write("**Historical RAG Context:**")
                            st.text(node_state.get("historical_context"))
                            st.write("**LLM Reasoning:**")
                            st.info(node_state.get("reasoning"))
                            st.write("**Proposed Weights:**")
                            st.json(node_state.get("current_weights"))
                            
                        elif node_name == "Simulator":
                            st.write("**Simulator Metrics:**")
                            st.json(node_state.get("simulation_results"))
                            
                        elif node_name == "Verifier":
                            success = node_state.get("constraints_satisfied")
                            if success:
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

        end_time = time.time()
        execution_time = end_time - start_time
        
        filepath = save_run_analytics(current_state, execution_time)

        is_success = current_state.get("constraints_satisfied", False)
        
        if is_success:
            st.success(f"Optimization Loop Complete! Target achieved. Analytics saved to `{filepath}`")
        else:
            st.error(f"Optimization loop stopped due to max iterations ({current_state.get('iteration_count')}). Target constraints were not met. Analytics saved to `{filepath}`")
        
        st.markdown(f"**Quick Stats:** Took **{execution_time:.2f}s**, used **{current_state.get('total_input_tokens', 0) + current_state.get('total_output_tokens', 0)} tokens**, over **{current_state.get('iteration_count', 0)} iterations**.")

if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
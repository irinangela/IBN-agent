import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
from nodes import intent_parser_node

def run_test(test_name: str, intent_text: str):
    print(f"\n--- Running Test: {test_name} ---")
    print(f"User Intent:\n{intent_text}\n")
    
    mock_state = {
        "user_intent": intent_text
    }
    
    try:
        result_state = intent_parser_node(mock_state)
        
        parsed_json = result_state.get("parsed_intent", {})
        print("Parsed JSON Output:")
        print(json.dumps(parsed_json, indent=2))
        
    except Exception as e:
        print(f"Error during parsing: {e}")

if __name__ == "__main__":
    # Test Case 0: Original prompt
    intent_0 = (
        "I need a placement that's mostly optimized for low latency, "
        "but I really can't have average latency go above 400ms. "
        "Try to keep security reasonably high too, but cost doesn't matter much to me. "
        "If you can't hit the latency target, security is the first thing I'm willing to give up on. "
    )

    # Test Case 1: Simple, cost is main objective, latency is a hard constraint, security can be relaxed
    intent_1 = (
        "Minimize cost at all costs. If latency ever goes above 200ms, "
        "then you can relax the security levels. "
    )
    
    # Test Case 2: Main objective is latency, cost has a hard constraint, and security has a soft preference
    intent_2 = (
        "I need a placement that's mostly optimized for low latency. We want the average latency "
        "to always be under 200ms. We also want security to be reasonably high, "
        "but cost shouldn't exceed 1000 dollars. If you can't hit the cost target, "
        "security is the first thing I'm willing to relax in terms of importance, not ever latency. "
    )
    
    # Test Case 3: Main objective is a balance between cost and security, but cost has a hard limit, and latency can be relaxed
    intent_3 = (
        "I want a balanced approach between cost and security. "
        "However, cost must absolutely be under 500 dollars. "
        "If you have to drop something, drop latency."
    )

    run_test("Test Case 0 (Original Prompt)", intent_0)
    run_test("Test Case 1 (Simple, cost is main objective, latency is a hard constraint, security can be relaxed)", intent_1)
    run_test("Test Case 2 (Main objective is latency, cost has a hard constraint, and security has a soft preference)", intent_2)
    run_test("Test Case 3 (Main objective is a balance between cost and security, but cost has a hard limit, and latency can be relaxed)", intent_3)
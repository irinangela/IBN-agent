import os
from dotenv import load_dotenv
load_dotenv()

from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver

from state import AgentState
from nodes import call_model, should_continue
from tools.tools import ALL_TOOLS

def build_graph():
    workflow = StateGraph(AgentState)   # Initialize Graph with State schema

    workflow.add_node("agent", call_model)
    workflow.add_node("tools", ToolNode(ALL_TOOLS)) # Add nodes

    workflow.set_entry_point("agent")   # Entry point START

    workflow.add_conditional_edges(     # Define conditional edges
        "agent",
        should_continue,
        {
            "tools": "tools",
            "end": END
        }
    )

    workflow.add_edge("tools", "agent") # Define direct edge from tools back to agent

    memory = MemorySaver()              # Memory saver to checkpoint the state after each node execution
    return workflow.compile(checkpointer=memory)

if __name__ == "__main__":
    app = build_graph()
    
    config = {"configurable": {"thread_id": "thesis_experiment_001"}}   # Specify a thread_id for memory checkpointing 
    
    print("Agent initialized. Type 'quit' to exit.\n")
    
    while True:
        user_query = input("User: ")
        if user_query.lower() in ['quit', 'exit']:
            break
            
        print("\n Agent is thinking ...\n")
        
        for event in app.stream({"messages": [{"role": "user", "content": user_query}]}, config, stream_mode="values"):
            last_message = event["messages"][-1]
            
            if hasattr(last_message, "tool_calls") and last_message.tool_calls:
                for tool in last_message.tool_calls:
                    print(f"[Calling Tool: {tool['name']} with args: {tool['args']}]")
            
            elif hasattr(last_message, "content") and last_message.content and last_message.type == "ai":
                print(f"\nAgent: {last_message.content}\n")
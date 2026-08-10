import streamlit as st
import os
import random
import time
from main import build_graph

st.set_page_config(page_title="")
st.title("")

if "agent" not in st.session_state:
    st.session_state.agent = build_graph()
    st.session_state.config = {"configurable": {"thread_id": "streamlit_ui_002"}}

if "messages" not in st.session_state:
    initial_greeting = random.choice([
        "Hello! I am your Assistant Agent. How can I help you today?",
    ])
    st.session_state.messages = [{"role": "assistant", "content": initial_greeting}]

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if msg.get("type") == "image":
            st.image(msg["content"]) 
        else:
            st.markdown(msg["content"])

if user_query := st.chat_input(""):
    st.session_state.messages.append({"role": "user", "type": "text", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    with st.chat_message("assistant"):
        with st.spinner("Analyzing data ..."):
            final_response = ""
            new_images = []

            for event in st.session_state.agent.stream(
                {"messages": [{"role": "user", "content": user_query}]}, 
                st.session_state.config, 
                stream_mode="values"
            ):
                last_message = event["messages"][-1]
                
                if last_message.type == "tool" and ".png" in str(last_message.content):
                    for word in last_message.content.split():
                        if ".png" in word:
                            clean_filename = word.strip("'.")
                            if clean_filename not in new_images:
                                new_images.append(clean_filename)

                if last_message.type == "ai" and last_message.content:
                    final_response = last_message.content
            
        message_placeholder = st.empty()
        typed_response = ""
        
        for chunk in final_response.split(' '):
            typed_response += chunk + " "
            time.sleep(0.02)
            message_placeholder.markdown(typed_response + "▌")
            
        message_placeholder.markdown(final_response)
        
        st.session_state.messages.append({"role": "assistant", "type": "text", "content": final_response})
        
        for img_file in new_images:
            if os.path.exists(img_file):
                st.image(img_file)
                st.session_state.messages.append({"role": "assistant", "type": "image", "content": img_file})
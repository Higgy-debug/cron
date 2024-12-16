import os
import streamlit as st
from io import StringIO
import re
import sys
from modules.history import ChatHistory
from modules.layout import Layout
from modules.utils import Utilities
from modules.sidebar import Sidebar

# Initialize session state
if "model" not in st.session_state:
    st.session_state["model"] = "your_default_model_name"

if "temperature" not in st.session_state:
    st.session_state["temperature"] = 0.7

if "history" not in st.session_state:
    st.session_state["history"] = []

# To be able to update the changes made to modules in localhost (press r)
def reload_module(module_name):
    import importlib
    import sys
    if module_name in sys.modules:
        importlib.reload(sys.modules[module_name])
    return sys.modules[module_name]

history_module = reload_module('modules.history')
layout_module = reload_module('modules.layout')
utils_module = reload_module('modules.utils')
sidebar_module = reload_module('modules.sidebar')

ChatHistory = history_module.ChatHistory
Layout = layout_module.Layout
Utilities = utils_module.Utilities
Sidebar = sidebar_module.Sidebar

st.set_page_config(layout="wide", page_icon="💬", page_title="Hirthick | Chat-Bot 🤖")

# Instantiate the main components
layout, sidebar, utils = Layout(), Sidebar(), Utilities()

layout.show_header("PDF, TXT, CSV")

uploaded_file = st.file_uploader("Upload a file", type=["pdf", "txt", "csv"])

if uploaded_file:
    # Configure the sidebar
    sidebar.show_options()

    # Initialize chat history
    history = ChatHistory()

    # Setup chatbot (replace with your "gguf" model setup)
    utils.setup_chatbot(uploaded_file, st.session_state["model"], st.session_state["temperature"])

    if st.session_state["ready"]:
        # Create containers for chat responses and user prompts
        response_container, prompt_container = st.container(), st.container()

        with prompt_container:
            # Display the prompt form
            is_ready, user_input = layout.prompt_form()

            # Initialize the chat history
            history.initialize(uploaded_file)

            # Reset the chat history if button clicked
            if st.session_state["reset_chat"]:
                history.reset(uploaded_file)

            if is_ready:
                # Update the chat history and display the chat messages
                history.append("user", user_input)

                old_stdout = sys.stdout
                sys.stdout = captured_output = StringIO()

                # Call your "gguf" model to generate response
                # output = your_gguf_model.generate_response(user_input)

                # For demonstration, let's assume output is generated
                output = st.session_state["chatbot"].conversational_chat(user_input)
                thoughts = captured_output.getvalue()
                cleaned_thoughts = re.sub(r'\x1b\[[0-9;]*[a-zA-Z]', '', thoughts)
                cleaned_thoughts = re.sub(r'\[1m>', '', cleaned_thoughts)

                history.append("assistant", output)

                # Display the agent's thoughts (if any)
                with st.expander("Display the agent's thoughts"):
                    st.write(cleaned_thoughts)

        history.generate_messages(response_container)
import sys

import streamlit as st

from config import Config
from chatbot import Chatbot


@st.cache_resource
def get_chatbot():
    if len(sys.argv) < 2:
        raise ValueError('Please provide a config file path as an positional argument')
    return Chatbot(Config(sys.argv[1]))


chatbot = get_chatbot()

st.title("Customer support")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# React to user input
if prompt := st.chat_input("What is up?"):
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        # st.markdown(response)
        response = st.write_stream(map(lambda ret: ret['answer'],
                                       filter(lambda ret: 'answer' in ret, chatbot.chat(prompt))))
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})
